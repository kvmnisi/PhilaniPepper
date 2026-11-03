/* Browser microphone recording with MediaRecorder.
   Recordings are converted to 16 kHz mono 16-bit WAV in the browser (Web
   Audio API) so the backend receives the format Whisper expects, whatever
   codec the browser recorded in. If conversion fails the original blob is
   sent and the backend falls back to ffmpeg. */
(function (global) {
  "use strict";

  var TARGET_RATE = 16000;

  function RecorderError(code, message) {
    this.name = "RecorderError";
    this.code = code;
    this.message = message;
  }
  RecorderError.prototype = Object.create(Error.prototype);

  function supportProblem() {
    if (global.isSecureContext === false) {
      return new RecorderError("insecure", "The microphone only works over HTTPS or on localhost.");
    }
    if (!global.navigator.mediaDevices || !global.navigator.mediaDevices.getUserMedia || !global.MediaRecorder) {
      return new RecorderError("unsupported", "This browser cannot record audio. Try a recent Chrome, Edge, Firefox or Safari.");
    }
    return null;
  }

  function pickMimeType() {
    var types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
    for (var i = 0; i < types.length; i++) {
      if (global.MediaRecorder.isTypeSupported && global.MediaRecorder.isTypeSupported(types[i])) { return types[i]; }
    }
    return "";
  }

  function permissionError(error) {
    var name = error && error.name;
    if (name === "NotAllowedError" || name === "SecurityError") {
      return new RecorderError("denied", "Microphone access was blocked. Allow the microphone for this page and try again.");
    }
    if (name === "NotFoundError" || name === "OverconstrainedError") {
      return new RecorderError("no-device", "No microphone was found.");
    }
    if (name === "NotReadableError") {
      return new RecorderError("busy", "The microphone is being used by another application.");
    }
    return new RecorderError("unknown", "The microphone could not be started.");
  }

  function readBlob(blob) {
    if (blob.arrayBuffer) { return blob.arrayBuffer(); }
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () { resolve(reader.result); };
      reader.onerror = function () { reject(reader.error); };
      reader.readAsArrayBuffer(blob);
    });
  }

  function encodeWav(samples, rate) {
    var buffer = new ArrayBuffer(44 + samples.length * 2);
    var view = new DataView(buffer);
    function text(offset, value) {
      for (var i = 0; i < value.length; i++) { view.setUint8(offset + i, value.charCodeAt(i)); }
    }
    text(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    text(8, "WAVE");
    text(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);          // PCM
    view.setUint16(22, 1, true);          // mono
    view.setUint32(24, rate, true);
    view.setUint32(28, rate * 2, true);   // byte rate
    view.setUint16(32, 2, true);          // block align
    view.setUint16(34, 16, true);         // bits per sample
    text(36, "data");
    view.setUint32(40, samples.length * 2, true);
    for (var i = 0, offset = 44; i < samples.length; i++, offset += 2) {
      var s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([view], { type: "audio/wav" });
  }

  function toWav16k(blob) {
    var Ctx = global.AudioContext || global.webkitAudioContext;
    var Offline = global.OfflineAudioContext || global.webkitOfflineAudioContext;
    if (!Ctx || !Offline) { return Promise.reject(new Error("Web Audio unavailable")); }
    return readBlob(blob).then(function (data) {
      var ctx = new Ctx();
      return new Promise(function (resolve, reject) {
        ctx.decodeAudioData(data, resolve, reject);
      }).then(function (decoded) {
        if (ctx.close) { ctx.close(); }
        var length = Math.max(1, Math.ceil(decoded.duration * TARGET_RATE));
        var offline = new Offline(1, length, TARGET_RATE);
        var source = offline.createBufferSource();
        source.buffer = decoded;
        source.connect(offline.destination);
        source.start(0);
        return new Promise(function (resolve, reject) {
          offline.oncomplete = function (event) { resolve(event.renderedBuffer); };
          var started = offline.startRendering();
          if (started && started.then) { started.then(resolve, reject); }
        });
      }).then(function (rendered) {
        return encodeWav(rendered.getChannelData(0), TARGET_RATE);
      });
    });
  }

  /* onLevel(0..1) is called while recording for a visual level meter. */
  function BrowserRecorder(onLevel) {
    this.onLevel = onLevel || function () {};
    this.stream = null;
    this.recorder = null;
    this.chunks = [];
    this.startedAt = 0;
    this.meter = null;
  }

  BrowserRecorder.supportProblem = supportProblem;

  BrowserRecorder.prototype.start = function () {
    var self = this;
    var problem = supportProblem();
    if (problem) { return Promise.reject(problem); }
    return global.navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
    }).then(function (stream) {
      self.stream = stream;
      self.chunks = [];
      var mime = pickMimeType();
      self.recorder = mime ? new global.MediaRecorder(stream, { mimeType: mime }) : new global.MediaRecorder(stream);
      self.recorder.ondataavailable = function (event) {
        if (event.data && event.data.size) { self.chunks.push(event.data); }
      };
      self.recorder.start();
      self.startedAt = Date.now();
      self.startMeter(stream);
    }, function (error) {
      throw permissionError(error);
    });
  };

  BrowserRecorder.prototype.startMeter = function (stream) {
    var Ctx = global.AudioContext || global.webkitAudioContext;
    if (!Ctx || !global.requestAnimationFrame) { return; }
    var self = this;
    try {
      var ctx = new Ctx();
      var analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      ctx.createMediaStreamSource(stream).connect(analyser);
      var data = new Uint8Array(analyser.fftSize);
      var meter = { ctx: ctx, running: true };
      var tick = function () {
        if (!meter.running) { return; }
        analyser.getByteTimeDomainData(data);
        var sum = 0;
        for (var i = 0; i < data.length; i++) { var v = (data[i] - 128) / 128; sum += v * v; }
        self.onLevel(Math.min(1, Math.sqrt(sum / data.length) * 4));
        global.requestAnimationFrame(tick);
      };
      this.meter = meter;
      tick();
    } catch (e) { this.meter = null; }
  };

  BrowserRecorder.prototype.release = function () {
    if (this.meter) {
      this.meter.running = false;
      if (this.meter.ctx.close) { this.meter.ctx.close(); }
      this.meter = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(function (track) { track.stop(); });
      this.stream = null;
    }
    this.onLevel(0);
  };

  BrowserRecorder.prototype.elapsedMs = function () {
    return this.startedAt ? Date.now() - this.startedAt : 0;
  };

  /* Resolves with {blob, durationMs, converted}. */
  BrowserRecorder.prototype.stop = function () {
    var self = this;
    var recorder = this.recorder;
    if (!recorder) { return Promise.reject(new RecorderError("idle", "Not recording.")); }
    var durationMs = this.elapsedMs();
    return new Promise(function (resolve) {
      recorder.onstop = function () { resolve(); };
      recorder.stop();
    }).then(function () {
      self.release();
      self.recorder = null;
      var original = new Blob(self.chunks, { type: recorder.mimeType || "audio/webm" });
      return toWav16k(original).then(function (wav) {
        return { blob: wav, durationMs: durationMs, converted: true };
      }, function () {
        return { blob: original, durationMs: durationMs, converted: false };
      });
    });
  };

  BrowserRecorder.prototype.cancel = function () {
    if (this.recorder && this.recorder.state !== "inactive") {
      this.recorder.onstop = null;
      this.recorder.stop();
    }
    this.recorder = null;
    this.release();
  };

  global.BrowserRecorder = BrowserRecorder;
  global.RecorderError = RecorderError;
})(window);
