/* HTTP client for the Python backend (see backend/main.py for the contract). */
(function (global) {
  "use strict";

  var config = global.AppConfig;

  function ApiError(code, message, stage, status) {
    this.name = "ApiError";
    this.code = code;
    this.message = message;
    this.stage = stage || null;
    this.status = status || 0;
  }
  ApiError.prototype = Object.create(Error.prototype);

  function url(path) {
    return config.apiBaseUrl + path;
  }

  function withTimeout(options, ms) {
    if (typeof AbortController === "undefined") { return { options: options, cancel: function () {} }; }
    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, ms);
    options.signal = controller.signal;
    return { options: options, cancel: function () { clearTimeout(timer); } };
  }

  function networkError(error) {
    if (error && error.name === "AbortError") {
      return new ApiError("timeout", "The assistant service took too long to answer.");
    }
    return new ApiError("network", "Cannot reach the assistant service at " + config.apiBaseUrl + ".");
  }

  function errorFromResponse(response) {
    return response.json().then(function (body) {
      var detail = body && (body.error || body.detail);
      if (detail && typeof detail === "object" && detail.message) {
        return new ApiError(detail.code || "server", detail.message, detail.stage, response.status);
      }
      var text = typeof detail === "string" ? detail : "The assistant service returned an error (" + response.status + ").";
      return new ApiError("server", text, null, response.status);
    }, function () {
      return new ApiError("server", "The assistant service returned an error (" + response.status + ").", null, response.status);
    });
  }

  function requestJson(path, options, timeoutMs) {
    var t = withTimeout(options || {}, timeoutMs || config.requestTimeoutMs);
    return fetch(url(path), t.options).then(function (response) {
      t.cancel();
      if (!response.ok) { return errorFromResponse(response).then(function (e) { throw e; }); }
      return response.json();
    }, function (error) {
      t.cancel();
      throw networkError(error);
    });
  }

  function postJson(path, body) {
    return requestJson(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  }

  /* Reads an NDJSON stream of stage events; resolves with the final result. */
  function readStream(response, onEvent) {
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    var finalResult = null;
    var finalError = null;

    function handleLine(line) {
      if (!line) { return; }
      var event = JSON.parse(line);
      if (event.stage === "result") { finalResult = event.result; }
      else if (event.stage === "error") {
        var e = event.error;
        finalError = new ApiError(e.code, e.message, e.stage, e.status);
      }
      onEvent(event);
    }

    function pump() {
      return reader.read().then(function (chunk) {
        if (chunk.done) {
          handleLine(buffer.trim());
          if (finalError) { throw finalError; }
          if (!finalResult) { throw new ApiError("server", "The response ended before a result arrived."); }
          return finalResult;
        }
        buffer += decoder.decode(chunk.value, { stream: true });
        var lines = buffer.split("\n");
        buffer = lines.pop();
        for (var i = 0; i < lines.length; i++) { handleLine(lines[i].trim()); }
        return pump();
      });
    }
    return pump();
  }

  /* fields: {branch, primary, session_id, ...}; audio: Blob or null; text: string or null. */
  function processTrial(fields, audio, text, onEvent) {
    var form = new FormData();
    var canStream = typeof TextDecoder !== "undefined" && typeof ReadableStream !== "undefined";
    Object.keys(fields).forEach(function (key) {
      if (fields[key] !== null && fields[key] !== undefined && fields[key] !== "") { form.append(key, fields[key]); }
    });
    if (audio) { form.append("audio", audio, "recording.wav"); }
    if (text) { form.append("text", text); }
    form.append("stream", canStream ? "true" : "false");

    var t = withTimeout({ method: "POST", body: form }, config.requestTimeoutMs);
    return fetch(url("/api/process"), t.options).then(function (response) {
      if (!response.ok) { return errorFromResponse(response).then(function (e) { throw e; }); }
      if (canStream && response.body && response.body.getReader) { return readStream(response, onEvent || function () {}); }
      return response.json();
    }, function (error) {
      throw networkError(error);
    }).then(function (result) {
      t.cancel();
      return result;
    }, function (error) {
      t.cancel();
      throw error;
    });
  }

  global.Api = {
    ApiError: ApiError,
    health: function () { return requestJson("/api/health", {}, config.healthTimeoutMs); },
    intents: function () { return requestJson("/api/intents"); },
    courses: function () { return requestJson("/api/courses"); },
    execute: function (intent, slots) { return postJson("/api/execute", { intent: intent, slots: slots || {} }); },
    processTrial: processTrial,
    exportUrl: function () { return url("/api/research/export"); }
  };
})(window);
