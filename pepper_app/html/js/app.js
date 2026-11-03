/* Application controller: state machine and wiring.
   States: idle -> listening -> uploading -> transcribing -> classifying -> executing -> success | error
   Three back ends for one UI:
     live   - Python backend over HTTP (browser, Vercel, Choregraphe simulator)
     demo   - simulated responses when no backend is reachable
     pepper - the robot records and calls the backend; the tablet mirrors state */
(function (global) {
  "use strict";

  var config = global.AppConfig;
  var UI = global.UI;
  var Api = global.Api;

  var INTENTS = ["greeting", "help", "ask_time", "ask_current_class", "ask_next_class", "ask_lecturer",
    "ask_course_time", "ask_course_venue", "ask_day_schedule", "list_courses", "ask_current_semester",
    "open_self_service", "OOS"];
  var BRANCH_HELP = {
    A: "A: isiZulu → English translation → English classifier.",
    B: "B: isiZulu → InkubaLM-0.4B classifies the intent directly.",
    both: "Both branches receive the identical transcript; Pepper executes the selected one."
  };

  var app = {
    state: "idle",
    backend: "checking",       // live | demo | pepper | checking
    recorder: null,
    bridge: null,
    branch: "A",
    timerId: null,
    autoStopId: null,
    trials: [],
    lastTrueIntent: null
  };

  function byId(id) { return document.getElementById(id); }

  function now() {
    return global.performance && performance.now ? performance.now() : Date.now();
  }

  function newSessionId() {
    var d = new Date();
    function pad(n) { return (n < 10 ? "0" : "") + n; }
    return "S" + d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + "-" + pad(d.getHours()) + pad(d.getMinutes());
  }

  /* ---------- State ---------- */

  function setState(state, options) {
    options = options || {};
    app.state = state;
    UI.setState(state, options.label);
    UI.setSteps(state, options);
  }

  function isBusy() {
    return ["uploading", "transcribing", "classifying", "executing"].indexOf(app.state) >= 0;
  }

  function studentMessage(error) {
    var code = error && error.code;
    if (code === "asr_unavailable") { return "Speech recognition is not available right now."; }
    if (code === "empty_transcript") { return "I didn't catch that. Please try again."; }
    if (code === "too_short") { return "That was too short. Hold on a moment longer before tapping stop."; }
    if (code === "network" || code === "timeout") { return "I can't reach the assistant service right now."; }
    return (error && error.message) || "Something went wrong. Please try again.";
  }

  function showError(error) {
    var stage = error && error.stage;
    var label = error && error.code === "denied" ? "Microphone blocked" : "Something went wrong";
    setState("error", { label: label, failedStage: stage || (app.state === "listening" ? "listening" : app.state) });
    var message = config.mode === "research" && error && error.message ? error.message : studentMessage(error);
    UI.setResponse(config.mode === "research" ? message : studentMessage(error));
    if (config.mode === "research" && error && error.code) { UI.setHint("Error code: " + error.code); }
    else { UI.setHint("Tap the microphone to try again."); }
  }

  /* ---------- Research controls ---------- */

  function researchFields() {
    var research = config.mode === "research";
    var fields = {
      branch: research ? app.branch : "A",
      primary: research && app.branch === "both" ? byId("primarySelect").value : null,
      session_id: research ? byId("sessionId").value.trim() : null,
      log: research && byId("logTrials").checked ? "true" : "false",
      save_audio: research && byId("saveAudio").checked ? "true" : "false"
    };
    if (research) {
      fields.reference_transcript = byId("refTranscript").value.trim();
      fields.true_intent = byId("trueIntent").value;
      fields.true_course = byId("trueCourse").value;
      fields.true_weekday = byId("trueWeekday").value;
    }
    return fields;
  }

  function clearTruth() {
    if (!byId("clearTruth").checked) { return; }
    ["refTranscript", "trueIntent", "trueCourse", "trueWeekday"].forEach(function (id) { byId(id).value = ""; });
  }

  function setBranch(value) {
    app.branch = value;
    var buttons = byId("branchControl").getElementsByTagName("button");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute("aria-checked", buttons[i].getAttribute("data-value") === value ? "true" : "false");
    }
    byId("primaryField").hidden = value !== "both";
    byId("branchHelp").textContent = BRANCH_HELP[value];
  }

  function setMode(mode) {
    config.setMode(mode);
    UI.setMode(mode);
  }

  /* ---------- Trial ---------- */

  function handleEvent(event, hadAudio) {
    switch (event.stage) {
      case "received":
        setState(hadAudio ? "transcribing" : "classifying", { skipped: hadAudio ? [] : ["listening", "transcribing"] });
        break;
      case "transcribing":
        setState("transcribing");
        break;
      case "transcribed":
        UI.setTranscript(event.transcript);
        break;
      case "classifying":
        setState("classifying", { skipped: hadAudio ? [] : ["listening", "transcribing"] });
        break;
      case "executing":
        setState("executing", { skipped: hadAudio ? [] : ["listening", "transcribing"] });
        break;
    }
  }

  function speak(text) {
    if (!byId("speakAloud").checked || !global.speechSynthesis || !global.SpeechSynthesisUtterance) { return; }
    global.speechSynthesis.cancel();
    var utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-ZA";
    global.speechSynthesis.speak(utterance);
  }

  function showResult(result, extras) {
    setState("success", { skipped: result.source === "text" ? ["listening", "transcribing"] : [] });
    UI.setTranscript(result.transcript);
    UI.setResponse(result.response);
    UI.renderResult(result);
    UI.setHint(result.status === "needs_slot" ? "Tap the microphone and say the course name." : "Tap the microphone to ask something else.");
    if (typeof result.sample_data === "boolean") { UI.setSampleNotice(result.sample_data); }
    speak(result.response);

    if (config.mode === "research" || result.logged) {
      app.trials.push({
        time: new Date().toLocaleTimeString(),
        result: result,
        trueIntent: extras.trueIntent || null,
        reference: extras.reference || null,
        clientMs: extras.clientMs
      });
      UI.renderTrial(result, extras);
      UI.renderHistory(app.trials);
    }
  }

  function runTrial(audio, text, recording) {
    var fields = researchFields();
    var extras = {
      trueIntent: fields.true_intent || null,
      reference: fields.reference_transcript || null,
      converted: recording ? recording.converted : null
    };
    var hadAudio = !!audio;
    var started = now();
    setState("uploading", { skipped: hadAudio ? [] : ["listening", "transcribing"] });
    UI.setResponse("One moment while I work that out…");
    UI.setHint("Please wait…");

    var onEvent = function (event) { handleEvent(event, hadAudio); };
    var pending = app.backend === "demo"
      ? global.DemoPipeline.run(hadAudio, text, onEvent)
      : Api.processTrial(fields, audio, text, onEvent);

    return pending.then(function (result) {
      extras.clientMs = app.backend === "demo" ? null : Math.round(now() - started);
      showResult(result, extras);
      if (config.mode === "research") { clearTruth(); }
    }, function (error) {
      showError(error);
      if (error && error.code === "network") { checkBackend(); }
    });
  }

  /* ---------- Browser recording ---------- */

  function stopTimers() {
    clearInterval(app.timerId);
    clearTimeout(app.autoStopId);
    app.timerId = app.autoStopId = null;
  }

  function startListening() {
    if (!app.recorder) { app.recorder = new global.BrowserRecorder(UI.setLevel); }
    UI.resetResult();
    UI.setTranscript(null);
    UI.setHint("Starting the microphone…");
    app.recorder.start().then(function () {
      setState("listening");
      UI.setResponse("I'm listening. Speak in isiZulu.");
      UI.setHint("Tap again when you have finished speaking.");
      UI.setTimer(0, config.maxRecordingMs);
      app.timerId = setInterval(function () { UI.setTimer(app.recorder.elapsedMs(), config.maxRecordingMs); }, 250);
      app.autoStopId = setTimeout(stopListening, config.maxRecordingMs);
    }, function (error) {
      showError(error);
    });
  }

  function stopListening() {
    if (app.state !== "listening") { return; }
    stopTimers();
    var tooShort = app.recorder.elapsedMs() < config.minRecordingMs;
    setState("uploading");
    app.recorder.stop().then(function (recording) {
      if (tooShort) {
        showError({ code: "too_short", message: "Recording was shorter than " + config.minRecordingMs + " ms.", stage: "listening" });
        return;
      }
      return runTrial(recording.blob, null, recording);
    }, function (error) {
      showError(error);
    });
  }

  /* ---------- Pepper runtime ---------- */

  function startPepper() {
    app.bridge = new global.PepperBridge();
    UI.setConnection("checking", "Connecting to Pepper…");
    app.bridge.connect().then(function () {
      app.backend = "pepper";
      UI.setConnection("pepper", "Pepper connected");
      app.bridge.on("state", function (state) {
        if (state.indexOf("error:") === 0) {
          showError({ code: "pepper", message: state.slice(6) });
        } else if (UI.STATE_LABELS[state]) {
          if (state === "listening") { UI.resetResult(); UI.setTranscript(null); UI.setResponse("I'm listening. Speak in isiZulu."); }
          setState(state);
        }
      });
      app.bridge.on("result", function (result) {
        if (result && result.error) { showError(result.error); return; }
        showResult(result, { clientMs: null });
      });
    }, function (error) {
      UI.setConnection("demo", "Pepper not connected");
      app.backend = "demo";
      if (config.mode === "research") { UI.setResponse(error.message); }
    });
  }

  /* ---------- Backend health ---------- */

  function checkBackend() {
    if (config.forceDemo) {
      app.backend = "demo";
      UI.setConnection("demo", "Demo mode");
      UI.renderEngines(null, "Demo mode forced by ?demo=1");
      return Promise.resolve(null);
    }
    UI.setConnection("checking", "Connecting…");
    return Api.health().then(function (health) {
      app.backend = "live";
      UI.setConnection("connected", "Assistant online");
      UI.renderEngines(health);
      UI.setSampleNotice(health.sample_timetable);
      loadResearchOptions();
      return health;
    }, function (error) {
      app.backend = "demo";
      UI.setConnection("demo", "Demo mode");
      UI.renderEngines(null, error.message);
      return null;
    });
  }

  function loadResearchOptions() {
    Api.intents().then(function (data) {
      UI.fillSelect(byId("trueIntent"), data.intents.map(function (i) { return { value: i.label, label: i.label }; }));
    }, function () {});
    Api.courses().then(function (data) {
      UI.fillSelect(byId("trueCourse"), data.courses.map(function (c) { return { value: c.code, label: c.code + " " + c.name }; }));
    }, function () {});
  }

  /* ---------- Wiring ---------- */

  function onMicClick() {
    if (isBusy()) { return; }
    if (app.backend === "pepper") {
      if (app.state === "listening") { app.bridge.stopListening(); }
      else { app.bridge.startListening({ branch: researchFields().branch, session_id: researchFields().session_id }); }
      return;
    }
    if (app.state === "listening") { stopListening(); }
    else { startListening(); }
  }

  /* Press and hold the crest for 1.5 s to toggle research mode. */
  function wireLongPress(node, ms, callback) {
    var timer = null;
    function start(event) {
      if (event.type === "mousedown" && event.button !== 0) { return; }
      cancel();
      timer = setTimeout(function () { timer = null; callback(); }, ms);
    }
    function cancel() { if (timer) { clearTimeout(timer); timer = null; } }
    node.addEventListener("touchstart", start, { passive: true });
    node.addEventListener("mousedown", start);
    ["touchend", "touchcancel", "mouseup", "mouseleave"].forEach(function (type) { node.addEventListener(type, cancel); });
    node.addEventListener("contextmenu", function (e) { e.preventDefault(); });
  }

  function downloadJson(name, data) {
    var blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = name;
    document.body.appendChild(link);
    link.click();
    setTimeout(function () { URL.revokeObjectURL(link.href); link.parentNode.removeChild(link); }, 0);
  }

  function wire() {
    byId("micButton").addEventListener("click", onMicClick);
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && app.state === "listening") { stopListening(); }
    });

    wireLongPress(byId("brand"), 1500, function () {
      setMode(config.mode === "research" ? "student" : "research");
    });
    byId("exitResearch").addEventListener("click", function () { setMode("student"); });

    var buttons = byId("branchControl").getElementsByTagName("button");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener("click", function () { setBranch(this.getAttribute("data-value")); });
    }

    byId("sessionId").value = config.read("isizulu.session") || newSessionId();
    byId("sessionId").addEventListener("change", function () { config.write("isizulu.session", this.value.trim()); });
    byId("newSession").addEventListener("click", function () {
      byId("sessionId").value = newSessionId();
      config.write("isizulu.session", byId("sessionId").value);
      app.trials = [];
      UI.renderHistory(app.trials);
    });

    byId("textForm").addEventListener("submit", function (event) {
      event.preventDefault();
      var text = byId("textInput").value.trim();
      if (!text || isBusy() || app.state === "listening") { return; }
      UI.resetResult();
      UI.setTranscript(text);
      runTrial(null, text, null);
    });

    byId("apiUrl").value = config.apiBaseUrl;
    byId("apiForm").addEventListener("submit", function (event) {
      event.preventDefault();
      config.setApiBaseUrl(byId("apiUrl").value.trim());
      byId("exportCsv").href = Api.exportUrl();
      checkBackend();
    });

    byId("exportCsv").href = Api.exportUrl();
    byId("exportJson").addEventListener("click", function () {
      downloadJson((byId("sessionId").value || "session") + ".json", app.trials);
    });

    var intents = byId("trueIntent");
    UI.fillSelect(intents, INTENTS.map(function (label) { return { value: label, label: label }; }));
  }

  function start() {
    UI.init();
    wire();
    setBranch("A");
    setMode(config.mode);
    setState("idle");
    if (config.runtime === "pepper") {
      startPepper();
    } else {
      checkBackend();
      var problem = global.BrowserRecorder.supportProblem();
      if (problem) { UI.setHint(problem.message); }
    }
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); }
  else { start(); }

  global.App = app;
})(window);
