/* Demo pipeline, used only when no backend is reachable (or ?demo=1).
   It exercises the UI states with simulated timing. Its output is marked
   demo: true, carries no latency figures, and does not claim to have
   understood anything. */
(function (global) {
  "use strict";

  var STEPS = [
    { delay: 350, event: { stage: "received" } },
    { delay: 650, event: { stage: "transcribing" } },
    { delay: 700, event: { stage: "transcribed", transcript: null } },
    { delay: 450, event: { stage: "classifying" } },
    { delay: 450, event: { stage: "executing" } }
  ];

  function demoResult(hadAudio, text) {
    var transcript = text || (hadAudio ? "(demo mode: speech is not transcribed)" : "");
    return {
      demo: true,
      session_id: "demo",
      trial_id: "demo-" + Date.now(),
      source: hadAudio ? "audio" : "text",
      transcript: transcript,
      branch: null,
      intent: "help",
      slots: {},
      confidence: null,
      translation: null,
      status: "ok",
      response: "I am in demo mode, so I cannot understand speech yet. When the assistant service is connected, " +
        "I can tell you about Honours exams, the academic calendar, School contacts and how to find places on campus.",
      action: { type: "say" },
      display: {
        kind: "help",
        items: ["the time", "the current or next exam", "when and where a course's exam is", "what is on a particular day",
          "the list of Honours courses", "where we are in the academic year", "who to contact in the School",
          "how to find a building or room", "opening Student Self-Service"]
      },
      branches: {},
      agreement: null,
      latency: { asr_ms: null, translation_ms: null, classification_ms: null, execution_ms: null, total_ms: null },
      logged: false
    };
  }

  function run(hadAudio, text, onEvent) {
    var result = demoResult(hadAudio, text);
    return new Promise(function (resolve) {
      var index = 0;
      function next() {
        if (index >= STEPS.length) { resolve(result); return; }
        var step = STEPS[index++];
        setTimeout(function () {
          var event = {};
          for (var key in step.event) { event[key] = step.event[key]; }
          if (event.stage === "transcribed") { event.transcript = result.transcript; }
          if (event.stage === "transcribing" && !hadAudio) { next(); return; }
          onEvent(event);
          next();
        }, step.delay);
      }
      next();
    });
  }

  global.DemoPipeline = { run: run };
})(window);
