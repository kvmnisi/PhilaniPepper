/* DOM rendering. No application logic lives here. */
(function (global) {
  "use strict";

  var doc = global.document;
  var STEP_ORDER = ["listening", "uploading", "transcribing", "classifying", "executing"];
  var STATE_LABELS = {
    idle: "Tap to speak isiZulu",
    listening: "Listening… tap to stop",
    uploading: "Sending…",
    transcribing: "Transcribing…",
    classifying: "Understanding…",
    executing: "Responding…",
    success: "Tap to ask again",
    error: "Something went wrong"
  };
  var MIC_LABELS = { idle: "Start speaking", listening: "Stop and send", success: "Start speaking", error: "Start speaking" };
  var BUSY = { uploading: 1, transcribing: 1, classifying: 1, executing: 1 };
  var STAND_INS = { "dev-rules": "dev-rules (stand-in)", "nllb-rules": "NLLB + rules (stand-in classifier)" };

  var $ = {};

  function byId(id) { return doc.getElementById(id); }

  /* el("p", {className: "x"}, ["text", childNode]) */
  function el(tag, attrs, children) {
    var node = doc.createElement(tag);
    if (attrs) {
      for (var key in attrs) {
        if (!attrs.hasOwnProperty(key) || attrs[key] === null || attrs[key] === undefined) { continue; }
        if (key === "className") { node.className = attrs[key]; }
        else if (key === "text") { node.textContent = attrs[key]; }
        else { node.setAttribute(key, attrs[key]); }
      }
    }
    (children || []).forEach(function (child) {
      if (child === null || child === undefined) { return; }
      node.appendChild(typeof child === "string" ? doc.createTextNode(child) : child);
    });
    return node;
  }

  function clear(node) {
    while (node.firstChild) { node.removeChild(node.firstChild); }
  }

  function capitalise(word) {
    return word ? word.charAt(0).toUpperCase() + word.slice(1) : "";
  }

  function ms(value) {
    if (value === null || value === undefined) { return "—"; }
    return (value < 10 ? value.toFixed(1) : Math.round(value)) + " ms";
  }

  function pct(value) {
    if (value === null || value === undefined) { return "—"; }
    return (value * 100).toFixed(1) + "%";
  }

  function slotsText(slots) {
    var keys = Object.keys(slots || {});
    if (!keys.length) { return "—"; }
    return keys.map(function (k) { return k + "=" + slots[k]; }).join(", ");
  }

  function init() {
    ["stateLabel", "micButton", "micHalo", "micHint", "timer", "steps", "transcript", "response", "resultCard",
      "tips", "connection", "demoBanner", "research", "engines", "devWarning", "trialDetail", "history",
      "sampleNote", "trueIntent", "trueCourse", "primaryField", "branchHelp"].forEach(function (id) { $[id] = byId(id); });
  }

  function setState(state, label) {
    doc.body.setAttribute("data-state", state);
    $.stateLabel.textContent = label || STATE_LABELS[state] || "";
    var busy = !!BUSY[state];
    $.micButton.disabled = busy;
    $.micButton.setAttribute("aria-pressed", state === "listening" ? "true" : "false");
    $.micButton.setAttribute("aria-label", busy ? "Working" : (MIC_LABELS[state] || "Start speaking"));
    $.micButton.setAttribute("aria-busy", busy ? "true" : "false");
  }

  /* skipped: list of steps not used in this trial (e.g. transcribing for text). */
  function setSteps(state, options) {
    options = options || {};
    var items = $.steps.getElementsByTagName("li");
    var active = STEP_ORDER.indexOf(state);
    var failedAt = state === "error" ? STEP_ORDER.indexOf(options.failedStage || "") : -1;
    for (var i = 0; i < items.length; i++) {
      var status = "pending";
      if (state === "success") { status = "done"; }
      else if (state === "error") { status = i < failedAt ? "done" : (i === failedAt ? "failed" : "pending"); }
      else if (active >= 0) { status = i < active ? "done" : (i === active ? "active" : "pending"); }
      if (options.skipped && options.skipped.indexOf(STEP_ORDER[i]) >= 0 && status === "done") { status = "pending"; }
      items[i].setAttribute("data-status", status);
    }
  }

  function setTimer(elapsedMs, maxMs) {
    var remaining = Math.max(0, Math.ceil((maxMs - elapsedMs) / 1000));
    var seconds = Math.floor(elapsedMs / 1000);
    $.timer.textContent = "0:" + (seconds < 10 ? "0" : "") + seconds + "  ·  " + remaining + "s left";
  }

  function setLevel(level) {
    $.micHalo.style.transform = "scale(" + (1 + level * 0.5).toFixed(3) + ")";
  }

  function setTranscript(text) {
    if (text) {
      $.transcript.textContent = text;
      $.transcript.className = "transcript";
    } else {
      $.transcript.textContent = "Your words will appear here.";
      $.transcript.className = "transcript is-empty";
    }
  }

  function setResponse(text) {
    $.response.textContent = text;
  }

  function setHint(text) {
    $.micHint.textContent = text;
  }

  function setConnection(status, text) {
    $.connection.setAttribute("data-status", status);
    $.connection.textContent = text;
    $.demoBanner.hidden = status !== "demo";
  }

  function setSampleNotice(isSample) {
    $.sampleNote.hidden = !isSample;
  }

  function setMode(mode) {
    doc.body.setAttribute("data-mode", mode);
    $.research.hidden = mode !== "research";
  }

  /* ---------- Result card ---------- */

  function sessionsTable(sessions, highlight) {
    var body = el("tbody", null, sessions.map(function (s) {
      var isHighlight = highlight && highlight.day === s.day && highlight.start === s.start;
      return el("tr", { className: isHighlight ? "is-highlight" : null }, [
        el("td", { text: capitalise(s.day) }),
        el("td", { className: "mono", text: s.start + "–" + s.end }),
        el("td", { text: s.venue || "To be confirmed" })
      ]);
    }));
    return el("table", { className: "table" }, [
      el("thead", null, [el("tr", null, [el("th", { text: "Day" }), el("th", { text: "Time" }), el("th", { text: "Venue" })])]),
      body
    ]);
  }

  function renderDisplay(display) {
    switch (display.kind) {
      case "course":
        return [
          el("div", { className: "result-head" }, [
            el("span", { className: "code-badge", text: display.code }),
            el("h3", { className: "result-title", text: display.name })
          ]),
          display.lecturer ? el("p", { className: "result-meta" }, ["Lecturer: ", el("strong", { text: display.lecturer })]) : null,
          display.sessions && display.sessions.length ? sessionsTable(display.sessions, display.highlight) : null
        ];
      case "schedule":
        if (!display.rows.length) {
          return [el("h3", { className: "result-title", text: capitalise(display.weekday) }), el("p", { className: "result-meta", text: "No classes scheduled." })];
        }
        return [
          el("h3", { className: "result-title", text: capitalise(display.weekday) }),
          el("table", { className: "table" }, [
            el("thead", null, [el("tr", null, [el("th", { text: "Time" }), el("th", { text: "Course" }), el("th", { text: "Venue" })])]),
            el("tbody", null, display.rows.map(function (r) {
              return el("tr", null, [
                el("td", { className: "mono", text: r.start + "–" + r.end }),
                el("td", null, [el("span", { className: "code-badge", text: r.code }), r.name]),
                el("td", { text: r.venue || "To be confirmed" })
              ]);
            }))
          ])
        ];
      case "course_list":
        return [
          el("h3", { className: "result-title", text: "Courses" }),
          el("ul", { className: "course-list" }, display.courses.map(function (c) {
            return el("li", null, [el("span", { className: "code-badge", text: c.code }), c.name]);
          }))
        ];
      case "time":
        return [el("div", { className: "big-clock", text: display.time }), el("div", { className: "big-date", text: display.date })];
      case "link":
        return [el("div", { className: "link-card" }, [
          el("a", { className: "btn btn-gold btn-large", href: display.url, target: "_blank", rel: "noopener noreferrer" },
            [display.label, el("span", { className: "ext", "aria-hidden": "true", text: "↗" })]),
          el("p", { text: "You will sign in on the official Wits page. Pepper never sees or stores your password." })
        ])];
      case "help":
        return [el("ul", { className: "chips" }, display.items.map(function (item) { return el("li", { text: item }); }))];
      case "semester":
        var s = display.semester || display.upcoming;
        if (!s) { return null; }
        return [
          el("h3", { className: "result-title", text: s.name }),
          el("p", { className: "result-meta", text: s.start + " to " + s.end })
        ];
      case "needs_slot":
        return display.courses ? [el("ul", { className: "chips" }, display.courses.map(function (c) { return el("li", { text: c }); }))] : null;
      default:
        return null;
    }
  }

  function renderResult(result) {
    var card = $.resultCard;
    clear(card);
    var parts = result && result.display ? renderDisplay(result.display) : null;
    if (!parts) {
      card.hidden = true;
      $.tips.hidden = false;
      return;
    }
    parts.forEach(function (part) { if (part) { card.appendChild(part); } });
    card.hidden = false;
    $.tips.hidden = true;
  }

  function resetResult() {
    clear($.resultCard);
    $.resultCard.hidden = true;
    $.tips.hidden = false;
  }

  /* ---------- Research panel ---------- */

  function tag(text, kind) {
    return el("span", { className: "tag tag-" + kind, text: text });
  }

  function renderEngines(health, error) {
    clear($.engines);
    function row(label, value) {
      $.engines.appendChild(el("dt", { text: label }));
      $.engines.appendChild(el("dd", { text: value }));
    }
    if (!health) {
      row("Status", error || "Not connected");
      $.devWarning.hidden = true;
      return;
    }
    var e = health.engines;
    row("Status", "connected · v" + health.version);
    row("ASR", e.asr.engine + (e.asr.model ? " · " + e.asr.model : "") + (e.asr.loaded ? "" : " (not loaded)"));
    row("Branch A", e.branch_a.engine + (e.branch_a.classifier ? " · " + e.branch_a.classifier : ""));
    row("Branch B", e.branch_b.engine + (e.branch_b.adapter ? " · " + e.branch_b.adapter : ""));
    row("Pepper adapter", e.pepper_adapter);
    if (e.load_errors) { row("Load errors", JSON.stringify(e.load_errors)); }
    row("Timetable", health.sample_timetable ? "sample data" : "configured");
    $.devWarning.hidden = !e.development_engines_active;
  }

  function branchColumn(result, name) {
    return result.branches && result.branches[name];
  }

  function renderTrial(result, extras) {
    var box = $.trialDetail;
    clear(box);
    if (result.demo) {
      box.appendChild(el("p", { className: "warning", text: "Demo-mode trial: nothing was transcribed or classified, and no timings were measured." }));
      return;
    }
    var general = [
      ["Session / trial", result.session_id + " / " + result.trial_id],
      ["Input", result.source === "audio" ? "audio" + (result.asr && result.asr.duration_s ? " · " + result.asr.duration_s + " s" : "") + (extras.converted === false ? " (sent as recorded; WAV conversion failed)" : "") : "typed transcript (ASR bypassed)"],
      ["ASR engine", result.asr ? result.asr.engine : "—"],
      ["ASR transcript", result.transcript],
      ["Reference", extras.reference || "—"],
      ["WER / CER", result.asr_evaluation ? pct(result.asr_evaluation.wer) + " / " + pct(result.asr_evaluation.cer) : "—"]
    ];
    var rows = general.map(function (r) {
      return el("tr", null, [el("th", { text: r[0] }), el("td", { className: r[0].indexOf("transcript") >= 0 || r[0] === "Reference" ? "text" : null, text: r[1] })]);
    });
    box.appendChild(el("table", { className: "detail-table" }, [el("tbody", null, rows)]));

    var names = ["A", "B"].filter(function (n) { return branchColumn(result, n); });
    var head = el("tr", null, [el("th", { text: "" })].concat(names.map(function (n) {
      return el("th", null, ["Branch " + n + (n === result.primary ? " (executed)" : "")]);
    })));
    function branchRow(label, getter, isText) {
      return el("tr", null, [el("th", { text: label })].concat(names.map(function (n) {
        var value = getter(branchColumn(result, n), n);
        return el("td", { className: isText ? "text" : null }, [value]);
      })));
    }
    var truth = extras.trueIntent;
    var branchRows = [
      branchRow("Engine", function (b) { return STAND_INS[b.engine] ? tag(STAND_INS[b.engine], "warn") : b.engine; }),
      branchRow("Translation", function (b) { return b.translation_note ? b.translation_note : (b.translation || "—"); }, true),
      branchRow("Intent", function (b) {
        var node = el("span", null, [b.intent + " "]);
        if (truth) { node.appendChild(b.intent === truth ? tag("correct", "ok") : tag("wrong: " + truth, "bad")); }
        return node;
      }),
      branchRow("Slots", function (b) { return slotsText(b.slots); }),
      branchRow("Confidence", function (b) { return b.confidence === null || b.confidence === undefined ? "—" : b.confidence.toFixed(3); }),
      branchRow("Translation latency", function (b) { return ms(b.latency.translation_ms); }),
      branchRow("Classification latency", function (b) { return ms(b.latency.classification_ms); }),
      branchRow("Response", function (b) { return b.dispatch ? b.dispatch.response : "—"; }, true)
    ];
    box.appendChild(el("div", { className: "table-scroll" }, [
      el("table", { className: "detail-table" }, [el("thead", null, [head]), el("tbody", null, branchRows)])
    ]));

    var lat = result.latency || {};
    var summary = [
      ["Agreement", result.agreement === null || result.agreement === undefined ? "—" : (result.agreement ? tag("A and B agree", "ok") : tag("A and B disagree", "bad"))],
      ["ASR latency", ms(lat.asr_ms)],
      ["Execution latency", ms(lat.execution_ms)],
      ["Server total", ms(lat.total_ms)],
      ["Client round trip", ms(extras.clientMs)],
      ["Final Pepper response", result.response],
      ["Pepper adapter", result.adapter ? result.adapter.adapter + (result.adapter.delivered ? " · delivered" : (result.adapter.error ? " · " + result.adapter.error : "")) : "—"],
      ["Logged", result.logged ? tag("written to research CSV", "neutral") : "not logged"]
    ];
    box.appendChild(el("table", { className: "detail-table" }, [el("tbody", null, summary.map(function (r) {
      return el("tr", null, [el("th", { text: r[0] }), el("td", { className: r[0] === "Final Pepper response" ? "text" : null }, [r[1]])]);
    }))]));
  }

  function renderHistory(trials) {
    var box = $.history;
    clear(box);
    if (!trials.length) {
      box.appendChild(el("p", { className: "muted", text: "Trials you run appear here." }));
      return;
    }
    var head = el("tr", null, ["#", "Time", "Transcript", "A", "B", "True", "Server ms", "Logged"].map(function (h) { return el("th", { text: h }); }));
    var rows = trials.slice().reverse().map(function (t, i) {
      var r = t.result;
      function cell(name) {
        var b = r.branches && r.branches[name];
        if (!b) { return el("td", { text: "—" }); }
        var kind = t.trueIntent ? (b.intent === t.trueIntent ? "ok" : "bad") : "neutral";
        return el("td", null, [tag(b.intent, kind)]);
      }
      return el("tr", null, [
        el("td", { text: String(trials.length - i) }),
        el("td", { text: t.time }),
        el("td", { className: "text", text: r.transcript }),
        cell("A"),
        cell("B"),
        el("td", { text: t.trueIntent || "—" }),
        el("td", { text: r.latency && r.latency.total_ms !== null ? String(Math.round(r.latency.total_ms)) : "—" }),
        el("td", { text: r.logged ? "yes" : "no" })
      ]);
    });
    box.appendChild(el("div", { className: "table-scroll" }, [
      el("table", { className: "detail-table" }, [el("thead", null, [head]), el("tbody", null, rows)])
    ]));
  }

  function fillSelect(select, options) {
    var keep = select.options[0];
    clear(select);
    select.appendChild(keep);
    options.forEach(function (o) { select.appendChild(el("option", { value: o.value, text: o.label })); });
  }

  global.UI = {
    init: init,
    el: el,
    setState: setState,
    setSteps: setSteps,
    setTimer: setTimer,
    setLevel: setLevel,
    setTranscript: setTranscript,
    setResponse: setResponse,
    setHint: setHint,
    setConnection: setConnection,
    setSampleNotice: setSampleNotice,
    setMode: setMode,
    renderResult: renderResult,
    resetResult: resetResult,
    renderEngines: renderEngines,
    renderTrial: renderTrial,
    renderHistory: renderHistory,
    fillSelect: fillSelect,
    STATE_LABELS: STATE_LABELS
  };
})(window);
