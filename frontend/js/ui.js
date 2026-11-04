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

  var DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  var MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December"];

  /* "2026-11-10" -> "Tuesday 10 November 2026" (parsed as a local date, no time zone shift). */
  function longDate(iso) {
    if (!iso) { return ""; }
    var p = iso.split("-");
    var d = new Date(+p[0], +p[1] - 1, +p[2]);
    return DAY_NAMES[d.getDay()] + " " + d.getDate() + " " + MONTH_NAMES[d.getMonth()] + " " + d.getFullYear();
  }

  function sessionsTable(sessions) {
    return el("table", { className: "table" }, [
      el("thead", null, [el("tr", null, [el("th", { text: "Day" }), el("th", { text: "Time" }), el("th", { text: "Venue" })])]),
      el("tbody", null, sessions.map(function (s) {
        return el("tr", null, [
          el("td", { text: capitalise(s.day) }),
          el("td", { className: "mono", text: s.start + "–" + s.end }),
          el("td", { text: s.venue || "To be confirmed" })
        ]);
      }))
    ]);
  }

  function pin(point, className) {
    return el("span", { className: className, style: "left:" + (point.x * 100) + "%;top:" + (point.y * 100) + "%" }, [
      el("span", { className: "pin-dot" }),
      el("span", { className: "pin-label", text: point.label })
    ]);
  }

  /* Campus map with "You are here" and destination pins; scrolls to the destination. */
  function campusMap(map) {
    if (!map || !map.markers || !map.markers.length) { return null; }
    var pins = [];
    if (map.here) { pins.push(pin(map.here, "pin pin-here")); }
    map.markers.forEach(function (m) { pins.push(pin(m, "pin pin-target")); });
    var frame = el("div", { className: "map-frame" }, [el("img", { src: map.image, alt: "Wits Braamfontein campus map" })].concat(pins));
    // Centre between "you are here" and the destination so the route is in view.
    var target = map.markers[0];
    var focusX = map.here ? (map.here.x + target.x) / 2 : target.x;
    var focusY = map.here ? (map.here.y + target.y) / 2 : target.y;
    var scroller = el("div", { className: "map-scroll", "data-focus-x": focusX, "data-focus-y": focusY }, [frame]);
    var toggle = el("button", { className: "btn btn-outline btn-small", type: "button", text: "Show whole map" });
    toggle.addEventListener("click", function () {
      var full = scroller.className.indexOf("is-full") < 0;
      scroller.className = full ? "map-scroll is-full" : "map-scroll";
      toggle.textContent = full ? "Zoom in" : "Show whole map";
      if (!full) { focusMap(scroller); }
    });
    return el("figure", { className: "campus-map" }, [
      scroller,
      el("figcaption", null, [
        el("span", { className: "legend legend-here", text: "You are here" }),
        el("span", { className: "legend legend-target", text: map.markers.map(function (m) { return m.label; }).join(", ") }),
        toggle
      ])
    ]);
  }

  function focusMap(scroller) {
    var frame = scroller.firstChild;
    var x = parseFloat(scroller.getAttribute("data-focus-x"));
    var y = parseFloat(scroller.getAttribute("data-focus-y"));
    scroller.scrollLeft = Math.max(0, x * frame.offsetWidth - scroller.clientWidth / 2);
    scroller.scrollTop = Math.max(0, y * frame.offsetHeight - scroller.clientHeight / 2);
  }

  function eventsTable(rows) {
    return el("table", { className: "table" }, [
      el("thead", null, [el("tr", null, [el("th", { text: "Time" }), el("th", { text: "Course" }), el("th", { text: "Venue" })])]),
      el("tbody", null, rows.map(function (r) {
        return el("tr", null, [
          el("td", { className: "mono", text: r.start + "–" + r.end }),
          el("td", null, [el("span", { className: "code-badge", text: r.code }), r.name,
            r.kind === "exam" ? el("span", { className: "kind-note", text: " · " + (r.exam_type || "exam") }) : null]),
          el("td", { text: r.venue })
        ]);
      }))
    ]);
  }

  function contactCard(person) {
    var lines = [];
    if (person.email) { lines.push(el("li", null, ["Email ", el("a", { href: "mailto:" + person.email, text: person.email })])); }
    if (person.phone) { lines.push(el("li", null, ["Phone ", el("a", { href: "tel:" + person.phone.replace(/\s+/g, ""), text: person.phone })])); }
    if (person.office) { lines.push(el("li", { text: "Office " + person.office })); }
    if (person.fax) { lines.push(el("li", { text: "Fax " + person.fax })); }
    return el("div", { className: "contact" }, [
      el("p", { className: "contact-role", text: person.role }),
      el("h3", { className: "result-title", text: person.name }),
      el("ul", { className: "contact-lines" }, lines)
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
          display.sessions && display.sessions.length ? sessionsTable(display.sessions) : null
        ].concat((display.exams || []).map(function (exam) {
          return el("div", { className: "exam" }, [
            el("p", { className: "exam-kind", text: capitalise(exam.type || "exam") }),
            el("p", { className: "exam-when" }, [el("strong", { text: exam.day }), " · " + exam.start + "–" + exam.end]),
            el("p", { className: "exam-where", text: exam.venue })
          ]);
        })).concat([campusMap(display.map)]);
      case "schedule":
        return [
          el("h3", { className: "result-title", text: display.title }),
          display.rows.length ? eventsTable(display.rows) : el("p", { className: "result-meta", text: "Nothing scheduled." }),
          campusMap(display.map)
        ];
      case "course_list":
        return [
          el("h3", { className: "result-title", text: "Honours courses" }),
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
      case "capabilities":
        return [
          el("h3", { className: "result-title", text: display.title }),
          el("ul", { className: "chips" }, display.items.map(function (item) { return el("li", { text: item }); }))
        ];
      case "help":
        return [el("ul", { className: "chips" }, display.items.map(function (item) { return el("li", { text: item }); }))];
      case "calendar":
        if (!display.period && !display.next) { return null; }
        var parts = [];
        if (display.period) {
          parts.push(el("p", { className: "contact-role", text: "Now" }));
          parts.push(el("h3", { className: "result-title", text: display.period.name }));
          parts.push(el("p", { className: "result-meta", text: longDate(display.period.start) + (display.period.end ? " to " + longDate(display.period.end) : " onwards") }));
        }
        if (display.next) {
          parts.push(el("p", { className: "result-meta" }, ["Next: ", el("strong", { text: display.next.name }), ", from " + longDate(display.next.start)]));
        }
        return parts;
      case "contact":
        return display.people.map(contactCard);
      case "location":
        var where = [];
        if (display.building && display.building !== display.name) { where.push(display.building); }
        if (display.campus) { where.push(display.campus + " Campus"); }
        if (display.grid) { where.push("map grid " + display.grid); }
        return [
          el("h3", { className: "result-title", text: display.name }),
          where.length ? el("p", { className: "result-meta", text: where.join(" · ") }) : null,
          display.note ? el("p", { className: "result-meta", text: display.note }) : null,
          campusMap(display.map)
        ];
      case "needs_slot":
        return display.choices ? [
          el("p", { className: "result-meta", text: "Try one of these:" }),
          el("ul", { className: "chips" }, display.choices.map(function (c) { return el("li", { text: c }); }))
        ] : null;
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
    var maps = card.getElementsByClassName("map-scroll");
    for (var i = 0; i < maps.length; i++) { focusMapWhenLoaded(maps[i]); }
  }

  function focusMapWhenLoaded(scroller) {
    var img = scroller.getElementsByTagName("img")[0];
    if (img.complete) { focusMap(scroller); } else { img.addEventListener("load", function () { focusMap(scroller); }); }
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

  var POLICY_LABELS = {
    verb_unknown_object: "verb + unknown subject: refused",
    verb_object_mismatch: "verb cannot apply: refused + offer",
    object_offer: "no verb: offered capabilities",
    verb_offer: "verb alone: offered capabilities",
    execute: "verb alone: help"
  };

  function frameText(frame) {
    if (!frame) { return "\u2014"; }
    var parts = ["verb: " + (frame.verb || "none"), "subject: " + (frame.objects && frame.objects.length ? frame.objects.join(", ") : "none")];
    if (frame.unknown && frame.unknown.length) { parts.push("unknown: " + frame.unknown.join(" ")); }
    return parts.join(" \u00b7 ");
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
      ["Verb / subject", frameText(result.frame)],
      ["Pepper did", result.policy ? el("span", null, [(result.executed_intent || result.intent) + " ", tag(POLICY_LABELS[result.policy] || result.policy, "warn")]) : (result.executed_intent || result.intent)],
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
