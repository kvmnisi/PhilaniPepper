/* Pepper tablet bridge (NAOqi 2.5, QiMessaging JS).
   On Pepper the tablet has no usable microphone, so the page does not record.
   Tapping the mic raises an ALMemory event; pepper_app/scripts/task_executor.py
   records from the head microphones, calls the backend, makes Pepper speak,
   and raises state/result events that this page displays. */
(function (global) {
  "use strict";

  var EVENTS = {
    start: "IsiZuluAssistant/StartListening",
    stop: "IsiZuluAssistant/StopListening",
    state: "IsiZuluAssistant/State",
    result: "IsiZuluAssistant/Result"
  };
  var QI_SCRIPT = "/libs/qimessaging/2/qimessaging.js";

  // QiMessaging 2 returns jQuery-style promises (done/fail); support both.
  function settle(future, onOk, onFail) {
    if (future && future.done) { future.done(onOk).fail(onFail || function () {}); }
    else if (future && future.then) { future.then(onOk, onFail); }
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = src;
      script.onload = resolve;
      script.onerror = function () { reject(new Error("QiMessaging is not available")); };
      document.head.appendChild(script);
    });
  }

  function PepperBridge() {
    this.memory = null;
    this.handlers = { state: [], result: [] };
  }

  PepperBridge.prototype.connect = function () {
    var self = this;
    var ready = global.QiSession ? Promise.resolve() : loadScript(QI_SCRIPT);
    return ready.then(function () {
      return new Promise(function (resolve, reject) {
        var timer = setTimeout(function () { reject(new Error("Timed out connecting to NAOqi")); }, 8000);
        global.QiSession(function (session) {
          settle(session.service("ALMemory"), function (memory) {
            clearTimeout(timer);
            self.memory = memory;
            self.subscribe(EVENTS.state, "state");
            self.subscribe(EVENTS.result, "result");
            resolve(self);
          }, function (error) { clearTimeout(timer); reject(error); });
        }, function () {
          clearTimeout(timer);
          reject(new Error("Disconnected from NAOqi"));
        });
      });
    });
  };

  PepperBridge.prototype.subscribe = function (eventName, kind) {
    var self = this;
    settle(this.memory.subscriber(eventName), function (subscriber) {
      subscriber.signal.connect(function (value) {
        var payload = value;
        if (kind === "result" && typeof value === "string") {
          try { payload = JSON.parse(value); } catch (e) { payload = { response: value }; }
        }
        self.handlers[kind].forEach(function (handler) { handler(payload); });
      });
    });
  };

  PepperBridge.prototype.on = function (kind, handler) {
    this.handlers[kind].push(handler);
  };

  PepperBridge.prototype.startListening = function (options) {
    this.memory.raiseEvent(EVENTS.start, JSON.stringify(options || {}));
  };

  PepperBridge.prototype.stopListening = function () {
    this.memory.raiseEvent(EVENTS.stop, "");
  };

  PepperBridge.EVENTS = EVENTS;
  global.PepperBridge = PepperBridge;
})(window);
