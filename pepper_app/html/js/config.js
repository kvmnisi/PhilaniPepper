/* Runtime configuration.
   API base URL, in priority order:
     1. ?api=<url> query parameter (remembered for next time)
     2. value saved from the research panel
     3. window.APP_ENV.apiBaseUrl (js/env.js, written at build time)
     4. http://localhost:8000
   Other query parameters: ?mode=research, ?demo=1, ?runtime=pepper */
(function (global) {
  "use strict";

  var env = global.APP_ENV || {};
  var KEY_API = "isizulu.apiBaseUrl";
  var KEY_MODE = "isizulu.mode";

  function param(name) {
    var match = new RegExp("[?&]" + name + "=([^&#]*)").exec(global.location.search);
    return match ? decodeURIComponent(match[1].replace(/\+/g, " ")) : null;
  }

  function read(key) {
    try { return global.localStorage.getItem(key); } catch (e) { return null; }
  }

  function write(key, value) {
    try {
      if (value === null || value === undefined || value === "") { global.localStorage.removeItem(key); }
      else { global.localStorage.setItem(key, value); }
    } catch (e) { /* storage unavailable: settings last for this page only */ }
  }

  function clean(url) {
    return (url || "").replace(/\/+$/, "");
  }

  var apiParam = param("api");
  if (apiParam) { write(KEY_API, clean(apiParam)); }

  var modeParam = param("mode");
  if (modeParam === "research" || modeParam === "student") { write(KEY_MODE, modeParam); }

  var config = {
    apiBaseUrl: clean(apiParam || read(KEY_API) || env.apiBaseUrl || "http://localhost:8000"),
    runtime: param("runtime") || env.runtime || "browser",
    forceDemo: param("demo") === "1",
    mode: read(KEY_MODE) === "research" ? "research" : "student",
    maxRecordingMs: 8000,
    minRecordingMs: 500,
    healthTimeoutMs: 4000,
    requestTimeoutMs: 90000,
    selfServiceUrl: "https://self-service.wits.ac.za/"
  };

  config.setApiBaseUrl = function (url) {
    config.apiBaseUrl = clean(url);
    write(KEY_API, config.apiBaseUrl);
  };

  config.setMode = function (mode) {
    config.mode = mode === "research" ? "research" : "student";
    write(KEY_MODE, config.mode);
  };

  config.read = read;
  config.write = write;

  global.AppConfig = config;
})(window);
