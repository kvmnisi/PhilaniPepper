/* Welcome section: the Start button scrolls to the assistant. */
(function (global) {
  "use strict";

  function prefersReducedMotion() {
    return !!(global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  function scrollToAssistant() {
    var target = document.getElementById("assistant");
    if (!target) { return; }
    try { target.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" }); }
    catch (e) { target.scrollIntoView(true); }
    try { target.focus({ preventScroll: true }); } catch (e2) { target.focus(); }
  }

  function init() {
    var starts = document.querySelectorAll("[data-start]");
    for (var i = 0; i < starts.length; i++) {
      starts[i].addEventListener("click", function (event) {
        event.preventDefault();
        scrollToAssistant();
      });
    }
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); }
  else { init(); }
})(window);
