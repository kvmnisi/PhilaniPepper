/* Welcome section: "what you can ask" slider and Start buttons.
   Behaves like the portfolio hero slider: auto-advances every 5 s with a
   progress fill on the active indicator, pauses for 8 s after manual
   navigation, and does not auto-play when reduced motion is requested. */
(function (global) {
  "use strict";

  var INTERVAL_MS = 5000;
  var RESUME_MS = 8000;

  function byId(id) { return document.getElementById(id); }

  function prefersReducedMotion() {
    return !!(global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  function scrollToAssistant() {
    var target = byId("assistant");
    if (!target) { return; }
    try { target.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" }); }
    catch (e) { target.scrollIntoView(true); }
    try { target.focus({ preventScroll: true }); } catch (e2) { target.focus(); }
  }

  function Slider(root) {
    this.track = byId("sliderTrack");
    this.slides = this.track.getElementsByClassName("slide");
    this.indicators = byId("sliderIndicators").getElementsByClassName("indicator");
    this.index = 0;
    this.timer = null;
    this.resumeTimer = null;
    this.autoplay = !prefersReducedMotion();
    var self = this;

    root.getElementsByClassName("slider-prev")[0].addEventListener("click", function () { self.manual(self.index - 1); });
    root.getElementsByClassName("slider-next")[0].addEventListener("click", function () { self.manual(self.index + 1); });
    for (var i = 0; i < this.indicators.length; i++) {
      this.indicators[i].addEventListener("click", function () { self.manual(parseInt(this.getAttribute("data-slide"), 10)); });
    }

    // Swipe on touch screens (Pepper's tablet, phones).
    var startX = null;
    root.addEventListener("touchstart", function (e) { startX = e.touches[0].clientX; }, { passive: true });
    root.addEventListener("touchend", function (e) {
      if (startX === null) { return; }
      var dx = e.changedTouches[0].clientX - startX;
      startX = null;
      if (Math.abs(dx) > 40) { self.manual(self.index + (dx < 0 ? 1 : -1)); }
    });

    // Pause while the tab is hidden.
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) { self.stop(); } else { self.start(); }
    });

    this.show(0);
    this.start();
  }

  Slider.prototype.show = function (index) {
    var count = this.slides.length;
    this.index = (index + count) % count;
    this.track.style.transform = "translateX(-" + (this.index * 100) + "%)";
    for (var i = 0; i < count; i++) {
      var active = i === this.index;
      this.slides[i].className = active ? "slide is-active" : "slide";
      this.slides[i].setAttribute("aria-hidden", active ? "false" : "true");
      var links = this.slides[i].getElementsByTagName("a");
      for (var j = 0; j < links.length; j++) { links[j].tabIndex = active ? 0 : -1; }
      this.indicators[i].className = "indicator" + (active ? " is-active" : "") + (active && this.timer ? " is-running" : "");
      this.indicators[i].setAttribute("aria-current", active ? "true" : "false");
    }
  };

  Slider.prototype.start = function () {
    if (!this.autoplay || this.timer) { return; }
    var self = this;
    this.timer = setInterval(function () { self.show(self.index + 1); }, INTERVAL_MS);
    this.show(this.index); // restarts the progress fill
  };

  Slider.prototype.stop = function () {
    clearInterval(this.timer);
    this.timer = null;
    this.show(this.index);
  };

  Slider.prototype.manual = function (index) {
    var self = this;
    this.stop();
    this.show(index);
    clearTimeout(this.resumeTimer);
    if (this.autoplay) { this.resumeTimer = setTimeout(function () { self.start(); }, RESUME_MS); }
  };

  function init() {
    var starts = document.querySelectorAll("[data-start]");
    for (var i = 0; i < starts.length; i++) {
      starts[i].addEventListener("click", function (event) {
        event.preventDefault();
        scrollToAssistant();
      });
    }
    var root = byId("welcomeSlider");
    if (root) { global.WelcomeSlider = new Slider(root); }
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); }
  else { init(); }
})(window);
