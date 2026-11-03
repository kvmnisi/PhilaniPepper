/* 3D Pepper for the robot panel (three.js r147, ColladaLoader).
   Loads only when the licensed meshes are installed locally
   (scripts/install_pepper_meshes.py) and WebGL works; otherwise the 2D SVG
   face stays. The robot is assembled from assets/pepper3d/pepper-model.json
   (derived from the Apache-2.0 Pepper URDF) and animated from the app state
   on <body data-state="...">. */
(function (global) {
  "use strict";

  var BASE = "assets/pepper3d/";
  var MESHES = BASE + "meshes/";
  var VENDOR = ["vendor/three/three.min.js", "vendor/three/ColladaLoader.js"];
  var reduceMotion = !!(global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches);

  // NAOqi "Stand" posture (radians).
  var STAND = {
    HeadYaw: 0, HeadPitch: -0.2, HipRoll: 0, HipPitch: -0.04, KneePitch: 0,
    LShoulderPitch: 1.55, LShoulderRoll: 0.14, LElbowYaw: -1.22, LElbowRoll: -0.52, LWristYaw: -0.86,
    RShoulderPitch: 1.55, RShoulderRoll: -0.14, RElbowYaw: 1.22, RElbowRoll: 0.52, RWristYaw: 0.86
  };
  // Per-state joint angles that replace STAND; target() adds motion on top.
  var POSES = {
    idle: {},
    listening: { HeadPitch: -0.38, HipPitch: -0.08 },                      // looks up at the student
    thinking: { HeadPitch: -0.1, RShoulderPitch: 0.95, RElbowRoll: 1.45, RElbowYaw: 1.7, RWristYaw: 1.2 },   // hand to chin
    success: { HeadPitch: -0.25, LShoulderPitch: 0.75, LShoulderRoll: 0.45, LElbowRoll: -0.35, LElbowYaw: -1.5, LWristYaw: -1.5 }, // presents the answer
    error: { HeadPitch: 0.3, HipPitch: 0.06, LShoulderRoll: 0.05, RShoulderRoll: -0.05 }                  // looks down
  };
  var STATE_POSE = { idle: "idle", listening: "listening", uploading: "thinking", transcribing: "thinking",
    classifying: "thinking", executing: "thinking", success: "success", error: "error" };

  function hasWebGL() {
    try {
      var canvas = document.createElement("canvas");
      return !!(global.WebGLRenderingContext && (canvas.getContext("webgl") || canvas.getContext("experimental-webgl")));
    } catch (e) { return false; }
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = src;
      script.onload = resolve;
      script.onerror = function () { reject(new Error("Could not load " + src)); };
      document.head.appendChild(script);
    });
  }

  function getJson(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) { throw new Error(url + " " + r.status); }
      return r.json();
    });
  }

  function meshesInstalled() {
    return fetch(MESHES + "NOTICE.txt", { method: "HEAD" }).then(function (r) { return r.ok; }, function () { return false; });
  }

  /* URDF rpy = fixed-axis roll, pitch, yaw = three.js Euler order ZYX. */
  function applyOrigin(object, xyz, rpy) {
    object.position.set(xyz[0], xyz[1], xyz[2]);
    object.rotation.set(rpy[0], rpy[1], rpy[2], "ZYX");
  }

  function Pepper3D(container, model) {
    var THREE = global.THREE;
    this.THREE = THREE;
    this.container = container;
    this.joints = {};
    this.current = {};
    this.state = document.body.getAttribute("data-state") || "idle";
    this.spin = 0;
    this.spinTarget = 0;
    this.lastDrag = 0;
    this.running = true;
    this.dragging = false;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(24, 1, 0.05, 20);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 2));
    if (THREE.sRGBEncoding) { this.renderer.outputEncoding = THREE.sRGBEncoding; }
    this.renderer.domElement.className = "pepper-canvas";

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xd9dee8, 0.95));
    var key = new THREE.DirectionalLight(0xffffff, 0.75);
    key.position.set(1.5, 2.5, 3);
    this.scene.add(key);
    var rim = new THREE.DirectionalLight(0xffe3a3, 0.35);   // warm gold rim light
    rim.position.set(-2, 1.5, -2);
    this.scene.add(rim);

    // URDF: X forward, Y left, Z up. three.js: Y up, camera looks down -Z.
    // robot: Z-up -> Y-up (X stays forward); facing: forward X -> +Z (towards the camera).
    this.turntable = new THREE.Group();
    var facing = new THREE.Group();
    facing.rotation.y = -Math.PI / 2;
    this.robot = new THREE.Group();
    this.robot.rotation.x = -Math.PI / 2;
    facing.add(this.robot);
    this.turntable.add(facing);
    this.scene.add(this.turntable);

    this.build(model);
    this.bindDrag();
  }

  Pepper3D.prototype.build = function (model) {
    var THREE = this.THREE;
    var links = {};
    var self = this;
    function linkGroup(name) {
      if (!links[name]) { links[name] = new THREE.Group(); links[name].name = name; }
      return links[name];
    }
    this.robot.add(linkGroup(model.root));
    model.joints.forEach(function (j) {
      var frame = new THREE.Group();
      applyOrigin(frame, j.xyz, j.rpy);
      var motion = new THREE.Group();
      frame.add(motion);
      motion.add(linkGroup(j.child));
      linkGroup(j.parent).add(frame);
      if (j.type === "revolute" || j.type === "continuous") {
        self.joints[j.name] = { node: motion, axis: new THREE.Vector3(j.axis[0], j.axis[1], j.axis[2]).normalize() };
      }
    });

    var loader = new THREE.ColladaLoader();
    var jobs = [];
    Object.keys(model.links).forEach(function (name) {
      model.links[name].forEach(function (visual) {
        jobs.push(new Promise(function (resolve, reject) {
          loader.load(MESHES + visual.mesh, function (collada) {
            var mesh = collada.scene;
            mesh.quaternion.set(0, 0, 0, 1);       // keep URDF's Z-up frame
            mesh.scale.set(1, 1, 1);
            mesh.traverse(function (node) {
              if (node.isMesh && node.geometry && !node.geometry.attributes.normal) { node.geometry.computeVertexNormals(); }
            });
            var holder = new THREE.Group();
            applyOrigin(holder, visual.xyz, visual.rpy);
            holder.scale.set(visual.scale[0], visual.scale[1], visual.scale[2]);
            holder.add(mesh);
            linkGroup(name).add(holder);
            resolve();
          }, undefined, function () { reject(new Error("Could not load " + visual.mesh)); });
        }));
      });
    });
    this.ready = Promise.all(jobs).then(function () { self.frame(); });
  };

  /* Fit the camera to the assembled robot. */
  Pepper3D.prototype.frame = function () {
    var THREE = this.THREE;
    this.pose(STAND, 1);
    var box = new THREE.Box3().setFromObject(this.turntable);
    var size = box.getSize(new THREE.Vector3());
    var center = box.getCenter(new THREE.Vector3());
    this.turntable.position.sub(center);   // centre the robot on the origin
    var fov = this.camera.fov * Math.PI / 180;
    var distance = (size.y / 2) / Math.tan(fov / 2) * 1.12;
    this.camera.position.set(0, size.y * 0.06, distance);
    this.camera.lookAt(0, size.y * 0.02, 0);
    this.resize();
  };

  Pepper3D.prototype.resize = function () {
    var w = this.container.clientWidth || 260;
    var h = this.container.clientHeight || 320;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  };

  /* Ease every joint towards `target` (blend 1 = jump). */
  Pepper3D.prototype.pose = function (target, blend) {
    var names = Object.keys(this.joints);
    for (var i = 0; i < names.length; i++) {
      var name = names[i];
      var goal = target[name] !== undefined ? target[name] : 0;
      var now = this.current[name] !== undefined ? this.current[name] : goal;
      now += (goal - now) * blend;
      this.current[name] = now;
      var joint = this.joints[name];
      joint.node.quaternion.setFromAxisAngle(joint.axis, now);
    }
  };

  Pepper3D.prototype.target = function (t) {
    var name = STATE_POSE[this.state] || "idle";
    var pose = {};
    var key;
    for (key in STAND) { pose[key] = STAND[key]; }
    for (key in POSES[name]) { pose[key] = POSES[name][key]; }
    if (!reduceMotion) {
      if (name === "idle") {
        pose.HeadYaw += Math.sin(t * 0.45) * 0.12;
        pose.HipPitch += Math.sin(t * 0.9) * 0.015;
        pose.LElbowRoll += Math.sin(t * 0.9) * 0.03;
        pose.RElbowRoll -= Math.sin(t * 0.9) * 0.03;
      } else if (name === "listening") {
        pose.HeadYaw += Math.sin(t * 0.7) * 0.04;
      } else if (name === "thinking") {
        pose.HeadYaw += Math.sin(t * 1.4) * 0.3;
      } else if (name === "success") {
        pose.HeadPitch += Math.sin(t * 5) * 0.05;
      }
    }
    return pose;
  };

  Pepper3D.prototype.bindDrag = function () {
    var self = this;
    var canvas = this.renderer.domElement;
    var startX = null;
    var startSpin = 0;
    function down(x) { startX = x; startSpin = self.spinTarget; self.dragging = true; }
    function move(x) {
      if (startX === null) { return; }
      self.spinTarget = startSpin + (x - startX) * 0.012;
      self.lastDrag = Date.now();
    }
    function up() { startX = null; self.dragging = false; self.lastDrag = Date.now(); }
    canvas.addEventListener("mousedown", function (e) { down(e.clientX); });
    global.addEventListener("mousemove", function (e) { move(e.clientX); });
    global.addEventListener("mouseup", up);
    canvas.addEventListener("touchstart", function (e) { down(e.touches[0].clientX); }, { passive: true });
    canvas.addEventListener("touchmove", function (e) { move(e.touches[0].clientX); }, { passive: true });
    canvas.addEventListener("touchend", up);
  };

  Pepper3D.prototype.start = function () {
    var self = this;
    var clock = new this.THREE.Clock();
    new MutationObserver(function () {
      self.state = document.body.getAttribute("data-state") || "idle";
    }).observe(document.body, { attributes: true, attributeFilter: ["data-state"] });
    document.addEventListener("visibilitychange", function () { self.running = !document.hidden; if (self.running) { loop(); } });
    global.addEventListener("resize", function () { self.resize(); });

    function loop() {
      if (!self.running) { return; }
      var t = clock.getElapsedTime();
      self.pose(self.target(t), reduceMotion ? 1 : 0.08);
      // After a drag, swing back to face the student.
      if (!self.dragging && Date.now() - self.lastDrag > 2500) { self.spinTarget *= 0.94; }
      self.spin += (self.spinTarget - self.spin) * 0.15;
      self.turntable.rotation.y = self.spin;
      self.renderer.render(self.scene, self.camera);
      global.requestAnimationFrame(loop);
    }
    loop();
  };

  function init() {
    var container = document.querySelector(".robot-panel .pepper");
    if (!container || !hasWebGL() || !global.fetch || !global.Promise) { return; }
    meshesInstalled().then(function (installed) {
      if (!installed) { return null; }
      return loadScript(VENDOR[0]).then(function () { return loadScript(VENDOR[1]); })
        .then(function () { return getJson(BASE + "pepper-model.json"); })
        .then(function (model) {
          var viewer = new Pepper3D(container, model);
          return viewer.ready.then(function () {
            container.appendChild(viewer.renderer.domElement);
            container.className += " has-3d";
            viewer.resize();
            viewer.start();
            global.Pepper3DViewer = viewer;
          });
        });
    }).catch(function (error) {
      // Any failure keeps the 2D face.
      if (global.console) { console.warn("3D Pepper unavailable, using the 2D face:", error.message); }
    });
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); }
  else { init(); }
})(window);
