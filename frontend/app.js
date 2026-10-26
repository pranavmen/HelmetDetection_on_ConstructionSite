/* HardHat Vision — one-page demo frontend
   Upload an image -> YOLO11s detection on the local Flask server.
   Video analysis runs the full pipeline (ByteTrack + zone + dwell) server-side. */
"use strict";

/* ---------------------------------------------------------------- helpers */
const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

const COLORS = { helmet: "#2ed573", head: "#ff4757" };

const state = {
  img: null,              // HTMLImageElement for the uploaded image
  detections: [],
  counts: { helmet: 0, head: 0 },
  conf: 0.5,
  hover: -1,
  lastBlob: null,         // re-sent when the confidence slider changes
  lastFileName: "frame.jpg",
};

function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.innerHTML = msg;
  $("toastWrap").appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = ".4s"; setTimeout(() => el.remove(), 422); }, 4500);
}

function setRangeFill(el) {
  const pct = ((el.value - el.min) / (el.max - el.min)) * 100;
  el.style.setProperty("--fill", pct + "%");
}

function bump(el) { el.parentElement.classList.remove("bump"); void el.offsetWidth; el.parentElement.classList.add("bump"); }

function setStat(el, v) {
  if (String(el.textContent) !== String(v)) { el.textContent = v; bump(el); }
}

/* ---------------------------------------------------------------- drawing */
function drawBox(d, i) {
  const [x1, y1, x2, y2] = d.bbox;
  const color = COLORS[d.cls] || "#aaa";
  const hot = state.hover === i;

  ctx.lineWidth = hot ? 3.5 : 2;
  ctx.strokeStyle = color;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  // label chip
  const label = `${d.cls} ${d.conf.toFixed(2)}`;
  ctx.font = "700 13px system-ui, sans-serif";
  const tw = ctx.measureText(label).width + 14;
  let ly = y1 - 22; if (ly < 2) ly = y1 + 2;
  ctx.fillStyle = color;
  ctx.fillRect(x1, ly, tw, 20);
  ctx.fillStyle = "#0b0d10";
  ctx.fillText(label, x1 + 7, ly + 14.5);

  // helmet colour → role line
  if (d.cls === "helmet" && d.color && d.color !== "unknown") {
    ctx.font = "600 11.5px system-ui, sans-serif";
    const t2 = `${d.color} · ${d.role}`;
    const w2 = ctx.measureText(t2).width + 12;
    ctx.fillStyle = "rgba(11,13,16,.82)";
    ctx.fillRect(x1, y1 + (y2 - y1) - 18, w2, 17);
    ctx.fillStyle = "#ffd76e";
    ctx.fillText(t2, x1 + 6, y1 + (y2 - y1) - 5.5);
  }
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.img) {
    ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = "#101418";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }
  state.detections.forEach(drawBox);
}

/* ---------------------------------------------------------------- rendering */
function renderStats() {
  setStat($("stHelmet"), state.counts.helmet || 0);
  setStat($("stHead"), state.counts.head || 0);
}

function renderDetections() {
  const dets = state.detections;
  $("detCount").textContent = `${dets.length} object${dets.length === 1 ? "" : "s"}`;
  if (!dets.length) {
    $("detList").innerHTML = '<div class="det-empty">No detections in this image.</div>';
    return;
  }
  $("detList").innerHTML = dets.map((d, i) => {
    const cls = d.cls === "helmet" ? "det-helmet-row" : "det-head-row";
    const icon = d.cls === "helmet" ? "⛑" : "✕";
    let tags = `<span class="tag">${Math.round(d.conf * 100)}%</span>`;
    if (d.cls === "helmet" && d.color && d.color !== "unknown")
      tags += `<span class="tag" style="color:#ffd76e;border-color:#ffd21e44">● ${d.color} → ${d.role}</span>`;
    return `<div class="det-item ${cls} ${state.hover === i ? "hover" : ""}" data-i="${i}">
      <div class="det-top"><span class="det-cls">${icon} ${d.cls}</span>
      <span class="det-conf"><i style="width:${Math.round(d.conf * 100)}%"></i></span>
      <span class="det-conf-num">${d.conf.toFixed(2)}</span></div>
      <div class="det-tags">${tags}</div></div>`;
  }).join("");
  [...$("detList").querySelectorAll(".det-item")].forEach((el) => {
    el.onmouseenter = () => { state.hover = +el.dataset.i; draw(); };
    el.onmouseleave = () => { state.hover = -1; draw(); };
  });
}

function renderAll() { renderStats(); renderDetections(); draw(); }

/* ---------------------------------------------------------------- inference */
async function detectBlob(blob, name = "frame.jpg") {
  $("canvasLoading").classList.remove("hidden");
  try {
    const fd = new FormData();
    fd.append("image", blob, name);
    fd.append("conf", state.conf);
    const j = await (await fetch("/api/detect", { method: "POST", body: fd })).json();
    if (!j.ok) throw new Error(j.error || "detection failed");
    state.detections = j.detections;
    state.counts = j.counts;
    $("inferChip").classList.remove("chip-hidden");
    $("inferChip").textContent = `${j.ms} ms`;
    return true;
  } catch (e) {
    toast("Detection failed: " + e.message, "err");
    return false;
  } finally {
    $("canvasLoading").classList.add("hidden");
  }
}

async function loadImageEl(src) {
  const img = new Image();
  img.src = src;
  await img.decode();
  return img;
}

async function runUploadFile(file) {
  if (!file || !file.type.startsWith("image/")) { toast("Please choose an image file", "err"); return; }
  state.lastBlob = file;
  state.lastFileName = file.name;
  $("emptyHint").classList.add("hidden");
  state.img = await loadImageEl(URL.createObjectURL(file));
  canvas.width = state.img.naturalWidth;
  canvas.height = state.img.naturalHeight;
  state.detections = []; state.counts = { helmet: 0, head: 0 };
  renderAll();
  if (await detectBlob(file, file.name)) renderAll();
}

/* ---------------------------------------------------------------- box hover */
function canvasPoint(e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * (canvas.width / r.width),
    y: (e.clientY - r.top) * (canvas.height / r.height),
  };
}
canvas.addEventListener("pointermove", (e) => {
  const p = canvasPoint(e);
  let hit = -1;
  state.detections.forEach((d, i) => {
    const [x1, y1, x2, y2] = d.bbox;
    if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) hit = i;
  });
  if (hit !== state.hover) { state.hover = hit; renderDetections(); draw(); }
});
canvas.addEventListener("pointerleave", () => {
  if (state.hover !== -1) { state.hover = -1; renderDetections(); draw(); }
});

/* drag & drop images */
const wrap = $("canvasWrap");
["dragenter", "dragover"].forEach((ev) => wrap.addEventListener(ev, (e) => {
  e.preventDefault();
  $("dropOverlay").classList.remove("hidden");
}));
["dragleave", "drop"].forEach((ev) => wrap.addEventListener(ev, (e) => {
  e.preventDefault();
  $("dropOverlay").classList.add("hidden");
}));
wrap.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) runUploadFile(f);
});
$("fileInput").addEventListener("change", (e) => runUploadFile(e.target.files[0]));

/* ---------------------------------------------------------------- controls */
$("confSlider").addEventListener("input", (e) => {
  state.conf = parseFloat(e.target.value);
  $("confVal").textContent = state.conf.toFixed(2);
  setRangeFill(e.target);
  clearTimeout(state._confT);
  state._confT = setTimeout(() => {
    if (state.lastBlob) detectBlob(state.lastBlob, state.lastFileName).then(renderAll);
  }, 350);
});
setRangeFill($("confSlider"));

/* ---------------------------------------------------------------- video analysis */
let videoFile = null;
let pollTimer = null;

function setVideoFile(f) {
  if (!f) return;
  if (f.size > 200 * 1024 * 1024) { toast("Video too large (max 200 MB)", "err"); return; }
  videoFile = f;
  $("videoFileName").textContent = `${f.name} · ${(f.size / 1e6).toFixed(1)} MB`;
  $("videoFileChip").classList.remove("hidden");
  $("videoRun").disabled = false;
}
$("videoDrop").addEventListener("click", () => $("videoInput").click());
$("videoInput").addEventListener("change", (e) => setVideoFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => $("videoDrop").addEventListener(ev, (e) => {
  e.preventDefault(); $("videoDrop").classList.add("drag");
}));
["dragleave", "drop"].forEach((ev) => $("videoDrop").addEventListener(ev, (e) => {
  e.preventDefault(); $("videoDrop").classList.remove("drag");
}));
$("videoDrop").addEventListener("drop", (e) => setVideoFile(e.dataTransfer.files[0]));
$("videoClear").addEventListener("click", () => {
  videoFile = null; $("videoFileChip").classList.add("hidden");
  $("videoInput").value = ""; $("videoRun").disabled = true;
});

$("videoRun").addEventListener("click", async () => {
  if (!videoFile) return;
  const fd = new FormData();
  fd.append("video", videoFile);
  fd.append("conf", state.conf);
  $("videoRun").disabled = true;
  $("videoProgress").classList.remove("hidden");
  $("videoResult").classList.add("hidden");
  $("videoStatusText").textContent = "Uploading + starting analysis…";
  try {
    const r = await fetch("/api/video", { method: "POST", body: fd });
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || ("HTTP " + r.status));
    pollVideoJob(j.job_id);
  } catch (e) {
    toast("Could not start analysis: " + e.message, "err");
    $("videoRun").disabled = false;
    $("videoProgress").classList.add("hidden");
  }
});

function fmtEta(sec) {
  if (!isFinite(sec) || sec < 0) return "…";
  if (sec < 60) return `${Math.ceil(sec)}s`;
  return `${Math.floor(sec / 60)}m ${Math.ceil(sec % 60)}s`;
}
function pollVideoJob(id) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const j = (await (await fetch(`/api/video/${id}`)).json()).job;
      if (!j) throw new Error("unknown job");
      const pct = Math.round((j.progress || 0) * 100);
      $("videoBar").style.width = pct + "%";
      $("videoPct").textContent = pct + "%";
      if (j.status === "running") {
        $("videoStatusText").textContent = `Analysing frame ${j.frames}/${j.total}…`;
        const eta = j.elapsed && j.frames ? (j.elapsed / j.frames) * (j.total - j.frames) : NaN;
        $("videoSub").textContent = j.violations.length
          ? `${j.elapsed}s elapsed · ~${fmtEta(eta)} left · ⛔ ${j.violations.length} violation${j.violations.length > 1 ? "s" : ""} fired`
          : `${j.elapsed}s elapsed · ~${fmtEta(eta)} left`;
      } else {
        clearInterval(pollTimer);
        $("videoRun").disabled = false;
        if (j.status === "done") {
          $("videoBar").style.width = "100%";
          $("videoPct").textContent = "100%";
          $("videoStatusText").textContent = `Done — ${j.frames} frames in ${j.elapsed}s` + (j.truncated ? " (capped at 3000 frames)" : "");
          $("videoSub").textContent = "";
          showVideoResult(j);
          refreshViolCount();
          toast(`✅ Analysis complete — ${j.violations.length} violation${j.violations.length === 1 ? "" : "s"} logged`, "ok");
        } else {
          $("videoStatusText").textContent = "Failed";
          $("videoSub").textContent = j.error || "unknown error";
          toast("Video analysis failed: " + (j.error || "unknown"), "err");
        }
      }
    } catch (e) {
      clearInterval(pollTimer);
      $("videoRun").disabled = false;
      toast("Lost contact with the video job: " + e.message, "err");
    }
  }, 800);
}
function showVideoResult(j) {
  $("videoResult").classList.remove("hidden");
  $("resultVideo").src = j.video + "?t=" + Date.now();
  $("videoViolCount").textContent = `${j.violations.length} violation${j.violations.length === 1 ? "" : "s"}`;
  $("videoViolList").innerHTML = j.violations.length
    ? j.violations.map((v) => `<li>⛔ ${v.reason} · frame ${v.frame}</li>`).join("")
    : '<li class="muted">No violations fired.</li>';
}

/* ---------------------------------------------------------------- violations count */
async function refreshViolCount() {
  try {
    const j = await (await fetch("/api/violations")).json();
    setStat($("stViol"), (j.violations || []).length);
  } catch (e) { /* server offline — keep last count */ }
}

/* ---------------------------------------------------------------- server info */
async function loadInfo() {
  try {
    const j = await (await fetch("/api/info")).json();
    if (!j.ok) throw new Error();
    const m = j.model;
    $("serverStatus").innerHTML =
      `<span class="dot dot-ok"></span><span>${m.file.replace("_best.pt", "")} · ${m.device} · ready</span>`;
    if (!m.trained_weights)
      $("serverStatus").innerHTML =
        `<span class="dot dot-loading"></span><span>fallback weights (train first!)</span>`;
    if (j.violations_count) setStat($("stViol"), j.violations_count);
  } catch (e) {
    $("serverStatus").innerHTML = '<span class="dot dot-err"></span><span>server offline</span>';
  }
}

/* ---------------------------------------------------------------- init */
document.addEventListener("DOMContentLoaded", () => {
  loadInfo();
  refreshViolCount();
});
