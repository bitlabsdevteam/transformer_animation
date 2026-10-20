import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- layout constants
const CELL = 0.12;          // footprint of one bar
const BAR_H = 1.4;          // height of a bar at |value| == slab scale
const STAGE_GAP = 1.5;      // gap between slabs in a row
const ROW_PAD = 3.4;        // gap between rows (blocks)
const TILE_GAP = 0.18;      // gap between attention-head tiles
const LOGIT_W = CELL * 3.2;

const SHORT = {
  embed: 'Embedding', ln1: 'LayerNorm 1', q: 'Query', k: 'Key', v: 'Value', attn: 'Attention (12 heads)',
  ctx: 'Σ P·V', attn_out: 'Attn proj', resid1: '+ Residual', ln2: 'LayerNorm 2', fc: 'MLP fc (3072)',
  gelu: 'GELU', mlp_out: 'MLP proj', resid2: '+ Residual', ln_f: 'Final LN', logits: 'Logits (top 50)',
};

const COLORS = {
  base: new THREE.Color('#1b2333'),
  valPos: new THREE.Color('#f59e0b'), valNeg: new THREE.Color('#3b82f6'),
  gradPos: new THREE.Color('#e879f9'), gradNeg: new THREE.Color('#22d3ee'),
  attn: new THREE.Color('#a78bfa'), white: new THREE.Color('#ffffff'),
  plate: new THREE.Color('#121927'), plateFwd: new THREE.Color('#3a2a0a'), plateBwd: new THREE.Color('#3a1240'),
};

const state = {
  data: null, slabs: [], mode: 'val', step: -1, dir: 'fwd', timer: null, speed: 1,
  selection: null, detailAbort: null, sampled: null, lastText: '',
};

// ---------------------------------------------------------------- three.js setup
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color('#0b0f17');
scene.fog = new THREE.Fog('#0b0f17', 28, 95);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500);
camera.position.set(-6, 14, 18);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(6, 0, -4);
scene.add(new THREE.HemisphereLight('#c7d7ff', '#0b0f17', 1.1));
const sun = new THREE.DirectionalLight('#ffffff', 1.3);
sun.position.set(20, 40, 30);
scene.add(sun);
let world = new THREE.Group();
scene.add(world);

const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const pulse = new THREE.Group();
const pulseCore = new THREE.Mesh(new THREE.SphereGeometry(0.22, 20, 20), new THREE.MeshBasicMaterial({ color: '#fbbf24' }));
const pulseHalo = new THREE.Mesh(new THREE.SphereGeometry(0.6, 20, 20),
  new THREE.MeshBasicMaterial({ color: '#fbbf24', transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending, depthWrite: false }));
pulse.add(pulseCore, pulseHalo);
pulse.visible = false;
scene.add(pulse);
const marker = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
  new THREE.MeshBasicMaterial({ color: '#ffffff', wireframe: true }));
marker.visible = false;
scene.add(marker);

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

// ---------------------------------------------------------------- tweens
const tweens = new Set();
function tween(dur, fn, done) {
  const tw = { t0: performance.now(), dur: Math.max(dur, 1), fn, done };
  tweens.add(tw);
  return tw;
}
const ease = (k) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);

function animate(now) {
  for (const tw of tweens) {
    const k = Math.min(1, (now - tw.t0) / tw.dur);
    tw.fn(ease(k));
    if (k >= 1) { tweens.delete(tw); tw.done?.(); }
  }
  if (pulse.visible) pulseHalo.scale.setScalar(1 + 0.15 * Math.sin(now / 120));
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

// ---------------------------------------------------------------- helpers
function decode(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

const fmt = (v) => {
  if (typeof v !== 'number') return String(v);
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(3);
  return v.toFixed(4);
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const showTok = (s) => JSON.stringify(s).slice(1, -1);
const tex = (s, display = false) => (window.katex
  ? window.katex.renderToString(s, { displayMode: display, throwOnError: false })
  : `<code>${esc(s)}</code>`);

const spriteCache = new Map();
function textSprite(text, { size = 0.32, color = '#cbd5e1', weight = 500, align = 'center' } = {}) {
  const key = `${text}|${color}|${weight}`;
  let tex = spriteCache.get(key);
  if (!tex) {
    const c = document.createElement('canvas');
    const ctx = c.getContext('2d');
    const font = `${weight} 48px "IBM Plex Sans", system-ui, sans-serif`;
    ctx.font = font;
    c.width = Math.ceil(ctx.measureText(text).width) + 16;
    c.height = 64;
    ctx.font = font;
    ctx.fillStyle = color;
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 8, 34);
    tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.userData.aspect = c.width / c.height;
    spriteCache.set(key, tex);
  }
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false, transparent: true, fog: true }));
  s.scale.set(size * tex.userData.aspect, size, 1);
  if (align === 'right') s.center.set(1, 0.5);
  if (align === 'left') s.center.set(0, 0.5);
  return s;
}

function stageTitle(s) {
  const where = s.layer < 0 ? '' : s.layer < state.data.config.n_layer ? `Block ${s.layer + 1} · ` : '';
  return `${where}${SHORT[s.name]}`;
}

// ---------------------------------------------------------------- scene building
function buildScene(data) {
  scene.remove(world);
  world.traverse((o) => { if (o.material && o.material !== marker.material) o.material.dispose?.(); if (o.geometry && o.geometry !== boxGeo) o.geometry.dispose(); });
  world = new THREE.Group();
  scene.add(world);
  state.slabs = [];

  const T = data.T;
  const depth = Math.max(T, 4) * CELL;
  const bins = data.config.bins;
  const tile = Math.max((depth - TILE_GAP) / 2, 0.9);
  const attnDepth = 2 * tile + TILE_GAP;
  const rowPitch = Math.max(depth, attnDepth) + ROW_PAD;

  const rowOf = (s) => (s.layer < 0 ? 0 : s.layer + 1);
  const cursor = new Map();

  data.stages.forEach((st, order) => {
    const row = Math.min(rowOf(st), data.config.n_layer + 1);
    const zc = -row * rowPitch;
    const sd = st.kind === 'attn' ? attnDepth : depth;
    const z0 = zc - sd / 2;
    let w;
    if (st.kind === 'vec') w = bins * CELL;
    else if (st.kind === 'attn') w = 6 * (tile + TILE_GAP) - TILE_GAP;
    else w = st.shape[1] * LOGIT_W;
    const x0 = cursor.get(row) ?? 0;
    cursor.set(row, x0 + w + STAGE_GAP);

    const val = decode(st.val);
    const grad = decode(st.grad);
    const pos = [];
    const vidx = [];
    const map = [];
    if (st.kind === 'vec') {
      for (let t = 0; t < T; t++) for (let b = 0; b < bins; b++) {
        pos.push(x0 + (b + 0.5) * CELL, z0 + (t + 0.5) * CELL);
        vidx.push(t * bins + b);
        map.push(t, b, 0);
      }
    } else if (st.kind === 'attn') {
      const c = tile / T;
      for (let h = 0; h < st.shape[0]; h++) {
        const ox = x0 + (h % 6) * (tile + TILE_GAP);
        const oz = z0 + Math.floor(h / 6) * (tile + TILE_GAP);
        for (let t = 0; t < T; t++) for (let j = 0; j <= t; j++) {
          pos.push(ox + (j + 0.5) * c, oz + (t + 0.5) * c);
          vidx.push(h * T * T + t * T + j);
          map.push(t, j, h);
        }
      }
    } else {
      for (let i = 0; i < st.shape[1]; i++) {
        pos.push(x0 + (i + 0.5) * LOGIT_W, zc);
        vidx.push(i);
        map.push(T - 1, i, 0);
      }
    }
    const count = vidx.length;
    const mat = new THREE.MeshLambertMaterial({ transparent: true, opacity: 1 });
    const mesh = new THREE.InstancedMesh(boxGeo, mat, count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.setColorAt(0, COLORS.base);
    mesh.frustumCulled = false;

    const group = new THREE.Group();
    group.add(mesh);
    world.add(group);

    const plateMat = new THREE.MeshBasicMaterial({ color: COLORS.plate, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide });
    const plate = new THREE.Mesh(new THREE.PlaneGeometry(w + 0.3, sd + 0.3), plateMat);
    plate.rotation.x = -Math.PI / 2;
    plate.position.set(x0 + w / 2, -0.002, zc);
    world.add(plate);

    const label = textSprite(stageTitle(st), { size: 0.36, color: '#94a3b8' });
    label.position.set(x0 + w / 2, BAR_H + 0.5, z0 - 0.2);
    world.add(label);

    const slab = {
      st, order, key: st.key, name: st.name, kind: st.kind, layer: st.layer, val, grad,
      pos: new Float32Array(pos), vidx: new Int32Array(vidx), map: new Int32Array(map),
      mesh, mat, group, plate, label, x0, z0, w, depth: sd, zc, row, tile, T,
      center: new THREE.Vector3(x0 + w / 2, 0, zc), progress: 1,
      box: new THREE.Box3(new THREE.Vector3(x0, -1.35 * BAR_H, z0), new THREE.Vector3(x0 + w, 1.35 * BAR_H, z0 + sd)),
    };
    plate.userData.slab = slab;
    state.slabs.push(slab);
  });

  // token labels at the start of each row
  const rows = new Set(state.slabs.map((s) => s.row));
  for (const row of rows) {
    const zc = -row * rowPitch;
    for (let t = 0; t < T; t++) {
      const s = textSprite(showTok(data.tokens[t]), { size: Math.min(0.26, CELL * 1.6), color: t === T - 1 ? '#fde68a' : '#94a3b8', align: 'right' });
      s.position.set(-0.3, 0.05, zc - depth / 2 + (t + 0.5) * CELL);
      world.add(s);
    }
    const rl = row === 0 ? 'Embedding' : row <= data.config.n_layer ? `Block ${row}` : 'Output';
    const s = textSprite(rl, { size: 0.55, color: '#e2e8f0', weight: 700, align: 'right' });
    s.position.set(-0.3, 0.9, zc - depth / 2 - 0.2);
    world.add(s);
  }

  // logits labels
  const logitSlab = state.slabs.find((s) => s.kind === 'logits');
  data.top.slice(0, 9).forEach((tk, i) => {
    const s = textSprite(showTok(tk.token), { size: 0.24, color: tk.id === data.target.id ? '#f0abfc' : '#cbd5e1' });
    s.position.set(logitSlab.x0 + (i + 0.5) * LOGIT_W, BAR_H * 1.3 + 0.3 + (i % 3) * 0.32, logitSlab.zc);
    world.add(s);
  });

  // data-flow path through slab centres + residual skip arcs
  const pathMat = new THREE.LineBasicMaterial({ color: '#334155' });
  for (const row of new Set(state.slabs.map((s) => s.row))) {
    const pts = state.slabs.filter((s) => s.row === row).map((s) => s.center.clone().setY(-0.05));
    world.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), pathMat));
  }
  const arcMat = new THREE.MeshBasicMaterial({ color: '#6d5bd0', transparent: true, opacity: 0.35 });
  const byKey = Object.fromEntries(state.slabs.map((s) => [s.key, s]));
  for (let l = 0; l < data.config.n_layer; l++) {
    const r1 = byKey[`L${l}.resid1`], r2 = byKey[`L${l}.resid2`], ln1 = byKey[`L${l}.ln1`];
    const start = new THREE.Vector3(ln1.x0 - 0.6, 0, ln1.zc);
    for (const [a, b] of [[start, r1.center], [r1.center, r2.center]]) {
      const mid = a.clone().lerp(b, 0.5).setY(BAR_H + 2.2 + Math.abs(b.x - a.x) * 0.04);
      const curve = new THREE.QuadraticBezierCurve3(a.clone().setY(0.1), mid, b.clone().setY(0.1));
      world.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 40, 0.022, 6), arcMat));
    }
  }
  applyMode();
}

const tmpM = new THREE.Matrix4();
const tmpC = new THREE.Color();
function applySlab(slab) {
  const grad = state.mode === 'grad';
  const arr = grad ? slab.grad : slab.val;
  let scale = grad ? slab.st.gscale : slab.st.vscale;
  let offset = 0;
  if (slab.kind === 'logits' && !grad) {
    let mn = Infinity, mx = -Infinity;
    for (const v of arr) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
    offset = mn - (mx - mn) * 0.15;
    scale = mx - offset;
  }
  const n = slab.vidx.length;
  const fp = slab.kind === 'attn' ? (slab.tile / slab.T) * 0.88 : slab.kind === 'logits' ? LOGIT_W * 0.8 : CELL * 0.86;
  for (let i = 0; i < n; i++) {
    const v = arr[slab.vidx[i]] - offset;
    const k = Math.min(Math.abs(v) / (scale || 1), 1.3);
    const h = Math.max(k * BAR_H * (slab.kind === 'attn' && !grad ? 0.6 : 1), 0.006);
    const sgn = v < 0 ? -1 : 1;
    tmpM.makeScale(fp, h, fp);
    tmpM.setPosition(slab.pos[2 * i], (sgn * h) / 2, slab.pos[2 * i + 1]);
    slab.mesh.setMatrixAt(i, tmpM);
    const c = Math.pow(Math.min(k, 1), 0.55);
    if (!grad && slab.kind === 'attn') {
      tmpC.copy(COLORS.base).lerp(COLORS.attn, Math.min(1, c * 1.4));
      if (c > 0.7) tmpC.lerp(COLORS.white, (c - 0.7) / 0.6);
    } else if (!grad && slab.kind === 'logits') {
      tmpC.copy(COLORS.base).lerp(COLORS.valPos, 0.3 + 0.7 * c);
    } else {
      const end = grad ? (sgn > 0 ? COLORS.gradPos : COLORS.gradNeg) : (sgn > 0 ? COLORS.valPos : COLORS.valNeg);
      tmpC.copy(COLORS.base).lerp(end, c);
    }
    slab.mesh.setColorAt(i, tmpC);
  }
  slab.mesh.instanceMatrix.needsUpdate = true;
  slab.mesh.instanceColor.needsUpdate = true;
  slab.mesh.boundingSphere = null; // InstancedMesh caches it for raycasting
}

function applyMode() {
  state.slabs.forEach(applySlab);
  $('mode-val').classList.toggle('on', state.mode === 'val');
  $('mode-grad').classList.toggle('on', state.mode === 'grad');
  renderTimeline();
  if (state.selection) refreshMarker();
}

function setProgress(slab, p) {
  slab.progress = p;
  slab.group.scale.y = 0.03 + 0.97 * p;
  slab.mat.opacity = 0.18 + 0.82 * p;
  slab.label.material.opacity = 0.35 + 0.65 * p;
}

// ---------------------------------------------------------------- camera
function setFog(near, far) { scene.fog.near = near; scene.fog.far = far; }

function focus(slab, dur = 700) {
  setFog(28, 95);
  const target = slab.center.clone();
  const span = Math.max(slab.w, slab.depth * 2.2, 8);
  const offset = new THREE.Vector3(0, 0.62 * span + 3.5, 0.7 * span + 5);
  const fromT = controls.target.clone();
  const fromP = camera.position.clone();
  const toP = target.clone().add(offset);
  tween(dur, (k) => {
    controls.target.lerpVectors(fromT, target, k);
    camera.position.lerpVectors(fromP, toP, k);
  });
}

function overview() {
  const box = new THREE.Box3();
  state.slabs.forEach((s) => box.expandByPoint(new THREE.Vector3(s.x0, 0, s.z0)).expandByPoint(new THREE.Vector3(s.x0 + s.w, 0, s.z0 + s.depth)));
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  const fromT = controls.target.clone(), fromP = camera.position.clone();
  state.slabs.forEach((s) => setProgress(s, 1));
  const toP = c.clone().add(new THREE.Vector3(0, 0.55 * size, 0.3 * size));
  const dist = toP.distanceTo(c);
  setFog(dist * 0.9, dist * 2.2);
  tween(900, (k) => { controls.target.lerpVectors(fromT, c, k); camera.position.lerpVectors(fromP, toP, k); });
}

// ---------------------------------------------------------------- playback
function stepDuration() { return 520 / state.speed; }

function stopPlay() {
  clearTimeout(state.timer);
  state.timer = null;
  $('pause').disabled = true;
}

function revealUpTo(idx) {
  // forward: everything at or before idx is computed; backward: everything at or after idx has a gradient
  state.slabs.forEach((s, i) => setProgress(s, state.dir === 'fwd' ? (i <= idx ? 1 : 0) : (i >= idx ? 1 : 0)));
}

function play(dir) {
  stopPlay();
  state.dir = dir;
  state.mode = dir === 'fwd' ? 'val' : 'grad';
  applyMode();
  const n = state.slabs.length;
  state.step = dir === 'fwd' ? -1 : n;
  state.slabs.forEach((s) => setProgress(s, 0));
  pulseCore.material.color.set(dir === 'fwd' ? '#fbbf24' : '#f0abfc');
  pulseHalo.material.color.set(dir === 'fwd' ? '#fbbf24' : '#f0abfc');
  $('pause').disabled = false;
  advance();
}

function advance() {
  const n = state.slabs.length;
  const next = state.step + (state.dir === 'fwd' ? 1 : -1);
  if (next < 0 || next >= n) { stopPlay(); pulse.visible = false; return; }
  goToStep(next, true);
  state.timer = setTimeout(advance, stepDuration());
}

function goToStep(idx, animated) {
  const prev = state.slabs[state.step];
  state.step = idx;
  const slab = state.slabs[idx];
  const dur = stepDuration();
  state.slabs.forEach((s) => s.plate.material.color.copy(COLORS.plate));
  slab.plate.material.color.copy(state.dir === 'fwd' ? COLORS.plateFwd : COLORS.plateBwd);
  if (animated) {
    setProgress(slab, 0);
    tween(dur * 0.85, (k) => setProgress(slab, k));
    if (prev) {
      const a = prev.center.clone().setY(0.4), b = slab.center.clone().setY(0.4);
      const mid = a.clone().lerp(b, 0.5).setY(1.6 + a.distanceTo(b) * 0.08);
      const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
      pulse.visible = true;
      tween(dur * 0.6, (k) => pulse.position.copy(curve.getPoint(k)));
    }
  } else {
    revealUpTo(idx);
  }
  focus(slab, Math.min(700, dur * 0.9));
  updateStageLabel();
  renderTimeline();
  if (state.speed <= 2 || !state.timer) autoInspect(slab);
}

function updateStageLabel() {
  const s = state.slabs[state.step];
  if (!s) { $('stage-label').textContent = ''; return; }
  const dirTxt = state.dir === 'fwd' ? 'forward' : 'backward';
  const n = state.slabs.length;
  const pos = state.dir === 'fwd' ? state.step + 1 : n - state.step;
  $('stage-label').innerHTML = `<b>${esc(stageTitle(s.st))}</b> · ${dirTxt} ${pos}/${n}`;
}

// ---------------------------------------------------------------- inspection
function autoInspect(slab) {
  const T = state.data.T;
  const grad = state.mode === 'grad';
  if (slab.kind === 'attn') {
    const head = state.selection?.head ?? 0;
    return inspect(slab, { token: T - 1, head });
  }
  if (slab.kind === 'logits') return inspect(slab, { token: T - 1, dim: state.data.target.id, idx: state.data.top.findIndex((x) => x.id === state.data.target.id) });
  const arr = grad ? slab.grad : slab.val;
  const bins = state.data.config.bins;
  let best = 0;
  for (let b = 1; b < bins; b++) if (Math.abs(arr[(T - 1) * bins + b]) > Math.abs(arr[(T - 1) * bins + best])) best = b;
  return inspect(slab, { token: T - 1, bin: best });
}

async function inspect(slab, sel) {
  state.selection = { slab, ...sel };
  refreshMarker();
  state.detailAbort?.abort();
  const ctl = new AbortController();
  state.detailAbort = ctl;
  const body = {
    run_id: state.data.run_id, key: slab.key, token: sel.token, mode: state.mode === 'grad' ? 'backward' : 'forward',
    bin: sel.bin ?? null, dim: sel.dim ?? null, head: sel.head ?? 0, key_token: sel.keyToken ?? null,
  };
  try {
    const res = await fetch('/api/detail', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: ctl.signal });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    renderInspector(await res.json());
  } catch (err) {
    if (err.name !== 'AbortError') $('inspector').innerHTML = `<p class="verdict bad">${esc(err.message)}</p>`;
  }
}

function cellPosition(slab, sel) {
  const T = slab.T;
  if (slab.kind === 'vec') {
    return { x: slab.x0 + (sel.bin + 0.5) * CELL, z: slab.z0 + (sel.token + 0.5) * CELL, s: CELL, i: sel.token * state.data.config.bins + sel.bin };
  }
  if (slab.kind === 'attn') {
    const c = slab.tile / T, h = sel.head ?? 0, j = sel.keyToken ?? 0;
    return { x: slab.x0 + (h % 6) * (slab.tile + TILE_GAP) + (j + 0.5) * c, z: slab.z0 + Math.floor(h / 6) * (slab.tile + TILE_GAP) + (sel.token + 0.5) * c, s: c, i: h * T * T + sel.token * T + j };
  }
  return { x: slab.x0 + (sel.idx + 0.5) * LOGIT_W, z: slab.zc, s: LOGIT_W, i: sel.idx };
}

function refreshMarker() {
  const sel = state.selection;
  if (!sel || (sel.slab.kind === 'attn' && sel.keyToken == null) || (sel.slab.kind === 'logits' && sel.idx < 0)) { marker.visible = false; return; }
  const p = cellPosition(sel.slab, sel);
  const arr = state.mode === 'grad' ? sel.slab.grad : sel.slab.val;
  const scale = (state.mode === 'grad' ? sel.slab.st.gscale : sel.slab.st.vscale) || 1;
  const v = sel.slab.kind === 'logits' && state.mode !== 'grad' ? 1 : arr[p.i] / scale;
  const h = Math.max(Math.min(Math.abs(v), 1.3) * BAR_H, 0.05) + 0.12;
  marker.visible = true;
  marker.scale.set(p.s * 1.6, h, p.s * 1.6);
  marker.position.set(p.x, (v < 0 ? -h : h) / 2, p.z);
}

function renderSteps(steps) {
  if (!steps?.length) return '';
  return `<table class="steps">${steps.map((s) => `<tr><td>${tex(s.label)}</td><td class="num">${fmt(s.value)}</td></tr>`).join('')}</table>`;
}

function renderTerms(terms, resultLabel) {
  if (!terms) return '';
  const rows = terms.rows.map((r) => `<tr><td>${esc(r.la)}·${esc(r.lb)}</td><td class="num">${fmt(r.a)} × ${fmt(r.b)}</td><td class="num">${fmt(r.prod)}</td></tr>`).join('');
  const more = terms.n > terms.rows.length ? `<tr><td colspan="3" class="more">… ${terms.n - terms.rows.length} more terms (largest |term| shown)</td></tr>` : '';
  return `<table>${rows}${more}<tr class="sum"><td>Σ all ${terms.n} terms</td><td></td><td class="num">${fmt(terms.sum)}</td></tr></table>${resultLabel ?? ''}`;
}

function renderInspector(d) {
  const f = d.forward, b = d.backward;
  const sel = state.selection;
  if (d.key_token != null && sel) { sel.keyToken = d.key_token; sel.head = d.head; refreshMarker(); }
  const chips = [
    d.layer != null ? `block ${d.layer + 1}` : null,
    `token ${d.token} “${esc(showTok(d.token_str))}”`,
    d.dim != null ? (d.dim_label != null ? `vocab ${d.dim} “${esc(showTok(d.dim_label))}”` : `dim ${d.dim}`) : null,
    d.head != null ? `head ${d.head + 1}` : null,
    d.key_token != null ? `key ${d.key_token} “${esc(showTok(d.key_token_str))}”` : null,
  ].filter(Boolean).map((c) => `<span class="chip">${c}</span>`).join('');
  const attnRow = f.row ? `<div class="attn-row" title="softmax row for this query">${f.row.map((r) => `<div class="${r.j === d.key_token ? 'sel' : ''}" style="height:${Math.max(2, r.p * 100)}%" title="${esc(showTok(r.tok))}: ${fmt(r.p)}"></div>`).join('')}</div>` : '';
  const fOk = f.stored != null
    ? `<div class="verdict ${f.ok ? 'ok' : 'bad'}">= ${fmt(f.result)} &nbsp;${f.ok ? '✓ matches' : '✗ differs from'} stored activation ${fmt(f.stored)}</div>`
    : `<div class="verdict ok">= ${fmt(f.result)}</div>`;
  const bOk = `<div class="verdict ${b.ok ? 'ok' : 'bad'}">hand-derived ${fmt(b.manual)} vs autograd ${fmt(b.autograd)} &nbsp;${b.ok ? '✓' : '✗'} |Δ| = ${b.abs_err.toExponential(1)}</div>`;
  $('inspector').innerHTML = `
    <h2>${esc(d.title)}</h2>
    <p class="desc">${esc(d.desc)}</p>
    <div class="chips">${chips}</div>
    <div class="calc fwd">
      <h3>Forward pass</h3>
      <div class="formula">${tex(f.formula, true)}</div>
      ${renderTerms(f.terms)}
      ${renderSteps(f.steps)}
      ${attnRow}
      ${fOk}
    </div>
    <div class="calc bwd">
      <h3>Backward pass · ∂L/∂(this)</h3>
      <div class="formula">${tex(b.formula, true)}</div>
      ${renderTerms(b.terms)}
      ${renderSteps(b.steps)}
      ${bOk}
    </div>
    <p class="desc">L = −log p(“${esc(showTok(state.data.target.token))}” | text) = ${fmt(state.data.loss)}. Gradients are of this loss.</p>`;
}

// ---------------------------------------------------------------- picking
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
function pick(ev) {
  if (!state.slabs.length) return null;
  const r = canvas.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  // cheap box test first, then exact per-instance raycast on the few candidate slabs
  const hits = [];
  for (const slab of state.slabs) {
    if (raycaster.ray.intersectsBox(slab.box)) slab.mesh.raycast(raycaster, hits);
  }
  if (!hits.length) return null;
  hits.sort((a, b) => a.distance - b.distance);
  const hit = hits[0];
  const slab = state.slabs.find((s) => s.mesh === hit.object);
  const i = hit.instanceId;
  const t = slab.map[3 * i], u = slab.map[3 * i + 1], h = slab.map[3 * i + 2];
  if (slab.kind === 'vec') return { slab, token: t, bin: u };
  if (slab.kind === 'attn') return { slab, token: t, keyToken: u, head: h };
  return { slab, token: t, idx: u, dim: state.data.top[u].id };
}

function describe(p) {
  const { slab } = p;
  const pos = cellPosition(slab, p);
  const v = slab.val[pos.i], g = slab.grad[pos.i];
  const tk = showTok(state.data.tokens[p.token]);
  let where = `token “${tk}”`;
  if (slab.kind === 'vec') where += ` · dims ${p.bin * (slab.st.dim / state.data.config.bins)}–${(p.bin + 1) * (slab.st.dim / state.data.config.bins) - 1}`;
  if (slab.kind === 'attn') where = `head ${p.head + 1} · “${tk}” → “${showTok(state.data.tokens[p.keyToken])}”`;
  if (slab.kind === 'logits') where = `“${showTok(state.data.top[p.idx].token)}”`;
  return `${stageTitle(slab.st)}<br>${esc(where)}<br>value ${fmt(v)} · ∂L/∂ ${fmt(g)}`;
}

let hoverRaf = 0;
canvas.addEventListener('pointermove', (ev) => {
  cancelAnimationFrame(hoverRaf);
  hoverRaf = requestAnimationFrame(() => {
    const p = pick(ev);
    const tip = $('tooltip');
    if (!p) { tip.hidden = true; return; }
    tip.innerHTML = describe(p);
    tip.style.left = `${ev.clientX + 14}px`;
    tip.style.top = `${ev.clientY + 14}px`;
    tip.hidden = false;
  });
});
canvas.addEventListener('pointerleave', () => { $('tooltip').hidden = true; });
let downAt = null;
canvas.addEventListener('pointerdown', (ev) => { downAt = [ev.clientX, ev.clientY]; });
canvas.addEventListener('pointerup', (ev) => {
  if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4) return;
  const p = pick(ev);
  if (p) inspect(p.slab, p);
});

// ---------------------------------------------------------------- timeline
function renderTimeline() {
  const d = state.data;
  if (!d) return;
  const rows = new Map();
  state.slabs.forEach((s, i) => { if (!rows.has(s.row)) rows.set(s.row, []); rows.get(s.row).push(i); });
  const norms = [...rows.values()].map((idxs) => Math.max(...state.slabs[idxs[idxs.length - 1]].st.gnorm));
  const maxLog = Math.max(...norms.map((n) => Math.log10(n + 1e-12)));
  const minLog = Math.min(...norms.map((n) => Math.log10(n + 1e-12)));
  const cur = state.slabs[state.step];
  let html = `<h4>Layers · ‖∂L/∂x‖</h4>`;
  [...rows.entries()].forEach(([row, idxs], r) => {
    const name = row === 0 ? 'Embedding' : row <= d.config.n_layer ? `Block ${row}` : 'Output';
    const done = state.step >= 0 && (state.dir === 'fwd' ? idxs[0] <= state.step : idxs[idxs.length - 1] >= state.step);
    const pct = ((Math.log10(norms[r] + 1e-12) - minLog) / (maxLog - minLog || 1)) * 100;
    html += `<div class="row ${cur && cur.row === row ? 'active' : ''} ${done ? 'done' : ''}" data-step="${idxs[0]}">
      <span>${name}</span><span class="bar" title="max ‖∂L/∂x‖ = ${fmt(norms[r])}"><i style="width:${Math.max(4, pct)}%"></i></span></div>`;
  });
  $('timeline').innerHTML = html;
}
$('timeline').addEventListener('click', (ev) => {
  const row = ev.target.closest('.row');
  if (!row) return;
  stopPlay();
  goToStep(Number(row.dataset.step), false);
});

// ---------------------------------------------------------------- predictions (mirrors the original's sampling)
function softmax(xs) {
  const m = Math.max(...xs);
  const e = xs.map((x) => (x === -Infinity ? 0 : Math.exp(x - m)));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

function samplingProbs() {
  const temp = Number($('temp').value);
  const type = document.querySelector('input[name=samp]:checked').value;
  const val = Number($('samp-val').value);
  const top = [...state.data.top].sort((a, b) => b.logit - a.logit).map((t) => ({ ...t, scaled: t.logit / temp }));
  if (type === 'top-k') {
    const p = softmax(top.map((t, i) => (i < val ? t.scaled : -Infinity)));
    return top.map((t, i) => ({ ...t, p: p[i], cut: i >= val }));
  }
  const p = softmax(top.map((t) => t.scaled));
  let cum = 0, cutoff = p.length - 1;
  for (let i = 0; i < p.length; i++) { cum += p[i]; if (cum >= val) { cutoff = i; break; } }
  const kept = p.slice(0, cutoff + 1), z = kept.reduce((a, b) => a + b, 0);
  return top.map((t, i) => ({ ...t, p: i <= cutoff ? p[i] / z : 0, cut: i > cutoff }));
}

function renderPredictions() {
  const probs = samplingProbs();
  let r = Math.random(), sampled = probs[probs.length - 1];
  for (const t of probs) { r -= t.p; if (r < 0) { sampled = t; break; } }
  state.sampled = sampled;
  $('pred-list').innerHTML = probs.slice(0, 8).map((t) => `
    <li data-id="${t.id}" class="${t.id === state.data.target.id ? 'target' : ''} ${t.id === sampled.id ? 'sampled' : ''} ${t.cut ? 'cut' : ''}">
      <span class="tok">${esc(showTok(t.token))}</span>
      <span class="pbar"><i style="width:${(t.p * 100).toFixed(1)}%"></i></span>
      <span>${(t.p * 100).toFixed(1)}%</span>
    </li>`).join('');
  $('target-tok').textContent = showTok(state.data.target.token);
  $('trunc-note').hidden = !state.data.truncated_from;
  $('trunc-note').textContent = state.data.truncated_from ? `Input is ${state.data.truncated_from} tokens; showing the last ${state.data.T}.` : '';
  $('loss').textContent = fmt(state.data.loss);
  $('predictions').hidden = false;
}

$('pred-list').addEventListener('click', (ev) => {
  const li = ev.target.closest('li');
  if (li) runModel(state.lastText, Number(li.dataset.id), 'bwd');
});
$('append').addEventListener('click', () => {
  if (!state.sampled) return;
  $('text').value = state.lastText + state.sampled.token;
  runModel($('text').value, null, 'fwd');
});

// ---------------------------------------------------------------- run
async function runModel(text, target = null, autoplay = 'fwd') {
  stopPlay();
  const status = $('status');
  status.textContent = target == null ? 'Running GPT-2 forward + backward…' : 'Re-running backward pass for the new target…';
  status.hidden = false;
  $('run').disabled = true;
  try {
    const res = await fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, target }) });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    state.data = await res.json();
    state.lastText = text;
    state.selection = null;
    state.step = -1;
    buildScene(state.data);
    renderPredictions();
    ['play-fwd', 'play-bwd', 'prev', 'next'].forEach((id) => { $(id).disabled = false; });
    status.hidden = true;
    if (autoplay) play(autoplay);
  } catch (err) {
    status.textContent = `Failed: ${err.message}`;
  } finally {
    $('run').disabled = false;
  }
}

// ---------------------------------------------------------------- controls
$('run-form').addEventListener('submit', (ev) => { ev.preventDefault(); runModel($('text').value, null, 'fwd'); });
$('examples').addEventListener('change', (ev) => { if (ev.target.value) { $('text').value = ev.target.value; runModel(ev.target.value); } ev.target.value = ''; });
$('play-fwd').addEventListener('click', () => play('fwd'));
$('play-bwd').addEventListener('click', () => play('bwd'));
$('pause').addEventListener('click', stopPlay);
$('overview').addEventListener('click', () => { stopPlay(); overview(); });
$('speed').addEventListener('input', (ev) => { state.speed = Number(ev.target.value); });
$('prev').addEventListener('click', () => { stopPlay(); const n = state.slabs.length; goToStep(Math.max(0, Math.min(n - 1, state.step - 1)), false); });
$('next').addEventListener('click', () => { stopPlay(); const n = state.slabs.length; goToStep(Math.max(0, Math.min(n - 1, state.step + 1)), false); });
$('mode-val').addEventListener('click', () => { state.mode = 'val'; applyMode(); });
$('mode-grad').addEventListener('click', () => { state.mode = 'grad'; applyMode(); });
$('temp').addEventListener('input', (ev) => { $('temp-out').textContent = ev.target.value; if (state.data) renderPredictions(); });
function syncSampling() {
  const type = document.querySelector('input[name=samp]:checked').value;
  const r = $('samp-val');
  if (type === 'top-k') { r.min = 1; r.max = 20; r.step = 1; if (Number(r.value) < 1) r.value = 5; $('samp-out').textContent = `k=${r.value}`; }
  else { r.min = 0.05; r.max = 1; r.step = 0.05; if (Number(r.value) > 1) r.value = 0.9; $('samp-out').textContent = `p=${r.value}`; }
  if (state.data) renderPredictions();
}
document.querySelectorAll('input[name=samp]').forEach((el) => el.addEventListener('change', syncSampling));
$('samp-val').addEventListener('input', syncSampling);
window.addEventListener('keydown', (ev) => {
  if (ev.target.tagName === 'INPUT' || !state.data) return;
  if (ev.key === 'ArrowRight') $('next').click();
  if (ev.key === 'ArrowLeft') $('prev').click();
  if (ev.key === ' ') { ev.preventDefault(); state.timer ? stopPlay() : play(state.dir); }
});

resize();
requestAnimationFrame(animate);
runModel($('text').value, null, 'fwd');
