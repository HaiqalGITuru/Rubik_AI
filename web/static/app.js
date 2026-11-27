import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';
import { RubikCube } from '/static/cube3d.js';
import { NetworkView } from '/static/network.js';

const $ = id => document.getElementById(id);
const fmt = (n, d = 0) => Number(n).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });

// ------------------------------------------------------------------ //
// 3D cube
// ------------------------------------------------------------------ //
const view = $('cube-view');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
const CAMERA_BASE = new THREE.Vector3(6.4, 5.4, 8.2); // white on top, green in front, red on the right
camera.position.copy(CAMERA_BASE);
scene.add(new THREE.HemisphereLight('#ffffff', '#b8c0cc', 2.2));
const sun = new THREE.DirectionalLight('#ffffff', 1.6);
sun.position.set(4, 8, 6);
scene.add(sun);

const cube = new RubikCube();
scene.add(cube);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 6;
controls.maxDistance = 16;

function resize() {
  const { width, height } = view.getBoundingClientRect();
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view);
resize();

let last = performance.now();
renderer.setAnimationLoop(now => {
  cube.update(now - last);
  last = now;
  controls.update();
  renderer.render(scene, camera);
});

// ------------------------------------------------------------------ //
// App state
// ------------------------------------------------------------------ //
let network = null;
let applied = [];          // every move since the cube was solved (sent to the server)
let busy = false;
let history = [];

const speed = () => Number($('speed').value);
const solveMs = () => 480 / speed();
const scrambleMs = () => 150 / speed();

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? 'POST' : 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'The server is not responding. Make sure python -m web.server is still running.');
  return data;
}

function setStatus(text, thinking = false) {
  $('status').textContent = text;
  $('status').classList.toggle('thinking', thinking);
}

function setBusy(b) {
  busy = b;
  for (const id of ['btn-scramble', 'btn-solve', 'btn-reset', 'beam']) $(id).disabled = b;
  $('scramble-form').querySelector('button').disabled = b;
}

function renderMoves(el, moves, emptyText) {
  el.replaceChildren();
  if (!moves.length) {
    const s = document.createElement('span');
    s.className = 'empty';
    s.textContent = emptyText;
    el.append(s);
    return [];
  }
  return moves.map(m => {
    const s = document.createElement('span');
    s.className = 'move';
    s.textContent = m;
    el.append(s);
    return s;
  });
}

async function analyzeCurrent() {
  const { frame } = await api('/api/analyze', { moves: applied });
  network.show(frame, 600);
}

// ------------------------------------------------------------------ //
// Scramble & apply a scramble
// ------------------------------------------------------------------ //
async function runScramble(moves) {
  setBusy(true);
  cube.reset();
  applied = [];
  clearSolution();
  renderMoves($('scramble-moves'), moves, 'Not scrambled yet');
  $('scramble-text').value = moves.join(' ');
  setStatus('Scrambling');
  for (const m of moves) {
    await cube.move(m, scrambleMs());
    applied.push(m);
  }
  setStatus(`Scrambled with ${moves.length} moves`);
  await analyzeCurrent();
  setBusy(false);
}

$('btn-scramble').addEventListener('click', async () => {
  try {
    const { moves } = await api('/api/scramble', {});
    await runScramble(moves);
  } catch (e) { showError(e); }
});

$('scramble-form').addEventListener('submit', async ev => {
  ev.preventDefault();
  $('scramble-error').textContent = '';
  try {
    const { moves } = await api('/api/parse', { text: $('scramble-text').value });
    await runScramble(moves);
  } catch (e) {
    $('scramble-error').textContent = e.message;
  }
});

$('btn-reset').addEventListener('click', async () => {
  cube.reset();
  applied = [];
  clearSolution();
  renderMoves($('scramble-moves'), [], 'Not scrambled yet');
  setStatus('Cube is solved');
  await analyzeCurrent();
});

// ------------------------------------------------------------------ //
// Solve with AI
// ------------------------------------------------------------------ //
function clearSolution() {
  renderMoves($('solution-moves'), [], 'Scramble the cube, then press Solve with AI');
  renderMoves($('kociemba-moves'), [], 'The baseline appears after the AI finishes');
  $('btn-copy').disabled = true;
}

$('btn-solve').addEventListener('click', async () => {
  if (!applied.length) {
    setStatus('The cube is already solved. Scramble it first.');
    return;
  }
  setBusy(true);
  const beam = Number($('beam').value);
  setStatus(`The AI is thinking (search width ${fmt(beam)})`, true);
  let res;
  try {
    res = await api('/api/solve', { moves: applied, beam });
  } catch (e) {
    showError(e);
    setBusy(false);
    return;
  }
  if (!res.solved) {
    setStatus(res.message);
    setBusy(false);
    return;
  }

  const chips = renderMoves($('solution-moves'), res.moves, '');
  $('btn-copy').disabled = false;
  if (res.kociemba) renderMoves($('kociemba-moves'), res.kociemba.moves, '');
  renderStats(res);
  const bars = renderConfidence(res.steps);
  const scrambleText = applied.join(' ');

  for (let i = 0; i < res.moves.length; i++) {
    chips.forEach((c, k) => { c.classList.toggle('current', k === i); c.classList.toggle('done', k < i); });
    bars.forEach((b, k) => b.classList.toggle('current', k === i));
    setStatus(`Move ${i + 1} of ${res.moves.length}: ${res.moves[i]}`);
    network.show(res.frames[i], solveMs() * 0.8);
    await cube.move(res.moves[i], solveMs());
  }
  chips.forEach(c => c.classList.remove('current', 'done'));
  bars.forEach(b => b.classList.remove('current'));
  network.show(res.frames[res.frames.length - 1], 600);
  applied = [];
  setStatus(`Solved in ${res.moves.length} moves`);
  addHistory(scrambleText, res);
  setBusy(false);
});

$('btn-copy').addEventListener('click', async () => {
  const text = [...$('solution-moves').querySelectorAll('.move')].map(e => e.textContent).join(' ');
  try {
    await navigator.clipboard.writeText(text);
    $('btn-copy').textContent = 'Copied';
    setTimeout(() => { $('btn-copy').textContent = 'Copy'; }, 1500);
  } catch { /* clipboard not available */ }
});

$('speed').addEventListener('input', () => {
  $('speed-value').textContent = `${speed()}x`;
});

function showError(e) {
  setStatus(e.message);
}

// ------------------------------------------------------------------ //
// Statistics, confidence chart, history
// ------------------------------------------------------------------ //
function renderStats(res) {
  const s = res.stats, k = res.kociemba;
  const rows = [
    ['Moves', `${s.length}`, k ? `${k.length}` : '–', k && s.length <= k.length ? 'ai' : k ? 'k' : ''],
    ['Thinking time', s.time_ms >= 1000 ? `${fmt(s.time_ms / 1000, 2)} s` : `${fmt(s.time_ms)} ms`,
      k ? `${fmt(k.time_ms, 1)} ms` : '–', k ? 'k' : ''],
    ['Positions evaluated', fmt(s.nodes), '–', ''],
    ['Mean confidence', `${fmt(s.avg_confidence * 100)}%`, '–', ''],
  ];
  $('stats-body').replaceChildren(...rows.map(([label, a, b, win]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<th>${label}</th><td class="${win === 'ai' ? 'win' : ''}">${a}</td><td class="${win === 'k' ? 'win' : ''}">${b}</td>`;
    return tr;
  }));
}

function renderConfidence(steps) {
  const W = 640, H = 170, top = 10, bottom = 28, left = 30;
  const n = steps.length, gap = 3;
  const bw = Math.min(26, Math.max(4, (W - left - gap * n) / n));
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'AI confidence at every move of the solution');
  for (const p of [0, 0.5, 1]) {
    const y = top + (H - top - bottom) * (1 - p);
    const line = document.createElementNS(ns, 'line');
    Object.entries({ x1: left, x2: W, y1: y, y2: y, stroke: '#DFE4EB' }).forEach(([a, v]) => line.setAttribute(a, v));
    const t = document.createElementNS(ns, 'text');
    Object.entries({ x: 0, y: y + 3 }).forEach(([a, v]) => t.setAttribute(a, v));
    t.textContent = `${p * 100}%`;
    svg.append(line, t);
  }
  const bars = steps.map((s, i) => {
    const h = (H - top - bottom) * s.confidence;
    const x = left + i * (bw + gap) + gap;
    const rect = document.createElementNS(ns, 'rect');
    Object.entries({ x, y: H - bottom - h, width: bw, height: Math.max(1, h), rx: 2, class: 'bar' })
      .forEach(([a, v]) => rect.setAttribute(a, v));
    const title = document.createElementNS(ns, 'title');
    title.textContent = `Move ${i + 1} (${s.move}): ${Math.round(s.confidence * 100)}%`;
    rect.append(title);
    const label = document.createElementNS(ns, 'text');
    Object.entries({ x: x + bw / 2, y: H - 10, 'text-anchor': 'middle' }).forEach(([a, v]) => label.setAttribute(a, v));
    label.textContent = s.move;
    svg.append(rect, label);
    return rect;
  });
  $('confidence').replaceChildren(svg);
  return bars;
}

function addHistory(scramble, res) {
  history.unshift({ scramble, res });
  $('history-body').replaceChildren(...history.map((h, i) => {
    const tr = document.createElement('tr');
    const s = h.res.stats;
    tr.innerHTML = `<td>${history.length - i}</td>
      <td class="mono" title="${h.scramble}">${h.scramble}</td>
      <td>${s.length} moves</td>
      <td>${s.time_ms >= 1000 ? `${fmt(s.time_ms / 1000, 2)} s` : `${fmt(s.time_ms)} ms`}</td>
      <td>${h.res.kociemba ? `${h.res.kociemba.length} moves` : '–'}</td>
      <td>${fmt(s.beam)}</td>`;
    return tr;
  }));
}

// ------------------------------------------------------------------ //
// Start
// ------------------------------------------------------------------ //
(async () => {
  try {
    const info = await api('/api/info');
    $('info-steps').textContent = `${fmt(info.train_steps)} steps`;
    $('info-params').textContent = `${fmt(info.parameters / 1e6, 2)} million`;
    network = new NetworkView($('network'), { moves: info.moves, layerSizes: info.layers, showNeurons: info.show_neurons, size: 3 });
    clearSolution();
    setStatus('Cube is solved');
    await analyzeCurrent();
  } catch (e) {
    showError(e);
  }
})();
