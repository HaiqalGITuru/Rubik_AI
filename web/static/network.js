// Visualisasi jaringan saraf dengan aktivasi ASLI dari model.
// Input: 54 stiker kubus. Tersembunyi: sampel neuron dari setiap lapisan.
// Output: probabilitas 12 gerakan; gerakan yang dipilih AI disorot.

import { COLORS } from './cube3d.js';

const INK = '#18202C';
const MUTED = '#8A93A3';
const RULE = '#DFE4EB';
const SIGNAL = [36, 83, 232]; // biru sinyal

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function mix(a, b, t) { return a + (b - a) * t; }

export class NetworkView {
  constructor(canvas, config) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.configure(config, false);
    this.prev = null;
    this.frame = null;
    this.wave = 1;                          // 0..1, posisi gelombang sinyal kiri -> kanan
    this.waveMs = 700;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    requestAnimationFrame(t => this.loop(t));
  }

  // Atur model yang divisualisasikan
  configure({ moves, layerSizes, showNeurons, size = 3 }, relayout = true) {
    this.moves = moves;                     // 12 gerakan
    this.layerSizes = layerSizes;           // ukuran asli, misal [324, 1024, 512, 512, 512, 12]
    this.showNeurons = showNeurons;         // jumlah neuron yang digambar per lapisan tersembunyi
    this.size = size;
    this.prev = null;
    this.frame = null;
    if (relayout) this.layout();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = this.canvas.getBoundingClientRect();
    this.w = width; this.h = height;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.layout();
  }

  layout() {
    const padTop = 34, padBottom = 18;
    const inputW = Math.min(150, this.w * 0.22);
    const outputW = Math.min(150, this.w * 0.26);
    const left = 12 + inputW + 26;
    const right = this.w - outputW - 14;
    const cols = this.showNeurons.length;
    this.cols = this.showNeurons.map((n, c) => {
      const x = left + ((right - left) * (c + 0.5)) / cols;
      const span = this.h - padTop - padBottom;
      const gap = span / n;
      return Array.from({ length: n }, (_, i) => ({ x, y: padTop + gap * (i + 0.5) }));
    });
    this.nodeR = Math.max(3, Math.min(7, (this.h - padTop - padBottom) / Math.max(...this.showNeurons) / 2 - 2));
    this.inputBox = { x: 12, w: inputW, top: padTop };
    this.outputBox = { x: this.w - outputW, w: outputW - 8, top: padTop, bottom: this.h - padBottom };
    const rowH = (this.outputBox.bottom - this.outputBox.top) / this.moves.length;
    this.outputs = this.moves.map((_, i) => ({ x: this.outputBox.x, y: this.outputBox.top + rowH * (i + 0.5), h: rowH }));
  }

  // Tampilkan frame baru: {input, hidden: [[...]], output: [...], chosen?}
  show(frame, durationMs = 700) {
    this.prev = this.frame;
    this.frame = frame;
    this.waveMs = Math.max(150, durationMs);
    this.wave = reduceMotion ? 1 : 0;
  }

  loop(t) {
    const dt = this.lastT ? t - this.lastT : 16;
    this.lastT = t;
    if (this.wave < 1) this.wave = Math.min(1, this.wave + dt / this.waveMs);
    this.draw();
    requestAnimationFrame(tt => this.loop(tt));
  }

  // Seberapa jauh gelombang sudah melewati kolom ke-k (0..1), kolom: input, hidden..., output
  reached(k) {
    const total = this.cols.length + 2;
    return Math.max(0, Math.min(1, this.wave * total - k));
  }

  value(layer, i) {
    const now = this.frame ? this.frame.hidden[layer][i] : 0;
    const before = this.prev ? this.prev.hidden[layer][i] : 0;
    return mix(before, now, this.reached(layer + 1));
  }

  draw() {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.font = '500 11px "IBM Plex Sans", system-ui, sans-serif';
    ctx.textBaseline = 'middle';

    // Judul kolom (ukuran lapisan asli)
    ctx.fillStyle = MUTED;
    ctx.textAlign = 'center';
    ctx.fillText(`${6 * this.size * this.size} stiker`, this.inputBox.x + this.inputBox.w / 2, 14);
    this.cols.forEach((col, c) => {
      ctx.fillText(this.layerSizes[c + 1].toLocaleString('id-ID'), col[0].x, 14);
    });
    ctx.fillText(`${this.moves.length} gerakan`, this.outputBox.x + this.outputBox.w / 2, 14);

    this.drawEdges();
    this.drawInput();
    this.cols.forEach((col, c) => col.forEach((p, i) => {
      const v = this.value(c, i);
      ctx.beginPath();
      ctx.arc(p.x, p.y, this.nodeR, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${SIGNAL.join(',')},${0.06 + 0.94 * v})`;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = v > 0.5 ? `rgb(${SIGNAL.join(',')})` : RULE;
      ctx.stroke();
    }));
    this.drawOutput();
  }

  drawEdges() {
    const { ctx } = this;
    ctx.lineWidth = 1;
    // input -> lapisan pertama: garis dari tepi kotak input
    const first = this.cols[0];
    const ix = this.inputBox.x + this.inputBox.w + 4;
    first.forEach((p, i) => {
      const v = this.value(0, i);
      if (v < 0.15) return;
      ctx.strokeStyle = `rgba(${SIGNAL.join(',')},${0.18 * v})`;
      ctx.beginPath();
      ctx.moveTo(ix, this.h / 2); ctx.lineTo(p.x - this.nodeR, p.y); ctx.stroke();
    });
    for (let c = 0; c < this.cols.length - 1; c++) {
      const a = this.cols[c], b = this.cols[c + 1];
      a.forEach((pa, i) => {
        const va = this.value(c, i);
        b.forEach((pb, j) => {
          const s = va * this.value(c + 1, j);
          ctx.strokeStyle = s > 0.04 ? `rgba(${SIGNAL.join(',')},${0.05 + 0.3 * s})` : 'rgba(24,32,44,0.035)';
          ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
        });
      });
    }
    // lapisan terakhir -> output
    const last = this.cols[this.cols.length - 1];
    const out = this.frame ? this.frame.output : null;
    const r = this.reached(this.cols.length + 1);
    this.outputs.forEach((o, k) => {
      const p = out ? out[k] * r : 0;
      last.forEach((pl, i) => {
        const s = this.value(this.cols.length - 1, i) * Math.sqrt(p);
        if (s < 0.05) return;
        ctx.strokeStyle = `rgba(${SIGNAL.join(',')},${0.25 * s})`;
        ctx.beginPath(); ctx.moveTo(pl.x, pl.y); ctx.lineTo(o.x - 4, o.y); ctx.stroke();
      });
    });
  }

  drawInput() {
    const { ctx } = this;
    const state = this.frame ? this.frame.input : null;
    // Jaring kubus mini: U di atas, L F R B di tengah, D di bawah
    const n = this.size, nn = n * n;
    const cell = Math.max(3, Math.floor(this.inputBox.w / (4 * n)));
    const netW = cell * 4 * n, netH = cell * 3 * n;
    const x0 = this.inputBox.x + (this.inputBox.w - netW) / 2;
    const y0 = this.h / 2 - netH / 2;
    const place = { U: [1, 0], L: [0, 1], F: [1, 1], R: [2, 1], B: [3, 1], D: [1, 2] };
    'URFDLB'.split('').forEach((f, fi) => {
      const [gx, gy] = place[f];
      for (let k = 0; k < nn; k++) {
        const color = state ? COLORS[state[fi * nn + k]] : '#EEF1F5';
        ctx.fillStyle = color;
        const x = x0 + (gx * n + (k % n)) * cell, y = y0 + (gy * n + Math.floor(k / n)) * cell;
        ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        ctx.strokeStyle = 'rgba(24,32,44,0.22)';
        ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      }
    });
  }

  drawOutput() {
    const { ctx } = this;
    const out = this.frame ? this.frame.output : null;
    const chosen = this.frame && this.frame.chosen !== undefined ? this.frame.chosen : -1;
    const r = this.reached(this.cols.length + 1);
    const labelW = 36, barX = this.outputBox.x + labelW, barW = this.outputBox.w - labelW - 34;
    this.outputs.forEach((o, k) => {
      const p = out ? out[k] * r : 0;
      const isChosen = k === chosen && r > 0.6;
      ctx.textAlign = 'left';
      ctx.font = `${isChosen ? 700 : 500} 12px "IBM Plex Mono", ui-monospace, monospace`;
      ctx.fillStyle = isChosen ? `rgb(${SIGNAL.join(',')})` : INK;
      ctx.fillText(this.moves[k], this.outputBox.x, o.y);
      const bh = Math.min(10, o.h * 0.5);
      ctx.fillStyle = '#EEF1F5';
      ctx.fillRect(barX, o.y - bh / 2, barW, bh);
      ctx.fillStyle = isChosen ? `rgb(${SIGNAL.join(',')})` : '#A9B3C2';
      ctx.fillRect(barX, o.y - bh / 2, barW * p, bh);
      ctx.font = '500 10px "IBM Plex Sans", system-ui, sans-serif';
      ctx.fillStyle = MUTED;
      ctx.fillText(`${Math.round(p * 100)}%`, barX + barW + 5, o.y);
    });
  }
}