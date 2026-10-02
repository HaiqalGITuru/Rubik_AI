// Kubus rubik 3D (Three.js).
// Konvensi sama dengan simulator Python: x -> kanan (R), y -> atas (U), z -> depan (F).
// Gerakan searah jarum jam (dilihat dari wajahnya) = rotasi -90 derajat terhadap normal wajah.

import * as THREE from 'three';

// Urutan warna sama dengan Python: 0=putih(U) 1=merah(R) 2=hijau(F) 3=kuning(D) 4=oranye(L) 5=biru(B)
export const COLORS = ['#FFFFFF', '#D7262E', '#14A050', '#FFD21F', '#FF7A1A', '#1E5BD8'];
const FACES = 'URFDLB';
const NORMALS = {
  U: [0, 1, 0], R: [1, 0, 0], F: [0, 0, 1],
  D: [0, -1, 0], L: [-1, 0, 0], B: [0, 0, -1],
};

// Indeks stiker (0..53) dari posisi kubus kecil + arah hadap, sama persis dengan rubik/moves.py
export function stickerIndex([x, y, z], [nx, ny, nz]) {
  let face, row, col;
  if (ny === 1) { face = 'U'; col = x + 1; row = z + 1; }
  else if (ny === -1) { face = 'D'; col = x + 1; row = 1 - z; }
  else if (nz === 1) { face = 'F'; col = x + 1; row = 1 - y; }
  else if (nz === -1) { face = 'B'; col = 1 - x; row = 1 - y; }
  else if (nx === 1) { face = 'R'; col = 1 - z; row = 1 - y; }
  else { face = 'L'; col = z + 1; row = 1 - y; }
  return FACES.indexOf(face) * 9 + row * 3 + col;
}

function colorOfNormal(n) {
  return FACES.indexOf(Object.keys(NORMALS).find(f => NORMALS[f].every((v, i) => v === n[i])));
}

function roundedSquare(size, radius) {
  const s = size / 2, r = radius, shape = new THREE.Shape();
  shape.moveTo(-s + r, -s);
  shape.lineTo(s - r, -s); shape.quadraticCurveTo(s, -s, s, -s + r);
  shape.lineTo(s, s - r); shape.quadraticCurveTo(s, s, s - r, s);
  shape.lineTo(-s + r, s); shape.quadraticCurveTo(-s, s, -s, s - r);
  shape.lineTo(-s, -s + r); shape.quadraticCurveTo(-s, -s, -s + r, -s);
  return new THREE.ShapeGeometry(shape, 4);
}

const easeInOut = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export class RubikCube extends THREE.Group {
  constructor() {
    super();
    this.bodyGeo = new THREE.BoxGeometry(0.96, 0.96, 0.96);
    this.bodyMat = new THREE.MeshStandardMaterial({ color: '#1C222D', roughness: 0.6 });
    this.stickerGeo = roundedSquare(0.84, 0.12);
    this.stickerMats = COLORS.map(c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.35 }));
    this.queue = [];
    this.active = null;
    this.reset();
  }

  reset() {
    this.queue.forEach(job => job.resolve());
    this.queue = [];
    this.active = null;
    this.clear();
    this.cubies = [];
    this.stickers = [];
    for (let x = -1; x <= 1; x++)
      for (let y = -1; y <= 1; y++)
        for (let z = -1; z <= 1; z++) {
          if (!x && !y && !z) continue;
          const cubie = new THREE.Group();
          cubie.position.set(x, y, z);
          cubie.add(new THREE.Mesh(this.bodyGeo, this.bodyMat));
          for (const n of Object.values(NORMALS)) {
            if (n[0] && n[0] !== x) continue;
            if (n[1] && n[1] !== y) continue;
            if (n[2] && n[2] !== z) continue;
            const color = colorOfNormal(n);
            const sticker = new THREE.Mesh(this.stickerGeo, this.stickerMats[color]);
            const normal = new THREE.Vector3(...n);
            sticker.position.copy(normal).multiplyScalar(0.481);
            sticker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
            sticker.userData = { color, normal };
            cubie.add(sticker);
            this.stickers.push({ mesh: sticker, cubie });
          }
          this.add(cubie);
          this.cubies.push(cubie);
        }
  }

  // Tambahkan gerakan ke antrian animasi. durationMs = 0 berarti langsung.
  move(notation, durationMs = 300) {
    return new Promise(resolve => {
      this.queue.push({ notation, durationMs, resolve });
      if (durationMs <= 0 && !this.active) this.update(0);
    });
  }

  _start(job) {
    const n = new THREE.Vector3(...NORMALS[job.notation[0]]);
    const turns = job.notation.endsWith('2') ? 2 : job.notation.endsWith("'") ? -1 : 1;
    const pivot = new THREE.Group();
    this.add(pivot);
    for (const c of this.cubies) if (Math.round(c.position.dot(n)) === 1) pivot.attach(c);
    this.active = { ...job, pivot, axis: n, angle: -turns * Math.PI / 2, elapsed: 0 };
  }

  _finish() {
    const { pivot, resolve } = this.active;
    pivot.quaternion.setFromAxisAngle(this.active.axis, this.active.angle);
    pivot.updateMatrixWorld(true);
    const m = new THREE.Matrix4();
    for (const c of [...pivot.children]) {
      this.attach(c);
      c.position.round();
      m.makeRotationFromQuaternion(c.quaternion);
      m.elements = m.elements.map(Math.round);
      c.quaternion.setFromRotationMatrix(m);
    }
    this.remove(pivot);
    this.active = null;
    resolve();
  }

  // Dipanggil setiap frame dengan selisih waktu (ms)
  update(dt) {
    for (;;) {
      if (!this.active) {
        if (!this.queue.length) return;
        this._start(this.queue.shift());
      }
      const a = this.active;
      a.elapsed += dt;
      dt = 0;
      const t = a.durationMs <= 0 ? 1 : Math.min(1, a.elapsed / a.durationMs);
      a.pivot.quaternion.setFromAxisAngle(a.axis, a.angle * easeInOut(t));
      if (t < 1) return;
      this._finish();
    }
  }

  get busy() {
    return Boolean(this.active || this.queue.length);
  }

  // Baca 54 warna stiker (format sama dengan Cube.state di Python)
  getState() {
    const state = new Array(54);
    for (const { mesh, cubie } of this.stickers) {
      const n = mesh.userData.normal.clone().applyQuaternion(cubie.quaternion).round();
      const p = cubie.position.clone().round();
      state[stickerIndex(p.toArray(), n.toArray())] = mesh.userData.color;
    }
    return state;
  }
}
