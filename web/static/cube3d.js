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
const TOKEN = /^(\d*)([URFDLBurfdlbxyz])(w?)(['2]?)$/;

// Indeks stiker dari posisi kubus kecil (koordinat digandakan) + arah hadap,
// sama persis dengan rubik/moves.py.
export function stickerIndex(n, [x, y, z], [nx, ny, nz]) {
  const m = n - 1;
  let face, row, col;
  if (ny === 1) { face = 'U'; col = (x + m) / 2; row = (z + m) / 2; }
  else if (ny === -1) { face = 'D'; col = (x + m) / 2; row = (m - z) / 2; }
  else if (nz === 1) { face = 'F'; col = (x + m) / 2; row = (m - y) / 2; }
  else if (nz === -1) { face = 'B'; col = (m - x) / 2; row = (m - y) / 2; }
  else if (nx === 1) { face = 'R'; col = (m - z) / 2; row = (m - y) / 2; }
  else { face = 'L'; col = (z + m) / 2; row = (m - y) / 2; }
  return FACES.indexOf(face) * n * n + row * n + col;
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
const snapHalf = v => Math.round(v * 2) / 2;  // bulatkan posisi ke kelipatan 0,5

export class RubikCube extends THREE.Group {
  constructor(n = 3) {
    super();
    this.bodyGeo = new THREE.BoxGeometry(0.96, 0.96, 0.96);
    this.bodyMat = new THREE.MeshStandardMaterial({ color: '#1C222D', roughness: 0.6 });
    this.stickerGeo = roundedSquare(0.84, 0.12);
    this.stickerMats = COLORS.map(c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.35 }));
    this.queue = [];
    this.active = null;
    this.reset(n);
  }

  // Bangun ulang kubus dalam keadaan selesai.
  reset(n = this.n) {
    this.n = n;
    this.queue.forEach(job => job.resolve());
    this.queue = [];
    this.active = null;
    this.clear();
    this.cubies = [];
    this.stickers = [];
    const h = (n - 1) / 2;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        for (let k = 0; k < n; k++) {
          const [x, y, z] = [i - h, j - h, k - h];
          if (Math.abs(x) < h && Math.abs(y) < h && Math.abs(z) < h) continue; // bagian dalam, tak terlihat
          const cubie = new THREE.Group();
          cubie.position.set(x, y, z);
          cubie.add(new THREE.Mesh(this.bodyGeo, this.bodyMat));
          for (const nv of Object.values(NORMALS)) {
            const along = nv[0] * x + nv[1] * y + nv[2] * z;
            if (Math.abs(along - h) > 1e-6) continue; // stiker hanya di permukaan luar
            const color = colorOfNormal(nv);
            const sticker = new THREE.Mesh(this.stickerGeo, this.stickerMats[color]);
            const normal = new THREE.Vector3(...nv);
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

  // Ubah token notasi menjadi {sumbu, lapisan, sudut}. Mendukung U, U', U2, 2U, Uw, u, 3Rw, x/y/z.
  _parse(token) {
    const mt = TOKEN.exec(token);
    if (!mt) throw new Error(`Gerakan tidak valid: ${token}`);
    let [, num, face, wide, suffix] = mt;
    const turns = suffix === '2' ? 2 : suffix === "'" ? -1 : 1;
    let layers;
    if ('xyz'.includes(face)) {
      face = { x: 'R', y: 'U', z: 'F' }[face];
      layers = Array.from({ length: this.n }, (_, i) => i + 1);
    } else {
      if (face === face.toLowerCase()) { face = face.toUpperCase(); wide = 'w'; }
      const k = num ? Number(num) : (wide ? 2 : 1);
      layers = wide ? Array.from({ length: k }, (_, i) => i + 1) : [k];
    }
    const h = (this.n - 1) / 2;
    return { axis: new THREE.Vector3(...NORMALS[face]), levels: layers.map(l => h - (l - 1)), angle: -turns * Math.PI / 2 };
  }

  // Tambahkan gerakan ke antrian animasi. durationMs = 0 berarti langsung.
  move(notation, durationMs = 300) {
    return new Promise(resolve => {
      this.queue.push({ notation, durationMs, resolve });
      if (durationMs <= 0 && !this.active) this.update(0);
    });
  }

  _start(job) {
    const { axis, levels, angle } = this._parse(job.notation);
    const pivot = new THREE.Group();
    this.add(pivot);
    for (const c of this.cubies) {
      const along = c.position.dot(axis);
      if (levels.some(l => Math.abs(along - l) < 1e-3)) pivot.attach(c);
    }
    this.active = { ...job, pivot, axis, angle, elapsed: 0 };
  }

  _finish() {
    const { pivot, resolve } = this.active;
    pivot.quaternion.setFromAxisAngle(this.active.axis, this.active.angle);
    pivot.updateMatrixWorld(true);
    const m = new THREE.Matrix4();
    for (const c of [...pivot.children]) {
      this.attach(c);
      c.position.set(snapHalf(c.position.x), snapHalf(c.position.y), snapHalf(c.position.z));
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

  _index(mesh, cubie) {
    const nrm = mesh.userData.normal.clone().applyQuaternion(cubie.quaternion).round();
    const p = cubie.position.clone().multiplyScalar(2).round(); // koordinat digandakan
    return stickerIndex(this.n, p.toArray(), nrm.toArray());
  }

  // Baca warna stiker (format sama dengan state di Python)
  getState() {
    const state = new Array(6 * this.n * this.n);
    for (const { mesh, cubie } of this.stickers) state[this._index(mesh, cubie)] = mesh.userData.color;
    return state;
  }

  // Warnai ulang semua stiker sesuai state dari server
  setState(state) {
    for (const { mesh, cubie } of this.stickers) {
      const color = state[this._index(mesh, cubie)];
      mesh.userData.color = color;
      mesh.material = this.stickerMats[color];
    }
  }
}