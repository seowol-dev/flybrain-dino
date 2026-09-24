// 초파리(Drosophila melanogaster) 절차적 모델.
// 실제 비율 참고: 몸길이 약 2.5 mm, 겹눈이 머리 측면의 대부분을 덮고 낱눈은 약 700개.
// 여기서는 몸길이 1.0 단위로 만든다.
import * as THREE from '../vendor/three.module.js';

function hexTexture(size = 512, cell = 15) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const x = c.getContext('2d');
  x.fillStyle = '#a3141c'; x.fillRect(0, 0, size, size);
  const r = size / cell / 2;
  const dx = r * 1.732, dy = r * 1.5;
  for (let row = -1; row * dy < size + r; row++) {
    for (let col = -1; col * dx < size + dx; col++) {
      const cx = col * dx + (row % 2 ? dx / 2 : 0), cy = row * dy;
      const g = x.createRadialGradient(cx - r * .3, cy - r * .3, r * .1, cx, cy, r);
      g.addColorStop(0, '#f2606a'); g.addColorStop(.5, '#c2202c'); g.addColorStop(1, '#75090f');
      x.fillStyle = g;
      x.beginPath();
      for (let k = 0; k < 6; k++) {
        const a = Math.PI / 6 + k * Math.PI / 3;
        const px = cx + r * 0.94 * Math.cos(a), py = cy + r * 0.94 * Math.sin(a);
        k ? x.lineTo(px, py) : x.moveTo(px, py);
      }
      x.closePath(); x.fill();
    }
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

function stripeTexture(size = 512) {
  // D. melanogaster 복부: 황갈색 바탕 + 마디마다 검은 가로띠, 끝으로 갈수록 검어진다
  const c = document.createElement('canvas');
  c.width = 64; c.height = size;
  const x = c.getContext('2d');
  const g = x.createLinearGradient(0, 0, 0, size);
  g.addColorStop(0.0, '#9c7335'); g.addColorStop(0.35, '#c79c4e');
  g.addColorStop(0.72, '#6b4a20'); g.addColorStop(1.0, '#1d1509');
  x.fillStyle = g; x.fillRect(0, 0, 64, size);
  for (let i = 0; i < 6; i++) {
    const y = size * (0.30 + i * 0.115);
    const h = size * 0.052;
    const gg = x.createLinearGradient(0, y, 0, y + h);
    gg.addColorStop(0, 'rgba(18,12,6,0.10)');
    gg.addColorStop(0.5, 'rgba(18,12,6,0.99)');
    gg.addColorStop(1, 'rgba(18,12,6,0.10)');
    x.fillStyle = gg; x.fillRect(0, y, 64, h);
  }
  return new THREE.CanvasTexture(c);
}

function thoraxTexture(size = 256) {
  // 흉부: 황갈색 + 세로 줄 세 개 (trident)
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const x = c.getContext('2d');
  const bg = x.createLinearGradient(0, 0, 0, size);
  bg.addColorStop(0, '#6b4e26'); bg.addColorStop(0.4, '#b78a45');
  bg.addColorStop(1, '#8a6633');
  x.fillStyle = bg; x.fillRect(0, 0, size, size);
  // 세로 줄 세 개 — 좌우로 이어지도록 전체 높이에 그린다
  for (const fx of [0.22, 0.5, 0.78]) {
    const g2 = x.createLinearGradient(size * (fx - 0.05), 0, size * (fx + 0.05), 0);
    g2.addColorStop(0, 'rgba(48,33,17,0)'); g2.addColorStop(0.5, 'rgba(48,33,17,0.40)');
    g2.addColorStop(1, 'rgba(48,33,17,0)');
    x.fillStyle = g2; x.fillRect(size * (fx - 0.05), 0, size * 0.10, size);
  }
  return new THREE.CanvasTexture(c);
}

function segment(len, r0, r1, mat) {
  const g = new THREE.CylinderGeometry(r1, r0, len, 10, 1, false);
  g.translate(0, -len / 2, 0);
  return new THREE.Mesh(g, mat);
}

function makeLeg(mat, { len = [0.20, 0.26, 0.24], rad = 0.0105 } = {}) {
  const root = new THREE.Group();
  const femur = segment(len[0], rad, rad * 0.85, mat);
  const kneeG = new THREE.Group(); kneeG.position.y = -len[0];
  const tibia = segment(len[1], rad * 0.85, rad * 0.6, mat);
  const ankleG = new THREE.Group(); ankleG.position.y = -len[1];
  const tarsus = segment(len[2], rad * 0.6, rad * 0.3, mat);
  root.add(femur, kneeG); kneeG.add(tibia, ankleG); ankleG.add(tarsus);
  // 잔털
  for (const [seg, l] of [[tibia, len[1]], [tarsus, len[2]]]) {
    for (let i = 0; i < 4; i++) {
      const h = new THREE.Mesh(new THREE.ConeGeometry(rad * 0.16, rad * 2.2, 4), mat);
      h.position.set(0, -l * (0.2 + i * 0.22), rad * 0.7);
      h.rotation.x = -1.1;
      seg.add(h);
    }
  }
  return { root, knee: kneeG, ankle: ankleG };
}

export function makeFly() {
  const fly = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({
    map: stripeTexture(), color: 0xffffff, roughness: 0.52, metalness: 0.12,
  });
  const thoraxMat = new THREE.MeshStandardMaterial({
    map: thoraxTexture(), color: 0xffffff, roughness: 0.42, metalness: 0.16 });
  const headMat = new THREE.MeshStandardMaterial({ color: 0xa87f40, roughness: 0.45, metalness: 0.15 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x5c4426, roughness: 0.52, metalness: 0.1 });
  const eyeMat = new THREE.MeshStandardMaterial({
    map: hexTexture(), roughness: 0.28, metalness: 0.05,
    emissive: 0x3a0a10, emissiveIntensity: 0.55,
  });
  const wingMat = new THREE.MeshStandardMaterial({
    color: 0xeaf2fc, transparent: true, opacity: 0.26, roughness: 0.15,
    metalness: 0.0, side: THREE.DoubleSide, depthWrite: false,
    emissive: 0x25384f, emissiveIntensity: 0.4,
  });

  // 흉부 (thorax)
  const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.175, 30, 22), thoraxMat);
  thorax.scale.set(1.05, 1.08, 1.30);
  thorax.position.set(0, 0.01, 0.02);
  fly.add(thorax);
  // 흉부 강모
  for (let i = 0; i < 10; i++) {
    const b = new THREE.Mesh(new THREE.ConeGeometry(0.0042, 0.050, 4), darkMat);
    const a = (i / 10) * Math.PI * 2;
    b.position.set(Math.cos(a) * 0.09, 0.17, Math.sin(a) * 0.12 - 0.02);
    b.rotation.set(-0.35 + Math.sin(a) * 0.2, 0, Math.cos(a) * 0.25);
    fly.add(b);
  }

  // 복부 (abdomen) — 뒤쪽(+z). 띠는 몸 장축에 수직으로 감긴다.
  // 구의 극축(y)을 z 로 돌린 뒤 늘려야 텍스처 띠가 몸통을 고리처럼 감는다.
  const abGeo = new THREE.SphereGeometry(0.155, 30, 24);
  abGeo.rotateX(Math.PI / 2);                  // 극축 y → z
  const abPos = abGeo.attributes.position;
  for (let i = 0; i < abPos.count; i++) {      // 뒤로 갈수록 가늘게 (원추형 복부)
    const z = abPos.getZ(i);
    const k = 1.0 - 0.26 * Math.max(0, z / 0.155);
    abPos.setX(i, abPos.getX(i) * k);
    abPos.setY(i, abPos.getY(i) * k);
  }
  abGeo.computeVertexNormals();
  const abdomen = new THREE.Mesh(abGeo, bodyMat);
  abdomen.scale.set(1.06, 0.98, 1.58);
  abdomen.position.set(0, -0.012, 0.345);
  abdomen.rotation.x = -0.10;
  fly.add(abdomen);

  // 머리 (head) — 앞쪽(-z)
  const head = new THREE.Group();
  head.position.set(0, 0.038, -0.205);
  const skull = new THREE.Mesh(new THREE.SphereGeometry(0.145, 30, 24), headMat);
  skull.scale.set(1.0, 0.98, 0.74);
  head.add(skull);
  // 겹눈 — 실제 초파리는 겹눈이 머리 측면의 대부분을 덮는다 (낱눈 약 700개)
  const eyeGeo = new THREE.SphereGeometry(0.122, 34, 28);
  for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(eyeGeo, eyeMat);
    eye.scale.set(0.70, 1.06, 0.88);
    eye.position.set(s * 0.078, 0.010, -0.002);
    eye.rotation.z = s * 0.17;
    head.add(eye);
  }
  // 홑눈 3개
  for (const [ox, oz] of [[0, -0.02], [-0.032, 0.005], [0.032, 0.005]]) {
    const o = new THREE.Mesh(new THREE.SphereGeometry(0.012, 10, 8),
      new THREE.MeshStandardMaterial({ color: 0xd9903a, emissive: 0x5a2a05, roughness: 0.3 }));
    o.position.set(ox, 0.104, oz + 0.02);
    head.add(o);
  }
  // 더듬이 + 주둥이
  for (const s of [-1, 1]) {
    const ant = new THREE.Mesh(new THREE.CapsuleGeometry(0.014, 0.05, 4, 8), darkMat);
    ant.position.set(s * 0.032, -0.028, -0.098);
    ant.rotation.set(1.15, 0, s * 0.25);
    head.add(ant);
    const arista = new THREE.Mesh(new THREE.ConeGeometry(0.004, 0.10, 4), darkMat);
    arista.position.set(s * 0.046, 0.004, -0.128);
    arista.rotation.set(0.6, 0, s * 0.7);
    head.add(arista);
  }
  const face = new THREE.Mesh(new THREE.SphereGeometry(0.058, 18, 14),
    new THREE.MeshStandardMaterial({ color: 0xd9c9a4, roughness: 0.35, metalness: 0.25 }));
  face.scale.set(0.85, 1.25, 0.55);
  face.position.set(0, -0.018, -0.095);
  head.add(face);
  const prob = new THREE.Mesh(new THREE.CapsuleGeometry(0.03, 0.05, 4, 10), darkMat);
  prob.position.set(0, -0.092, -0.062);
  prob.rotation.x = 0.35;
  head.add(prob);
  fly.add(head);

  // 날개
  const wingShape = new THREE.Shape();
  wingShape.moveTo(0, 0);
  wingShape.bezierCurveTo(0.14, 0.16, 0.70, 0.27, 1.02, 0.06);
  wingShape.bezierCurveTo(0.74, -0.16, 0.18, -0.14, 0, 0);
  const wingGeo = new THREE.ShapeGeometry(wingShape, 30);
  wingGeo.rotateX(-Math.PI / 2);      // 날개면을 수평(XZ)으로 눕힌다. 장축은 +X.
  const wings = [];
  for (const s of [-1, 1]) {
    const w = new THREE.Mesh(wingGeo, wingMat);
    // 장축 +X 를 뒤쪽(+Z)으로 돌리고 바깥으로 조금 벌린다. 쉴 때 날개는 복부 위에 포갠다.
    w.position.set(s * 0.052, 0.095, 0.055);
    w.rotation.set(0.10, -Math.PI / 2 + s * 0.34, 0);
    w.scale.setScalar(0.92);
    fly.add(w);
    wings.push(w);
  }

  // 다리 6개 (앞/가운데/뒤)
  const legs = [];
  const legSpec = [
    { z: -0.11, x: 0.105, spread: 0.45, tilt: -0.55, len: [0.17, 0.21, 0.17] },  // 앞다리 — 자판을 짚는다
    { z: 0.03, x: 0.125, spread: 1.05, tilt: 0.05, len: [0.19, 0.24, 0.18] },
    { z: 0.16, x: 0.115, spread: 1.55, tilt: 0.55, len: [0.21, 0.28, 0.20] },
  ];
  for (const spec of legSpec) {
    for (const s of [-1, 1]) {
      const { root, knee, ankle } = makeLeg(darkMat, { len: spec.len });
      root.position.set(s * spec.x, -0.085, spec.z);
      root.rotation.set(spec.tilt, 0, s * spec.spread);
      knee.rotation.set(-0.35, 0, -s * (spec.spread + 0.45));
      ankle.rotation.set(0.85, 0, s * 0.18);
      fly.add(root);
      legs.push({ root, knee, ankle, side: s, spec });
    }
  }

  // 그림자/접지용 기준
  fly.userData = { head, thorax, abdomen, wings, legs, eyeMat };
  return fly;
}

/** 프레임별 애니메이션. jumpPulse 는 0~1 (도약 직후 1). */
export function animateFly(fly, t, { jumpPulse = 0, act = 0 } = {}) {
  const u = fly.userData;
  // 잔떨림 (살아 있는 느낌)
  fly.position.y = Math.sin(t * 2.7) * 0.004 + jumpPulse * 0.03;
  fly.rotation.z = Math.sin(t * 1.7) * 0.012;
  u.head.rotation.x = Math.sin(t * 0.9) * 0.05 - 0.05 + jumpPulse * -0.18;
  u.head.rotation.y = Math.sin(t * 0.43) * 0.09;
  // 겹눈 발광 = 시각 입력 세기
  u.eyeMat.emissiveIntensity = 0.45 + act * 1.4;
  // 날개: 평소 접힘, 도약 시 퍼덕
  const flap = jumpPulse > 0.02 ? Math.sin(t * 90) * 0.6 * jumpPulse : 0;
  u.wings.forEach((w, i) => {
    const s = i === 0 ? -1 : 1;
    w.rotation.y = -Math.PI / 2 + s * 0.34 + s * flap * 0.5;
    w.rotation.x = 0.10 + flap * 0.55;
  });
  // 앞다리: 자판 두드리기 (도약할 때 누른다)
  u.legs.forEach((L, i) => {
    if (L.spec.z < -0.1) {
      const press = jumpPulse * 0.5 + Math.max(0, Math.sin(t * 3 + i)) * 0.03;
      L.knee.rotation.x = -0.2 - press;
      L.ankle.rotation.x = 0.5 + press * 0.8;
    } else {
      L.knee.rotation.x = -0.2 + Math.sin(t * 1.3 + i) * 0.02;
    }
  });
}
