// 실제 측정 기반 초파리 몸 모델 로더.
//
// 원본: TuragaLab/flybody (Apache-2.0) 의 MuJoCo 성체 Drosophila melanogaster 모델.
//   Vaxenburg et al., "Whole-body simulation of realistic fruit fly locomotion".
// tools/build_fly_glb.py 가 MuJoCo XML 의 바디 트리와 재질을 읽어
//   web/assets/fly.glb        — 월드 좌표로 구운 메시 85개 (정점 5만)
//   web/assets/fly.parts.json — 바디 67개의 부모/피벗/소속 메시
// 로 내보낸다. 여기서 피벗을 기준으로 관절 계층을 복원해 머리·날개·앞다리를 움직인다.
import * as THREE from '../vendor/three.module.js';
import { GLTFLoader } from '../vendor/GLTFLoader.js';

const SKIN = {
  body: 0xb0682a, lower: 0xcb9a5f, black: 0x121212,
  red: 0xc4241a, ocelli: 0x2a1408, brown: 0x3a1b0d, membrane: 0x8aa8c4,
};

export async function loadFly(base = './assets') {
  const gltf = await new GLTFLoader().loadAsync(`${base}/fly.glb`);
  const parts = await (await fetch(`${base}/fly.parts.json`)).json();

  // 메시 이름 → 메시
  const meshes = new Map();
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    // GLB 에는 위치만 들어 있다(용량을 줄이려고 법선을 뺐다). 없으면 조명 계산이 0 이 되어
    // 전부 검게 나오므로 여기서 계산한다. 용접된 메시라 부드러운 법선이 나온다.
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    meshes.set(o.name, o);
  });

  // 바디별 그룹을 피벗 위치에 만들고 부모에 매단다
  const groups = new Map();
  const root = new THREE.Group();
  root.name = 'fly';
  const order = Object.keys(parts).sort((a, b) => depth(parts, a) - depth(parts, b));
  for (const name of order) {
    const p = parts[name];
    const g = new THREE.Group();
    g.name = name;
    const parent = p.parent ? groups.get(p.parent) : null;
    const pp = p.parent ? parts[p.parent].pivot : [0, 0, 0];
    g.position.set(p.pivot[0] - pp[0], p.pivot[1] - pp[1], p.pivot[2] - pp[2]);
    (parent || root).add(g);
    groups.set(name, g);
    // 메시는 월드 좌표로 구워져 있으므로 피벗만큼 되돌려 붙인다
    for (const gn of p.geoms) {
      const m = meshes.get(gn);
      if (!m) continue;
      m.position.sub(new THREE.Vector3(...p.pivot));
      m.castShadow = true;
      tune(m, gn);
      g.add(m);
    }
  }
  // 쉴 때 날개는 복부 위로 포갠다 (MuJoCo 기본 자세는 펼친 상태다)
  for (const s of ['left', 'right']) {
    const w = groups.get(`wing_${s}`);
    if (w) w.rotation.set(0, 0, 0);
  }
  root.userData = { groups, parts };
  return root;
}

function depth(parts, name) {
  let d = 0, cur = name;
  while (parts[cur] && parts[cur].parent) { cur = parts[cur].parent; d++; if (d > 40) break; }
  return d;
}

// 복부 가로띠. MuJoCo 모델은 몸이 단색 황갈색이지만 실제 D. melanogaster 는 각 배마디
// 뒤쪽에 검은 띠가 있고 끝으로 갈수록 어두워진다. 몸 장축(z) 기준 톱니 함수로 정점색을
// 칠해 이를 만든다. (형태가 아니라 색만 더한 것이다.)
function abdomenBands(mesh) {
  const g = mesh.geometry;
  const pos = g.attributes.position;
  g.computeBoundingBox();
  const bb = g.boundingBox;
  const z0 = bb.min.z, z1 = bb.max.z, span = Math.max(z1 - z0, 1e-6);
  const col = new Float32Array(pos.count * 3);
  const light = new THREE.Color(0xc79a53), dark = new THREE.Color(0x2b1c0c);
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const u = (pos.getZ(i) - z0) / span;          // 0 = 앞, 1 = 뒤
    const edge = Math.pow(Math.max(0, u - 0.42) / 0.58, 1.4);   // 마디 뒤쪽이 어둡다
    c.copy(light).lerp(dark, Math.min(1, edge));
    col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
}

function tune(mesh, name) {
  const m = mesh.material;
  const hex = (k) => new THREE.Color(SKIN[k]);
  const set = (color, opts = {}) => {
    mesh.material = new THREE.MeshPhysicalMaterial({
      color, roughness: 0.42, metalness: 0.12, clearcoat: 0.45, clearcoatRoughness: 0.35,
      ...opts,
    });
  };
  if (/_red\b|_red__|red/.test(name)) {
    // 겹눈 — 살짝 발광시켜 시각 입력 세기를 표현한다
    set(hex('red'), { roughness: 0.28, clearcoat: 0.9, clearcoatRoughness: 0.12,
      emissive: new THREE.Color(0x3a0206), emissiveIntensity: 0.5 });
    mesh.userData.isEye = true;
  } else if (/ocelli/.test(name)) {
    set(hex('ocelli'), { roughness: 0.2, clearcoat: 1.0 });
  } else if (/black|bristle/.test(name)) {
    set(hex('black'), { roughness: 0.55, clearcoat: 0.25 });
  } else if (/wing/.test(name)) {
    mesh.material = new THREE.MeshPhysicalMaterial({
      color: 0xdfeaf6, roughness: 0.08, metalness: 0.0, transparent: true, opacity: 0.30,
      transmission: 0.0, side: THREE.DoubleSide, depthWrite: false,
      iridescence: 0.9, iridescenceIOR: 1.25, clearcoat: 1.0,
    });
  } else if (/abdomen/.test(name)) {
    abdomenBands(mesh);
    set(0xffffff, { vertexColors: true, roughness: 0.38, clearcoat: 0.55 });
    mesh.material.vertexColors = true;
  } else if (/lower/.test(name)) {
    set(hex('lower'));
  } else if (/brown/.test(name)) {
    set(hex('brown'));
  } else if (/thorax/.test(name)) {
    set(0x9c6a2e, { roughness: 0.36, clearcoat: 0.6 });
  } else {
    set(m && m.color ? m.color : hex('body'));
  }
}

/** 프레임 애니메이션. jumpPulse 0~1, act 0~1(시각 입력 세기). */
export function animateFlyModel(fly, t, { jumpPulse = 0, act = 0 } = {}) {
  const G = fly.userData.groups;
  const g = (n) => G.get(n);

  fly.position.y = (fly.userData.baseY ?? 0) + Math.sin(t * 2.6) * 0.0025 + jumpPulse * 0.018;
  fly.rotation.z = Math.sin(t * 1.6) * 0.008;

  const head = g('head');
  if (head) {
    head.rotation.x = -0.04 + Math.sin(t * 0.85) * 0.035 - jumpPulse * 0.12;
    head.rotation.y = Math.sin(t * 0.41) * 0.07;
  }
  // 겹눈 발광 = 시각 입력
  fly.traverse((o) => {
    if (o.isMesh && o.userData.isEye) o.material.emissiveIntensity = 0.35 + act * 1.5;
  });
  // 날개: 도약 순간에만 퍼덕
  const flap = jumpPulse > 0.02 ? Math.sin(t * 85) * 0.75 * jumpPulse : 0;
  for (const [s, sign] of [['left', 1], ['right', -1]]) {
    const w = g(`wing_${s}`);
    if (w) { w.rotation.z = sign * flap; w.rotation.y = sign * flap * 0.35; }
  }
  // 앞다리(T1): 자판 누르기
  for (const s of ['left', 'right']) {
    const press = jumpPulse * 0.55 + Math.max(0, Math.sin(t * 2.6 + (s === 'left' ? 0 : 1.1))) * 0.025;
    const fe = g(`femur_T1_${s}`), ti = g(`tibia_T1_${s}`);
    if (fe) fe.rotation.x = press * 0.45;
    if (ti) ti.rotation.x = -press * 0.9;
  }
  // 나머지 다리는 아주 약하게 흔들어 살아 있는 느낌만
  for (const T of ['T2', 'T3']) {
    for (const s of ['left', 'right']) {
      const ti = g(`tibia_${T}_${s}`);
      if (ti) ti.rotation.x = Math.sin(t * 1.15 + (T === 'T2' ? 0 : 0.8)) * 0.02;
    }
  }
}
