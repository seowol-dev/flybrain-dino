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
import { bakeDetail } from './fly_detail.js';

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

// 부위 분류 — MuJoCo geom 이름(`body__geom`)에서 재질 종류를 읽는다.
function kindOf(name) {
  if (/_red\b|red$|_red_/.test(name)) return 'eye';
  if (/ocelli/.test(name)) return 'ocelli';
  if (/wing/.test(name)) return 'wing';
  if (/bristle|_black/.test(name)) return 'bristle';
  if (/abdomen/.test(name)) return 'abdomen';
  return 'cuticle';
}

// 정점색의 밝기로 거칠기를 변조한다. 골(어두운 곳)은 거칠고 융기(밝은 곳)는 매끈하다.
// 표면이 '진짜'로 보이는 데는 알베도보다 광택 변화가 더 크게 기여한다. UV 가 없어
// roughnessMap 을 못 쓰므로 셰이더에 직접 주입한다.
function roughnessFromColor(mat, amount) {
  mat.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <roughnessmap_fragment>',
      `#include <roughnessmap_fragment>
       #ifdef USE_COLOR
         float _lum = dot(vColor.rgb, vec3(0.2126, 0.7152, 0.0722));
         roughnessFactor = clamp(roughnessFactor + (1.0 - _lum) * ${amount.toFixed(3)}, 0.03, 1.0);
       #endif`);
  };
  mat.customProgramCacheKey = () => `rgh${amount}`;
  return mat;
}

// 부위별 물리 재질. 실제 초파리 큐티클은 얇은 왁스층이 있어 살짝 번들거리고,
// 겹눈은 각막 렌즈 때문에 젖은 듯한 강한 정반사가 난다.
function materialFor(kind, THREE) {
  const common = { vertexColors: true, color: 0xffffff };
  switch (kind) {
    case 'eye':
      return roughnessFromColor(new THREE.MeshPhysicalMaterial({
        ...common, roughness: 0.12, metalness: 0.0,
        clearcoat: 1.0, clearcoatRoughness: 0.04,       // 각막 렌즈
        specularIntensity: 1.0, specularColor: new THREE.Color(0xfff0ec),
        emissive: new THREE.Color(0x40060a), emissiveIntensity: 0.35,
      }), 0.55);
    case 'ocelli':
      return new THREE.MeshPhysicalMaterial({
        color: 0x3a2008, roughness: 0.12, clearcoat: 1.0, clearcoatRoughness: 0.03,
        emissive: new THREE.Color(0x1a0d02), emissiveIntensity: 0.4,
      });
    case 'wing':
      return new THREE.MeshPhysicalMaterial({
        ...common, transparent: true, opacity: 0.34, depthWrite: false,
        roughness: 0.06, metalness: 0.0, side: THREE.DoubleSide,
        iridescence: 1.0, iridescenceIOR: 1.32,
        iridescenceThicknessRange: [180, 520],          // 얇은 막 간섭 → 무지개빛
        clearcoat: 1.0, clearcoatRoughness: 0.04,
      });
    case 'bristle':
      return new THREE.MeshPhysicalMaterial({
        ...common, roughness: 0.52, metalness: 0.05, clearcoat: 0.25, clearcoatRoughness: 0.45,
      });
    case 'abdomen':
      return roughnessFromColor(new THREE.MeshPhysicalMaterial({
        ...common, roughness: 0.30, metalness: 0.05,
        clearcoat: 0.30, clearcoatRoughness: 0.32,
        sheen: 0.18, sheenRoughness: 0.6, sheenColor: new THREE.Color(0xffd9a0),
      }), 0.62);
    default:                                            // cuticle
      return roughnessFromColor(new THREE.MeshPhysicalMaterial({
        ...common, roughness: 0.28, metalness: 0.07,
        clearcoat: 0.32, clearcoatRoughness: 0.30,
        sheen: 0.20, sheenRoughness: 0.55, sheenColor: new THREE.Color(0xffd9a0),
      }), 0.66);
  }
}

function tune(mesh, name) {
  const kind = kindOf(name);
  // 지오메트리에서 요철·반점을 계산해 정점색으로 굽는다 (UV 가 없어 텍스처를 못 쓴다)
  if (kind !== 'ocelli') bakeDetail(mesh, kind === 'bristle' ? 'bristle' : kind, THREE);
  mesh.material = materialFor(kind, THREE);
  if (kind === 'eye') mesh.userData.isEye = true;
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
  // 다리는 IK(ik.js)가 담당한다 — 여기서 건드리면 충돌한다.
}
