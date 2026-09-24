// 정점 단위 표면 디테일 굽기 — UV 없는 메시에 텍스처 느낌을 만든다.
//
// 이 모델에는 UV 가 없어서 이미지 텍스처를 입힐 수 없다. 대신 지오메트리 자체에서
// 세 가지를 계산해 정점색(COLOR_0)으로 굽는다.
//
//   1) 요철(cavity) : 이웃 정점들의 평균 위치가 법선 '안쪽'에 있으면 오목한 곳이다.
//                     낱눈 사이 골, 배마디 틈, 강모 뿌리가 어두워진다. 형태가 드러나는
//                     가장 큰 요인이다.
//   2) 곡면 노출     : 바깥으로 볼록한 곳은 빛을 더 받는다(얕은 AO 대용).
//   3) 반점(mottle)  : 3D 값잡음. 실제 큐티클은 균일한 플라스틱이 아니다.
//
// 모두 지오메트리에서 나오므로 좌표계·스케일이 바뀌어도 같은 결과가 나온다.

const H = (x, y, z) => {
  // 결정적 3D 해시 (0~1)
  let h = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(z | 0, 2147483647);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
};

function valueNoise(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const s = (t) => t * t * (3 - 2 * t);
  const u = s(xf), v = s(yf), w = s(zf);
  let n = 0;
  for (let dz = 0; dz < 2; dz++) {
    for (let dy = 0; dy < 2; dy++) {
      for (let dx = 0; dx < 2; dx++) {
        const wt = (dx ? u : 1 - u) * (dy ? v : 1 - v) * (dz ? w : 1 - w);
        n += wt * H(xi + dx, yi + dy, zi + dz);
      }
    }
  }
  return n;
}

/** 이웃 평균 위치 기준 오목도 (0 = 평평/볼록, 1 = 깊이 오목). */
function cavity(geo) {
  const pos = geo.attributes.position, nrm = geo.attributes.normal;
  const n = pos.count;
  const sum = new Float32Array(n * 3);
  const cnt = new Uint32Array(n);
  const idx = geo.index;
  const add = (a, b) => {
    sum[a * 3] += pos.getX(b); sum[a * 3 + 1] += pos.getY(b); sum[a * 3 + 2] += pos.getZ(b);
    cnt[a]++;
  };
  if (idx) {
    for (let i = 0; i < idx.count; i += 3) {
      const a = idx.getX(i), b = idx.getX(i + 1), c = idx.getX(i + 2);
      add(a, b); add(a, c); add(b, a); add(b, c); add(c, a); add(c, b);
    }
  }
  // **국소 모서리 길이**로 정규화한다. 메시 반지름으로 나누면 큰 매끈한 면(흉부·복부)에서
  // 값이 0 에 수렴해 아무 디테일도 안 나온다. 요철은 국소적인 성질이다.
  const out = new Float32Array(n);
  let edgeSum = 0, edgeN = 0;
  for (let i = 0; i < n; i++) {
    if (!cnt[i]) continue;
    const dx = sum[i * 3] / cnt[i] - pos.getX(i);
    const dy = sum[i * 3 + 1] / cnt[i] - pos.getY(i);
    const dz = sum[i * 3 + 2] / cnt[i] - pos.getZ(i);
    const len = Math.hypot(dx, dy, dz);
    edgeSum += len; edgeN++;
    // 법선 방향 성분: 양수면 이웃이 바깥 → 오목
    out[i] = dx * nrm.getX(i) + dy * nrm.getY(i) + dz * nrm.getZ(i);
  }
  const mean = Math.max(edgeSum / Math.max(edgeN, 1), 1e-9);
  for (let i = 0; i < n; i++) out[i] = Math.max(0, Math.min(1, (out[i] / mean) * 4.2));
  return out;
}

/**
 * 정점색을 굽는다.
 * kind: 'eye' | 'cuticle' | 'abdomen' | 'wing' | 'bristle'
 */
export function bakeDetail(mesh, kind, THREE) {
  const g = mesh.geometry;
  if (!g.attributes.normal) g.computeVertexNormals();
  const pos = g.attributes.position;
  const n = pos.count;
  const cav = cavity(g);
  g.computeBoundingBox();
  const bb = g.boundingBox;
  const span = Math.max(bb.max.z - bb.min.z, 1e-6);
  const col = new Float32Array(n * 3);

  const P = {
    eye:      { base: [0.70, 0.050, 0.030], dark: [0.13, 0.010, 0.012], cavK: 0.85, noise: 620, noiseAmt: 0.18, dorsal: 0.0 },
    cuticle:  { base: [0.55, 0.325, 0.115], dark: [0.055, 0.028, 0.010], cavK: 1.15, noise: 46,  noiseAmt: 0.30, dorsal: 0.34 },
    abdomen:  { base: [0.60, 0.39, 0.15],   dark: [0.035, 0.020, 0.009], cavK: 1.05, noise: 40,  noiseAmt: 0.22, dorsal: 0.20 },
    wing:     { base: [0.86, 0.90, 0.96],   dark: [0.30, 0.38, 0.50],   cavK: 1.30, noise: 90,  noiseAmt: 0.05, dorsal: 0.0 },
    bristle:  { base: [0.10, 0.072, 0.045], dark: [0.02, 0.015, 0.010], cavK: 0.50, noise: 60,  noiseAmt: 0.06, dorsal: 0.0 },
  }[kind] || { base: [0.5, 0.3, 0.12], dark: [0.12, 0.07, 0.03], cavK: 0.8, noise: 60, noiseAmt: 0.2, dorsal: 0.1 };

  for (let i = 0; i < n; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    let t = cav[i] * P.cavK;                       // 0 = 볼록, 1 = 오목
    const nz = valueNoise(x * P.noise, y * P.noise, z * P.noise);
    t = Math.min(1, t + (nz - 0.5) * P.noiseAmt);
    // 등쪽(+y)이 어둡다 — 실제 초파리는 흉배와 배등판이 진하고 배쪽이 밝다
    if (P.dorsal > 0) {
      const uy = (y - bb.min.y) / Math.max(bb.max.y - bb.min.y, 1e-6);
      t = Math.min(1, t + uy * P.dorsal);
    }
    // 복부는 마디 뒤쪽이 검다 (실제 D. melanogaster)
    if (kind === 'abdomen') {
      const u = (z - bb.min.z) / span;
      t = Math.min(1, t + Math.pow(Math.max(0, u - 0.44) / 0.56, 1.35) * 0.95);
    }
    // 겹눈은 가장자리로 갈수록 어둡다
    if (kind === 'eye') {
      const r = Math.hypot(x - (bb.min.x + bb.max.x) / 2, y - (bb.min.y + bb.max.y) / 2);
      const rmax = Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y) / 2;
      t = Math.min(1, t + Math.pow(Math.min(1, r / Math.max(rmax, 1e-6)), 3.0) * 0.45);
    }
    const k = Math.max(0, Math.min(1, t));
    col[i * 3] = P.base[0] + (P.dark[0] - P.base[0]) * k;
    col[i * 3 + 1] = P.base[1] + (P.dark[1] - P.base[1]) * k;
    col[i * 3 + 2] = P.base[2] + (P.dark[2] - P.base[2]) * k;
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return g;
}
