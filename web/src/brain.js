// 초파리 전뇌 점구름. FlyWire soma 좌표에 실측 발화를 입힌다.
import * as THREE from '../vendor/three.module.js';

// super_class 별 기본색 (어두운 기본 + 발화 시 밝아짐)
const PALETTE = {
  optic: [0.18, 0.40, 0.62],
  central: [0.42, 0.34, 0.55],
  visual_projection: [0.85, 0.52, 0.22],
  descending: [0.92, 0.28, 0.35],
  motor: [0.95, 0.75, 0.25],
  visual_centrifugal: [0.30, 0.60, 0.50],
  endocrine: [0.60, 0.60, 0.60],
};

const VERT = `
attribute float aAct;
attribute vec3 aBase;
attribute float aSize;
varying float vAct;
varying vec3 vBase;
uniform float uScale;     // 뷰포트 높이 / (2·tan(fov/2)) — 월드 반지름을 픽셀로
uniform float uMax;
void main() {
  vAct = aAct; vBase = aBase;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float px = aSize * uScale * (1.0 + 1.6 * aAct) / max(-mv.z, 0.05);
  gl_PointSize = clamp(px, 1.0, uMax);
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = `
varying float vAct;
varying vec3 vBase;
uniform float uDim;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float fall = exp(-r2 * 9.0);
  vec3 hot = mix(vBase, vec3(1.0, 0.96, 0.85), clamp(vAct * 1.1, 0.0, 0.92));
  float a = fall * (uDim + 1.35 * vAct);
  gl_FragColor = vec4(hot * (0.55 + 1.9 * vAct), a);
}`;

export class Brain {
  constructor(run, { scale = 1.0 } = {}) {
    this.run = run;
    this.N = run.N;
    this.group = new THREE.Group();

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(run.pos, 3));
    const base = new Float32Array(this.N * 3);
    const size = new Float32Array(this.N);
    const classes = run.meta.classes;
    for (let i = 0; i < this.N; i++) {
      const c = PALETTE[classes[run.cls[i]]] || [0.5, 0.5, 0.5];
      base[i * 3] = c[0]; base[i * 3 + 1] = c[1]; base[i * 3 + 2] = c[2];
      const sc = classes[run.cls[i]];
      // 월드 단위 반지름 (뇌 전체가 대략 반지름 1)
      size[i] = (sc === 'descending' || sc === 'motor') ? 0.0115
        : (sc === 'visual_projection' ? 0.0062 : 0.0034);
    }
    this.act = new Float32Array(this.N);
    g.setAttribute('aBase', new THREE.BufferAttribute(base, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    this.actAttr = new THREE.BufferAttribute(this.act, 1);
    this.actAttr.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aAct', this.actAttr);
    this.mat = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG,
      uniforms: { uScale: { value: 900 * scale }, uMax: { value: 26.0 }, uDim: { value: 0.16 } },
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(g, this.mat);
    this.group.add(this.points);

    // 배경: 표본에 없는 나머지 뉴런 (구조만 보여 준다)
    const bg = new THREE.BufferGeometry();
    bg.setAttribute('position', new THREE.BufferAttribute(run.backdrop, 3));
    this.backdrop = new THREE.Points(bg, new THREE.PointsMaterial({
      color: 0x223049, size: 0.0035, sizeAttenuation: true,
      transparent: true, opacity: 0.20, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    this.group.add(this.backdrop);
  }

  // 활동 프레임 적용 (지수 감쇠로 잔광을 남긴다)
  setFrame(f, decay = 0.72) {
    const N = this.N, A = this.run.act;
    const off = Math.max(0, Math.min(this.run.F - 1, f)) * N;
    const a = this.act;
    for (let i = 0; i < N; i++) {
      const v = A[off + i] * 0.5;
      a[i] = Math.max(a[i] * decay, v > 1 ? 1 : v);
    }
    this.actAttr.needsUpdate = true;
  }

  setDim(v) { this.mat.uniforms.uDim.value = v; }

  /** 뷰포트가 바뀔 때 호출. 점 크기를 화면 높이/시야각에 맞춘다. */
  setViewport(heightPx, fovDeg) {
    const s = heightPx / (2 * Math.tan((fovDeg * Math.PI / 180) / 2));
    this.mat.uniforms.uScale.value = s / this.group.scale.x;
    this.mat.uniforms.uMax.value = Math.max(8, heightPx * 0.02);
  }
}
