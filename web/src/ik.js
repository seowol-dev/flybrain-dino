// 다리 역기구학 (CCD).
//
// 왜 필요한가: 몸통이 조금만 움직여도 다리 전체가 따라 움직여 발이 허공을 미끄러진다.
// 발끝을 바닥·자판에 **고정**해 두고 관절을 역으로 푸는 것이 IK 다. 앞다리가 스페이스바를
// 실제로 누르는 것도 목표점을 내려 주면 된다.
//
// three.js 의 CCDIKSolver 는 SkinnedMesh + Bone 전용이라 여기서는 쓸 수 없다
// (이 모델은 glTF 노드를 Group 으로 재구성한 계층이다). CCD 자체는 단순해서 직접 푼다.
//
// CCD: 말단 관절부터 뿌리 쪽으로 가며, 각 관절에서 (관절→발끝) 벡터를 (관절→목표) 벡터에
// 맞추도록 회전시킨다. 몇 번 돌리면 수렴한다. 관절마다 기준 자세에서 벗어나는 각을
// 제한해 다리가 뒤집히지 않게 한다.
import * as THREE from '../vendor/three.module.js';

const _jp = new THREE.Vector3(), _ep = new THREE.Vector3();
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _axis = new THREE.Vector3();
const _qd = new THREE.Quaternion(), _qw = new THREE.Quaternion();
const _qp = new THREE.Quaternion(), _qn = new THREE.Quaternion();
const _qr = new THREE.Quaternion();

export class LegIK {
  /**
   * @param {THREE.Object3D[]} joints  뿌리→말단 순서 (예: coxa, femur, tibia, tarsus)
   * @param {THREE.Object3D}   tip     발끝 (예: claw). joints 의 자손이어야 한다.
   * @param {object} opt  limits(관절별 최대 이탈각, rad), iterations, step(한 번에 도는 최대각)
   */
  constructor(joints, tip, opt = {}) {
    this.joints = joints;
    this.tip = tip;
    this.rest = joints.map((j) => j.quaternion.clone());
    this.limits = opt.limits ?? joints.map((_, i) => [0.45, 0.95, 1.15, 0.85][i] ?? 0.8);
    this.iterations = opt.iterations ?? 4;
    this.step = opt.step ?? 0.34;
    this.tolerance = opt.tolerance ?? 1e-4;
  }

  /** 기준 자세로 되돌린다. */
  reset() {
    this.joints.forEach((j, i) => j.quaternion.copy(this.rest[i]));
  }

  tipWorld(out = new THREE.Vector3()) {
    return out.setFromMatrixPosition(this.tip.matrixWorld);
  }

  /** target: 월드 좌표 Vector3 */
  solve(target) {
    const J = this.joints;
    for (let it = 0; it < this.iterations; it++) {
      for (let i = J.length - 1; i >= 0; i--) {
        const j = J[i];
        _jp.setFromMatrixPosition(j.matrixWorld);
        _ep.setFromMatrixPosition(this.tip.matrixWorld);
        if (_ep.distanceToSquared(target) < this.tolerance * this.tolerance) return true;
        _a.subVectors(_ep, _jp);
        _b.subVectors(target, _jp);
        if (_a.lengthSq() < 1e-14 || _b.lengthSq() < 1e-14) continue;
        _a.normalize(); _b.normalize();
        const dot = Math.max(-1, Math.min(1, _a.dot(_b)));
        let angle = Math.acos(dot);
        if (angle < 1e-5) continue;
        angle = Math.min(angle, this.step);
        _axis.crossVectors(_a, _b);
        if (_axis.lengthSq() < 1e-14) continue;
        _axis.normalize();

        // 월드에서의 보정 회전을 관절의 로컬 회전으로 옮긴다
        _qd.setFromAxisAngle(_axis, angle);
        j.getWorldQuaternion(_qw);
        j.parent.getWorldQuaternion(_qp);
        _qn.copy(_qp).invert().multiply(_qd).multiply(_qw);
        j.quaternion.copy(_qn);
        this._clamp(i);
        j.updateMatrixWorld(true);
      }
    }
    return false;
  }

  /** 기준 자세에서 너무 많이 벗어나지 않게 자른다. */
  _clamp(i) {
    const j = this.joints[i], rest = this.rest[i], max = this.limits[i];
    _qr.copy(rest).invert().multiply(j.quaternion);
    const ang = 2 * Math.acos(Math.min(1, Math.abs(_qr.w)));
    if (ang > max) {
      _qn.copy(j.quaternion);
      j.quaternion.slerpQuaternions(rest, _qn, max / ang);
    }
  }
}

/**
 * 파리의 6 다리 IK 체인을 만든다.
 * @param {THREE.Object3D} fly  loadFly() 결과 (userData.groups 필요)
 */
export function buildLegIK(fly) {
  const G = fly.userData.groups;
  const legs = [];
  for (const T of ['T1', 'T2', 'T3']) {
    for (const side of ['left', 'right']) {
      const names = [`coxa_${T}_${side}`, `femur_${T}_${side}`, `tibia_${T}_${side}`, `tarsus_${T}_${side}`];
      const joints = names.map((n) => G.get(n)).filter(Boolean);
      const tip = G.get(`claw_${T}_${side}`);
      if (joints.length < 3 || !tip) continue;
      legs.push({
        name: `${T}_${side}`, tier: T, side,
        isFront: T === 'T1',
        solver: new LegIK(joints, tip, {
          // 앞다리는 자판을 눌러야 하므로 조금 더 자유롭게
          limits: T === 'T1' ? [0.55, 1.15, 1.35, 1.0] : [0.40, 0.85, 1.05, 0.75],
          iterations: T === 'T1' ? 5 : 3,
        }),
        target: new THREE.Vector3(),
        rest: new THREE.Vector3(),
      });
    }
  }
  fly.userData.legs = legs;
  return legs;
}
