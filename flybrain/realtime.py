"""실시간 시각 폐회로 실행기.

VisualBrain : 망막지도 → 광수용체 자극 → MLX GPU 전뇌 LIF → 시각/하행 뉴런 발화율.
FrameClock  : 프레임 예산(기본 60 fps = 16.67 ms) 대비 실제 소요를 재고 생물시간을
              벽시계에 맞춘다. 늦어진 양(slip) 통계를 남긴다.

실시간 예산 (이전 세션 실측, Apple M5):
  MLX 엔진 RTF ≈ 0.46x — 생물시간 1초에 벽시계 0.46초. 실시간보다 2.2배 빠르다.
  단 run(eval_every=) 은 64 이상이어야 한다 (1이면 커널 launch 오버헤드로 5배 느려짐).
  → 프레임당 16.67 ms 예산 중 약 7.7 ms 가 적분, 나머지가 인코드/디코드/렌더 여유.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .atlas import Atlas
from .data import Connectome, load_connectome, weights
from .lif import LIFParams
from .config import PATHS
from .retina import RetinaEncoder, load_eye_maps

# T4/T5 의 '지연 가지' — 이 세포유형의 출력 시냅스는 느린 시냅스 풀로 보낸다.
# 기본운동검출기(EMD)의 방향 선택성은 빠른 가지와 느린 가지를 비교해서 생긴다.
#   T4 (ON) : 빠름 Mi1, Tm3    /  느림 Mi9(글루탐산), Mi4(GABA), CT1(GABA)
#   T5 (OFF): 빠름 Tm1, Tm2    /  느림 Tm9(지속형), CT1
# 근거: Arenz et al. 2017 (Curr Biol) — Mi9/Mi4 가 Mi1/Tm3 보다 느리다;
#       Serbe et al. 2016 (Neuron) — Tm9 지속형, Tm1/Tm2 과도형;
#       Meier & Borst 2019 (Curr Biol) — CT1 은 느린 GABA성 지연 가지.
SLOW_TYPES: tuple[str, ...] = ("Mi9", "Mi4", "CT1", "Tm9")


def slow_pre_mask(cx: Connectome, atlas: Atlas, types=SLOW_TYPES) -> np.ndarray:
    """느린 시냅스 풀을 쓸 시냅스전 뉴런 마스크 (N,) bool."""
    m = np.zeros(len(cx.neurons), bool)
    for t in types:
        idx = atlas.by_type(t)
        if len(idx):
            m[idx] = True
    return m


# 시각 경로 기본 기록 그룹 (이름 → cell_type)
VISUAL_GROUPS: dict[str, tuple[str, ...]] = {
    "L1": ("L1",), "L2": ("L2",), "L3": ("L3",),
    "Mi1": ("Mi1",), "Tm1": ("Tm1",), "Tm2": ("Tm2",), "Tm9": ("Tm9",),
    "T4a": ("T4a",), "T4b": ("T4b",), "T4c": ("T4c",), "T4d": ("T4d",),
    "T5a": ("T5a",), "T5b": ("T5b",), "T5c": ("T5c",), "T5d": ("T5d",),
    "LPLC2": ("LPLC2",), "LC4": ("LC4",), "LPLC1": ("LPLC1",),
    "DNp01": ("DNp01",), "DNp11": ("DNp11",), "DNp07": ("DNp07",), "DNp10": ("DNp10",),
}


@dataclass
class VisualBrain:
    """망막지도가 붙은 전뇌 LIF. 프레임 단위로 휘도를 넣고 발화율을 받는다.

    **작동점(operating point) 보정이 왜 필요한가**

    이 모델은 Shiu et al. 2024 의 균일 LIF (역치까지 7 mV, 시냅스 1개 0.275 mV) 인데,
    시각엽 회로는 중앙뇌와 성질이 다르다.

      1. 광수용체 R1-6 은 히스타민성 = 이 모델에서 **억제**다. 즉 '빛 → 억제' 만 있고
         '어둠 → 흥분' 이 없다. 실제 라미나 단극세포는 어둠에서 탈분극해 있고(Laughlin 1981)
         빛이 그 탈분극을 누르는 방식으로 신호를 만든다.
      2. 시각엽 컬럼 뉴런 대부분(L1~L5, Mi, Tm, Dm, CT1 …)은 애초에 **스파이크를 내지 않는
         등급전위(graded) 뉴런**이다. 이들을 스파이킹 LIF 로 근사하려면 작동점을 역치 근처에
         두어야 위아래로 변조될 수 있다. 그냥 두면 컬럼 연결(시냅스 5~50개 = 1~14 mV)이
         역치에 못 미쳐 시각 경로 전체가 침묵한다.

    그래서 시각엽 내재뉴런(super_class='optic', 77,530개)에만 뉴런별 지속 탈분극(tonic bias)을
    주되, 그 값을 손으로 정하지 않고 **균일 회색 화면에서 목표 기저 발화율이 되도록 항상성
    보정**으로 자동 결정한다(calibrate_optic_bias). 자유 파라미터는 목표 발화율 하나뿐이다.

    시각투사뉴런(LPLC2/LC4)과 하행뉴런(DNp01)은 보정 대상이 아니다 — 실제로도 기저 발화가
    거의 없는 희소(sparse) 부호화 뉴런이고, 이들이 켜지는 것 자체가 측정 대상이기 때문이다.
    """
    cx: Connectome | None = None
    preset: str = "curated"
    params: LIFParams = field(default_factory=LIFParams)
    seed: int = 0
    target_hz: float = 10.0          # 시각엽 내재뉴런의 균일 회색 기저 발화율 목표
    calibrate: bool = True
    encoder_kwargs: dict = field(default_factory=dict)
    silence: tuple = ()
    eyes_used: tuple = ("left", "right")
    extra_groups: dict = field(default_factory=dict)
    slow_types: tuple = SLOW_TYPES      # () 로 두면 모든 시냅스가 같은 시정수 (이전 동작)

    def __post_init__(self):
        from .lif_mlx import MLXLIFNetwork
        import mlx.core as mx

        self.cx = self.cx or load_connectome()
        self.atlas = Atlas(self.cx)
        all_eyes = load_eye_maps(self.cx, self.atlas)
        self.eyes = {s: all_eyes[s] for s in self.eyes_used}
        self.encoder = RetinaEncoder(self.eyes, **self.encoder_kwargs)
        self.slow_pre = (slow_pre_mask(self.cx, self.atlas, self.slow_types)
                         if len(self.slow_types) else None)
        self.net = MLXLIFNetwork(weights(self.cx, self.preset), self.params, seed=self.seed,
                                 slow_pre=self.slow_pre)
        self.optic = np.flatnonzero((self.cx.neurons.super_class == "optic").fillna(False).to_numpy(bool))
        self.bias_mv = np.zeros(len(self.optic))
        if self.calibrate:
            self.bias_mv = calibrate_optic_bias(self)
            self.net.set_bias(self.optic, mv=self.bias_mv)
            self.net.reset()
            self.encoder.reset()
        if len(self.silence):
            self.net.silence(np.asarray(self.silence))

        self.groups: dict[str, np.ndarray] = {}
        for name, types in VISUAL_GROUPS.items():
            idx = np.concatenate([self.atlas.by_type(t) for t in types])
            if len(idx):
                self.groups[name] = idx
                for side in ("left", "right"):
                    si = np.concatenate([self.atlas.by_type(t, side) for t in types])
                    if len(si):
                        self.groups[f"{name}_{side[0].upper()}"] = si
        self.groups.update(self.extra_groups)
        self._names = list(self.groups)
        flat = np.concatenate([self.groups[n] for n in self._names])
        self._offs = np.cumsum([len(self.groups[n]) for n in self._names])[:-1]
        self._gather = mx.array(flat.astype(np.int64))
        self._mx = mx
        self._prev = np.zeros(len(flat), np.int64)
        self._step_debt = 0.0

    def reset(self):
        """시행 사이 초기화. net.reset() 만 부르면 안 된다 — 누적 발화수(cnt)는 0 으로
        돌아가는데 _prev 는 이전 값을 들고 있어서 첫 프레임 발화율이 거대한 음수가 된다.
        작동점 bias 와 인코더 순응 상태도 함께 되돌린다."""
        self.net.reset()
        if self.calibrate:
            self.net.set_bias(self.optic, mv=self.bias_mv)
        if len(self.silence):
            self.net.silence(np.asarray(self.silence))
        self.encoder.reset()
        self._prev = np.zeros_like(self._prev)
        self._step_debt = 0.0

    # -- 프레임 --------------------------------------------------------------
    def frame(self, lum: dict[str, np.ndarray], dt_s: float, eval_every: int = 64) -> dict[str, float]:
        """컬럼 휘도를 넣고 dt_s 만큼 생물시간을 진행한 뒤 그룹 평균 발화율(Hz)을 돌려준다."""
        idx, rate = self.encoder.encode(lum, dt_s)
        self.net.set_stimulus((idx, rate))
        # 생물시간 오차 누적 방지: dt_s 를 정수 스텝으로 나누되 나머지를 다음 프레임에 넘긴다
        self._step_debt += dt_s * 1e3 / self.params.dt
        n_steps = int(self._step_debt)
        self._step_debt -= n_steps
        self._run_steps(n_steps, eval_every)
        return self._read(n_steps * self.params.dt * 1e-3)

    def _run_steps(self, n_steps: int, eval_every: int):
        net = self.net
        for k in range(n_steps):
            net._tick()
            if (k + 1) % eval_every == 0:
                self._mx.eval(*net._state(), *net._ring)
        self._mx.eval(*net._state(), *net._ring)

    def _read(self, dur_s: float) -> dict[str, float]:
        cnt = np.array(self._mx.take(self.net.cnt, self._gather)).astype(np.int64)
        d = cnt - self._prev
        self._prev = cnt
        dur = max(dur_s, 1e-6)
        return {n: float(p.mean()) / dur for n, p in zip(self._names, np.split(d, self._offs))}

    def group_size(self, name: str) -> int:
        return len(self.groups.get(name, ()))


# ---------------------------------------------------------------------------
# 시각엽 작동점(tonic bias) 항상성 보정
# ---------------------------------------------------------------------------
def _bias_cache_path(vb: "VisualBrain") -> "object":
    e = vb.encoder
    p = vb.params
    slow = "-".join(vb.slow_types) if len(vb.slow_types) else "none"
    tag = (f"{vb.preset}_t{vb.target_hz:g}_b{e.r_base:g}_g{e.r_gain:g}_l{e.lum_ref:g}"
           f"_e{''.join(x[0] for x in sorted(vb.eyes))}_s{vb.seed}"
           f"_ts{p.tau_slow:g}_ws{vb.net.w_slow:.4f}_sl{slow}")
    return PATHS["cache"] / f"optic_bias_{tag}.npy"


def calibrate_optic_bias(vb: "VisualBrain", n_iter: int = 60, window_ms: float = 200.0,
                         step_mv: float = 1.2, lo: float = -20.0, hi: float = 90.0,
                         force: bool = False, verbose: bool = True) -> np.ndarray:
    """균일 회색 화면에서 시각엽 내재뉴런이 target_hz 로 발화하도록 뉴런별 tonic bias 를 찾는다.

    스파이킹 LIF 로 등급전위(graded) 시각엽을 근사할 때 필요한 작동점 보정이다.
    네트워크를 멈추지 않고 계속 돌리면서 창(window_ms)마다 발화율을 재고
        bias ← bias + step · tanh(1 - rate/target)
    로 갱신한다. 되먹임이 있는 회로이므로 작은 보폭으로 여러 번 돈다.
    결과는 data/cache 에 저장한다 (자극 세기/목표치가 바뀌면 다시 계산).
    """
    path = _bias_cache_path(vb)
    if path.exists() and not force:
        b = np.load(path)
        if len(b) == len(vb.optic):
            if verbose:
                print(f"[flybrain] 시각엽 작동점 보정 캐시 사용: {path.name}", flush=True)
            return b
    import mlx.core as mx
    from .visual import UniformField

    net, optic = vb.net, vb.optic
    gray = UniformField(vb.encoder.lum_ref).columns(vb.eyes, 0.0)
    idx, rate = vb.encoder.encode(gray, 0.0)
    net.set_stimulus((idx, rate))
    gather = mx.array(optic.astype(np.int64))
    bias = np.zeros(len(optic))
    steps = int(round(window_ms / vb.params.dt))
    dur_s = window_ms * 1e-3
    prev = np.zeros(len(optic), np.int64)
    if verbose:
        print(f"[flybrain] 시각엽 작동점 보정 ({len(optic):,}개 뉴런 → {vb.target_hz:g} Hz, "
              f"{n_iter}회 × {window_ms:g} ms) ...", flush=True)
    for it in range(n_iter):
        for k in range(steps):
            net._tick()
            if (k + 1) % 64 == 0:
                mx.eval(*net._state(), *net._ring)
        mx.eval(*net._state(), *net._ring)
        cnt = np.array(mx.take(net.cnt, gather)).astype(np.int64)
        r = (cnt - prev) / dur_s
        prev = cnt
        bias = np.clip(bias + step_mv * np.tanh(1.0 - r / vb.target_hz), lo, hi)
        net.set_bias(optic, mv=bias)
        if verbose and (it + 1) % 15 == 0:
            print(f"           {it + 1:3d}/{n_iter}  평균 {r.mean():5.1f} Hz  "
                  f"중앙값 {np.median(r):5.1f} Hz  침묵 {np.mean(r == 0):.0%}  "
                  f"bias {np.median(bias):5.1f} mV", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, bias)
    return bias


# ---------------------------------------------------------------------------
# 프레임 시계 / slip 통계
# ---------------------------------------------------------------------------
@dataclass
class FrameClock:
    """생물시간을 벽시계에 맞춘다. 프레임 예산을 넘긴 양(slip)을 기록한다.

    realtime=False 면 기다리지 않고 최대 속도로 돌리되 프레임 소요시간은 그대로 잰다
    (실시간 유지가 가능했는지는 그대로 판정할 수 있다).
    """
    fps: float = 60.0
    realtime: bool = True
    budget_ms: float = field(init=False)
    work_ms: list = field(default_factory=list)
    slip_ms: list = field(default_factory=list)
    _t0: float = 0.0
    _frame: int = 0

    def __post_init__(self):
        self.budget_ms = 1000.0 / self.fps

    def start(self):
        self._t0 = time.perf_counter()
        self._frame = 0

    def tick(self, work_start: float):
        """프레임 작업이 끝난 직후 호출. work_start 는 작업 시작 시각(perf_counter)."""
        now = time.perf_counter()
        self.work_ms.append((now - work_start) * 1e3)
        self._frame += 1
        deadline = self._t0 + self._frame * self.budget_ms * 1e-3
        late = now - deadline
        if self.realtime and late < 0:
            time.sleep(-late)
            late = 0.0
        self.slip_ms.append(max(late, 0.0) * 1e3)

    def stats(self) -> dict:
        w = np.array(self.work_ms) if self.work_ms else np.zeros(1)
        s = np.array(self.slip_ms) if self.slip_ms else np.zeros(1)
        return {
            "n_frames": len(self.work_ms),
            "budget_ms": round(self.budget_ms, 2),
            "work_mean_ms": round(float(w.mean()), 2),
            "work_p50_ms": round(float(np.percentile(w, 50)), 2),
            "work_p95_ms": round(float(np.percentile(w, 95)), 2),
            "work_max_ms": round(float(w.max()), 2),
            "over_budget_frac": round(float((w > self.budget_ms).mean()), 4),
            "slip_total_ms": round(float(s.sum()), 1),
            "slip_max_ms": round(float(s.max()), 2),
            "realtime_ok": bool(np.percentile(w, 95) <= self.budget_ms),
        }

    def report(self) -> str:
        st = self.stats()
        return (f"프레임 {st['n_frames']}개 · 예산 {st['budget_ms']} ms | "
                f"작업 평균 {st['work_mean_ms']} / p50 {st['work_p50_ms']} / p95 {st['work_p95_ms']} / "
                f"최대 {st['work_max_ms']} ms | 예산 초과 {st['over_budget_frac']:.1%} | "
                f"누적 지연 {st['slip_total_ms']} ms (최대 {st['slip_max_ms']} ms) | "
                f"실시간 {'유지' if st['realtime_ok'] else '실패'}")
