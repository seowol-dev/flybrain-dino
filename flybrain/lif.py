"""Leaky integrate-and-fire 전뇌 네트워크 (numpy / scipy.sparse 구현).

모델과 상수는 Shiu et al. 2024 (Nature, "A Drosophila computational brain model reveals
sensorimotor processing") 의 Brian2 모델을 그대로 따른다:

    dv/dt = (v_0 - v + g) / t_mbr      (refractory 동안 정지)
    dg/dt = -g / tau                   (refractory 동안 정지)
    spike : v > v_th  ->  v = v_rst, g = 0, refractory t_rfc
    synapse: pre 스파이크 후 t_dly 지연 뒤 g += w,   w = w_syn × 부호 × 시냅스 수

외부 자극(광유전학 활성 모델)은 Poisson 과정으로 강제 스파이크를 일으킨다 (Shiu: r_poi=150 Hz,
자극 뉴런은 refractory 없음).

두 지수 감쇠는 선형이므로 스텝마다 정확해(exact) 로 적분한다 (Brian2 method='linear' 와 동일).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class LIFParams:
    v_0: float = -52.0      # mV, 휴지 전위
    v_rst: float = -52.0    # mV, 리셋 전위
    v_th: float = -45.0     # mV, 역치
    t_mbr: float = 20.0     # ms, 막 시정수
    tau: float = 5.0        # ms, 시냅스 시정수
    t_rfc: float = 2.2      # ms, 불응기
    t_dly: float = 1.8      # ms, 시냅스 지연
    w_syn: float = 0.275    # mV, 시냅스 1개당 가중치
    dt: float = 0.1         # ms, 적분 스텝
    r_poi: float = 150.0    # Hz, 기본 자극 빈도
    # --- 선택적 확장 (Shiu 원 모델에는 없음; 0 이면 꺼짐) ---
    stp_U: float = 0.0      # 단기 시냅스 억압: 스파이크당 자원 소모 비율 (Tsodyks-Markram), 0=off
    stp_tau_rec: float = 0.0  # ms, 자원 회복 시정수
    adapt_inc: float = 0.0  # mV, 스파이크당 적응 전류 증가량 (spike-frequency adaptation), 0=off
    adapt_tau: float = 0.0  # ms, 적응 전류 감쇠 시정수


@dataclass
class SpikeRecord:
    """스파이크 기록 (스텝 단위)."""
    steps: list = field(default_factory=list)
    ids: list = field(default_factory=list)

    def append(self, step: int, idx: np.ndarray):
        if len(idx):
            self.steps.append(np.full(len(idx), step, dtype=np.int64))
            self.ids.append(idx.astype(np.int64))

    def arrays(self):
        if not self.ids:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(self.steps), np.concatenate(self.ids)

    def clear(self):
        self.steps.clear()
        self.ids.clear()


class LIFNetwork:
    """전뇌 LIF 네트워크.

    Parameters
    ----------
    W_pre : csr_matrix (N,N)  행=pre, 열=post, 값=부호×시냅스수
    params : LIFParams
    seed : int
    """

    def __init__(self, W_pre: sp.csr_matrix, params: LIFParams | None = None, seed: int = 0):
        self.p = params or LIFParams()
        p = self.p
        self.n = W_pre.shape[0]
        # 가중치(mV) — float32 로 메모리 절약 (15M nnz → 60MB)
        self.W = sp.csr_matrix((W_pre.data.astype(np.float32) * np.float32(p.w_syn), W_pre.indices, W_pre.indptr), shape=W_pre.shape)
        self.rng = np.random.default_rng(seed)

        self.delay_steps = max(1, int(round(p.t_dly / p.dt)))
        self.rfc_steps = int(round(p.t_rfc / p.dt))
        # exact integration 상수
        self.e_m = np.float32(np.exp(-p.dt / p.t_mbr))
        self.e_s = np.float32(np.exp(-p.dt / p.tau))
        self.a = np.float32(p.tau / (p.tau - p.t_mbr))  # 특수해 계수

        self.v = np.full(self.n, p.v_0, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        # 단기 시냅스 억압 (presynaptic 자원 x ∈ [0,1])
        self.use_stp = p.stp_U > 0 and p.stp_tau_rec > 0
        self.x = np.ones(self.n, dtype=np.float32)
        self._last_spk_step = np.zeros(self.n, dtype=np.int64)
        # 적응 전류 (mV, v 방정식에 -a 로 들어감)
        self.use_adapt = p.adapt_inc > 0 and p.adapt_tau > 0
        self.ad = np.zeros(self.n, dtype=np.float32)
        if self.use_adapt:
            self.e_a = np.float32(np.exp(-p.dt / p.adapt_tau))
            self.a_ad = np.float32(p.adapt_tau / (p.adapt_tau - p.t_mbr))
        self.rfc_left = np.zeros(self.n, dtype=np.int32)
        self.step_i = 0
        self._ring: list[np.ndarray] = [np.zeros(0, np.int64) for _ in range(self.delay_steps + 1)]
        self._x_at_spike: list[np.ndarray] = [np.zeros(0, np.float32) for _ in range(self.delay_steps + 1)]
        self.silenced = np.zeros(self.n, dtype=bool)
        self._any_silenced = False

        # 외부 Poisson 자극: 뉴런 인덱스 -> 스텝당 발화 확률
        self._stim_idx = np.zeros(0, np.int64)
        self._stim_p = np.zeros(0, np.float64)
        self._no_rfc = np.zeros(self.n, dtype=bool)
        # 지속 전류 주입 (mV 단위 상수 g 오프셋) — 선택적
        self.bias = np.zeros(self.n, dtype=np.float32)
        self._any_bias = False
        self.record = SpikeRecord()
        self.recording = True

    # -- 자극 ---------------------------------------------------------------
    def set_stimulus(self, rates: dict[int, float] | tuple[np.ndarray, np.ndarray] | None):
        """rates: {neuron_idx: Hz} 또는 (idx 배열, Hz 배열). None 이면 자극 제거."""
        if rates is None or (isinstance(rates, dict) and not rates):
            self._stim_idx = np.zeros(0, np.int64)
            self._stim_p = np.zeros(0)
            self._no_rfc[:] = False
            return
        if isinstance(rates, dict):
            idx = np.fromiter(rates.keys(), dtype=np.int64, count=len(rates))
            r = np.fromiter(rates.values(), dtype=np.float64, count=len(rates))
        else:
            idx, r = np.asarray(rates[0], np.int64), np.asarray(rates[1], np.float64)
        keep = r > 0
        self._stim_idx = idx[keep]
        self._stim_p = np.clip(r[keep] * self.p.dt * 1e-3, 0, 1)
        self._no_rfc[:] = False
        self._no_rfc[self._stim_idx] = True

    def stimulate(self, idx, rate: float | None = None):
        """편의 함수: idx 뉴런들을 rate Hz 로 자극 (기존 자극에 추가)."""
        rate = self.p.r_poi if rate is None else rate
        idx = np.asarray(idx, np.int64)
        d = dict(zip(self._stim_idx.tolist(), (self._stim_p / (self.p.dt * 1e-3)).tolist()))
        for i in idx:
            d[int(i)] = rate
        self.set_stimulus(d)

    def set_bias(self, idx=None, mv: float = 0.0):
        """지속 탈분극 전류(tonic bias). mv = 정상상태에서 휴지전위 위로 올라가는 값(mV).

        v_th - v_0 = 7 mV 이므로 mv=7 이면 딱 역치, 그 위면 지속 발화한다.
        MLXLIFNetwork.set_bias 와 같은 정의다.
        """
        e_s = float(np.exp(-self.p.dt / self.p.tau))
        c = mv * (1.0 - e_s) / e_s
        if idx is None:
            self.bias[:] = c
        else:
            self.bias[np.asarray(idx, np.int64)] = c
        self._any_bias = bool(np.any(self.bias))

    def silence(self, idx):
        """뉴런 억제 (Kir2.1 식): 발화 자체를 막고 출력 시냅스도 0.
        나머지 네트워크에 대한 효과는 Shiu 의 silencing(출력 시냅스 0)과 같고,
        억제된 뉴런 자신의 기록 발화율도 0 이 된다 (운동뉴런 억제 실험에 필요)."""
        idx = np.asarray(idx, np.int64)
        self.silenced[idx] = True
        self._any_silenced = True
        W = self.W
        for i in idx:
            W.data[W.indptr[i]:W.indptr[i + 1]] = 0.0

    def reset(self):
        self.v[:] = self.p.v_0
        self.g[:] = 0
        self.rfc_left[:] = 0
        self.step_i = 0
        self._ring = [np.zeros(0, np.int64) for _ in range(self.delay_steps + 1)]
        self._x_at_spike = [np.zeros(0, np.float32) for _ in range(self.delay_steps + 1)]
        self.x[:] = 1
        self._last_spk_step[:] = 0
        self.ad[:] = 0
        self.record.clear()

    # -- 적분 ---------------------------------------------------------------
    @property
    def t_ms(self) -> float:
        return self.step_i * self.p.dt

    def step(self) -> np.ndarray:
        """한 스텝(dt) 진행. 이번 스텝에 발화한 뉴런 인덱스를 돌려준다."""
        p = self.p
        L = len(self._ring)
        # 1) 불응기 판정. Brian2 는 spike 후 timestep(t_rfc) 스텝 동안(= s+1 .. s+21) 불응.
        active = self.rfc_left <= 0
        # 2) 지연된 시냅스 입력 전달. Brian2 에서 g 는 '(unless refractory)' 이므로
        #    불응기 중 도착한 시냅스 입력은 버려진다 (Brian2 conditional write 규칙과 동일).
        arriving = self._ring[(self.step_i - self.delay_steps) % L]
        if len(arriving):
            if self.use_stp:
                # 도착한 스파이크는 발화 시점의 자원 x 로 가중 (발화 시 기록해 둔 값)
                eff = self._x_at_spike[(self.step_i - self.delay_steps) % L]
                dg = np.asarray(eff @ self.W[arriving], dtype=np.float32).ravel()
            else:
                dg = np.asarray(self.W[arriving].sum(axis=0), dtype=np.float32).ravel()
            dg[~active] = 0.0
            self.g += dg
        if self._any_bias:
            self.g[active] += self.bias[active]
        # 3) 막전위 / 시냅스 변수 정확해 적분 (불응기 아닌 뉴런만)
        v, g = self.v, self.g
        A = g * self.a
        if self.use_adapt:
            B = -self.ad * self.a_ad
            v_new = p.v_0 + (v - p.v_0 - A - B) * self.e_m + A * self.e_s + B * self.e_a
            self.ad *= self.e_a
        else:
            v_new = p.v_0 + (v - p.v_0 - A) * self.e_m + A * self.e_s
        g_new = g * self.e_s
        np.copyto(v, v_new, where=active)
        np.copyto(g, g_new, where=active)
        self.rfc_left -= 1
        # 4) 역치 통과
        fired = np.flatnonzero((v > p.v_th) & active)
        # 5) 외부 Poisson 자극 (강제 스파이크, 자극 뉴런은 불응기 없음 — Shiu 모델과 동일)
        hit = None
        if len(self._stim_idx):
            hit = self._stim_idx[self.rng.random(len(self._stim_idx)) < self._stim_p]
            if len(hit):
                fired = np.union1d(fired, hit)
            else:
                hit = None
        if self._any_silenced and len(fired):
            fired = fired[~self.silenced[fired]]
        if len(fired):
            v[fired] = p.v_rst
            g[fired] = 0.0
            # 자극 대상 뉴런은 Shiu 모델처럼 불응기 0
            self.rfc_left[fired] = np.where(self._no_rfc[fired], 0, self.rfc_steps - 1)
        fired_out = fired[~self.silenced[fired]] if (len(fired) and self._any_silenced) else fired
        if self.use_adapt and len(fired):
            self.ad[fired] += np.float32(p.adapt_inc)
        if self.use_stp:
            # 회복: x ← 1 - (1 - x)·exp(-Δt/τ_rec)  (마지막 스파이크 이후 경과 시간 기준, 발화 뉴런만 갱신)
            if len(fired_out):
                el = (self.step_i - self._last_spk_step[fired_out]) * p.dt
                xr = 1.0 - (1.0 - self.x[fired_out]) * np.exp(-el / p.stp_tau_rec)
                self._x_at_spike[self.step_i % L] = xr.astype(np.float32)
                self.x[fired_out] = xr * (1.0 - p.stp_U)
                self._last_spk_step[fired_out] = self.step_i
            else:
                self._x_at_spike[self.step_i % L] = np.zeros(0, np.float32)
        self._ring[self.step_i % L] = fired_out
        if self.recording:
            self.record.append(self.step_i, fired)
        self.step_i += 1
        return fired

    def run(self, duration_ms: float, progress: bool = False) -> None:
        n_steps = int(round(duration_ms / self.p.dt))
        it = range(n_steps)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="sim", unit="step", leave=False)
        for _ in it:
            self.step()

    # -- 분석 ---------------------------------------------------------------
    def spikes(self, t_from_ms: float = 0.0, t_to_ms: float | None = None):
        """(time_ms, neuron_idx) 배열."""
        steps, ids = self.record.arrays()
        t = steps * self.p.dt
        m = t >= t_from_ms
        if t_to_ms is not None:
            m &= t < t_to_ms
        return t[m], ids[m]

    def rates(self, t_from_ms: float = 0.0, t_to_ms: float | None = None) -> np.ndarray:
        """구간 평균 발화율 (Hz), shape (N,)."""
        t_to_ms = self.t_ms if t_to_ms is None else t_to_ms
        _, ids = self.spikes(t_from_ms, t_to_ms)
        counts = np.bincount(ids, minlength=self.n)
        dur = max(t_to_ms - t_from_ms, self.p.dt) * 1e-3
        return counts / dur

    def group_rate(self, idx, t_from_ms: float, t_to_ms: float | None = None) -> float:
        """그룹 평균 발화율 (Hz/neuron)."""
        idx = np.asarray(idx)
        if len(idx) == 0:
            return 0.0
        return float(self.rates(t_from_ms, t_to_ms)[idx].mean())
