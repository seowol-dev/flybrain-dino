"""Apple Silicon GPU(Metal) LIF 엔진 — lif.LIFNetwork 와 같은 동역학, 같은 인터페이스.

한 스텝(0.1 ms)은 Metal 커널 두 번으로 처리한다.
  1) propagate : 1.8 ms 전에 발화한 뉴런의 출력 시냅스만 순회해 시냅스후 뉴런에
                 부호 있는 시냅스 수를 int32 원자적 덧셈 (결정적, 순서 무관).
                 작업 단위 = (시냅스전 뉴런, 최대 CHUNK 개의 시냅스) → 부하 분산.
  2) update    : 모든 뉴런을 한 번에 — 불응기 판정, 입력 반영, 정확해 적분, 역치,
                 Poisson 자극(해시 난수), 리셋, 발화 기록, 발화 수 누적.

numpy 엔진(lif.py)과의 차이: Poisson 난수열이 다르다(통계적으로 동일), STP/적응 미지원.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import mlx.core as mx

from .lif import LIFParams

CHUNK = 64

_HEADER = """
inline uint pcg_hash(uint x) {
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
"""

_PROP_SRC = """
    uint k = thread_position_in_grid.x;
    if (k >= n_items[0]) return;
    int pre = item_pre[k];
    if (fired[pre] == 0) return;
    int e1 = item_end[k];
    for (int e = item_start[k]; e < e1; ++e) {
        atomic_fetch_add_explicit(&acc[post[e]], wint[e], memory_order_relaxed);
    }
"""

_UPD_SRC = """
    uint i = thread_position_in_grid.x;
    uint n = (uint)ncount[0];
    if (i >= n) return;
    // prm: 0 v_0, 1 v_rst, 2 v_th, 3 w_syn, 4 e_m, 5 e_s, 6 a
    float v0 = prm[0], vr = prm[1], vth = prm[2], wsyn = prm[3], em = prm[4], es = prm[5], a = prm[6];
    int rfc_steps = iprm[0];
    uint tick = (uint)iprm[1];
    uint seed = (uint)iprm[2];
    float v = v_in[i];
    float g = g_in[i];
    int r = rfc_in[i];
    bool active = r <= 0;
    if (active) {
        g += (float)acc[i] * wsyn + bias[i];
        float A = g * a;
        v = v0 + (v - v0 - A) * em + A * es;
        g = g * es;
    }
    r -= 1;
    bool f = active && (v > vth);
    float p = stim_p[i];
    bool hit = false;
    if (p > 0.0f) {
        uint h = pcg_hash(i ^ pcg_hash(tick ^ pcg_hash(seed)));
        float u = (float)(h >> 8) * (1.0f / 16777216.0f);
        hit = u < p;
        f = f || hit;
    }
    if (silenced[i] != 0) f = false;
    if (f) {
        v = vr; g = 0.0f;
        r = (p > 0.0f) ? 0 : (rfc_steps - 1);
    }
    v_out[i] = v; g_out[i] = g; rfc_out[i] = r;
    fired_out[i] = f ? 1 : 0;
    cnt_out[i] = cnt_in[i] + (f ? 1 : 0);
"""


class MLXLIFNetwork:
    """GPU 전뇌 LIF. 사용법은 lif.LIFNetwork 와 같다 (stimulate/set_stimulus/silence/run/rates).

    기록은 발화 '수'만 누적한다(스파이크 시각 목록은 없음). rates(t0, t1) 은
    mark() 로 찍어 둔 구간 사이의 평균 발화율을 돌려준다.
    """

    def __init__(self, W_pre: sp.csr_matrix, params: LIFParams | None = None, seed: int = 0,
                 chunk: int = CHUNK):
        self.p = params or LIFParams()
        p = self.p
        if p.stp_U > 0 or p.adapt_inc > 0:
            raise NotImplementedError("GPU 엔진은 STP/적응을 지원하지 않습니다. lif.LIFNetwork 를 쓰세요.")
        W = W_pre.tocsr()
        W.sort_indices()
        if not np.allclose(W.data, np.round(W.data)):
            raise ValueError("GPU 엔진은 정수 시냅스 수 가중치가 필요합니다.")
        self.n = W.shape[0]
        n = self.n
        indptr = W.indptr.astype(np.int64)
        deg = np.diff(indptr)
        nchunks = np.maximum(1, -(-deg // chunk))
        nchunks[deg == 0] = 0
        item_pre = np.repeat(np.arange(n, dtype=np.int32), nchunks)
        first = np.repeat(np.cumsum(nchunks) - nchunks, nchunks)
        part = np.arange(len(item_pre)) - first
        start = indptr[item_pre] + part * chunk
        end = np.minimum(start + chunk, indptr[item_pre + 1])
        self._item_pre = mx.array(item_pre)
        self._item_start = mx.array(start.astype(np.int32))
        self._item_end = mx.array(end.astype(np.int32))
        self._n_items = mx.array(np.array([len(item_pre)], np.int32))
        self._post = mx.array(W.indices.astype(np.int32))
        self._wint = mx.array(np.round(W.data).astype(np.int32))
        self._ncount = mx.array(np.array([n], np.int32))
        self._W_host = W

        self.delay_steps = max(1, int(round(p.t_dly / p.dt)))
        self.rfc_steps = int(round(p.t_rfc / p.dt))
        e_m = np.exp(-p.dt / p.t_mbr)
        e_s = np.exp(-p.dt / p.tau)
        a = p.tau / (p.tau - p.t_mbr)
        self._prm = mx.array(np.array([p.v_0, p.v_rst, p.v_th, p.w_syn, e_m, e_s, a], np.float32))
        self.seed = int(seed) & 0x7FFFFFFF

        self._prop = mx.fast.metal_kernel(
            name="fly_propagate", input_names=["n_items", "item_pre", "item_start", "item_end", "post", "wint", "fired"],
            output_names=["acc"], source=_PROP_SRC, atomic_outputs=True)
        self._upd = mx.fast.metal_kernel(
            name="fly_update",
            input_names=["ncount", "prm", "iprm", "v_in", "g_in", "rfc_in", "acc", "stim_p", "bias", "silenced",
                         "cnt_in"],
            output_names=["v_out", "g_out", "rfc_out", "fired_out", "cnt_out"],
            source=_UPD_SRC, header=_HEADER)
        self._silenced_np = np.zeros(n, np.int32)
        self._stim_np = np.zeros(n, np.float32)
        self._bias_np = np.zeros(n, np.float32)
        self.reset()

    # -- 상태 ----------------------------------------------------------------
    def reset(self):
        n = self.n
        self.v = mx.full((n,), self.p.v_0, dtype=mx.float32)
        self.g = mx.zeros((n,), dtype=mx.float32)
        self.rfc = mx.zeros((n,), dtype=mx.int32)
        self.cnt = mx.zeros((n,), dtype=mx.int32)
        zero = mx.zeros((n,), dtype=mx.uint8)
        self._ring = [zero] * (self.delay_steps + 1)
        self.step_i = 0
        self._stim = mx.array(self._stim_np)
        self._bias = mx.array(self._bias_np)
        self._sil = mx.array(self._silenced_np)
        self._marks: dict = {}
        mx.eval(self.v, self.g, self.rfc, self.cnt, zero)

    @property
    def t_ms(self) -> float:
        return self.step_i * self.p.dt

    def set_stimulus(self, rates):
        """rates: {idx: Hz} | (idx 배열, Hz 배열) | None"""
        s = np.zeros(self.n, np.float32)
        if rates is not None and not (isinstance(rates, dict) and not rates):
            if isinstance(rates, dict):
                idx = np.fromiter(rates.keys(), np.int64, len(rates))
                r = np.fromiter(rates.values(), np.float64, len(rates))
            else:
                idx, r = np.asarray(rates[0], np.int64), np.asarray(rates[1], np.float64)
            s[idx] = np.clip(r * self.p.dt * 1e-3, 0, 1)
        self._stim_np = s
        self._stim = mx.array(s)

    def stimulate(self, idx, rate: float | None = None):
        rate = self.p.r_poi if rate is None else rate
        s = self._stim_np.copy()
        s[np.asarray(idx, np.int64)] = np.clip(rate * self.p.dt * 1e-3, 0, 1)
        self._stim_np = s
        self._stim = mx.array(s)

    def silence(self, idx):
        self._silenced_np[np.asarray(idx, np.int64)] = 1
        self._sil = mx.array(self._silenced_np)

    def unsilence(self, idx=None):
        if idx is None:
            self._silenced_np[:] = 0
        else:
            self._silenced_np[np.asarray(idx, np.int64)] = 0
        self._sil = mx.array(self._silenced_np)

    def set_bias(self, idx=None, mv: float = 0.0):
        """지속 탈분극 전류(tonic bias). mv = 정상상태에서 휴지전위 위로 올라가는 값(mV).

        v_th - v_0 = 7 mV 이므로 mv=7 이면 딱 역치, 그 위면 지속 발화한다.
        광수용체 R1-6 은 히스타민(억제)이므로 라미나 세포(L1~L3)가 어둠에서 탈분극해 있어야
        빛이 그것을 눌러 신호가 된다 (Laughlin 1981). 그 휴지 탈분극을 여기서 준다.
        idx=None 이면 전체를 mv 로 설정한다.
        """
        e_s = float(np.exp(-self.p.dt / self.p.tau))
        c = mv * (1.0 - e_s) / e_s          # 스텝당 g 증가량 → 정상상태 g = mv
        if idx is None:
            self._bias_np[:] = c
        else:
            self._bias_np[np.asarray(idx, np.int64)] = c
        self._bias = mx.array(self._bias_np)

    # -- 적분 ----------------------------------------------------------------
    def _tick(self):
        L = len(self._ring)
        fired_old = self._ring[(self.step_i - self.delay_steps) % L]
        acc = self._prop(
            inputs=[self._n_items, self._item_pre, self._item_start, self._item_end, self._post, self._wint, fired_old],
            grid=(int(self._n_items.item()) if not hasattr(self, "_ni") else self._ni, 1, 1),
            threadgroup=(256, 1, 1), output_shapes=[(self.n,)], output_dtypes=[mx.int32], init_value=0,
        )[0]
        iprm = mx.array(np.array([self.rfc_steps, self.step_i, self.seed], np.int32))
        v, g, rfc, fired, cnt = self._upd(
            inputs=[self._ncount, self._prm, iprm, self.v, self.g, self.rfc, acc, self._stim, self._bias,
                    self._sil, self.cnt],
            grid=(self.n, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(self.n,)] * 5,
            output_dtypes=[mx.float32, mx.float32, mx.int32, mx.uint8, mx.int32],
        )
        self.v, self.g, self.rfc, self.cnt = v, g, rfc, cnt
        self._ring[self.step_i % L] = fired
        self.step_i += 1

    def run(self, duration_ms: float, progress: bool = False, eval_every: int = 64):
        self._ni = int(self._n_items.item())
        n_steps = int(round(duration_ms / self.p.dt))
        for k in range(n_steps):
            self._tick()
            if (k + 1) % eval_every == 0:
                mx.eval(self.v, self.g, self.rfc, self.cnt, *self._ring)
        mx.eval(self.v, self.g, self.rfc, self.cnt, *self._ring)

    def step(self):
        self._ni = int(self._n_items.item())
        self._tick()

    # -- 기록 ----------------------------------------------------------------
    def counts(self) -> np.ndarray:
        """시작 이후 누적 발화 수."""
        return np.array(self.cnt)

    def mark(self, name="last"):
        """현재 시각과 누적 발화 수를 기록 (구간 발화율 계산용)."""
        self._marks[name] = (self.t_ms, self.counts())

    def rates_since(self, name="last") -> np.ndarray:
        t0, c0 = self._marks[name]
        dur = max(self.t_ms - t0, self.p.dt) * 1e-3
        return (self.counts() - c0) / dur

    def rates(self, t_from_ms: float = 0.0, t_to_ms: float | None = None) -> np.ndarray:
        """t_from_ms=0 이면 시작 이후 평균 발화율. 그 외 구간은 mark()/rates_since() 를 쓴다."""
        if t_from_ms != 0.0 or (t_to_ms is not None and abs(t_to_ms - self.t_ms) > 1e-9):
            raise ValueError("GPU 엔진은 임의 구간 rates 를 지원하지 않습니다. mark()/rates_since() 를 쓰세요.")
        return self.counts() / max(self.t_ms * 1e-3, 1e-9)
