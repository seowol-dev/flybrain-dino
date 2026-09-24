"""망막지도(retinotopy) 복구 + 화면 → 광수용체 인코더.

문제: FlyWire v783 에서 R1-6 광수용체 7,932개는 **soma 좌표가 없다**(망막이 촬상 볼륨 밖).
그래서 "어느 광수용체가 시야의 어느 방향을 보는가"를 좌표로는 알 수 없고, 지금까지처럼
광수용체 전체를 같은 세기로 자극하면 균일 조명이 되어 T4/T5(시공간 대비 검출기)는
원리적으로 켜지지 않는다.

해결: 컬럼 격자를 **Mi1** 으로 대신한다. Mi1 은 메둘라 컬럼당 정확히 1개이고 좌표가
거의 완비되어 있다(1,555/1,584). 그리고 광수용체는 라미나를 거쳐 자기 컬럼의 Mi1 로만
수렴한다:

    R1-6 → L1/L2/L3 (라미나, 같은 컬럼) → Mi1 (메둘라, 같은 컬럼)

따라서 |W[R→L]| @ |W[L→Mi1]| 의 행별 최대값이 그 광수용체의 컬럼이다. 실측상 이 배정은
거의 모호하지 않다(행 최대값이 행 합의 100% — 중앙값).

컬럼의 시야 방향은 Mi1 soma 좌표로 얻는다. 메둘라 세포체층은 눈을 감싸는 껍질이므로
구(sphere)를 최소제곱으로 맞춘 뒤 중심에서 본 방향을 (방위각, 고도) 로 쓴다.

좌표계: FAFB/FlyWire 복셀 (4, 4, 40) nm. +x = 오른쪽, +y = 배쪽(ventral), +z = 뒤쪽.
        (확인: Kenyon cell 세포체 z=193 µm ≫ AL PN z=49 µm → +z 는 후측,
               Kenyon cell y=132 µm ≪ AL PN y=260 µm → +y 는 복측)
        → 고도축 = -y(등쪽이 +), 방위각축 = -z(앞쪽이 +).

라미나→메둘라 시신경교차(chiasm)는 전후축을 뒤집으므로 `flip_az=True` 가 기본값이다.
이 부호는 `flybrain grating` 의 T4a~d 방향 선택성 측정으로 검증한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atlas import Atlas
from .config import PATHS
from .data import Connectome

VOXEL_UM = np.array([4.0, 4.0, 40.0]) / 1000.0   # FAFB 복셀 → µm
DORSAL = np.array([0.0, -1.0, 0.0])              # 등쪽 방향 (고도 +)
ANTERIOR = np.array([0.0, 0.0, -1.0])            # 앞쪽 방향 (방위각 +)
LAMINA_TYPES = ("L1", "L2", "L3")
SIDES = ("left", "right")


def _unit(v):
    return v / np.linalg.norm(v)


def _sphere_fit(P: np.ndarray) -> tuple[np.ndarray, float]:
    """점구름에 구를 최소제곱으로 맞춘다 → (중심, 반지름)."""
    A = np.hstack([2 * P, np.ones((len(P), 1))])
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(sol[3] + (c ** 2).sum()))


@dataclass
class EyeMap:
    """한쪽 눈의 망막지도.

    col_idx  : (C,) 컬럼 대표 뉴런(Mi1)의 모델 인덱스
    col_az/el: (C,) 컬럼의 시야 방향 (도). 방위각 + = 앞쪽, 고도 + = 등쪽
    pr_idx   : (P,) 컬럼이 배정된 광수용체(R1-6)의 모델 인덱스
    pr_col   : (P,) 각 광수용체가 속한 컬럼 번호 (0..C-1)
    """
    side: str
    col_idx: np.ndarray
    col_az: np.ndarray
    col_el: np.ndarray
    pr_idx: np.ndarray
    pr_col: np.ndarray
    sphere_r: float = 0.0
    resid_um: float = 0.0
    n_pr_total: int = 0

    @property
    def n_col(self) -> int:
        return len(self.col_idx)

    @property
    def n_pr(self) -> int:
        return len(self.pr_idx)

    def summary(self) -> str:
        return (f"[{self.side}] 컬럼 {self.n_col}개, 광수용체 {self.n_pr}/{self.n_pr_total}개 배정 "
                f"({self.n_pr / max(self.n_pr_total, 1):.0%}), 컬럼당 평균 {self.n_pr / max(self.n_col, 1):.1f}개, "
                f"방위각 {self.col_az.min():.0f}~{self.col_az.max():.0f}°, "
                f"고도 {self.col_el.min():.0f}~{self.col_el.max():.0f}°, "
                f"구 반지름 {self.sphere_r:.0f} µm (잔차 {self.resid_um:.1f} µm)")


def build_eye_map(cx: Connectome, atlas: Atlas, side: str, *, flip_az: bool = True,
                  flip_el: bool = False) -> EyeMap:
    """R1-6 → L1/L2/L3 → Mi1 연결로 광수용체를 메둘라 컬럼에 배정하고 컬럼의 시야 방향을 구한다."""
    import scipy.sparse as sp

    R = atlas.by_type("R1-6", side)
    L = np.concatenate([atlas.by_type(t, side) for t in LAMINA_TYPES])
    M = atlas.by_type("Mi1", side)
    soma = cx.neurons.iloc[M][["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    keep = np.isfinite(soma).all(axis=1)
    M, soma = M[keep], soma[keep] * VOXEL_UM

    W = abs(cx.W_pre).tocsr()
    S = (W[R][:, L] @ W[L][:, M]).tocsr()          # 광수용체 → (라미나 경유) → Mi1
    best = np.asarray(S.max(axis=1).todense()).ravel()
    col_of_pr = np.asarray(S.argmax(axis=1)).ravel()
    has_col = best > 0

    # 실제로 쓰인 컬럼만 남기고 0..C-1 로 다시 번호를 매긴다
    used = np.unique(col_of_pr[has_col])
    remap = np.full(len(M), -1, np.int64)
    remap[used] = np.arange(len(used))
    col_idx = M[used]

    # 컬럼 방향: 구 중심에서 본 단위벡터 → (방위각, 고도)
    c, r = _sphere_fit(soma)
    d = soma[used] - c
    resid = float(np.std(np.linalg.norm(soma - c, axis=1)))
    u = d / np.linalg.norm(d, axis=1)[:, None]
    m = _unit(u.mean(0))                            # 눈의 광축
    e_el = _unit(DORSAL - (DORSAL @ m) * m)
    e_az = ANTERIOR - (ANTERIOR @ m) * m
    e_az = _unit(e_az - (e_az @ e_el) * e_el)
    # 방위각등거리(azimuthal equidistant) 투영: 광축에서 벌어진 각 θ 를 그대로 반지름으로 쓴다.
    # 컵 모양 메둘라는 광축에서 100° 넘게 벌어진 컬럼이 있어 단순 (atan2) 방위각/고도로는
    # ±180° 에서 되감겨 지도가 찢어진다. 이 투영은 시야 120° 까지 되감김 없이 펼쳐진다.
    x, y, z = u @ m, u @ e_az, u @ e_el
    theta = np.degrees(np.arccos(np.clip(x, -1, 1)))
    phi = np.arctan2(z, y)
    az = theta * np.cos(phi) * (-1 if flip_az else 1)
    el = theta * np.sin(phi) * (-1 if flip_el else 1)

    return EyeMap(side=side, col_idx=col_idx, col_az=az, col_el=el,
                  pr_idx=R[has_col], pr_col=remap[col_of_pr[has_col]],
                  sphere_r=r, resid_um=resid, n_pr_total=len(R))


def map_frame(cx: Connectome, atlas: Atlas, side: str):
    """build_eye_map 과 같은 구면/축 기준을 다시 만들어 임의 뉴런을 지도 좌표로 투영한다.

    반환: project(idx) -> (n, 2) 배열 [방위각, 고도] (도). soma 좌표가 없는 뉴런은 NaN.
    T4/T5, Tm, LPLC 같은 뉴런의 수용장 위치를 알아야 할 때 쓴다(컬럼 배정이 없으므로
    세포체 위치를 같은 투영으로 넘긴다 — 세포유형별 깊이 차이는 방사 방향이라
    (방위각, 고도) 에는 거의 영향이 없다).
    """
    M = atlas.by_type("Mi1", side)
    soma = cx.neurons[["soma_x", "soma_y", "soma_z"]].to_numpy(float) * VOXEL_UM
    ok = np.isfinite(soma).all(axis=1)
    c, _ = _sphere_fit(soma[M[ok[M]]])
    u0 = soma[M[ok[M]]] - c
    u0 /= np.linalg.norm(u0, axis=1)[:, None]
    m = _unit(u0.mean(0))
    e_el = _unit(DORSAL - (DORSAL @ m) * m)
    e_az = ANTERIOR - (ANTERIOR @ m) * m
    e_az = _unit(e_az - (e_az @ e_el) * e_el)

    def project(idx, flip_az: bool = True, flip_el: bool = False):
        idx = np.asarray(idx, np.int64)
        out = np.full((len(idx), 2), np.nan)
        good = ok[idx]
        if not good.any():
            return out
        p = soma[idx[good]] - c
        u = p / np.linalg.norm(p, axis=1)[:, None]
        x, y, z = u @ m, u @ e_az, u @ e_el
        th = np.degrees(np.arccos(np.clip(x, -1, 1)))
        ph = np.arctan2(z, y)
        out[good, 0] = th * np.cos(ph) * (-1 if flip_az else 1)
        out[good, 1] = th * np.sin(ph) * (-1 if flip_el else 1)
        return out

    return project


def retinotopic_positions(cx: Connectome, atlas: Atlas, eye: EyeMap, *, max_hops: int = 2,
                          min_syn: int = 3) -> np.ndarray:
    """모든 뉴런에 망막지도 좌표 (방위각, 고도) 를 연결로 배정한다. (N, 2), 미배정은 NaN.

    Mi1 은 컬럼당 1개이고 위치가 이미 있다(EyeMap). 거기서 시냅스 수 가중 평균으로
    한 홉씩 퍼뜨린다. 세포체 위치로 투영하는 방법은 세포유형마다 껍질 반지름이 달라
    각도 규모가 어긋나므로(라미나 vs 메둘라) 쓰지 않는다.
    """
    N = len(cx.neurons)
    pos = np.full((N, 2), np.nan)
    pos[eye.col_idx, 0] = eye.col_az
    pos[eye.col_idx, 1] = eye.col_el
    W = abs(cx.W_pre).tocsr()
    W.data[W.data < min_syn] = 0
    W.eliminate_zeros()
    Wsym = (W + W.T).tocsr()          # 방향 무관하게 같은 컬럼 이웃을 찾는다
    for _ in range(max_hops):
        known = np.isfinite(pos[:, 0])
        src = np.where(known[:, None], np.nan_to_num(pos), 0.0)
        wk = Wsym @ known.astype(np.float64)               # 알려진 이웃으로의 시냅스 합
        acc = Wsym @ src                                    # 가중 좌표 합
        new = (~known) & (wk > 0)
        if not new.any():
            break
        pos[new] = acc[new] / wk[new, None]
    return pos


def frontal_groups(cx: Connectome, atlas: Atlas, types, radius_deg: float = 18.0,
                   eyes: dict | None = None, pos: np.ndarray | None = None,
                   inner_deg: float = 0.0, prefix: str = "F") -> dict[str, np.ndarray]:
    """세포유형별로 '지도 중심 radius_deg 안을 보는' 뉴런만 골라 그룹을 만든다.

    작은 물체의 루밍은 시야 전체 평균에서 묻힌다(선인장 각반너비 4° 면 컬럼의 약 2%).
    충돌 경로에 해당하는 중심 수용장만 읽으면 신호가 살아난다. 실제 초파리의 도피
    반응도 정면 루밍에 가장 강하다.

    inner_deg 를 주면 고리(annulus) 가 된다 — 중심 대비 주변을 재서 전역 밝기 변화를
    상쇄하는 중심-주변(center-surround) 비교를 만들 수 있다.
    """
    eyes = eyes or load_eye_maps(cx, atlas)
    out: dict[str, list] = {}
    for side, eye in eyes.items():
        p = retinotopic_positions(cx, atlas, eye) if pos is None else pos
        for t in types:
            idx = atlas.by_type(t, side)
            if not len(idx):
                continue
            q = p[idx]
            d = np.hypot(q[:, 0], q[:, 1])
            keep = np.isfinite(q).all(axis=1) & (d <= radius_deg) & (d >= inner_deg)
            if keep.any():
                out.setdefault(f"{prefix}_{t}", []).append(idx[keep])
    return {k: np.concatenate(v) for k, v in out.items()}


def _cache_path(flip_az: bool, flip_el: bool):
    return PATHS["cache"] / f"retina_783_az{int(flip_az)}_el{int(flip_el)}.npz"


def load_eye_maps(cx: Connectome | None = None, atlas: Atlas | None = None, *,
                  flip_az: bool = True, flip_el: bool = False, force: bool = False) -> dict[str, EyeMap]:
    """양쪽 눈의 망막지도. 결과는 data/cache 에 저장해 두 번째부터는 즉시 로드된다."""
    path = _cache_path(flip_az, flip_el)
    if path.exists() and not force:
        z = np.load(path)
        return {s: EyeMap(side=s, col_idx=z[f"{s}_col_idx"], col_az=z[f"{s}_col_az"], col_el=z[f"{s}_col_el"],
                          pr_idx=z[f"{s}_pr_idx"], pr_col=z[f"{s}_pr_col"],
                          sphere_r=float(z[f"{s}_sphere_r"]), resid_um=float(z[f"{s}_resid"]),
                          n_pr_total=int(z[f"{s}_n_pr_total"])) for s in SIDES}
    if cx is None:
        from .data import load_connectome
        cx = load_connectome()
    atlas = atlas or Atlas(cx)
    maps = {s: build_eye_map(cx, atlas, s, flip_az=flip_az, flip_el=flip_el) for s in SIDES}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"{s}_{k}": v for s, e in maps.items() for k, v in
                      (("col_idx", e.col_idx), ("col_az", e.col_az), ("col_el", e.col_el),
                       ("pr_idx", e.pr_idx), ("pr_col", e.pr_col),
                       ("sphere_r", np.array(e.sphere_r)), ("resid", np.array(e.resid_um)),
                       ("n_pr_total", np.array(e.n_pr_total)))})
    return maps


# ---------------------------------------------------------------------------
# 화면(휘도) → 광수용체 발화율
# ---------------------------------------------------------------------------
@dataclass
class RetinaEncoder:
    """컬럼별 휘도(0~1) → R1-6 Poisson 자극 빈도(Hz).

    R1-6 은 빛에 탈분극한다(밝을수록 많이 발화). 이 모델에서 히스타민은 억제이므로
    밝음 → 라미나 억제, 어둠 → 라미나 탈억제가 되어 부호가 생물학과 맞는다.
    그래서 어두운 물체(선인장)는 OFF 경로 L2/L3 → Tm1/Tm2 → T5 → LPLC2 를 구동한다.

    r_base  : 기준 휘도(lum_ref)에서의 발화율 (Hz)
    r_gain  : 휘도 1 차이당 발화율 변화 (Hz)
    lum_ref : 기준 휘도. 순응(tau_adapt_s>0)을 켜면 컬럼별 이동평균이 이 값을 대신한다.
    tau_adapt_s : 휘도 순응 시정수(초). 0 이면 순응 없음(정적 대비도 계속 신호가 된다).
    """
    eyes: dict[str, EyeMap]
    r_base: float = 45.0
    r_gain: float = 90.0
    r_max: float = 200.0
    lum_ref: float = 0.5
    tau_adapt_s: float = 0.0
    _adapt: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.reset()
        # 자극 인덱스는 고정이므로 한 번만 만들어 둔다 (프레임마다 재사용)
        self._idx = np.concatenate([self.eyes[s].pr_idx for s in SIDES if s in self.eyes])
        self._split = np.cumsum([self.eyes[s].n_pr for s in SIDES if s in self.eyes])[:-1]

    def reset(self):
        self._adapt = {s: np.full(e.n_col, self.lum_ref) for s, e in self.eyes.items()}

    def encode(self, lum: dict[str, np.ndarray], dt_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """lum[side] = (C,) 컬럼별 휘도 → (뉴런 인덱스, Hz)."""
        rates = []
        for s in SIDES:
            if s not in self.eyes:
                continue
            e, v = self.eyes[s], np.asarray(lum[s], float)
            ref = self.lum_ref
            if self.tau_adapt_s > 0 and dt_s > 0:
                a = self._adapt[s]
                k = 1.0 - np.exp(-dt_s / self.tau_adapt_s)
                a += k * (v - a)
                ref = a
            r = np.clip(self.r_base + self.r_gain * (v - ref), 0.0, self.r_max)
            rates.append(r[e.pr_col])
        return self._idx, np.concatenate(rates)

    def column_rates(self, lum: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """기록/그림용: 컬럼별 발화율(순응 상태를 바꾸지 않는다)."""
        out = {}
        for s, e in self.eyes.items():
            ref = self._adapt[s] if self.tau_adapt_s > 0 else self.lum_ref
            out[s] = np.clip(self.r_base + self.r_gain * (np.asarray(lum[s], float) - ref), 0.0, self.r_max)
        return out


def lamina_bias_neurons(atlas: Atlas) -> np.ndarray:
    """탄력적 탈분극(tonic bias)을 줄 라미나 단극세포 L1/L2/L3.

    이유: 광수용체가 히스타민성(억제)이라 '빛 → 억제'만 있고 '어둠 → 흥분'이 없다.
    실제 라미나 단극세포는 어둠에서 탈분극해 있고(Laughlin 1981) 빛이 그것을 누른다.
    이 모델에는 그 휴지 탈분극이 없으므로 LIFNetwork.set_bias 로 넣어 준다.
    이 bias 가 없으면 시각 경로 전체가 침묵한다(빛을 꺼도 아무 일도 일어나지 않는다).
    """
    return np.concatenate([atlas.by_type(t) for t in LAMINA_TYPES])
