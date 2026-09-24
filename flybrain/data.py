"""커넥톰 로딩/전처리.

원본:
  * Connectivity_783.parquet / Completeness_783.csv  — Shiu et al. 2024 (philshiu/Drosophila_brain_model)
    FlyWire v783 전체 138,639 뉴런, 15.09M 연결(시냅스 5,449만 개). 'Excitatory' 열은
    신경전달물질 예측(ACh/DA/5HT/OA = +1, GABA/Glu = -1)으로 결정된 부호.
  * neuron_annotations.tsv — Schlegel et al. 2024 세포 유형 주석 (super_class, cell_class, cell_type, side ...)

전처리 결과는 data/cache 에 npz + parquet 로 저장해 두 번째부터는 수 초 안에 로드된다.
"""
from __future__ import annotations

import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import PATHS, DOWNLOADS

ANNOTATION_COLS = [
    "root_id", "flow", "super_class", "cell_class", "cell_sub_class", "cell_type",
    "hemibrain_type", "top_nt", "top_nt_conf", "known_nt", "known_nt_source", "side", "nerve",
    "soma_x", "soma_y", "soma_z", "synonyms",
]

FAST_NT = ("acetylcholine", "gaba", "glutamate", "histamine", "dopamine", "serotonin", "octopamine")
MONOAMINES = ("dopamine", "serotonin", "octopamine")
PRESETS = ("shiu2024", "curated")


def _parse_known_nt(s) -> str | None:
    """'acetylcholine; sNPF' → 'acetylcholine'. 여러 전달물질이거나 '-negative' 만 있으면 None."""
    if not isinstance(s, str):
        return None
    toks = {t.strip() for t in re.split(r"[;,]", s)}
    pos = {t for t in toks if t in FAST_NT}
    return pos.pop() if len(pos) == 1 else None


def curate_nt(neurons: pd.DataFrame, min_type_size: int = 3) -> tuple[pd.Series, pd.Series]:
    """뉴런별 신경전달물질 보정.

    1) 문헌 기반 known_nt (Schlegel et al. 2024 주석) 가 단일 전달물질이면 사용
    2) 아니면 같은 cell_type (≥ min_type_size 개) 의 신뢰도 가중 다수결
    3) 아니면 FlyWire 예측(top_nt, Eckstein et al. 2024)
    반환: (nt, source) — source ∈ {'known','type','pred'}
    """
    known = neurons["known_nt"].map(_parse_known_nt)
    df = pd.DataFrame({
        "ct": neurons["cell_type"].astype(object),
        "nt": neurons["top_nt"].astype(object),
        "c": neurons["top_nt_conf"].astype(float).fillna(0.3),
    })
    ok = df.dropna(subset=["ct", "nt"])
    cons = (ok.groupby(["ct", "nt"]).c.sum().reset_index()
              .sort_values("c").drop_duplicates("ct", keep="last").set_index("ct").nt)
    size = ok.groupby("ct").size()
    big = set(size[size >= min_type_size].index)
    type_nt = df.ct.map(lambda c: cons.get(c) if c in big else None)
    out = np.array(known.astype(object).to_numpy(), dtype=object, copy=True)
    src = np.where(pd.notna(out), "known", "").astype(object)
    m = pd.isna(out) & pd.notna(type_nt.to_numpy())
    out[m] = type_nt.to_numpy()[m]
    src[m] = "type"
    m = pd.isna(out)
    out[m] = df.nt.to_numpy()[m]
    src[m] = "pred"
    return pd.Series(out, index=neurons.index, dtype=object), pd.Series(src, index=neurons.index, dtype=object)


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[flybrain] downloading {url}\n           -> {dst}", file=sys.stderr)

    def hook(n, bs, total):
        if total > 0 and n % 200 == 0:
            print(f"\r           {n * bs / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)

    urllib.request.urlretrieve(url, dst, reporthook=hook)
    print("", file=sys.stderr)


def ensure_raw_files(download: bool = True) -> None:
    """원본 파일이 없으면 내려받는다."""
    for key in ("connectivity", "completeness", "annotations"):
        p = PATHS[key]
        if not p.exists():
            if not download:
                raise FileNotFoundError(p)
            _download(DOWNLOADS[key], p)


@dataclass
class Connectome:
    """전처리된 커넥톰.

    Attributes
    ----------
    neurons : pd.DataFrame
        모델 인덱스 순서(0..N-1)의 뉴런 표. root_id 와 주석 열을 가진다.
    W_pre : scipy.sparse.csr_matrix, shape (N, N)
        행 = 시냅스전(pre) 뉴런, 열 = 시냅스후(post) 뉴런. 값 = 부호 × 시냅스 수 (정수).
        시뮬레이터에서 mV 단위 가중치는 w_syn × 값.
    """
    neurons: pd.DataFrame
    W_pre: sp.csr_matrix

    @property
    def n(self) -> int:
        return len(self.neurons)

    @property
    def n_synapses(self) -> int:
        return int(np.abs(self.W_pre.data).sum())

    @property
    def n_connections(self) -> int:
        return int(self.W_pre.nnz)

    def index_of(self, root_ids) -> np.ndarray:
        """FlyWire root_id → 모델 인덱스."""
        lut = self._lut()
        idx = lut.reindex(np.asarray(root_ids, dtype=np.int64))
        if idx.isna().any():
            missing = idx[idx.isna()].index.tolist()[:5]
            raise KeyError(f"모델에 없는 root_id: {missing} ...")
        return idx.to_numpy().astype(np.int64)

    def _lut(self) -> pd.Series:
        if not hasattr(self, "_lut_cache"):
            self._lut_cache = pd.Series(np.arange(self.n), index=self.neurons.root_id.to_numpy())
        return self._lut_cache

    def summary(self) -> str:
        sc = self.neurons.super_class.value_counts(dropna=False)
        lines = [
            f"FlyWire v783 connectome: {self.n:,} neurons, {self.n_connections:,} connections, "
            f"{self.n_synapses:,} synapses",
            "super_class: " + ", ".join(f"{k}={v:,}" for k, v in sc.items()),
        ]
        return "\n".join(lines)


def build_cache(min_synapses: int = 1, force: bool = False, download: bool = True) -> Connectome:
    """원본 → 캐시(npz/parquet). min_synapses 로 약한 연결을 걸러낼 수 있다(Shiu 모델은 1 = 전부 사용)."""
    cache_npz = PATHS["cache_npz"]
    cache_neu = PATHS["cache_neurons"]
    if cache_npz.exists() and cache_neu.exists() and not force:
        return load_connectome(min_synapses=min_synapses)

    ensure_raw_files(download=download)
    print("[flybrain] building cache (첫 실행에서만, 1~2분) ...", file=sys.stderr)
    comp = pd.read_csv(PATHS["completeness"], index_col=0)
    root_ids = comp.index.to_numpy().astype(np.int64)
    n = len(root_ids)

    ann = pd.read_csv(PATHS["annotations"], sep="\t", low_memory=False, usecols=ANNOTATION_COLS)
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    neurons = ann.reindex(root_ids).reset_index().rename(columns={"index": "root_id"})
    neurons["root_id"] = root_ids
    neurons.insert(0, "idx", np.arange(n))
    nt, src = curate_nt(neurons)
    neurons["nt_curated"] = nt.to_numpy()
    neurons["nt_source"] = src.to_numpy()
    for c in ("super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "top_nt", "known_nt",
              "known_nt_source", "side", "flow", "nerve", "synonyms", "nt_curated", "nt_source"):
        neurons[c] = neurons[c].astype("string")

    con = pd.read_parquet(
        PATHS["connectivity"],
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = con["Presynaptic_Index"].to_numpy(np.int32)
    post = con["Postsynaptic_Index"].to_numpy(np.int32)
    w = con["Excitatory x Connectivity"].to_numpy(np.int32)
    W = sp.coo_matrix((w, (pre, post)), shape=(n, n), dtype=np.int32).tocsr()
    W.sum_duplicates()

    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_npz, data=W.data, indices=W.indices, indptr=W.indptr, shape=np.array(W.shape))
    neurons.to_parquet(cache_neu, index=False)
    print(f"[flybrain] cache written: {cache_npz}", file=sys.stderr)
    return load_connectome(min_synapses=min_synapses)


def load_connectome(min_synapses: int = 1) -> Connectome:
    cache_npz = PATHS["cache_npz"]
    cache_neu = PATHS["cache_neurons"]
    if not (cache_npz.exists() and cache_neu.exists()):
        return build_cache(min_synapses=min_synapses)
    z = np.load(cache_npz)
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    if min_synapses > 1:
        W = W.multiply(np.abs(W) >= min_synapses).tocsr()
        W.eliminate_zeros()
    neurons = pd.read_parquet(cache_neu)
    return Connectome(neurons=neurons, W_pre=W)


# ---------------------------------------------------------------------------
# 가중치 프리셋
# ---------------------------------------------------------------------------
def weights(cx: Connectome, preset: str = "shiu2024", *, monoamine_sign: float | None = None,
            eln_scale: float | None = None) -> sp.csr_matrix:
    """시뮬레이터에 넣을 부호 있는 시냅스 수 행렬 (pre × post).

    preset='shiu2024'
        Shiu et al. 2024 원 모델 그대로. 부호 = 'Excitatory' 열 (ACh/DA/5HT/OA = +, GABA/Glu = −).
        Brian2 원본 대비 검증됨 (README 참고).
    preset='curated'
        폐회로 가상 세계용 보정 모델. 원 모델은 더듬이엽(AL)에서 자기유지 폭주(수천 뉴런이
        자극 종료 후에도 ~280 Hz 로 발화)가 일어나 후각/온도 입력이 모두 같은 패턴이 된다.
        보정 내용:
          1. 신경전달물질: 문헌 known_nt → cell_type 다수결 → 예측값 순 (curate_nt)
          2. 모노아민(DA/5HT/OA)은 빠른 시냅스 효과 0 (monoamine_sign=0; 느린 조절성 전달)
          3. 흥분성 국소뉴런 lLN1 (Shang 2007) 의 화학 시냅스 출력 × eln_scale(기본 0).
             이들은 주로 전기 시냅스(gap junction)로 작동하며(Yaksi & Wilson 2010),
             이 모델에는 전기 시냅스가 없다.
    """
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}")
    W = cx.W_pre
    if preset == "shiu2024":
        if monoamine_sign is None and eln_scale is None:
            return W
        monoamine_sign = 1.0 if monoamine_sign is None else monoamine_sign
        eln_scale = 1.0 if eln_scale is None else eln_scale
        nt = cx.neurons["top_nt"].astype(object).where(cx.neurons["top_nt"].notna(), None)
    else:
        monoamine_sign = 0.0 if monoamine_sign is None else monoamine_sign
        eln_scale = 0.0 if eln_scale is None else eln_scale
        nt = cx.neurons["nt_curated"].astype(object).where(cx.neurons["nt_curated"].notna(), None)
    sign_map = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
    sign_map.update({m: monoamine_sign for m in MONOAMINES})
    s = nt.map(sign_map).astype(float).fillna(1.0).to_numpy()
    if preset == "shiu2024":
        # 원 부호 유지, 모노아민만 교체
        base = np.sign(W.data).astype(np.float32)
        pre = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
        is_mono = nt.isin(MONOAMINES).to_numpy(dtype=bool)[pre]
        base[is_mono] = monoamine_sign
        data = base * np.abs(W.data)
    else:
        pre = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
        data = (s[pre] * np.abs(W.data)).astype(np.float32)
    eln = cx.neurons["cell_type"].isin(["lLN1_bc", "lLN1_a"]).fillna(False).to_numpy(dtype=bool)
    if eln_scale != 1.0:
        data = np.where(eln[pre], data * eln_scale, data)
    out = sp.csr_matrix((data.astype(np.float32), W.indices.copy(), W.indptr.copy()), shape=W.shape)
    out.eliminate_zeros()
    return out
