"""뉴런 검색 + 기능별 그룹 정의 (감각 채널, 하행 뉴런, 운동 뉴런).

그룹 이름은 SensoryEncoder / MotorDecoder 에서 사용된다. 각 그룹은 좌/우('left','right')로
나눠 조회할 수 있다 (FlyWire 'side' 주석 기준).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Connectome


# ---------------------------------------------------------------------------
# 감각 채널 정의: 이름 -> 선택 규칙 (열 == 값). 여러 규칙은 AND.
# ---------------------------------------------------------------------------
SENSORY_GROUPS: dict[str, dict] = {
    # 후각: ORN_<사구체> 는 접두어 매칭 (53 glomeruli)
    "orn": {"cell_class": "olfactory"},
    # 미각
    "sugar": {"cell_class": "gustatory", "cell_sub_class": "sugar/water"},
    "bitter": {"cell_class": "gustatory", "cell_sub_class": "bitter"},
    "salt_low": {"cell_class": "gustatory", "cell_sub_class": "low-salt"},
    "taste_peg": {"cell_class": "gustatory", "cell_sub_class": "taste peg"},
    # 시각
    "photoreceptor": {"cell_type": "R1-6"},
    "photoreceptor_r7": {"cell_type": "R7"},
    "photoreceptor_r8": {"cell_type": "R8"},
    "ocellar": {"cell_sub_class": "ocellar", "cell_class": "visual"},
    # 온도 / 습도
    "heat": {"cell_type": "TRN_VP2"},
    "cold": {"cell_type_prefix": "TRN_VP3"},
    "dry": {"cell_type": "HRN_VP4"},
    "moist": {"cell_type": "HRN_VP5"},
    # 기계감각
    "wind": {"cell_class": "mechanosensory", "cell_sub_class": "wind_gravity"},
    "sound": {"cell_class": "mechanosensory", "cell_sub_class": "auditory"},
    "touch_head": {"cell_class": "mechanosensory", "cell_sub_class": "head bristle"},
    "touch_eye": {"cell_class": "mechanosensory", "cell_sub_class": "eye bristle"},
    "groom_jo": {"cell_class": "mechanosensory", "cell_sub_class": "grooming"},
}

# 냄새 → 사구체 활성 프로파일 (상대 강도 0~1). 문헌 기반 근사치이며 실험에서 자유롭게 바꿀 수 있다.
ODORANTS: dict[str, dict[str, float]] = {
    # 사과식초(ACV): Semmelhack & Wang 2009 — DM1/VA2 유인, 고농도에서 DM5 회피
    "vinegar": {"DM1": 1.0, "VA2": 0.9, "DM4": 0.7, "DM2": 0.6, "VM2": 0.5, "DM5": 0.3},
    "vinegar_strong": {"DM1": 1.0, "VA2": 1.0, "DM4": 0.9, "DM2": 0.8, "VM2": 0.8, "DM5": 1.0},
    "yeast": {"DM1": 0.8, "VM2": 0.8, "VA2": 0.7, "DM4": 0.5, "VM7d": 0.5},
    "co2": {"V": 1.0},                       # Gr21a/Gr63a → V 사구체, 회피
    "geosmin": {"DA2": 1.0},                 # Or56a, 강한 회피
    "cva": {"DA1": 1.0, "DL3": 0.4},        # 페로몬 cVA (Or67d) → DA1
    "ethyl_acetate": {"DM2": 0.9, "DM4": 0.6, "DM5": 0.6, "VM5d": 0.5},
    "benzaldehyde": {"DL5": 1.0, "DM5": 0.5},
    "banana": {"DM2": 1.0, "DM4": 0.8, "VA2": 0.5, "DL1": 0.4},
}

# 하행/운동 뉴런 그룹: 이름 -> cell_type 목록
MOTOR_GROUPS: dict[str, list[str]] = {
    "walk_forward": ["DNp09", "DNa01"],      # DNp09: 전진 보행/구애 추적 (Bidaye 2020), DNa01: 보행 조향
    "turn": ["DNa02", "DNa01", "DNa03"],     # 동측(ipsilateral) 회전 (Rayshubskiy 2020; Braun 2024)
    "walk_backward": ["MDN"],                # moonwalker DN (Bidaye 2014)
    "escape": ["DNp01", "DNp11"],            # giant fiber 도약 탈출
    "takeoff": ["DNp07", "DNp10"],           # 이륙
    "feed": ["CB0701"],                      # MN9 (rostrum protractor, 주둥이 신전) — Shiu 2024 검증 뉴런
}
MOTOR_SUBCLASS_GROUPS: dict[str, str] = {
    "feed_ingestion": "ingestion_motor_neuron",
    "feed_proboscis": "proboscis_motor_neuron",
    "neck": "neck_motor_neuron",
    "antenna_motor": "antennal_motor_neuron",
}


@dataclass
class Atlas:
    """커넥톰 뉴런 표 위에서 이름/유형으로 뉴런 인덱스를 찾는다."""
    cx: Connectome
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def neurons(self) -> pd.DataFrame:
        return self.cx.neurons

    # -- 기본 검색 -----------------------------------------------------------
    def select(self, side: str | None = None, **rules) -> np.ndarray:
        """rules 예: cell_type='DNa02', cell_class='olfactory', cell_type_prefix='ORN_DM1'"""
        df = self.neurons
        m = np.ones(len(df), dtype=bool)
        for col, val in rules.items():
            if col.endswith("_prefix"):
                m &= df[col[:-7]].fillna("").str.startswith(val).to_numpy(dtype=bool)
            elif col.endswith("_regex"):
                m &= df[col[:-6]].fillna("").str.contains(val, regex=True).to_numpy(dtype=bool)
            elif isinstance(val, (list, tuple, set)):
                m &= df[col].isin(list(val)).fillna(False).to_numpy(dtype=bool)
            else:
                m &= (df[col] == val).fillna(False).to_numpy(dtype=bool)
        if side is not None:
            m &= (df["side"] == side).fillna(False).to_numpy(dtype=bool)
        return np.flatnonzero(m)

    def by_type(self, cell_types, side: str | None = None) -> np.ndarray:
        if isinstance(cell_types, str):
            cell_types = [cell_types]
        df = self.neurons
        m = df.cell_type.isin(cell_types).fillna(False).to_numpy(dtype=bool) | df.hemibrain_type.isin(cell_types).fillna(False).to_numpy(dtype=bool)
        if side is not None:
            m &= (df.side == side).fillna(False).to_numpy(dtype=bool)
        return np.flatnonzero(m)

    def by_root_id(self, root_ids) -> np.ndarray:
        return self.cx.index_of(root_ids)

    def search(self, pattern: str, limit: int = 50) -> pd.DataFrame:
        """cell_type / hemibrain_type / cell_class / synonyms 에서 정규식 검색."""
        df = self.neurons
        rx = re.compile(pattern, re.I)
        cols = ["cell_type", "hemibrain_type", "cell_class", "cell_sub_class", "synonyms"]
        m = np.zeros(len(df), dtype=bool)
        for c in cols:
            m |= df[c].fillna("").map(lambda s: bool(rx.search(s))).to_numpy()
        out = df.loc[m, ["idx", "root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "side", "top_nt"]]
        return out.head(limit)

    def describe(self, idx) -> pd.DataFrame:
        cols = ["idx", "root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "side", "top_nt"]
        return self.neurons.iloc[np.asarray(idx)][cols]

    # -- 기능 그룹 -----------------------------------------------------------
    def sensory(self, name: str, side: str | None = None) -> np.ndarray:
        key = ("sensory", name, side)
        if key not in self._cache:
            if name not in SENSORY_GROUPS:
                raise KeyError(f"unknown sensory group {name!r}; available: {list(SENSORY_GROUPS)}")
            self._cache[key] = self.select(side=side, **SENSORY_GROUPS[name])
        return self._cache[key]

    def glomerulus(self, glom: str, side: str | None = None) -> np.ndarray:
        """ORN_<glom> 뉴런."""
        return self.select(side=side, cell_type=f"ORN_{glom}")

    def glomeruli(self) -> list[str]:
        ct = self.neurons.cell_type.dropna()
        return sorted({c[4:] for c in ct if c.startswith("ORN_")})

    def motor(self, name: str, side: str | None = None) -> np.ndarray:
        key = ("motor", name, side)
        if key not in self._cache:
            if name in MOTOR_SUBCLASS_GROUPS:
                idx = self.select(side=side, cell_sub_class=MOTOR_SUBCLASS_GROUPS[name])
            elif name in MOTOR_GROUPS:
                idx = self.by_type(MOTOR_GROUPS[name], side=side)
            else:
                raise KeyError(f"unknown motor group {name!r}; available: {list(MOTOR_GROUPS) + list(MOTOR_SUBCLASS_GROUPS)}")
            self._cache[key] = idx
        return self._cache[key]

    def group_table(self) -> pd.DataFrame:
        rows = []
        for g in SENSORY_GROUPS:
            rows.append(("sensory", g, len(self.sensory(g, "left")), len(self.sensory(g, "right")), len(self.sensory(g))))
        for g in list(MOTOR_GROUPS) + list(MOTOR_SUBCLASS_GROUPS):
            rows.append(("motor", g, len(self.motor(g, "left")), len(self.motor(g, "right")), len(self.motor(g))))
        return pd.DataFrame(rows, columns=["kind", "group", "n_left", "n_right", "n_total"])
