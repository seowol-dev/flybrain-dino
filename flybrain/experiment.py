"""실험 러너.

ActivationExperiment : Shiu et al. 식 개방회로(open-loop) 실험 — 뉴런 집합을 활성/억제하고 전뇌 반응 측정.
ClosedLoopExperiment : 가상 세계 폐회로(closed-loop) — 감각 → 전뇌 LIF → 하행 뉴런 → 몸 움직임 → 새 감각.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .atlas import Atlas
from .data import Connectome, load_connectome, weights
from .interface import MotorDecoder, SensoryEncoder
from .lif import LIFNetwork, LIFParams
from .world import World


# ---------------------------------------------------------------------------
# 개방회로 활성화 실험
# ---------------------------------------------------------------------------
@dataclass
class ActivationExperiment:
    cx: Connectome
    params: LIFParams = field(default_factory=LIFParams)
    preset: str = "shiu2024"

    def __post_init__(self):
        self.W = weights(self.cx, self.preset)

    def run(self, excite, silence=(), rate: float | None = None, duration_ms: float = 1000.0,
            n_trials: int = 5, seed: int = 0, progress: bool = True) -> pd.DataFrame:
        """excite 뉴런을 rate Hz Poisson 으로 자극, silence 뉴런은 출력 차단.
        → 뉴런별 평균 발화율(Hz)과 표준편차 표 (발화한 뉴런만)."""
        rate = self.params.r_poi if rate is None else rate
        all_rates = []
        for k in range(n_trials):
            net = LIFNetwork(self.W, self.params, seed=seed + k)
            if len(silence):
                net.silence(silence)
            net.stimulate(excite, rate)
            t0 = time.time()
            net.run(duration_ms, progress=progress)
            if progress:
                print(f"  trial {k + 1}/{n_trials}: {time.time() - t0:.1f}s, spikes={len(net.spikes()[0]):,}")
            all_rates.append(net.rates(0, duration_ms))
        R = np.vstack(all_rates)
        df = self.cx.neurons[["idx", "root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "side", "top_nt"]].copy()
        df["rate_hz"] = R.mean(0)
        df["rate_std"] = R.std(0)
        df["stimulated"] = False
        df.loc[np.asarray(excite, int), "stimulated"] = True
        return df[df.rate_hz > 0].sort_values("rate_hz", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 폐회로 가상 세계 실험
# ---------------------------------------------------------------------------
@dataclass
class ClosedLoopExperiment:
    world: World
    cx: Connectome | None = None
    params: LIFParams = field(default_factory=LIFParams)
    preset: str = "curated"          # 'curated' (폐회로 권장) | 'shiu2024' (원 모델)
    control_dt_ms: float = 50.0      # 뇌→몸 업데이트 주기 (ms)
    seed: int = 0
    encoder_kwargs: dict = field(default_factory=dict)
    decoder_kwargs: dict = field(default_factory=dict)
    silence: tuple = ()               # 억제할 뉴런 인덱스 (유전적 억제 실험)
    excite: dict = field(default_factory=dict)  # {neuron_idx: Hz} 지속 광유전학 활성
    record_neurons: dict = field(default_factory=dict)  # {이름: 인덱스 배열} 추가 기록 그룹

    def __post_init__(self):
        self.cx = self.cx or load_connectome()
        self.atlas = Atlas(self.cx)
        self.net = LIFNetwork(weights(self.cx, self.preset), self.params, seed=self.seed)
        if len(self.silence):
            self.net.silence(np.asarray(self.silence))
        self.encoder = SensoryEncoder(self.atlas, seed=self.seed, **self.encoder_kwargs)
        self.decoder = MotorDecoder(self.atlas, seed=self.seed, log_groups=dict(self.record_neurons), **self.decoder_kwargs)
        self.log: list[dict] = []

    def run(self, duration_s: float, progress: bool = True, keep_spikes: bool = False) -> pd.DataFrame:
        n_ctrl = int(round(duration_s * 1000 / self.control_dt_ms))
        dt_s = self.control_dt_ms / 1000
        it = range(n_ctrl)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc=f"closed-loop {self.world.name}", unit="ctrl")
        steps_per = int(round(self.control_dt_ms / self.params.dt))
        for _ in it:
            percept = self.world.percept()
            idx, rate, summ = self.encoder.encode(percept)
            if self.excite:
                ex_i = np.fromiter(self.excite.keys(), np.int64)
                ex_r = np.fromiter(self.excite.values(), float)
                idx = np.concatenate([idx, ex_i])
                rate = np.concatenate([rate, ex_r])
            self.net.set_stimulus((idx, rate))
            t0 = self.net.t_ms
            for _ in range(steps_per):
                self.net.step()
            act = self.decoder.decode(self.net, t0, self.net.t_ms, dt_s)
            if not keep_spikes:
                self.net.record.clear()
                # rates() 가 방금 구간을 쓰므로 decode 이후에 지운다
            f = self.world.fly
            row = {
                "t": self.world.t, "x": f.x, "y": f.y, "heading": f.heading,
                "speed": act.speed, "omega": act.omega, "proboscis": act.proboscis, "jump": act.jump,
                "energy": f.energy, "n_stim": len(idx),
            }
            for k, v in act.rates.items():
                row["r_" + k] = v
            for ch, (l, r) in summ.items():
                if not ch.startswith("odor:"):
                    row[f"in_{ch}_L"], row[f"in_{ch}_R"] = l, r
            row["in_odor_total"] = sum(l + r for ch, (l, r) in summ.items() if ch.startswith("odor:"))
            self.log.append(row)
            self.world.step(dt_s, act.speed, act.omega, jump=act.jump, proboscis=act.proboscis)
        return self.trajectory()

    def trajectory(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

    def save(self, out_dir: str | Path, tag: str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df = self.trajectory()
        p = out / f"{tag}_trajectory.csv"
        df.to_csv(p, index=False)
        (out / f"{tag}_world.txt").write_text(self.world.describe())
        meta = {"control_dt_ms": self.control_dt_ms, "seed": self.seed, "preset": self.preset,
                "params": self.params.__dict__, "n_silenced": len(self.silence), "n_excited": len(self.excite)}
        (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return p
