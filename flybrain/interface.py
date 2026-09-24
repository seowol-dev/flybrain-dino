"""뇌 ↔ 세계 인터페이스.

SensoryEncoder : World.percept() → 감각 뉴런 Poisson 자극 빈도
MotorDecoder   : 하행/운동 뉴런 발화율 → 보행 속도·회전·주둥이 신전·도약
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atlas import Atlas

SIDES = ("left", "right")


@dataclass
class SensoryEncoder:
    """감각 채널 세기(0~1) → 뉴런별 자극 빈도(Hz).

    max_rate : 세기 1 일 때의 빈도 (Shiu 모델 기본 자극 150 Hz 와 동일)
    channel_scale : 채널별 배율 (예: 광수용체 8천 개는 낮게)
    subsample : 채널별 사용할 뉴런 비율 (큰 집단의 계산량 절감)
    """
    atlas: Atlas
    max_rate: float = 150.0
    channel_scale: dict[str, float] = field(default_factory=lambda: {
        "light": 0.25, "photoreceptor": 0.25, "sound": 0.6, "wind": 0.6,
        "touch_head": 0.6, "touch_eye": 0.6, "looming": 1.0,
    })
    subsample: dict[str, float] = field(default_factory=lambda: {"light": 0.3, "looming": 0.3, "touch_eye": 0.3})
    seed: int = 0
    # 채널 이름 → atlas 감각 그룹 이름
    channel_group: dict[str, str] = field(default_factory=lambda: {
        "light": "photoreceptor", "looming": "photoreceptor", "sugar": "sugar", "bitter": "bitter",
        "salt_low": "salt_low", "water": "moist", "heat": "heat", "cold": "cold", "dry": "dry",
        "moist": "moist", "wind": "wind", "sound": "sound", "touch_head": "touch_head",
        "touch_eye": "touch_eye", "groom": "groom_jo",
    })
    _idx: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        for ch, grp in self.channel_group.items():
            for side in SIDES:
                idx = self.atlas.sensory(grp, side)
                frac = self.subsample.get(ch, 1.0)
                if frac < 1.0 and len(idx):
                    idx = np.sort(rng.choice(idx, size=max(1, int(len(idx) * frac)), replace=False))
                self._idx[(ch, side)] = idx
        for g in self.atlas.glomeruli():
            for side in SIDES:
                self._idx[("odor:" + g, side)] = self.atlas.glomerulus(g, side)

    def neurons(self, channel: str, side: str) -> np.ndarray:
        return self._idx.get((channel, side), np.zeros(0, np.int64))

    def encode(self, percept: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        """→ (neuron_idx, rate_Hz, summary{channel: (L,R) rate})"""
        idxs, rates, summary = [], [], {}
        for ch, val in percept.items():
            if ch == "odor":
                for g, (l, r) in val.items():
                    for side, v in zip(SIDES, (l, r)):
                        idx = self.neurons("odor:" + g, side)
                        if len(idx) and v > 0:
                            idxs.append(idx)
                            rates.append(np.full(len(idx), v * self.max_rate))
                    summary["odor:" + g] = (l * self.max_rate, r * self.max_rate)
                continue
            if ch not in self.channel_group:
                continue
            l, r = val
            sc = self.channel_scale.get(ch, 1.0)
            for side, v in zip(SIDES, (l, r)):
                idx = self.neurons(ch, side)
                if len(idx) and v > 0:
                    idxs.append(idx)
                    rates.append(np.full(len(idx), v * sc * self.max_rate))
            summary[ch] = (l * sc * self.max_rate, r * sc * self.max_rate)
        if not idxs:
            return np.zeros(0, np.int64), np.zeros(0), summary
        idx = np.concatenate(idxs)
        rate = np.concatenate(rates)
        # 같은 뉴런이 여러 채널에서 자극되면 최대값
        order = np.argsort(idx, kind="stable")
        idx, rate = idx[order], rate[order]
        uniq, start = np.unique(idx, return_index=True)
        rate_max = np.maximum.reduceat(rate, start)
        return uniq, rate_max, summary


@dataclass
class Action:
    speed: float = 0.0          # mm/s
    omega: float = 0.0          # rad/s
    proboscis: bool = False
    jump: bool = False
    rates: dict = field(default_factory=dict)


@dataclass
class MotorDecoder:
    """하행 뉴런 발화율 → 운동 명령.

    gain_forward  : mm/s per Hz (walk_forward 그룹 평균 발화율)
    gain_backward : mm/s per Hz (MDN)
    gain_turn     : rad/s per Hz (turn 그룹 좌-우 발화율 차; 동측 회전)
    feed_threshold: Hz, MN9 발화율이 넘으면 주둥이 신전 (섭식 운동뉴런은 기록만)
    escape_threshold : Hz, giant fiber 발화율이 넘으면 도약
    spontaneous_speed / turn_noise : 뇌 입력이 없을 때의 자발 보행 (0 으로 끄면 순수 커넥톰 구동)
    """
    atlas: Atlas
    gain_forward: float = 0.25
    gain_backward: float = 0.25
    gain_turn: float = 0.06
    max_speed: float = 25.0
    max_omega: float = 6.0
    feed_threshold: float = 10.0
    escape_threshold: float = 8.0
    spontaneous_speed: float = 5.0
    feeding_stops_walking: bool = True
    jump_cooldown_s: float = 1.0      # 몸 규칙: 도약 후 다시 도약하기까지 최소 간격
    turn_noise: float = 0.8
    turn_persistence: float = 0.9
    seed: int = 0
    log_groups: dict[str, np.ndarray] = field(default_factory=dict)
    _rng: np.random.Generator = field(default=None, repr=False)
    _noise: float = 0.0
    _since_jump: float = 1e9

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        a = self.atlas
        self.g = {
            "fwd_L": a.motor("walk_forward", "left"), "fwd_R": a.motor("walk_forward", "right"),
            "turn_L": a.motor("turn", "left"), "turn_R": a.motor("turn", "right"),
            "back": a.motor("walk_backward"),
            "escape": a.motor("escape"),
            "takeoff": a.motor("takeoff"),
            "feed": a.motor("feed"),
            "feed_ing": a.motor("feed_ingestion"),
        }

    def decode(self, net, t_from_ms: float, t_to_ms: float, dt: float) -> Action:
        rates = net.rates(t_from_ms, t_to_ms)

        def gr(key):
            idx = self.g[key]
            return float(rates[idx].mean()) if len(idx) else 0.0

        r = {k: gr(k) for k in self.g}
        for name, idx in self.log_groups.items():
            r[name] = float(rates[idx].mean()) if len(idx) else 0.0
        fwd = 0.5 * (r["fwd_L"] + r["fwd_R"])
        speed = self.gain_forward * fwd - self.gain_backward * r["back"]
        # 자발 보행 (지속 랜덤 워크)
        self._noise = self.turn_persistence * self._noise + (1 - self.turn_persistence) * self._rng.normal(0, self.turn_noise)
        speed += self.spontaneous_speed
        omega = self.gain_turn * (r["turn_L"] - r["turn_R"]) + self._noise
        speed = float(np.clip(speed, -self.max_speed, self.max_speed))
        omega = float(np.clip(omega, -self.max_omega, self.max_omega))
        # 주둥이 신전은 MN9 (rostrum protractor) 가 결정 (Shiu 2024; Gordon & Scott 2009).
        # 섭식(ingestion) 운동뉴런은 기록만 한다.
        proboscis = r["feed"] > self.feed_threshold
        if proboscis and self.feeding_stops_walking:
            # 몸 규칙: 주둥이를 내밀고 먹는 동안에는 걷지 않는다
            speed, omega = 0.0, 0.0
        self._since_jump += dt
        jump = max(r["escape"], r["takeoff"]) > self.escape_threshold and self._since_jump >= self.jump_cooldown_s
        if jump:
            self._since_jump = 0.0
        return Action(speed=speed, omega=omega, proboscis=proboscis, jump=jump, rates=r)
