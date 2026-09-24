"""시각 자극 장면 — 컬럼의 (방위각, 고도) 격자에 휘도(0~1)를 그린다.

여기 있는 자극은 시각 경로가 **진짜로 작동하는지** 검증하기 위한 것이다.

  DriftingGrating : 움직이는 줄무늬. T4a~d / T5a~d 의 방향 선택성을 잰다.
                    시공간 대비가 있어야 T4/T5 가 켜지므로, 이게 되면 망막지도가 살아 있다는 뜻.
  LoomingDisc     : 확대되는 검은 원. LPLC2 → DNp01(거대섬유) 의 자연 유발자극.
  FlickerField    : 전체 화면 점멸. 대조군 — 공간 대비가 없으므로 T4/T5 는 거의 안 켜져야 한다.

방향 약속 (지도 좌표계): 방위각 + = 앞쪽, 고도 + = 등쪽.
  direction 0°   = +방위각 방향 이동 = back-to-front (regressive)
  direction 180° = -방위각 방향 이동 = front-to-back (progressive)
  direction 90°  = 위쪽,  270° = 아래쪽
문헌값: T4a/T5a = progressive, T4b/T5b = regressive, T4c = 위, T4d = 아래
(Maisak et al. 2013). `flybrain grating` 이 이 대응을 실측해 망막지도의 축 부호를 검증한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DIRECTIONS = {0: "back→front", 90: "위(up)", 180: "front→back", 270: "아래(down)"}


@dataclass
class Scene:
    """모든 자극의 공통 인터페이스: luminance(az, el, t) → (N,) 0~1."""

    def luminance(self, az: np.ndarray, el: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    def columns(self, eyes: dict, t: float) -> dict[str, np.ndarray]:
        """눈별 컬럼 휘도."""
        return {s: self.luminance(e.col_az, e.col_el, t) for s, e in eyes.items()}


@dataclass
class UniformField(Scene):
    """균일 조명. 지금까지의 자극 방식(= 광수용체 전체 같은 세기)과 같은 대조군."""
    lum: float = 0.5

    def luminance(self, az, el, t):
        return np.full(len(az), self.lum)


@dataclass
class FlickerField(Scene):
    """전체 화면이 같이 깜빡인다. 시간 대비는 있고 공간 대비는 없다."""
    freq_hz: float = 2.0
    contrast: float = 1.0
    mean: float = 0.5

    def luminance(self, az, el, t):
        v = self.mean + 0.5 * self.contrast * math.sin(2 * math.pi * self.freq_hz * t)
        return np.full(len(az), min(max(v, 0.0), 1.0))


@dataclass
class DriftingGrating(Scene):
    """움직이는 정현파 줄무늬.

    direction_deg : 이동 방향 (위 약속)
    period_deg    : 공간 주기(도). 초파리 최적치는 대략 20~30° (Δφ≈5° 의 4~6배)
    temporal_hz   : 시간 주파수(Hz). 최적치는 대략 1~4 Hz → 속도 = period × temporal
    """
    direction_deg: float = 0.0
    period_deg: float = 24.0
    temporal_hz: float = 2.0
    contrast: float = 1.0
    mean: float = 0.5
    start_s: float = 0.0

    def luminance(self, az, el, t):
        if t < self.start_s:
            return np.full(len(az), self.mean)
        a = math.radians(self.direction_deg)
        proj = az * math.cos(a) + el * math.sin(a)
        ph = 2 * math.pi * (proj / self.period_deg - self.temporal_hz * (t - self.start_s))
        return np.clip(self.mean + 0.5 * self.contrast * np.sin(ph), 0.0, 1.0)


@dataclass
class LoomingDisc(Scene):
    """정면으로 다가오는 검은 원 (OFF 루밍).

    일정 속도로 다가오는 반지름 l 의 물체는 각반지름이
        θ(t) = arctan( (l/v) / (t_c - t) )
    로 커진다. l/v (ms) 가 루밍의 표준 파라미터다. 파리의 거대섬유는 θ 가 일정 임계각에
    도달할 때 발화한다 (von Reiser & Dickinson 2002; Card & Dickinson 2008).

    l_over_v_ms : l/v (ms). 작을수록 급격히 다가온다.
    t_hit_s     : 충돌 예정 시각(초)
    theta_max   : 이 각반지름에서 멈춘다(도). 충돌 후 화면을 덮는 것을 막는다.
    """
    l_over_v_ms: float = 40.0
    t_hit_s: float = 1.0
    az0: float = 0.0
    el0: float = 0.0
    theta_max: float = 60.0
    bg: float = 1.0
    fg: float = 0.0

    def theta_deg(self, t: float) -> float:
        dt = self.t_hit_s - t
        if dt <= 0:
            return self.theta_max
        th = math.degrees(math.atan((self.l_over_v_ms * 1e-3) / dt))
        return min(th, self.theta_max)

    def luminance(self, az, el, t):
        th = self.theta_deg(t)
        d = np.hypot(az - self.az0, el - self.el0)
        # 가장자리를 1° 폭으로 부드럽게 (컬럼 간격 ~5° 보다 작게 두어 aliasing 만 줄인다)
        w = np.clip((th - d) / 1.0, 0.0, 1.0)
        return self.bg + (self.fg - self.bg) * w


@dataclass
class RecedingDisc(LoomingDisc):
    """멀어지는 검은 원 — 루밍의 음성 대조군 (LPLC2/DNp01 이 반응하면 안 된다)."""

    def theta_deg(self, t: float) -> float:
        return super().theta_deg(self.t_hit_s * 2 - t) if t < self.t_hit_s * 2 else 0.0


# ---------------------------------------------------------------------------
# 방향 선택성 지표
# ---------------------------------------------------------------------------
def dsi(pref: float, null: float) -> float:
    """방향 선택성 지수 (Direction Selectivity Index) = (pref - null) / (pref + null)."""
    s = pref + null
    return float((pref - null) / s) if s > 0 else 0.0


def preferred_direction(rates_by_dir: dict[float, float]) -> tuple[float, float]:
    """벡터 합으로 선호 방향(도)과 방향성 강도(0~1)를 구한다."""
    ang = np.radians(np.array(list(rates_by_dir.keys()), float))
    r = np.array(list(rates_by_dir.values()), float)
    if r.sum() <= 0:
        return float("nan"), 0.0
    vx, vy = (r * np.cos(ang)).sum(), (r * np.sin(ang)).sum()
    return float(np.degrees(np.arctan2(vy, vx)) % 360), float(np.hypot(vx, vy) / r.sum())
