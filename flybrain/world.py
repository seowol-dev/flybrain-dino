"""2차원 가상 아레나 (단위: mm, 초, 라디안).

초파리 몸체(Fly)와 환경 자극(냄새 원천, 먹이 패치, 빛, 온도, 바람, 시간 이벤트)을 담고,
`percept()` 로 좌/우 안테나·눈 위치에서 감지되는 자극 세기(0~1)를 돌려준다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .atlas import ODORANTS


def _gauss(dx, dy, sigma):
    return math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))


@dataclass
class Fly:
    x: float = 50.0
    y: float = 50.0
    heading: float = 0.0          # rad, +x 축 기준 반시계
    speed: float = 0.0            # mm/s (부호: 후진 <0)
    omega: float = 0.0            # rad/s
    proboscis: bool = False
    energy: float = 0.0           # 섭식 누적량
    satiety_capacity: float = 3.0 # 이만큼 먹으면 배부름 (hunger→0). inf 로 두면 포만 없음
    hunger0: float = 1.0          # 시작 배고픔 (0~1)
    jumps: int = 0
    body_length: float = 2.5      # mm
    antenna_sep: float = 0.6      # mm, 좌우 안테나 간격
    eye_azimuth: float = math.radians(45)

    @property
    def hunger(self) -> float:
        """배고픔 0~1. 뇌 모델에는 호르몬/대사 상태가 없으므로 몸 수준 변수로 두고
        당(sugar) 감각 뉴런 이득을 조절한다 (배고픈 파리일수록 당 GRN 반응이 큼)."""
        if not math.isfinite(self.satiety_capacity):
            return self.hunger0
        return max(0.0, self.hunger0 - self.energy / self.satiety_capacity)

    def antenna_positions(self):
        cx = self.x + math.cos(self.heading) * self.body_length * 0.5
        cy = self.y + math.sin(self.heading) * self.body_length * 0.5
        nx, ny = -math.sin(self.heading), math.cos(self.heading)  # 왼쪽 법선
        d = self.antenna_sep / 2
        return (cx + nx * d, cy + ny * d), (cx - nx * d, cy - ny * d)

    def eye_rays(self, dists=(2.0, 5.0, 10.0)):
        """좌/우 눈이 바라보는 지점들."""
        out = []
        for sgn in (+1, -1):
            a = self.heading + sgn * self.eye_azimuth
            out.append([(self.x + math.cos(a) * r, self.y + math.sin(a) * r) for r in dists])
        return out[0], out[1]


@dataclass
class OdorSource:
    x: float
    y: float
    odorant: str | dict[str, float] = "vinegar"   # ODORANTS 키 또는 {glomerulus: 강도}
    concentration: float = 1.0
    sigma: float = 20.0          # mm, 가우시안 플룸 반경
    label: str = ""

    def profile(self) -> dict[str, float]:
        return ODORANTS[self.odorant] if isinstance(self.odorant, str) else dict(self.odorant)

    def conc_at(self, x, y) -> float:
        return self.concentration * _gauss(x - self.x, y - self.y, self.sigma)


@dataclass
class Patch:
    """바닥의 맛 자극 (sugar / bitter / salt_low / water)."""
    x: float
    y: float
    radius: float = 8.0
    kind: str = "sugar"
    intensity: float = 1.0
    depletable: bool = False
    amount: float = 1.0

    def contains(self, x, y) -> bool:
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.radius ** 2


@dataclass
class LightField:
    """밝기 필드. kind: 'uniform' | 'half' (x>x0 밝음) | 'spot' (가우시안) | 'gradient' (x 방향)."""
    kind: str = "uniform"
    intensity: float = 0.5
    x0: float = 50.0
    y0: float = 50.0
    sigma: float = 25.0
    dark_level: float = 0.02

    def at(self, x, y) -> float:
        if self.kind == "uniform":
            return self.intensity
        if self.kind == "half":
            return self.intensity if x > self.x0 else self.dark_level
        if self.kind == "spot":
            return self.dark_level + self.intensity * _gauss(x - self.x0, y - self.y0, self.sigma)
        if self.kind == "gradient":
            return self.dark_level + self.intensity * min(max(x / (2 * self.x0), 0.0), 1.0)
        raise ValueError(self.kind)


@dataclass
class ThermalZone:
    """온도 구역. delta>0 뜨거움(heat), delta<0 차가움(cold). 0~1 스케일."""
    x: float
    y: float
    radius: float = 25.0
    delta: float = 1.0
    soft: bool = True

    def at(self, x, y) -> float:
        d2 = (x - self.x) ** 2 + (y - self.y) ** 2
        if self.soft:
            return self.delta * math.exp(-d2 / (2 * self.radius ** 2))
        return self.delta if d2 <= self.radius ** 2 else 0.0


@dataclass
class WindField:
    direction: float = 0.0   # 바람이 불어오는 방향 (rad)
    speed: float = 0.0       # 0~1


@dataclass
class TimedStimulus:
    """시간 구간 동안 특정 감각 채널을 직접 켠다. 예: looming, 소리, 빛 펄스, 접촉."""
    channel: str
    t_start: float
    t_end: float
    intensity: float = 1.0
    side: str = "both"       # 'both' | 'left' | 'right'
    ramp: bool = False       # True 면 t_start→t_end 로 선형 증가 (looming 근사)

    def value(self, t: float) -> float:
        if not (self.t_start <= t < self.t_end):
            return 0.0
        if self.ramp:
            return self.intensity * (t - self.t_start) / max(self.t_end - self.t_start, 1e-9)
        return self.intensity


@dataclass
class World:
    width: float = 100.0
    height: float = 100.0
    fly: Fly = field(default_factory=Fly)
    odor_sources: list[OdorSource] = field(default_factory=list)
    patches: list[Patch] = field(default_factory=list)
    light: LightField = field(default_factory=lambda: LightField("uniform", 0.3))
    thermal: list[ThermalZone] = field(default_factory=list)
    wind: WindField = field(default_factory=WindField)
    events: list[TimedStimulus] = field(default_factory=list)
    boundary: str = "wall"          # 'wall' | 'wrap'
    odor_lateral_gain: float = 8.0  # 좌우 안테나 농도차 증폭 (실제 초파리의 좌우 비교 민감도 근사)
    t: float = 0.0
    name: str = "arena"

    # -- 자극 계산 -----------------------------------------------------------
    def odor_at(self, x, y) -> dict[str, float]:
        glom: dict[str, float] = {}
        for s in self.odor_sources:
            c = s.conc_at(x, y)
            if c < 1e-4:
                continue
            for g, w in s.profile().items():
                glom[g] = min(1.0, glom.get(g, 0.0) + c * w)
        return glom

    def patch_at(self, x, y) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self.patches:
            if p.contains(x, y) and (not p.depletable or p.amount > 0):
                out[p.kind] = max(out.get(p.kind, 0.0), p.intensity)
        return out

    def _lr(self, a: float, b: float) -> tuple[float, float]:
        """좌우 농도 (a,b) 를 평균 + 증폭된 차이로 변환."""
        m = 0.5 * (a + b)
        if m <= 0:
            return 0.0, 0.0
        d = (a - b) / (a + b + 1e-9) * self.odor_lateral_gain
        return float(np.clip(m * (1 + d), 0, 1)), float(np.clip(m * (1 - d), 0, 1))

    def percept(self) -> dict:
        """현재 초파리가 감지하는 자극.  반환 예:
        {'odor': {'DM1': (L,R), ...}, 'sugar': (L,R), 'light': (L,R), 'heat': (L,R), ... 'events': {...}}"""
        f = self.fly
        (lx, ly), (rx, ry) = f.antenna_positions()
        P: dict = {}
        # 후각 (좌우 안테나)
        oL, oR = self.odor_at(lx, ly), self.odor_at(rx, ry)
        odor = {}
        for g in set(oL) | set(oR):
            odor[g] = self._lr(oL.get(g, 0.0), oR.get(g, 0.0))
        P["odor"] = odor
        # 미각 (몸 위치의 패치) — 좌우 동일
        for kind, v in self.patch_at(f.x, f.y).items():
            if kind == "sugar":
                v *= f.hunger
            P[kind] = (v, v)
        # 시각 (좌/우 눈 방향의 밝기)
        eL, eR = f.eye_rays()
        bl = float(np.mean([self.light.at(x, y) for x, y in eL]))
        br = float(np.mean([self.light.at(x, y) for x, y in eR]))
        P["light"] = (min(bl, 1.0), min(br, 1.0))
        # 온도
        heat = cold = 0.0
        for z in self.thermal:
            v = z.at(f.x, f.y)
            if v > 0:
                heat += v
            else:
                cold += -v
        if heat:
            P["heat"] = (min(heat, 1.0),) * 2
        if cold:
            P["cold"] = (min(cold, 1.0),) * 2
        # 바람 (상대 방향에 따라 좌우 비대칭)
        if self.wind.speed > 0:
            rel = self.wind.direction - f.heading
            asym = 0.5 * math.sin(rel)
            P["wind"] = (float(np.clip(self.wind.speed * (1 + asym), 0, 1)), float(np.clip(self.wind.speed * (1 - asym), 0, 1)))
        # 시간 이벤트
        for ev in self.events:
            v = ev.value(self.t)
            if v <= 0:
                continue
            l, r = P.get(ev.channel, (0.0, 0.0))
            if ev.side in ("both", "left"):
                l = max(l, v)
            if ev.side in ("both", "right"):
                r = max(r, v)
            P[ev.channel] = (min(l, 1.0), min(r, 1.0))
        return P

    # -- 동역학 --------------------------------------------------------------
    def step(self, dt: float, speed: float, omega: float, jump: bool = False, proboscis: bool = False) -> None:
        f = self.fly
        f.speed, f.omega, f.proboscis = speed, omega, proboscis
        f.heading = (f.heading + omega * dt) % (2 * math.pi)
        dist = speed * dt
        if jump:
            dist += 6.0
            f.jumps += 1
        nx = f.x + math.cos(f.heading) * dist
        ny = f.y + math.sin(f.heading) * dist
        if self.boundary == "wrap":
            nx %= self.width
            ny %= self.height
        else:
            if nx < 0 or nx > self.width:
                nx = min(max(nx, 0.0), self.width)
                f.heading = (math.pi - f.heading) % (2 * math.pi)
            if ny < 0 or ny > self.height:
                ny = min(max(ny, 0.0), self.height)
                f.heading = (-f.heading) % (2 * math.pi)
        f.x, f.y = nx, ny
        # 섭식
        if proboscis:
            for p in self.patches:
                if p.kind == "sugar" and p.contains(f.x, f.y) and (not p.depletable or p.amount > 0):
                    eaten = p.intensity * dt
                    f.energy += eaten
                    if p.depletable:
                        p.amount = max(0.0, p.amount - eaten)
        self.t += dt

    def describe(self) -> str:
        parts = [f"{self.name}: {self.width}x{self.height} mm, boundary={self.boundary}"]
        for s in self.odor_sources:
            parts.append(f"  odor {s.odorant if isinstance(s.odorant, str) else 'custom'} @({s.x:.0f},{s.y:.0f}) c={s.concentration} sigma={s.sigma}")
        for p in self.patches:
            parts.append(f"  patch {p.kind} @({p.x:.0f},{p.y:.0f}) r={p.radius}")
        parts.append(f"  light {self.light.kind} I={self.light.intensity}")
        for z in self.thermal:
            parts.append(f"  thermal delta={z.delta} @({z.x:.0f},{z.y:.0f}) r={z.radius}")
        if self.wind.speed:
            parts.append(f"  wind dir={math.degrees(self.wind.direction):.0f}deg speed={self.wind.speed}")
        for e in self.events:
            parts.append(f"  event {e.channel} {e.t_start}-{e.t_end}s I={e.intensity} side={e.side}")
        return "\n".join(parts)
