"""크롬 공룡게임 — 초파리 1인칭 시점.

**왜 1인칭인가**

원래 크롬 공룡게임은 측면 스크롤이라 선인장이 화면을 옆으로 지나가기만 하고 확대되지
않는다. 그런 측면 이동을 읽으려면 T4/T5 방향선택성이 필요한데, 이 모델에서는 그것이
작동하지 않는다(PLAN_realtime_dino.md 의 측정 참고: 뉴런별 |DSI| 중앙값 0.002).

그래서 파리는 공룡의 눈으로 세계를 본다. 다가오는 선인장은 각크기가 1/거리 로 커지는
**루밍(looming)** 자극이 되고, 루밍은 실제 초파리에서 도피 도약을 일으키는 자연
유발자극이다. 즉 게임을 파리가 실제로 풀 수 있는 문제로 바꾼 것이지, 파리가 못 하는 일을
억지로 시키는 것이 아니다.

좌표: 시야 방향 (방위각 az, 고도 el) 도 단위. 정면 az=0. 지평선 el=0 기준으로 땅은 아래.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .visual import Scene


# ---------------------------------------------------------------------------
# 게임 물리
# ---------------------------------------------------------------------------
@dataclass
class Cactus:
    dist: float           # m, 눈까지 남은 거리
    half_w: float = 0.35  # m, 반너비
    height: float = 0.50  # m, 높이 (땅에서)


@dataclass
class DinoGame:
    """크롬 공룡게임 물리. 길이 단위 m, 시간 s.

    공룡은 제자리에 있고 세계가 다가온다(1인칭). 점프는 초기 속도를 주는 한 번의 충격이고,
    공중에서는 중력만 받는다. 속도는 시간이 지나며 조금씩 빨라진다(원 게임과 같다).
    """
    # 스케일은 루밍이 파리에게 보이도록 잡았다: 충돌 1초 전(d=2 m) 각반너비 약 10°,
    # 0.5초 전 약 19°, 0.25초 전 약 35°.
    # 도약 최고높이 = jump_v²/(2·g) = 0.96 m. 선인장(≤0.48 m) 위에 머무는 시간은
    # 충돌 전 0.07~0.52초 구간이다 — 디코더가 이 창 안에서 터져야 넘는다.
    speed: float = 2.0              # m/s, 현재 전진 속도
    speed_max: float = 3.2
    speed_ramp: float = 0.04        # m/s per s
    gravity: float = 22.0           # m/s²
    jump_v: float = 6.5             # m/s, 도약 초기 속도
    eye_h: float = 0.25             # m, 서 있을 때 눈높이
    y: float = 0.0                  # m, 땅 위 높이
    vy: float = 0.0
    spacing: tuple[float, float] = (4.0, 8.0)    # m, 선인장 간격 범위 (2~4초)
    cacti: list = field(default_factory=list)
    t: float = 0.0
    score: int = 0                  # 넘은 선인장 수
    dead: bool = False
    rng: np.random.Generator | None = None
    _next_spawn: float = 0.0

    def __post_init__(self):
        self.rng = self.rng or np.random.default_rng(0)
        self._next_spawn = 6.0
        self._spawn()

    @property
    def airborne(self) -> bool:
        return self.y > 1e-6

    def _spawn(self):
        lo, hi = self.spacing
        h = float(self.rng.uniform(0.34, 0.48))
        w = float(self.rng.uniform(0.28, 0.44))
        self.cacti.append(Cactus(dist=self._next_spawn, half_w=w, height=h))
        self._next_spawn += float(self.rng.uniform(lo, hi))

    def jump(self) -> bool:
        """땅에 있을 때만 도약. 실제로 뛰었으면 True."""
        if self.airborne or self.dead:
            return False
        self.vy = self.jump_v
        return True

    def step(self, dt: float) -> None:
        if self.dead:
            return
        self.t += dt
        self.speed = min(self.speed + self.speed_ramp * dt, self.speed_max)
        # 수직 운동
        if self.airborne or self.vy > 0:
            self.y += self.vy * dt
            self.vy -= self.gravity * dt
            if self.y <= 0:
                self.y, self.vy = 0.0, 0.0
        # 세계가 다가온다
        d = self.speed * dt
        for c in self.cacti:
            c.dist -= d
        self._next_spawn -= d
        # 충돌 / 통과 판정 (몸통 반너비 0.15 m 로 본다)
        body = 0.12
        for c in list(self.cacti):
            if c.dist < body and c.dist > -body:
                if self.y < c.height:
                    self.dead = True
            elif c.dist <= -body:
                self.cacti.remove(c)
                self.score += 1
        while self._next_spawn < 6.0:
            self._spawn()

    def nearest(self) -> Cactus | None:
        ahead = [c for c in self.cacti if c.dist > 0]
        return min(ahead, key=lambda c: c.dist) if ahead else None

    def theta_nearest(self) -> float:
        """가장 가까운 선인장의 각반너비(도) — 루밍 세기의 물리적 기준값."""
        c = self.nearest()
        if c is None:
            return 0.0
        return math.degrees(math.atan(c.half_w / max(c.dist, 1e-3)))


# ---------------------------------------------------------------------------
# 1인칭 렌더링 (시야 방향 → 휘도)
# ---------------------------------------------------------------------------
@dataclass
class DinoScene(Scene):
    """게임 상태를 파리 시야에 그린다. 밝은 하늘/땅 위의 어두운 선인장.

    선인장은 거리 d, 반너비 w, 높이 h 의 직립 판으로 본다.
      방위각 반각 = arctan(w / d)
      아래 끝  el = arctan((0 − eye) / d),  위 끝 el = arctan((h − eye) / d)
    눈높이 eye = eye_h + y (점프하면 눈이 올라가 선인장이 시야에서 내려간다).
    """
    game: DinoGame | None = None
    sky: float = 1.0
    ground: float = 0.72
    cactus: float = 0.03
    horizon_soft: float = 1.5      # 도, 경계 부드럽게 (컬럼 간격 ~2.2° 보다 작게)

    def luminance(self, az, el, t):
        g = self.game
        eye = g.eye_h + g.y
        out = np.where(el >= 0.0, self.sky, self.ground).astype(np.float32)
        # 지평선을 부드럽게 (앨리어싱만 줄인다)
        w = np.clip((el + self.horizon_soft) / (2 * self.horizon_soft), 0.0, 1.0)
        out = self.ground + (self.sky - self.ground) * w
        for c in g.cacti:
            if c.dist <= 0.02:
                continue
            haz = math.degrees(math.atan(c.half_w / c.dist))
            el_lo = math.degrees(math.atan((0.0 - eye) / c.dist))
            el_hi = math.degrees(math.atan((c.height - eye) / c.dist))
            s = self.horizon_soft
            mx = np.clip((haz - np.abs(az)) / s, 0.0, 1.0)
            my = np.clip((el - el_lo) / s, 0.0, 1.0) * np.clip((el_hi - el) / s, 0.0, 1.0)
            out = out + (self.cactus - out) * (mx * my)
        return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 점프 디코더
# ---------------------------------------------------------------------------
# 루밍 판별에 쓰는 집단 신호. PLAN_realtime_dino.md 의 측정 결과:
#   균일 회색 : Mi1 10.03, L1  9.99  → 지표 ≈ 0
#   접근(loom): Mi1 12.14, L1  4.94  → 지표 ≈ +7
#   후퇴(대조): Mi1  8.49, L1 12.24  → 지표 ≈ −4
# LPLC2 → DNp01(거대섬유) 경로를 쓰지 못하는 이유는 README 의 '알려진 한계' 참고.
# 어두운 물체가 정면 수용장에 들어오면 그 컬럼의 광수용체(히스타민=억제)가 조용해지고
# 라미나 단극세포가 탈억제되어 발화가 올라간다. 전역 밝기 변화는 중심과 주변에 같이
# 나타나므로, 중심 − 주변 으로 빼면 '정면에 어두운 물체' 만 남는다. LC/LPLC 가 하는
# 국소 풀링과 같은 구조를 디코더에서 만든 것이다.
LAMINA_READ = ("L1", "L2", "L3", "Tm1", "Tm2")
ON_GROUPS = ("Mi1", "T4a", "T4b", "T4c", "T4d")      # 참고 기록용
OFF_GROUPS = LAMINA_READ


@dataclass
class LoomDecoder:
    """시엽 집단 발화율 → 도약 결정.

    지표 = mean(중심 라미나) − mean(주변 라미나) 를 느린 이동평균으로 뺀 값(순응)이다.
    절대 발화율은 작동점 보정에 따라 달라지므로 변화분만 쓴다. 도약 후에는
    refractory_s 동안 다시 뛰지 않는다(실제 파리도 도약 후 재도약까지 시간이 걸린다).
    """
    # 유형별로 평균을 내면 작은 그룹(중심 L1 18개)의 양자화 잡음이 지표를 지배한다.
    # 중심/주변을 각각 하나로 풀링한 그룹을 쓴다.
    on_groups: tuple = ("F_pool",)      # 중심
    off_groups: tuple = ("S_pool",)     # 주변 (고리)
    threshold: float = 1.5
    tau_fast_s: float = 0.03
    tau_slow_s: float = 1.0
    refractory_s: float = 0.45
    _fast: float = 0.0
    _slow: float = 0.0
    _primed: bool = False
    _last_jump: float = -1e9

    def reset(self):
        self._fast = self._slow = 0.0
        self._primed = False
        self._last_jump = -1e9

    def index(self, rates: dict[str, float]) -> float:
        on = [rates[g] for g in self.on_groups if g in rates]
        off = [rates[g] for g in self.off_groups if g in rates]
        if not on or not off:
            return 0.0
        return float(np.mean(on) - np.mean(off))

    def update(self, rates: dict[str, float], dt_s: float, t: float) -> tuple[bool, float, float]:
        """(도약할까, 원시 지표, 순응 후 신호) 를 돌려준다."""
        raw = self.index(rates)
        if not self._primed:
            self._fast = self._slow = raw
            self._primed = True
        else:
            self._fast += (1 - math.exp(-dt_s / self.tau_fast_s)) * (raw - self._fast)
            self._slow += (1 - math.exp(-dt_s / self.tau_slow_s)) * (raw - self._slow)
        sig = self._fast - self._slow
        fire = sig > self.threshold and (t - self._last_jump) > self.refractory_s
        if fire:
            self._last_jump = t
        return fire, raw, sig
