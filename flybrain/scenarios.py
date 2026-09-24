"""미리 준비된 가상 세계 실험 시나리오.

각 함수는 (World, dict(실험 옵션)) 을 돌려준다. CLI: `flybrain run <이름>`.
"""
from __future__ import annotations

import math

from .world import Fly, LightField, OdorSource, Patch, ThermalZone, TimedStimulus, WindField, World


def feeding(seed_heading: float = 0.0):
    """먹이 찾기/섭식: 당 패치, 쓴맛 패치, 당+쓴맛 혼합 패치. 배고픔(포만) 포함.
    검증된 경로: 당 GRN → ... → MN9 (주둥이 신전) — Shiu et al. 2024."""
    w = World(
        width=80, height=80, name="feeding",
        fly=Fly(x=12, y=40, heading=seed_heading, satiety_capacity=2.0),
        patches=[
            Patch(30, 40, radius=6, kind="sugar"),
            Patch(50, 40, radius=6, kind="bitter"),
            Patch(68, 40, radius=6, kind="sugar"), Patch(68, 40, radius=6, kind="bitter"),
        ],
        light=LightField("uniform", 0.3),
    )
    return w, {}


def sugar_vs_bitter_mix():
    """당만 있는 패치 vs 당+쓴맛 패치 위에 파리를 올려놓고 섭식 여부 비교 (쓴맛에 의한 섭식 억제)."""
    worlds = {}
    for name, kinds in {"sugar_only": ["sugar"], "sugar+bitter": ["sugar", "bitter"], "bitter_only": ["bitter"]}.items():
        w = World(width=40, height=40, name=name, fly=Fly(x=20, y=20, satiety_capacity=math.inf),
                  patches=[Patch(20, 20, radius=15, kind=k) for k in kinds])
        worlds[name] = w
    return worlds


def odor_plume(odorant: str = "vinegar"):
    """냄새 원천 하나. 후각 경로 → 하행 뉴런 → 조향. (모델 한계: README '후각' 절 참고)"""
    w = World(
        width=120, height=80, name=f"odor_{odorant}",
        fly=Fly(x=20, y=40, heading=math.radians(30)),
        odor_sources=[OdorSource(90, 40, odorant=odorant, concentration=1.0, sigma=25)],
    )
    return w, {}


def wind_puff():
    """2초에 옆에서 강한 바람 → Johnston 기관 바람 뉴런 → 이륙/탈출 하행 뉴런 → 도약."""
    w = World(
        width=80, height=80, name="wind_puff", fly=Fly(x=40, y=40),
        events=[TimedStimulus("wind", 2.0, 2.5, intensity=1.0, side="left")],
    )
    return w, {}


def looming():
    """빛 점멸/접근 자극(광수용체 램프). 모델에서 광수용체→하행 경로는 약하다 — 한계 확인용."""
    w = World(
        width=80, height=80, name="looming", fly=Fly(x=40, y=40),
        events=[TimedStimulus("looming", 1.0, 2.0, intensity=1.0, side="left", ramp=True)],
    )
    return w, {}


def head_touch():
    """머리 기계감각모 자극 → 주둥이 운동뉴런/하행 뉴런 반응."""
    w = World(
        width=60, height=60, name="head_touch", fly=Fly(x=30, y=30),
        events=[TimedStimulus("touch_head", 1.0, 2.0, intensity=1.0, side="left")],
    )
    return w, {}


def thermal_arena():
    """오른쪽 절반이 뜨거운 아레나 (heat 수용체 TRN_VP2)."""
    w = World(
        width=100, height=60, name="thermal", fly=Fly(x=50, y=30, heading=0.0),
        thermal=[ThermalZone(100, 30, radius=35, delta=1.0)],
    )
    return w, {}


SCENARIOS = {
    "feeding": feeding,
    "odor": odor_plume,
    "wind": wind_puff,
    "looming": looming,
    "touch": head_touch,
    "thermal": thermal_arena,
}
