import math

from flybrain.world import Fly, OdorSource, Patch, TimedStimulus, World


def test_odor_gradient_lateralized():
    w = World(fly=Fly(x=50, y=50, heading=0.0), odor_sources=[OdorSource(50, 80, odorant="vinegar", sigma=20)])
    P = w.percept()
    l, r = P["odor"]["DM1"]
    assert l > r  # 원천이 왼쪽(+y)에 있으면 왼쪽 안테나가 더 강함


def test_patch_and_hunger():
    w = World(fly=Fly(x=10, y=10, satiety_capacity=1.0), patches=[Patch(10, 10, radius=3, kind="sugar")])
    assert w.percept()["sugar"][0] == 1.0
    for _ in range(20):
        w.step(0.1, 0.0, 0.0, proboscis=True)
    assert w.fly.energy > 0.9
    assert w.percept().get("sugar", (0, 0))[0] < 0.2


def test_walls_and_events():
    w = World(width=10, height=10, fly=Fly(x=9, y=5, heading=0.0), events=[TimedStimulus("wind", 0.0, 1.0)])
    w.step(1.0, 10.0, 0.0)
    assert 0 <= w.fly.x <= 10
    assert abs(w.fly.heading - math.pi) < 1e-6
    assert "wind" not in w.percept()  # t=1.0 → 이벤트 종료
