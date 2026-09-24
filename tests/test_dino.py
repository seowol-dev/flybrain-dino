"""공룡게임 물리 / 1인칭 렌더링 / 도약 디코더 단위 테스트."""
import numpy as np
import pytest

from flybrain.dino import Cactus, DinoGame, DinoScene, LoomDecoder


def test_jump_window_clears_obstacle():
    """도약 최고높이가 선인장보다 높고, 적절한 시점에 뛰면 넘는다."""
    g = DinoGame(rng=np.random.default_rng(0))
    apex = g.jump_v ** 2 / (2 * g.gravity)
    assert apex > 0.48, "도약 최고높이가 선인장 최대 높이보다 커야 한다"
    g.cacti = [Cactus(dist=1.0, half_w=0.35, height=0.45)]
    g._next_spawn = 1e9
    # 충돌 0.3 초 전에 도약
    while g.cacti and g.cacti[0].dist > 0.3 * g.speed:
        g.step(1 / 240)
    assert g.jump()
    for _ in range(240):
        g.step(1 / 240)
    assert not g.dead and g.score == 1


def test_no_jump_means_collision():
    g = DinoGame(rng=np.random.default_rng(0))
    for _ in range(600):
        g.step(1 / 120)
        if g.dead:
            break
    assert g.dead


def test_cannot_double_jump():
    g = DinoGame(rng=np.random.default_rng(0))
    assert g.jump()
    g.step(1 / 120)
    assert not g.jump()


def test_looming_grows_and_darkens():
    """다가올수록 각크기가 커지고 정면 휘도가 어두워진다 (루밍)."""
    g = DinoGame(rng=np.random.default_rng(0))
    g.cacti = [Cactus(dist=4.0, half_w=0.35, height=0.45)]
    g._next_spawn = 1e9
    sc = DinoScene(game=g)
    # 시야 격자를 깔고 '어두워진 면적'이 늘어나는지 본다 (루밍의 정의)
    a, e = np.meshgrid(np.linspace(-40, 40, 81), np.linspace(-25, 25, 51))
    az, el = a.ravel(), e.ravel()
    th, dark = [], []
    for _ in range(5):
        th.append(g.theta_nearest())
        dark.append(float((sc.luminance(az, el, g.t) < 0.3).mean()))
        for _ in range(30):
            g.step(1 / 60)
    assert th == sorted(th), "각반너비가 단조 증가해야 한다"
    assert dark == sorted(dark), "어두운 면적이 단조 증가해야 한다"
    assert dark[-1] > 3 * dark[0] > 0, "면적이 뚜렷하게 커져야 한다"


def test_decoder_refractory_and_threshold():
    d = LoomDecoder(threshold=1.0, tau_fast_s=0.01, tau_slow_s=10.0, refractory_s=0.5)
    rates = {"F_pool": 10.0, "S_pool": 10.0}
    for k in range(20):
        d.update(rates, 1 / 60, k / 60)
    fired = [d.update({"F_pool": 40.0, "S_pool": 10.0}, 1 / 60, (20 + k) / 60)[0] for k in range(30)]
    assert sum(fired) == 1, "불응기 동안 한 번만 발화해야 한다"


def test_decoder_ignores_global_brightness():
    """중심과 주변이 같이 변하면(전역 밝기 변화) 지표가 거의 움직이지 않는다."""
    d = LoomDecoder(threshold=1.0, tau_fast_s=0.02, tau_slow_s=1.0)
    for k in range(60):
        d.update({"F_pool": 10.0, "S_pool": 10.0}, 1 / 60, k / 60)
    sig = [d.update({"F_pool": 30.0, "S_pool": 30.0}, 1 / 60, (60 + k) / 60)[2] for k in range(30)]
    assert max(abs(s) for s in sig) < 0.5
