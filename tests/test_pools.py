"""시냅스 풀 2개(빠름/느림) 엔진 확장 테스트."""
import numpy as np
import scipy.sparse as sp

from flybrain.lif import LIFParams
from flybrain.lif_mlx import MLXLIFNetwork


def _chain():
    return sp.csr_matrix(np.array([[0.0, 30.0, 0.0], [0, 0, 0], [0, 0, 0]]))


def test_slow_pool_is_slower_at_equal_charge():
    """전하량을 보존하면 느린 풀은 같은 입력에 대해 더 약하게(넓게 퍼져) 작용한다."""
    W = _chain()
    fast = MLXLIFNetwork(W, LIFParams(), seed=1)
    fast.stimulate([0], 150.0)
    fast.run(200.0)
    slow = MLXLIFNetwork(W, LIFParams(tau_slow=60.0), seed=1,
                         slow_pre=np.array([True, False, False]))
    slow.stimulate([0], 150.0)
    slow.run(200.0)
    assert fast.counts()[0] == slow.counts()[0]      # 자극 뉴런은 같다
    assert slow.counts()[1] < fast.counts()[1]


def test_slow_pool_without_charge_conservation_is_stronger():
    W = _chain()
    p = LIFParams(tau_slow=60.0, w_slow_scale=1.0)
    net = MLXLIFNetwork(W, p, seed=1, slow_pre=np.array([True, False, False]))
    net.stimulate([0], 150.0)
    net.run(200.0)
    ref = MLXLIFNetwork(W, LIFParams(), seed=1)
    ref.stimulate([0], 150.0)
    ref.run(200.0)
    assert net.counts()[1] > ref.counts()[1]


def test_no_slow_pre_matches_single_pool():
    """slow_pre 를 주지 않으면 느린 풀이 비고 기존 동작과 같다."""
    W = _chain()
    net = MLXLIFNetwork(W, LIFParams(), seed=3)
    assert net._pool_mx[1][4] == 0
    net.stimulate([0], 150.0)
    net.run(100.0)
    assert net.counts()[1] > 0


def test_bias_is_modulated_by_inhibition():
    """지속 전류는 Poisson 강제 스파이크와 달리 억제성 입력에 깎인다."""
    W = sp.csr_matrix(np.array([[0.0, -40.0, 0.0], [0, 0, 0], [0, 0, 0]]))
    free = MLXLIFNetwork(W, LIFParams(), seed=5)
    free.set_bias([1], mv=9.0)
    free.run(500.0)
    inhib = MLXLIFNetwork(W, LIFParams(), seed=5)
    inhib.set_bias([1], mv=9.0)
    inhib.stimulate([0], 150.0)
    inhib.run(500.0)
    assert free.counts()[1] > 0
    assert inhib.counts()[1] < free.counts()[1]
