"""엔진 단위 테스트. Brian2 기준값은 Shiu et al. 2024 원본 model.py 를 Brian2 2.9 로 돌려 얻은 것."""
import numpy as np
import scipy.sparse as sp

from flybrain.lif import LIFNetwork, LIFParams


def _net(W, **kw):
    return LIFNetwork(sp.csr_matrix(np.asarray(W, dtype=np.int32)), LIFParams(**kw), seed=3)


def test_single_psp_matches_brian2():
    # Brian2: 40 시냅스 단일 스파이크 → 최대 탈분극 1.7324 mV
    W = np.zeros((2, 2)); W[0, 1] = 40
    net = _net(W)
    net.v[0] = 0.0  # 첫 스텝에 강제 발화
    vs = []
    for _ in range(150):
        net.step()
        vs.append(float(net.v[1]))
    assert abs(max(vs) + 52 - 1.7324) < 1e-3


def test_poisson_stimulus_rate():
    net = _net(np.zeros((1, 1)))
    net.stimulate([0], 150.0)
    net.run(20000)
    assert abs(net.rates()[0] - 150) < 8


def test_convergent_drive_matches_brian2():
    # Brian2 (20 s): 10 → 1, 40 시냅스씩 → 사후 뉴런 132.0 Hz
    W = np.zeros((11, 11)); W[:10, 10] = 40
    net = _net(W)
    net.stimulate(list(range(10)), 150.0)
    net.run(20000)
    assert abs(net.rates()[10] - 132.0) < 8


def test_refractory_blocks_synaptic_input():
    # 불응기 중 도착한 입력은 버려진다 (Brian2 'unless refractory' 규칙)
    W = np.zeros((2, 2)); W[0, 1] = 40
    net = _net(W)
    net.v[1] = 0.0   # 사후 뉴런이 먼저 발화 → 불응기 (21 스텝)
    net.v[0] = 0.0   # 사전 뉴런도 발화 → 18 스텝 후 도착 (불응기 안)
    for _ in range(30):
        net.step()
    assert net.g[1] == 0.0


def test_inhibition_and_silence():
    W = np.zeros((3, 3)); W[0, 2] = 60; W[1, 2] = -60
    net = _net(W)
    net.stimulate([0], 150.0)
    net.run(3000)
    r_exc = net.rates()[2]
    net2 = _net(W)
    net2.stimulate([0, 1], 150.0)
    net2.run(3000)
    assert net2.rates()[2] < r_exc
    net3 = _net(W)
    net3.silence([0])
    net3.stimulate([0], 150.0)
    net3.run(1000)
    assert net3.rates()[2] == 0


def test_adaptation_reduces_rate():
    W = np.zeros((2, 2)); W[0, 1] = 80
    base = _net(W); base.stimulate([0], 150.0); base.run(5000)
    ad = _net(W, adapt_inc=2.0, adapt_tau=100.0); ad.stimulate([0], 150.0); ad.run(5000)
    assert ad.rates()[1] < base.rates()[1]
