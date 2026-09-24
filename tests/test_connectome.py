"""실제 커넥톰이 필요한 통합 테스트 (데이터가 없으면 건너뜀)."""
import numpy as np
import pytest

from flybrain.config import PATHS

pytestmark = pytest.mark.skipif(not PATHS["cache_npz"].exists(), reason="flybrain setup 먼저 실행")


@pytest.fixture(scope="module")
def cx():
    from flybrain.data import load_connectome
    return load_connectome()


def test_sizes(cx):
    assert cx.n == 138639
    assert cx.n_connections == 15091983
    assert cx.n_synapses == 54492922


def test_groups(cx):
    from flybrain.atlas import Atlas
    a = Atlas(cx)
    assert len(a.sensory("sugar")) == 129
    assert len(a.motor("feed")) == 2
    assert len(a.glomeruli()) == 53


def test_presets(cx):
    from flybrain.data import weights
    assert weights(cx, "shiu2024") is cx.W_pre
    Wc = weights(cx, "curated")
    assert Wc.shape == cx.W_pre.shape and Wc.nnz < cx.W_pre.nnz


def test_sugar_drives_mn9(cx):
    from flybrain.atlas import Atlas
    from flybrain.lif import LIFNetwork
    a = Atlas(cx)
    net = LIFNetwork(cx.W_pre, seed=0)
    net.stimulate(a.sensory("sugar", "left"), 150.0)
    net.run(300)
    assert net.rates(0, 300)[a.motor("feed")].mean() > 30
