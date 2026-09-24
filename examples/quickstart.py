"""파이썬 API 빠른 예제: 커스텀 가상 세계 만들고 10초 실험.
실행: uv run python examples/quickstart.py"""
from flybrain import ClosedLoopExperiment, Fly, World
from flybrain.plots import report
from flybrain.replay import write_replay
from flybrain.world import OdorSource, Patch, TimedStimulus

world = World(
    width=100, height=100, name="quickstart",
    fly=Fly(x=20, y=50, heading=0.0),
    patches=[Patch(45, 50, radius=8, kind="sugar")],
    odor_sources=[OdorSource(80, 80, odorant="yeast", sigma=20)],
    events=[TimedStimulus("wind", 7.0, 7.5, side="left")],   # 7초에 바람 → 탈출
)
exp = ClosedLoopExperiment(world, preset="curated", seed=1)
df = exp.run(10.0)
report(df, world, "results/quickstart.png")
write_replay(df, world, "results/quickstart.html")
print(df[["t", "x", "y", "speed", "proboscis", "jump", "r_feed", "r_escape"]].iloc[::20].round(2))
