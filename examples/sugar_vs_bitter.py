"""배고픈 파리를 당 / 당+쓴맛 / 쓴맛 패치 위에 올려 섭식(주둥이 신전) 비율을 비교한다.
추가로 MN9 억제 대조군. 실행: uv run python examples/sugar_vs_bitter.py"""
import numpy as np
import pandas as pd

from flybrain import Atlas, ClosedLoopExperiment, load_connectome
from flybrain.scenarios import sugar_vs_bitter_mix

cx = load_connectome()
mn9 = tuple(Atlas(cx).motor("feed"))
rows = []
for seed in range(3):
    for name, world in sugar_vs_bitter_mix().items():
        conds = [(name, ())] + ([("sugar_only + MN9 억제", mn9)] if name == "sugar_only" else [])
        for label, sil in conds:
            w = sugar_vs_bitter_mix()[name]
            exp = ClosedLoopExperiment(w, cx=cx, seed=seed, silence=sil, decoder_kwargs={"spontaneous_speed": 0.0})
            df = exp.run(3.0, progress=False)
            rows.append({"조건": label, "seed": seed, "주둥이 신전 비율": df.proboscis.mean(),
                         "MN9 Hz": df.r_feed.mean(), "섭식 운동뉴런 Hz": df.r_feed_ing.mean()})
            print(rows[-1])
res = pd.DataFrame(rows).groupby("조건").mean(numeric_only=True).drop(columns="seed").round(2)
print(res.to_string())
res.to_csv("results/sugar_vs_bitter.csv")
