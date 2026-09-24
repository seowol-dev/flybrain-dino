"""공룡게임 실시간 폐회로 실행기."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .dino import DinoGame, DinoScene, LoomDecoder
from .realtime import FrameClock, VisualBrain


def run_dino(duration_s: float = 20.0, *, fps: float = 60.0, seed: int = 0,
             decoder: LoomDecoder | None = None, realtime: bool = True,
             brain: VisualBrain | None = None, silence_types: tuple = (),
             target_hz: float = 10.0, frontal_deg: float = 25.0,
             surround_deg: tuple = (35.0, 75.0), blind: bool = False,
             verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """생물시간 = 벽시계 시간으로 게임과 전뇌를 함께 돌린다.

    decoder=None 이면 도약하지 않는다(개방회로 — 루밍 지표 궤적 측정용).
    silence_types 로 뉴런 유형을 끄면 음성 대조 실험이 된다.
    blind=True 면 게임은 그대로 돌지만 파리에게는 균일 회색만 보여 준다 — 생존이
    시각 때문인지 우연한 도약 리듬 때문인지 가르는 대조군이다.
    """
    vb = brain
    if vb is None:
        from .atlas import Atlas
        from .data import load_connectome
        from .retina import frontal_groups, load_eye_maps
        from .dino import OFF_GROUPS, ON_GROUPS
        cx = load_connectome()
        atlas = Atlas(cx)
        from .dino import LAMINA_READ
        eyes = load_eye_maps(cx, atlas)
        types = tuple(dict.fromkeys(ON_GROUPS + OFF_GROUPS))
        fg = frontal_groups(cx, atlas, types, radius_deg=frontal_deg, eyes=eyes, prefix="F")
        fg.update(frontal_groups(cx, atlas, LAMINA_READ, radius_deg=surround_deg[1],
                                 inner_deg=surround_deg[0], eyes=eyes, prefix="S"))
        # 중심/주변 풀링 그룹 — 디코더가 실제로 읽는 신호
        for pre in ("F", "S"):
            parts = [v for k, v in fg.items() if k.startswith(pre + "_") and k[2:] in LAMINA_READ]
            if parts:
                fg[f"{pre}_pool"] = np.unique(np.concatenate(parts))
        sil = ()
        if silence_types:
            sil = tuple(np.concatenate([atlas.by_type(t) for t in silence_types]).tolist())
        vb = VisualBrain(cx=cx, target_hz=target_hz, seed=seed, silence=sil, extra_groups=fg)
    game = DinoGame(rng=np.random.default_rng(seed))
    scene = DinoScene(game=game)
    if blind:
        from .visual import UniformField
        scene_in = UniformField(0.72)     # 땅 밝기와 같은 균일 화면
    else:
        scene_in = scene
    clock = FrameClock(fps=fps, realtime=realtime)
    dt = 1.0 / fps
    if decoder is not None:
        decoder.reset()
    rows = []
    clock.start()
    n = int(round(duration_s * fps))
    for k in range(n):
        w0 = time.perf_counter()
        t = k * dt
        rates = vb.frame(scene_in.columns(vb.eyes, t), dt)
        jump = False
        sig = raw = float("nan")
        if decoder is not None:
            fire, raw, sig = decoder.update(rates, dt, t)
            if fire:
                jump = game.jump()
        game.step(dt)
        vis = sorted([c for c in game.cacti if 0.02 < c.dist < 8.0], key=lambda c: c.dist)[:3]
        rows.append({"t": t, "theta_deg": game.theta_nearest(),
                     "cacti": [[round(c.dist, 4), c.half_w, c.height] for c in vis],
                     "eye_h": game.eye_h,
                     "dist": game.nearest().dist if game.nearest() else np.nan,
                     "y": game.y, "speed": game.speed, "score": game.score,
                     "dead": game.dead, "jump": jump, "loom_raw": raw, "loom_sig": sig,
                     **{f"r_{a}": b for a, b in rates.items()}})
        clock.tick(w0)
        if game.dead:
            if verbose:
                print(f"[dino] 충돌 t={t:.2f}s score={game.score}", flush=True)
            break
    df = pd.DataFrame(rows)
    stats = {"score": int(game.score), "dead": bool(game.dead),
             "survived_s": float(df.t.iloc[-1]) if len(df) else 0.0,
             "jumps": int(df.jump.sum()), **clock.stats()}
    if verbose:
        print("[dino] " + clock.report(), flush=True)
    return df, stats
