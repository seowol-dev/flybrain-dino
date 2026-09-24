"""웹 시각화용 기록 내보내기.

공룡게임 폐회로를 한 번 돌리면서
  1) 게임 상태(프레임별)
  2) 뉴런별 발화(표본 집합, 낮은 표본율로 양자화)
  3) 뉴런 3D 좌표(FlyWire soma)
를 브라우저가 바로 읽을 수 있는 형식으로 저장한다.

뉴런 좌표는 실제 FAFB/FlyWire 좌표다(복셀 4,4,40 nm). 138,639개 중 118,086개(85%)에
좌표가 있고, 하행뉴런 1,299개와 운동뉴런 110개는 전부 있다.

크기 예산: 표본 뉴런 N × 프레임 F 바이트. 기본값(16k 뉴런 × 30초 × 20 Hz)이면 약 9.8 MB.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

VOXEL_UM = np.array([4e-3, 4e-3, 40e-3])

# 표본 배분 — 시각적으로 시엽이 살아 있어야 하고, 출력 뉴런은 전부 넣는다
SAMPLE_PLAN = {
    "descending": None,          # None = 전부
    "motor": None,
    "endocrine": None,
    "visual_centrifugal": None,
    "visual_projection": 3000,
    "optic": 8000,
    "central": 4000,
}


def choose_neurons(cx, rng: np.random.Generator) -> np.ndarray:
    n = cx.neurons
    xyz = n[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    ok = np.isfinite(xyz).all(axis=1)
    out = []
    for sc, k in SAMPLE_PLAN.items():
        idx = np.flatnonzero((n.super_class == sc).fillna(False).to_numpy(bool) & ok)
        if len(idx) == 0:
            continue
        out.append(idx if k is None or k >= len(idx) else rng.choice(idx, k, replace=False))
    return np.sort(np.concatenate(out))


def _norm_positions(xyz_um: np.ndarray) -> tuple[np.ndarray, dict]:
    """µm 좌표 → int16. 뇌 중심을 원점으로, 최대 반경을 10000 으로 맞춘다."""
    c = xyz_um.mean(axis=0)
    d = xyz_um - c
    scale = float(np.abs(d).max())
    q = np.clip(np.round(d / scale * 10000), -32000, 32000).astype(np.int16)
    return q, {"center_um": c.tolist(), "scale_um": scale, "quant": 10000}


def export_run(out_dir: str | Path = "web/data", *, duration_s: float = 30.0, seed: int = 2,
               fps: float = 60.0, sample_hz: float = 20.0, threshold: float = 1.8,
               tau_fast: float = 0.06, frontal_deg: float = 25.0, target_hz: float = 10.0,
               verbose: bool = True) -> dict:
    import mlx.core as mx

    from .atlas import Atlas
    from .data import load_connectome
    from .dino import LAMINA_READ, OFF_GROUPS, ON_GROUPS, DinoGame, DinoScene, LoomDecoder
    from .realtime import VisualBrain
    from .retina import frontal_groups, load_eye_maps

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    cx = load_connectome()
    atlas = Atlas(cx)
    eyes = load_eye_maps(cx, atlas)
    types = tuple(dict.fromkeys(ON_GROUPS + OFF_GROUPS))
    fg = frontal_groups(cx, atlas, types, radius_deg=frontal_deg, eyes=eyes, prefix="F")
    fg.update(frontal_groups(cx, atlas, LAMINA_READ, radius_deg=75.0, inner_deg=35.0,
                             eyes=eyes, prefix="S"))
    for pre in ("F", "S"):
        parts = [v for k, v in fg.items() if k.startswith(pre + "_") and k[2:] in LAMINA_READ]
        if parts:
            fg[f"{pre}_pool"] = np.unique(np.concatenate(parts))

    vb = VisualBrain(cx=cx, target_hz=target_hz, seed=seed, extra_groups=fg)
    sample = choose_neurons(cx, rng)
    gather = mx.array(sample.astype(np.int64))
    if verbose:
        print(f"[export] 표본 뉴런 {len(sample):,}개", flush=True)

    game = DinoGame(rng=np.random.default_rng(seed))
    scene = DinoScene(game=game)
    dec = LoomDecoder(threshold=threshold, tau_fast_s=tau_fast)
    dt = 1.0 / fps
    every = max(1, int(round(fps / sample_hz)))
    n_frames = int(round(duration_s * fps))

    acts: list[np.ndarray] = []
    frames: list[dict] = []
    prev = np.zeros(len(sample), np.int64)
    t0 = time.perf_counter()
    for k in range(n_frames):
        t = k * dt
        rates = vb.frame(scene.columns(vb.eyes, t), dt)
        fire, raw, sig = dec.update(rates, dt, t)
        jump = game.jump() if fire else False
        game.step(dt)
        vis = sorted([c for c in game.cacti if 0.02 < c.dist < 9.0], key=lambda c: c.dist)[:3]
        frames.append({
            "t": round(t, 4), "y": round(game.y, 4), "score": game.score,
            "dist": round(game.nearest().dist, 4) if game.nearest() else None,
            "theta": round(game.theta_nearest(), 3), "jump": bool(jump),
            "dead": bool(game.dead), "sig": round(float(sig), 4),
            "speed": round(game.speed, 3),
            "cacti": [[round(c.dist, 3), round(c.half_w, 3), round(c.height, 3)] for c in vis],
            "rates": {g: round(float(rates.get(g, 0.0)), 2) for g in ("F_pool", "S_pool")},
        })
        if k % every == 0:
            cnt = np.array(mx.take(vb.net.cnt, gather)).astype(np.int64)
            acts.append(np.clip(cnt - prev, 0, 255).astype(np.uint8))
            prev = cnt
        if game.dead:
            break
    wall = time.perf_counter() - t0

    act = np.stack(acts)                       # (F, N) uint8
    xyz = cx.neurons[["soma_x", "soma_y", "soma_z"]].to_numpy(float)[sample] * VOXEL_UM
    pos, pmeta = _norm_positions(xyz)

    # 배경 점구름: 표본에 없는 나머지 좌표 보유 뉴런
    allxyz = cx.neurons[["soma_x", "soma_y", "soma_z"]].to_numpy(float) * VOXEL_UM
    okall = np.isfinite(allxyz).all(axis=1)
    rest = np.setdiff1d(np.flatnonzero(okall), sample)
    bpos = np.clip(np.round((allxyz[rest] - np.array(pmeta["center_um"])) / pmeta["scale_um"] * 10000),
                   -32000, 32000).astype(np.int16)

    n = cx.neurons
    sc = n.super_class.to_numpy(object)[sample]
    classes = sorted({str(x) for x in sc})
    cls_id = np.array([classes.index(str(x)) for x in sc], np.uint8)
    # 눈에 띄게 표시할 뉴런들
    marks = {name: np.flatnonzero(np.isin(sample, atlas.by_type(name))).tolist()
             for name in ("DNp01", "DNp11", "LPLC2", "LC4", "T4a", "T5a", "Mi1", "L1", "L2")}
    pools = {k: np.flatnonzero(np.isin(sample, v)).tolist() for k, v in fg.items()
             if k in ("F_pool", "S_pool")}

    (out / "positions.bin").write_bytes(pos.tobytes())
    (out / "backdrop.bin").write_bytes(bpos.tobytes())
    (out / "activity.bin").write_bytes(act.tobytes())
    (out / "class.bin").write_bytes(cls_id.tobytes())
    meta = {
        "n_neurons": int(len(sample)), "n_backdrop": int(len(rest)),
        "n_frames_game": len(frames), "n_frames_act": int(act.shape[0]),
        "fps": fps, "sample_hz": sample_hz, "act_every": every,
        "classes": classes, "marks": marks, "pools": pools,
        "position": pmeta, "threshold": threshold,
        "score": int(game.score), "dead": bool(game.dead),
        "duration_s": round(frames[-1]["t"], 3) if frames else 0.0,
        "wall_s": round(wall, 2),
        "connectome": {"neurons": int(cx.n), "connections": int(cx.n_connections),
                       "synapses": int(cx.n_synapses), "version": "FlyWire v783"},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (out / "game.json").write_text(json.dumps(frames, ensure_ascii=False), encoding="utf-8")
    if verbose:
        mb = sum((out / f).stat().st_size for f in
                 ("positions.bin", "backdrop.bin", "activity.bin", "class.bin", "game.json")) / 1e6
        print(f"[export] {len(frames)} 프레임 · 선인장 {game.score}개 · {'충돌' if game.dead else '완주'} "
              f"· 활동 {act.shape} · 총 {mb:.1f} MB · {wall:.1f}s", flush=True)
    return meta
