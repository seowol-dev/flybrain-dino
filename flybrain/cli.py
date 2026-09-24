"""flybrain 명령줄 도구.

  flybrain setup                         데이터 내려받기 + 캐시 생성
  flybrain info                          커넥톰 요약 + 감각/운동 그룹 표
  flybrain search DNa02                  뉴런 검색 (정규식)
  flybrain validate                      Shiu et al. 2024 핵심 결과(당 GRN → MN9) 재현 확인
  flybrain activate --group sugar --side left [--silence-type ...]   개방회로 활성화 실험
  flybrain run feeding --duration 20     가상 세계 폐회로 실험 (시나리오: feeding, odor, wind, looming, touch, thermal)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .config import PATHS


def _load():
    from .atlas import Atlas
    from .data import load_connectome
    cx = load_connectome()
    return cx, Atlas(cx)


def cmd_setup(a):
    from .data import build_cache
    cx = build_cache(force=a.force)
    print(cx.summary())


def cmd_info(a):
    cx, at = _load()
    print(cx.summary())
    print()
    print(at.group_table().to_string(index=False))
    print(f"\n후각 사구체(ORN 유형): {len(at.glomeruli())}개 — {', '.join(at.glomeruli())}")


def cmd_search(a):
    _, at = _load()
    df = at.search(a.pattern, limit=a.limit)
    print(df.to_string(index=False) if len(df) else "(없음)")


def _resolve(at, cx, groups=(), types=(), roots=(), side=None):
    idx = []
    for g in groups or ():
        try:
            idx.append(at.sensory(g, side))
        except KeyError:
            idx.append(at.motor(g, side))
    for t in types or ():
        if t.startswith("ORN_"):
            idx.append(at.select(side=side, cell_type=t))
        else:
            idx.append(at.by_type(t, side))
    if roots:
        idx.append(cx.index_of([int(r) for r in roots]))
    return np.unique(np.concatenate(idx)) if idx else np.zeros(0, int)


def cmd_activate(a):
    from .experiment import ActivationExperiment
    from .lif import LIFParams
    from .plots import activation_bar
    cx, at = _load()
    exc = _resolve(at, cx, a.group, a.type, a.root_id, a.side)
    sil = _resolve(at, cx, a.silence_group, a.silence_type, a.silence_root_id, None)
    if len(exc) == 0:
        sys.exit("자극할 뉴런이 없습니다. --group / --type / --root-id 를 지정하세요.")
    print(f"자극 뉴런 {len(exc)}개 @ {a.rate} Hz, 억제 뉴런 {len(sil)}개, {a.trials}회 × {a.duration} ms, preset={a.preset}")
    t0 = time.time()
    df = ActivationExperiment(cx, LIFParams(), preset=a.preset).run(exc, sil, rate=a.rate, duration_ms=a.duration, n_trials=a.trials, seed=a.seed)
    print(f"완료 {time.time() - t0:.1f}s — 발화한 뉴런 {len(df):,}개")
    cols = ["root_id", "super_class", "cell_class", "cell_type", "side", "rate_hz", "rate_std"]
    print(df[~df.stimulated].head(a.top)[cols].round(2).to_string(index=False))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.name or "activate_" + "_".join((a.group or []) + (a.type or []))[:60]
    df.to_csv(out / f"{tag}.csv", index=False)
    activation_bar(df, out / f"{tag}.png", title=tag)
    print(f"\n저장: {out / (tag + '.csv')}\n      {out / (tag + '.png')}")


def cmd_validate(a):
    from .experiment import ActivationExperiment
    cx, at = _load()
    # Shiu et al. 2024 예제의 당 GRN 21개 (v630 ID; 20개가 v783 에 그대로 존재)
    sugar = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
             720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
             720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
             720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
             720575940611875570]
    ids = [i for i in sugar if i in set(cx.neurons.root_id)]
    df = ActivationExperiment(cx, preset="shiu2024").run(cx.index_of(ids), n_trials=a.trials, progress=False)
    mn9 = df[df.cell_type == "CB0701"]
    print(f"당 GRN {len(ids)}개 150 Hz 자극 → 발화한 뉴런 {len(df)}개")
    print(mn9[["root_id", "side", "rate_hz", "rate_std"]].round(1).to_string(index=False))
    ref = {720575940660219265: 79.5, 720575940618238523: 60.75}
    ok = True
    for rid, r_ref in ref.items():
        r = float(df.loc[df.root_id == rid, "rate_hz"].sum())
        good = abs(r - r_ref) < 15
        ok &= good
        print(f"  MN9 {rid}: {r:.1f} Hz (Brian2 원본 {r_ref} Hz) {'OK' if good else 'MISMATCH'}")
    print("검증 통과" if ok else "검증 실패")
    sys.exit(0 if ok else 1)


def cmd_run(a):
    from .experiment import ClosedLoopExperiment
    from .plots import report
    from .replay import write_replay
    from .scenarios import SCENARIOS
    if a.scenario not in SCENARIOS:
        sys.exit(f"알 수 없는 시나리오 {a.scenario!r}. 가능: {', '.join(SCENARIOS)}")
    kw = {"odorant": a.odorant} if a.scenario == "odor" else {}
    world, opts = SCENARIOS[a.scenario](**kw)
    cx, at = _load()
    silence = _resolve(at, cx, a.silence_group, a.silence_type, a.silence_root_id, None)
    excite = {int(i): a.excite_rate for i in _resolve(at, cx, a.excite_group, a.excite_type, a.excite_root_id, None)}
    print(world.describe())
    exp = ClosedLoopExperiment(world, cx=cx, preset=a.preset, seed=a.seed, control_dt_ms=a.control_dt,
                               silence=tuple(silence), excite=excite, **opts)
    t0 = time.time()
    df = exp.run(a.duration)
    tag = a.name or f"{a.scenario}_{a.preset}_s{a.seed}"
    out = Path(a.out)
    exp.save(out, tag)
    report(df, world, out / f"{tag}.png", title=f"{world.name} · preset={a.preset} · seed={a.seed}")
    write_replay(df, world, out / f"{tag}.html")
    f = world.fly
    print(f"\n완료 {time.time() - t0:.0f}s (생물 시간 {a.duration}s)")
    print(f"최종 위치 ({f.x:.1f}, {f.y:.1f}) mm, 이동거리 {np.hypot(np.diff(df.x), np.diff(df.y)).sum():.1f} mm, "
          f"섭식 {f.energy:.2f}, 주둥이 신전 비율 {df.proboscis.mean():.0%}, 도약 {f.jumps}회")
    print(f"저장: {out / (tag + '.png')}\n      {out / (tag + '.html')}  (브라우저에서 재생)\n      {out / (tag + '_trajectory.csv')}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="flybrain", description="FlyWire 초파리 전뇌 시뮬레이터")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="데이터 내려받기 + 캐시")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("info", help="커넥톰 요약")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("search", help="뉴런 검색")
    s.add_argument("pattern")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("validate", help="Shiu 2024 당→MN9 재현 확인")
    s.add_argument("--trials", type=int, default=8)
    s.set_defaults(fn=cmd_validate)

    def add_sel(s, prefix="", help_=""):
        s.add_argument(f"--{prefix}group", nargs="*", default=[], help=f"{help_}기능 그룹 (예: sugar bitter orn wind walk_backward)")
        s.add_argument(f"--{prefix}type", nargs="*", default=[], help=f"{help_}cell_type (예: DNa02 MDN ORN_DM1)")
        s.add_argument(f"--{prefix}root-id", nargs="*", default=[], help=f"{help_}FlyWire root_id")

    s = sub.add_parser("activate", help="개방회로 활성화/억제 실험")
    add_sel(s)
    add_sel(s, "silence-", "억제할 ")
    s.add_argument("--side", choices=["left", "right"], default=None)
    s.add_argument("--rate", type=float, default=150.0)
    s.add_argument("--duration", type=float, default=1000.0, help="ms")
    s.add_argument("--trials", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--preset", choices=["shiu2024", "curated"], default="shiu2024")
    s.add_argument("--top", type=int, default=30)
    s.add_argument("--name", default="")
    s.add_argument("--out", default=str(PATHS["results"]))
    s.set_defaults(fn=cmd_activate)

    s = sub.add_parser("run", help="가상 세계 폐회로 실험")
    s.add_argument("scenario")
    s.add_argument("--duration", type=float, default=10.0, help="생물 시간(초)")
    s.add_argument("--preset", choices=["shiu2024", "curated"], default="curated")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--control-dt", type=float, default=50.0, help="뇌→몸 업데이트 주기 (ms)")
    s.add_argument("--odorant", default="vinegar")
    add_sel(s, "silence-", "억제할 ")
    add_sel(s, "excite-", "지속 활성화할 ")
    s.add_argument("--excite-rate", type=float, default=150.0)
    s.add_argument("--name", default="")
    s.add_argument("--out", default=str(PATHS["results"]))
    s.set_defaults(fn=cmd_run)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
