"""시각화: 궤적, 시계열, 활성화 실험 결과."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .world import World

PATCH_COLORS = {"sugar": "#e9b949", "bitter": "#7b4fa3", "salt_low": "#4f8fc0", "water": "#6cc3d5"}
for _f in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False


def draw_world(ax, world: World):
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    # 빛/온도 배경
    if world.light.kind != "uniform" or world.thermal:
        xs = np.linspace(0, world.width, 120)
        ys = np.linspace(0, world.height, 80)
        X, Y = np.meshgrid(xs, ys)
        if world.thermal:
            T = sum(np.vectorize(z.at)(X, Y) for z in world.thermal)
            ax.imshow(T, extent=(0, world.width, 0, world.height), origin="lower", cmap="coolwarm", vmin=-1, vmax=1, alpha=0.35)
        else:
            L = np.vectorize(world.light.at)(X, Y)
            ax.imshow(L, extent=(0, world.width, 0, world.height), origin="lower", cmap="gray", alpha=0.3)
    for s in world.odor_sources:
        for k, a in ((1, 0.25), (2, 0.15), (3, 0.08)):
            ax.add_patch(plt.Circle((s.x, s.y), s.sigma * k * 0.6, color="#3a9d5d", alpha=a, lw=0))
        ax.plot(s.x, s.y, marker="*", color="#1f6b3a", ms=12)
        ax.annotate(s.odorant if isinstance(s.odorant, str) else "odor", (s.x, s.y), xytext=(4, 6), textcoords="offset points", fontsize=8)
    kinds_at = {}
    for p in world.patches:
        kinds_at.setdefault((p.x, p.y, p.radius), []).append(p.kind)
    for (x, y, r), kinds in kinds_at.items():
        if len(kinds) == 1:
            ax.add_patch(plt.Circle((x, y), r, color=PATCH_COLORS.get(kinds[0], "#999"), alpha=0.45, lw=0))
        else:
            ax.add_patch(plt.Circle((x, y), r, facecolor=PATCH_COLORS.get(kinds[0], "#999"), edgecolor=PATCH_COLORS.get(kinds[1], "#555"), hatch="///", alpha=0.5, lw=1.5))
        ax.annotate("+".join(kinds), (x, y), ha="center", va="center", fontsize=8)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")


def draw_trajectory(ax, df: pd.DataFrame, world: World):
    draw_world(ax, world)
    if df.empty:
        return
    pts = ax.scatter(df.x, df.y, c=df.t, cmap="viridis", s=6, zorder=3)
    ax.plot(df.x, df.y, color="k", lw=0.4, alpha=0.4, zorder=2)
    fe = df[df.proboscis]
    if len(fe):
        ax.scatter(fe.x, fe.y, marker="v", s=18, color="#d1495b", zorder=4, label="주둥이 신전(섭식)")
    jp = df[df.jump]
    if len(jp):
        ax.scatter(jp.x, jp.y, marker="x", s=40, color="#c1121f", zorder=5, label="도약")
    ax.plot(df.x.iloc[0], df.y.iloc[0], "o", color="#2b2d42", ms=6, label="시작")
    plt.colorbar(pts, ax=ax, label="시간 (s)", fraction=0.046)
    if len(fe) or len(jp):
        ax.legend(loc="upper left", fontsize=8)


def report(df: pd.DataFrame, world: World, path: str | Path, title: str = "") -> Path:
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.1, 1])
    ax0 = fig.add_subplot(gs[:, 0])
    draw_trajectory(ax0, df, world)
    ax0.set_title(title or world.name)
    t = df.t
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, df.speed, label="속도 mm/s")
    ax1.plot(t, df.omega * 5, label="회전 rad/s ×5", alpha=0.7)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylabel("운동")
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    for c, lab in (("r_fwd_L", "전진 DN 좌"), ("r_fwd_R", "전진 DN 우"), ("r_turn_L", "회전 DN 좌"), ("r_turn_R", "회전 DN 우"), ("r_back", "MDN 후진")):
        if c in df:
            ax2.plot(t, df[c], label=lab, lw=1)
    ax2.set_ylabel("Hz")
    ax2.legend(fontsize=7, ncol=3, loc="upper right")
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    for c, lab in (("r_feed", "MN9"), ("r_feed_ing", "섭식 운동뉴런"), ("r_escape", "Giant fiber/탈출"), ("r_takeoff", "이륙 DN")):
        if c in df:
            ax3.plot(t, df[c], label=lab, lw=1)
    ax3.set_ylabel("Hz")
    ax3.legend(fontsize=7, ncol=2, loc="upper right")
    ax4 = fig.add_subplot(gs[3, 1], sharex=ax1)
    ins = [c for c in df.columns if c.startswith("in_") and c.endswith("_L") and df[c].abs().sum() > 0]
    for c in ins:
        ch = c[3:-2]
        if ch in ("light",):
            continue
        ax4.plot(t, (df[c] + df[f"in_{ch}_R"]) / 2, label=ch, lw=1)
    if "in_odor_total" in df and df.in_odor_total.sum() > 0:
        ax4.plot(t, df.in_odor_total / max(df.in_odor_total.max(), 1e-9) * 150, label="냄새(정규화)", lw=1)
    ax4.set_ylabel("감각 입력 Hz")
    ax4.set_xlabel("시간 (s)")
    if ax4.lines:
        ax4.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def activation_bar(df: pd.DataFrame, path: str | Path, top: int = 30, title: str = "") -> Path:
    d = df[~df.stimulated].head(top).iloc[::-1]
    labels = [f"{ct if isinstance(ct, str) else '?'} ({s[0] if isinstance(s, str) else '?'}) {cc if isinstance(cc, str) else ''}"
              for ct, s, cc in zip(d.cell_type, d.side, d.cell_class)]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(d) + 1)))
    ax.barh(range(len(d)), d.rate_hz, xerr=d.rate_std, color="#3d5a80", alpha=0.85)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("평균 발화율 (Hz)")
    ax.set_title(title or "가장 강하게 반응한 뉴런 (자극 뉴런 제외)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
