#!/usr/bin/env python
"""MuJoCo 초파리 몸 모델(flybody) → 웹용 glTF.

출처: TuragaLab/flybody (Apache-2.0) 의 `flybody/fruitfly/assets/`.
      Vaxenburg et al., Whole-body physics simulation of fruit fly locomotion.
      해부학적으로 측정된 성체 Drosophila melanogaster 몸 모델이다.

MuJoCo XML 의 body 트리(pos/quat)와 geom 배치(pos/quat/mesh/material)를 그대로 읽어
계층을 유지한 채 glTF 로 내보낸다. 계층을 남겨야 브라우저에서 머리를 돌리고 날개를
퍼덕이고 앞다리로 자판을 누를 수 있다.

    uv run python tools/build_fly_glb.py /tmp/flybody/flybody/fruitfly/assets web/assets/fly.glb
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

# 몸통 좌우 대칭축 등은 MuJoCo 기준(+x 앞, +z 위)이다. 웹에서는 -z 가 앞이므로 마지막에 돌린다.
SKIP_CLASSES = {"collision"}


def quat_to_mat(q) -> np.ndarray:
    """MuJoCo 쿼터니언 (w, x, y, z) → 4x4."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(4)
    w, x, y, z = w / n, x / n, y / n, z / n
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return m


def euler_to_mat(e) -> np.ndarray:
    rx, ry, rz = e
    m = trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz")
    return m


def local_transform(el: ET.Element) -> np.ndarray:
    t = np.eye(4)
    if "quat" in el.attrib:
        t = quat_to_mat([float(v) for v in el.attrib["quat"].split()])
    elif "euler" in el.attrib:
        t = euler_to_mat([float(v) for v in el.attrib["euler"].split()])
    if "pos" in el.attrib:
        t[:3, 3] = [float(v) for v in el.attrib["pos"].split()]
    return t


def parse(asset_dir: Path, xml_name: str = "fruitfly.xml", root_matrix: np.ndarray | None = None):
    """XML → (평면화된 trimesh.Scene, 바디 트리 정보, 통계).

    계층은 glTF 노드 대신 별도 JSON 으로 넘긴다. 브라우저가 이름으로 메시를 모아
    피벗 기준 그룹으로 다시 묶는다(웹에서 머리·날개·앞다리를 움직이기 위해서다).
    """
    root = ET.parse(asset_dir / xml_name).getroot()

    default_scale = 1.0
    for d in root.iter("default"):
        for m in d.findall("mesh"):
            if "scale" in m.attrib:
                default_scale = float(m.attrib["scale"].split()[0])
    meshes: dict[str, tuple[Path, float]] = {}
    for m in root.iter("mesh"):
        if "name" in m.attrib and "file" in m.attrib:
            sc = float(m.attrib["scale"].split()[0]) if "scale" in m.attrib else default_scale
            meshes[m.attrib["name"]] = (asset_dir / m.attrib["file"], sc)

    mats: dict[str, np.ndarray] = {}
    for m in root.iter("material"):
        if "name" in m.attrib and "rgba" in m.attrib:
            mats[m.attrib["name"]] = np.array([float(v) for v in m.attrib["rgba"].split()])

    scene = trimesh.Scene()
    cache: dict[str, trimesh.Trimesh] = {}
    bodies: dict[str, dict] = {}
    stats = {"geoms": 0, "verts": 0}

    def load(name: str):
        if name not in cache:
            path, sc = meshes.get(name, (None, 1.0))
            if path is None or not path.exists():
                cache[name] = None
            else:
                g = trimesh.load(path, process=False, force="mesh")
                g.apply_scale(sc)
                cache[name] = g
        return cache[name]

    def walk(body: ET.Element, parent: str | None, world: np.ndarray):
        name = body.attrib.get("name", f"body_{id(body)}")
        w = world @ local_transform(body)
        bodies[name] = {"parent": parent, "pivot": w[:3, 3].tolist(), "geoms": []}
        for gm in body.findall("geom"):
            if gm.attrib.get("class") in SKIP_CLASSES:
                continue
            mesh_name = gm.attrib.get("mesh")
            if not mesh_name:
                continue
            g = load(mesh_name)
            if g is None:
                continue
            gg = g.copy()
            gg.apply_transform(w @ local_transform(gm))    # 월드 좌표로 굽는다
            rgba = mats.get(gm.attrib.get("material", ""),
                            mats.get("body", np.array([0.67, 0.35, 0.14, 1.0])))
            gg.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=(np.clip(rgba, 0, 1) * 255).astype(np.uint8),
                    metallicFactor=0.15, roughnessFactor=0.45,
                    alphaMode="BLEND" if rgba[3] < 0.999 else "OPAQUE"))
            gname = f"{name}__{gm.attrib.get('name', mesh_name)}"
            scene.add_geometry(gg, node_name=gname, geom_name=gname)
            bodies[name]["geoms"].append(gname)
            stats["geoms"] += 1
            stats["verts"] += len(gg.vertices)
        for child in body.findall("body"):
            walk(child, name, w)

    world_el = root.find("worldbody")
    base = root_matrix if root_matrix is not None else np.eye(4)
    for b in world_el.findall("body"):
        walk(b, None, base)
    return scene, bodies, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("assets", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--xml", default="fruitfly.xml")
    a = ap.parse_args()
    # MuJoCo(+x 앞, +z 위) → three.js(+y 위, 앞은 -z). 최상위 바디에 미리 곱한다.
    rot = trimesh.transformations.euler_matrix(-np.pi / 2, 0, np.pi / 2, "sxyz")
    scene, bodies, stats = parse(a.assets, a.xml, root_matrix=rot)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(trimesh.exchange.gltf.export_glb(scene))
    import json
    parts = a.out.with_suffix(".parts.json")
    parts.write_text(json.dumps(bodies, ensure_ascii=False), encoding="utf-8")
    print(f"바디 {len(bodies)}개 → {parts.name}")
    print(f"geom {stats['geoms']}개, 정점 {stats['verts']:,}개 → {a.out} "
          f"({a.out.stat().st_size / 1e6:.1f} MB)")
    b = scene.bounds
    print("경계:", np.round(b, 4).tolist(), " 크기:", np.round(b[1] - b[0], 4).tolist())


if __name__ == "__main__":
    main()
