#!/usr/bin/env python
"""전시 화면 스크린샷 도구 (개발/QA용).

    uv run python tools/shot.py out.png --t 12.5 --cam 1 --size 1600x900

web/ 을 임시 서버로 띄우고 headless Chromium 으로 렌더한 뒤 PNG 로 저장한다.
콘솔 오류가 있으면 그대로 출력한다.
"""
from __future__ import annotations

import argparse
import contextlib
import http.server
import socketserver
import threading
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "web"


@contextlib.contextmanager
def serve(directory: Path, port: int = 0):
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            yield httpd.server_address[1]
        finally:
            httpd.shutdown()


def shoot(out: Path, t: float | None, cam: int, size: tuple[int, int], kiosk: bool,
          wait_ms: int, selector: str | None):
    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    with serve(ROOT) as port, sync_playwright() as p:
        browser = p.chromium.launch(args=[
            "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
            "--disable-gpu", "--use-gl=angle",
        ])
        page = browser.new_page(viewport={"width": size[0], "height": size[1]},
                                device_scale_factor=1)
        page.on("console", lambda m: errors.append(f"[{m.type}] {m.text}") if m.type in ("error",) else None)
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
        page.goto(f"http://127.0.0.1:{port}/", wait_until="load")
        page.wait_for_function("() => window.__state !== undefined", timeout=90_000)
        page.wait_for_timeout(1200)
        if t is not None:
            page.evaluate("v => { window.__state.t = v; window.__state.playing = false; }", t)
        for _ in range(cam):
            page.click("#cam")
        if kiosk:
            page.click("#kiosk")
        page.wait_for_timeout(wait_ms)
        target = page.locator(selector) if selector else page
        target.screenshot(path=str(out), timeout=180_000, animations="disabled")
        browser.close()
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--t", type=float, default=None, help="재생 위치(초)로 고정")
    ap.add_argument("--cam", type=int, default=0, help="시점 버튼 누르는 횟수")
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--kiosk", action="store_true")
    ap.add_argument("--wait", type=int, default=900)
    ap.add_argument("--selector", default=None)
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.split("x"))
    errs = shoot(a.out, a.t, a.cam, (w, h), a.kiosk, a.wait, a.selector)
    print(f"저장: {a.out}")
    for e in errs[:20]:
        print(" ", e)


if __name__ == "__main__":
    main()
