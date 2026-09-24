# 전시용 웹 시각화

초파리 전뇌 시뮬레이션이 크롬 공룡게임을 하는 기록을 Three.js 로 재생한다.
빌드 단계가 없고 정적 파일만 있으므로 아무 정적 호스팅에나 올리면 된다.

```bash
cd web && python3 -m http.server 8000     # http://localhost:8000
```

## 무엇이 실제 데이터인가

| 화면 요소 | 출처 |
|---|---|
| 뇌 점구름의 위치 | FlyWire v783 의 실제 soma 좌표 (138,639개 중 118,086개에 좌표가 있다) |
| 점이 밝아지는 순간 | LIF 시뮬레이션에서 그 뉴런이 실제로 발화한 시점 (20 Hz 표본) |
| 모니터의 게임 화면 | 폐회로 실행 중의 실제 게임 상태 |
| 파리 시점 패널 | 광수용체에 투영된 장면을 같은 기하로 다시 그린 것 |
| 초파리 3D 모델 | **TuragaLab/flybody** 의 해부학적 성체 *D. melanogaster* 몸 모델 (Apache-2.0). 날개 시맥·강모·부절 마디까지 실제 형태다. 출처와 변경 내역은 [`assets/ATTRIBUTION.md`](assets/ATTRIBUTION.md) |

## 조작

| 키 | 동작 |
|---|---|
| `Space` | 재생 / 일시정지 |
| `c` | 시점 전환 (전체 · 어깨너머 · 뇌 · 옆모습 · 클로즈업) |
| `e` | 전시 모드 (해설 + 자동 시점 전환) |
| `f` | 전체화면 |
| 드래그 / 휠 | 궤도 회전 · 확대 |

`?kiosk=1` 을 붙여 열면 전시 모드로 바로 시작한다. 키오스크 부팅 스크립트에 쓴다.

```
https://<user>.github.io/<repo>/?kiosk=1
```

## 3D 모델 다시 만들기

```bash
git clone https://github.com/TuragaLab/flybody /tmp/flybody
uv run python tools/build_fly_glb.py /tmp/flybody/flybody/fruitfly/assets /tmp/fly_raw.glb
npx @gltf-transform/cli@4 weld /tmp/fly_raw.glb /tmp/fly_weld.glb
npx @gltf-transform/cli@4 simplify /tmp/fly_weld.glb web/assets/fly.glb --ratio 0.30 --error 0.0002
cp /tmp/fly_raw.parts.json web/assets/fly.parts.json
```

## 데이터 다시 만들기

```bash
uv run python -c "from flybrain.export_web import export_run; export_run(duration_s=30, seed=2)"
```

`web/data/` 에 다음이 생긴다.

| 파일 | 내용 | 크기 |
|---|---|---|
| `positions.bin` | 표본 뉴런 17,008개의 좌표 (int16 ×3) | 100 KB |
| `backdrop.bin` | 나머지 101,078개 좌표 (구조 배경용) | 592 KB |
| `activity.bin` | 프레임(600) × 뉴런(17,008) 발화 수 (uint8) | 9.7 MB |
| `class.bin` | 뉴런별 super_class 번호 | 17 KB |
| `game.json` | 60 fps 게임 상태 1,800 프레임 | 365 KB |
| `meta.json` | 메타데이터, 그룹 인덱스, 통계 | 10 KB |

## 화면 캡처

```bash
uv run python tools/shot.py out.png --t 22 --cam 1 --size 1600x900 [--kiosk]
```

headless Chromium 으로 렌더해 PNG 로 저장한다. 콘솔 오류도 함께 출력한다.
