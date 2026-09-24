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
| 초파리 3D 모델 | **TuragaLab/flybody** 의 해부학적 성체 *D. melanogaster* 몸 모델 (Apache-2.0). 날개 시맥·강모·부절 마디·겹눈 낱눈까지 실제 형태다. 출처와 변경 내역은 [`assets/ATTRIBUTION.md`](assets/ATTRIBUTION.md) |
| 초파리 표면 질감 | 지오메트리에서 계산한 것이지 사진 텍스처가 아니다. 아래 참고 |

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

## 표면 질감을 어떻게 만들었나

이 모델에는 UV 가 없어서 이미지 텍스처를 입힐 수 없다. 대신 `src/fly_detail.js` 가
**지오메트리 자체에서** 세 가지를 계산해 정점색(COLOR_0)으로 굽는다.

1. **요철(cavity)** — 이웃 정점들의 평균 위치가 법선 안쪽에 있으면 오목한 곳이다.
   낱눈 사이 골, 배마디 틈, 강모 뿌리가 어두워진다. 국소 모서리 길이로 정규화하므로
   메시 크기와 무관하다(메시 반지름으로 나누면 흉부 같은 큰 면에서 값이 0 으로 죽는다).
2. **반점(mottle)** — 3D 값잡음. 실제 큐티클은 균일한 플라스틱이 아니다.
3. **부위별 음영** — 등쪽(+y)이 어둡고, 복부는 마디 뒤쪽이 검다(실제 *D. melanogaster*).

그리고 `fly_model.js` 의 `roughnessFromColor()` 가 셰이더에 한 줄을 주입해
**정점색 밝기로 거칠기를 변조**한다. 골은 거칠고 융기는 매끈하다. 표면이 진짜로 보이는
데는 알베도보다 광택 변화가 더 크게 기여한다.

겹눈은 clearcoat 1.0 / clearcoatRoughness 0.04 로 각막 렌즈의 젖은 반사를, 날개는
iridescence 로 얇은 막 간섭(무지개빛)을 만든다.

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
