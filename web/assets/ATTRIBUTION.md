# 3D 모델 출처

## fly.glb — 성체 초파리 몸 모델

원본: **[TuragaLab/flybody](https://github.com/TuragaLab/flybody)** 의
`flybody/fruitfly/assets/` (MuJoCo 모델 `fruitfly.xml` + OBJ 메시 85개).

> Vaxenburg, R. et al. *Whole-body physics simulation of fruit fly locomotion.*
> Janelia Research Campus / Google DeepMind.

라이선스: **Apache License 2.0** — 전문은 이 폴더의 `flybody_LICENSE.txt` 에 있다.

### 원본에서 바꾼 것 (Apache-2.0 §4(b) 고지)

`tools/build_fly_glb.py` 가 다음을 수행했다.

1. MuJoCo XML 의 바디 트리(pos/quat)를 따라 각 geom 을 **월드 좌표로 구움**.
   관절 계층은 `fly.parts.json` (바디 67개의 부모·피벗·소속 메시) 으로 따로 내보내
   브라우저에서 복원한다.
2. 좌표계를 MuJoCo(+x 앞, +z 위) → three.js(+y 위, −z 앞) 로 회전.
3. 정점 용접 + 단순화(meshoptimizer, 비율 0.30, 오차 0.0002): 817,650 → 113737 정점, 12.4 MB → 2.6 MB.
   오차 허용치를 작게 잡아 겹눈의 **낱눈(ommatidia) 격자**가 지오메트리로 남게 했다.
4. 법선은 용량 때문에 뺐고 브라우저에서 계산한다.
5. 재질은 MuJoCo `<material>` 의 rgba 를 따르되, 복부에는 정점색으로 **가로띠**를 더했다.
   원본 모델은 몸이 단색 황갈색이지만 실제 *D. melanogaster* 는 배마디 뒤쪽이 검다.
   (형태는 건드리지 않았고 색만 더한 것이다.)

메시 형태 자체는 원본 그대로이며, 해부학적으로 측정된 성체 *Drosophila melanogaster* 다.
