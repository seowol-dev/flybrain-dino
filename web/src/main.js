import * as THREE from '../vendor/three.module.js';
import { OrbitControls } from '../vendor/OrbitControls.js';
import { EffectComposer } from '../vendor/EffectComposer.js';
import { RenderPass } from '../vendor/RenderPass.js';
import { UnrealBloomPass } from '../vendor/UnrealBloomPass.js';
import { OutputPass } from '../vendor/OutputPass.js';
import { loadRun } from './data.js';
import { Brain } from './brain.js';
import { makeFly, animateFly } from './fly.js';
import { loadFly, animateFlyModel } from './fly_model.js';
import { buildLegIK } from './ik.js';
import { makeDesk } from './desk.js';
import { drawSideView, drawFlyView } from './game2d.js';

const $ = (id) => document.getElementById(id);
// 전시 모드 타임라인 — 시점 · 해설을 시간에 맞춰 바꾼다. t 는 기록 재생 위치(초).
const SHOW = [
  { t: 0.0, cam: 0, title: '초파리가 크롬 공룡게임을 한다',
    body: '초파리 뇌 전체 — 뉴런 <b>138,639개</b>, 시냅스 <b>5,449만 개</b> — 를 그대로 시뮬레이션해서 화면을 보고 스페이스바를 누르게 했다.' },
  { t: 5.5, cam: 4, title: '이 파리가 실제로 게임을 풀고 있다',
    body: '앞다리가 자판을 짚고 있다. 도약 명령은 사람이 짠 규칙이 아니라 <b>뇌 시뮬레이션의 발화</b>에서 나온다.' },
  { t: 11.0, cam: 2, title: '점 하나가 뉴런 하나다',
    body: '위치는 전자현미경으로 측정한 <b>실제 FlyWire 좌표</b>. 밝아지는 것은 그 순간 실제로 발화한 뉴런이다.' },
  { t: 17.0, cam: 1, title: '파리가 보는 것',
    body: '겹눈 광수용체 <b>6,826개</b>에 화면이 투영된다. 다가오는 선인장은 각크기가 커지는 <b>루밍</b> 자극 — 실제 초파리가 도망칠 때 쓰는 바로 그 신호다.' },
  { t: 23.0, cam: 0, title: '눈을 가리면 즉시 죽는다',
    body: '같은 뇌, 같은 게임에서 화면만 균일한 회색으로 바꾸면 <b>6번 모두 첫 선인장에서 충돌</b>했다. 생존은 시각에서 온다.' },
  { t: 27.0, cam: 3, title: '실시간이다',
    body: '생물 시간 1초 = 벽시계 1초. Apple M5 GPU 에서 프레임당 <b>12~15 ms</b> (예산 16.7 ms).' },
];

// 파리 배치 — 모델 교체 시 여기만 만진다
// 다리 끝이 책상면(y=-0.25)에 닿도록 맞춘 값이다
// 몸길이 0.294, 날개폭 0.612, 높이 0.208 (실측). 다리 끝이 책상면에 닿도록 y 를 잡는다.
const FLY = { scale: 2.60, pos: [0.0, 0.145, -0.16], yaw: 0.04 };

const state = {
  t: 0, playing: true, speed: 1, camIdx: 0, kiosk: false,
  lastJump: -9, capIdx: -1, camLerp: 0,
};

const CAMS = [
  { pos: [1.70, 0.92, 2.55], tgt: [0.10, 0.10, -0.70], name: '전체' },
  { pos: [0.03, 0.36, 0.95], tgt: [0.0, 0.08, -1.50], name: '파리 어깨너머' },
  { pos: [2.30, 2.05, -1.55], tgt: [1.08, 1.22, -2.40], name: '뇌' },
  { pos: [-1.30, 0.16, -0.16], tgt: [0.0, 0.02, -0.16], name: '파리 옆모습' },
  { pos: [-0.62, 0.22, 0.34], tgt: [-0.10, -0.06, -0.32], name: '앞다리와 스페이스바' },
];

async function main() {
  const run = await loadRun();
  $('loadmsg').textContent = '장면 구성 중…';
  $('n_neu').textContent = run.meta.connectome.neurons.toLocaleString();
  $('n_syn').textContent = run.meta.connectome.synapses.toLocaleString();
  $('g_mark').style.left = `${(1 / 2.4) * 100}%`;   // 임계선 위치 (막대 최대 = 임계×2.4)

  const canvas = $('gl');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070c);
  scene.fog = new THREE.FogExp2(0x05070c, 0.055);

  const camera = new THREE.PerspectiveCamera(42, 2, 0.05, 200);
  camera.position.set(...CAMS[0].pos);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(...CAMS[0].tgt);
  controls.enableDamping = true; controls.dampingFactor = 0.07;
  controls.minDistance = 0.6; controls.maxDistance = 14;

  // 조명
  scene.add(new THREE.AmbientLight(0x2a3752, 1.45));
  const key = new THREE.DirectionalLight(0xdfe9ff, 1.9);
  key.position.set(3.2, 4.0, 2.6); scene.add(key);
  const rim = new THREE.DirectionalLight(0x6f96d8, 1.1);
  rim.position.set(-3.5, 1.4, -3.0); scene.add(rim);
  const fill = new THREE.PointLight(0xffd0a0, 3.2, 7, 2);
  fill.position.set(-1.5, 0.85, 1.3); scene.add(fill);

  // 책상 · 모니터 · 자판
  const desk = makeDesk();
  scene.add(desk.group);
  const sctx = desk.canvas.getContext('2d');

  // 초파리 — 실측 기반 모델(flybody). 실패하면 절차적 모델로 되돌린다.
  let fly, animate;
  try {
    fly = await loadFly();
    animate = animateFlyModel;
    fly.scale.setScalar(FLY.scale);
  } catch (e) {
    console.warn('fly.glb 로드 실패, 절차적 모델 사용:', e);
    fly = makeFly();
    animate = animateFly;
    fly.scale.setScalar(0.80);
  }
  fly.position.set(FLY.pos[0], FLY.pos[1], FLY.pos[2]);
  fly.rotation.y = FLY.yaw;
  fly.userData.baseY = FLY.pos[1];
  scene.add(fly);

  // ── 다리 IK ────────────────────────────────────────────────────────────
  // 발끝을 책상·자판에 고정한다. 몸통이 흔들려도 발이 미끄러지지 않고,
  // 앞다리는 도약 순간 스페이스바를 실제로 눌러 내린다.
  let legs = [];
  if (fly.userData.groups) {
    fly.updateMatrixWorld(true);
    legs = buildLegIK(fly);
    const S = desk.surfaces;
    for (const leg of legs) {
      leg.solver.tipWorld(leg.rest);
      const inKb = leg.rest.x > S.kb.x0 && leg.rest.x < S.kb.x1
                && leg.rest.z > S.kb.z0 && leg.rest.z < S.kb.z1;
      leg.target.copy(leg.rest);
      leg.target.y = inKb ? S.kbY : S.deskY;
      if (leg.isFront) {
        // 앞다리는 스페이스바 위에 얹는다. 자연스러운 x 는 그대로 두고 z 만 맞춘다
        // (억지로 모으면 IK 가 다리를 비틀어 어색해진다).
        const x = Math.max(S.space.x0 + 0.03, Math.min(S.space.x1 - 0.03, leg.rest.x));
        leg.target.set(x, S.space.y, S.space.z);
      }
      leg.home = leg.target.clone();
    }
    window.__legs = legs;
  }

  // 뇌 홀로그램
  const brain = new Brain(run, { scale: 1.0 });
  brain.group.position.set(1.10, 1.22, -2.40);
  brain.group.scale.setScalar(0.78);
  brain.group.rotation.y = -0.5;
  scene.add(brain.group);
  // 머리 → 뇌 연결선
  const linkGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0.12, 0.22, -0.46), new THREE.Vector3(0.70, 0.82, -1.48),
    new THREE.Vector3(1.05, 1.15, -2.36)]);
  scene.add(new THREE.Line(linkGeo, new THREE.LineBasicMaterial({
    color: 0x3f7fb8, transparent: true, opacity: 0.12 })));

  // 포스트 프로세싱
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.42, 0.62, 0.86);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  const eyeCanvas = $('eyecanvas'), ectx = eyeCanvas.getContext('2d');

  function resize() {
    const w = innerWidth, h = innerHeight;
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    bloom.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    brain.setViewport(h * renderer.getPixelRatio(), camera.fov);
  }
  addEventListener('resize', resize); resize();

  const DUR = run.meta.duration_s;
  const clock = new THREE.Clock();

  const camPos = new THREE.Vector3(), camTgt = new THREE.Vector3();
  camPos.fromArray(CAMS[0].pos); camTgt.fromArray(CAMS[0].tgt);

  function setCaption(i) {
    if (i === state.capIdx) return;
    state.capIdx = i;
    const el = $('caption');
    el.classList.remove('on');
    setTimeout(() => {
      if (i < 0) return;
      const sh = SHOW[i];
      el.querySelector('h2').textContent = sh.title;
      el.querySelector('p').innerHTML = sh.body;
      el.classList.add('on');
    }, i < 0 ? 0 : 320);
  }

  function showIndexAt(t) {
    let k = 0;
    for (let i = 0; i < SHOW.length; i++) if (t >= SHOW[i].t) k = i;
    return k;
  }

  function tick() {
    requestAnimationFrame(tick);
    const dt = Math.min(clock.getDelta(), 0.1);
    if (state.playing) state.t = (state.t + dt * state.speed) % DUR;

    const gi = Math.min(run.game.length - 1, Math.floor(state.t * run.meta.fps));
    const fr = run.game[gi];
    const ai = Math.floor(state.t * run.meta.sample_hz);
    brain.setFrame(ai);

    if (fr.jump) state.lastJump = state.t;
    const since = state.t - state.lastJump;
    const pulse = since >= 0 && since < 0.45 ? Math.exp(-since * 7) : 0;

    // 화면
    drawSideView(sctx, desk.canvas.width, desk.canvas.height, fr, state.t);
    desk.tex.needsUpdate = true;
    drawFlyView(ectx, eyeCanvas.width, eyeCanvas.height, fr);

    // 파리
    const vis = Math.min(1, (fr.rates?.F_pool ?? 0) / 25);
    animate(fly, state.t, { jumpPulse: pulse, act: vis });
    // 몸통이 움직인 뒤에 IK 를 푼다 (발끝은 그 자리에 남는다)
    if (legs.length) {
      fly.updateMatrixWorld(true);
      for (const leg of legs) {
        leg.target.copy(leg.home);
        if (leg.isFront) leg.target.y -= pulse * 0.016;     // 스페이스바를 누른다
        leg.solver.solve(leg.target);
      }
    }
    desk.spaceKey.position.y = 0.038 - pulse * 0.02;
    desk.spaceKey.material.emissive.setHex(pulse > 0.15 ? 0x2a6f97 : 0x000000);

    // 뇌 회전
    brain.group.rotation.y += dt * 0.09;

    // HUD
    $('g_t').textContent = state.t.toFixed(1) + ' s';
    $('g_score').textContent = fr.score;
    $('g_dist').textContent = fr.dist == null ? '–' : fr.dist.toFixed(2) + ' m';
    $('g_theta').innerHTML = fr.theta.toFixed(1) + '<small>°</small>';
    const sig = fr.sig ?? 0;
    $('g_sig').textContent = sig.toFixed(2);
    const TH = run.meta.threshold, FULL = TH * 2.4;
    $('g_fill').style.width = `${Math.max(0, Math.min(1, sig / FULL)) * 100}%`;
    $('loom').classList.toggle('fire', sig > TH);
    if ((state.frameNo = (state.frameNo | 0) + 1) % 6 === 0) {
      let n = 0; for (let i = 0; i < brain.N; i++) if (brain.act[i] > 0.12) n++;
      $('g_act').textContent = n.toLocaleString();
    }
    if (pulse > 0.1) $('g_cap').innerHTML = '<b style="color:var(--hot)">⬆︎ 도약 — 스페이스바</b>';
    else if (state.frameNo % 30 === 0) $('g_cap').textContent = '임계를 넘으면 도약 — 실제 초파리의 도피 반사와 같은 자극이다.';
    $('seek').value = String((state.t / DUR) * 1000);

    if (state.kiosk) {
      const k = showIndexAt(state.t);
      setCaption(k);
      const c = CAMS[SHOW[k].cam];
      camPos.lerp(new THREE.Vector3().fromArray(c.pos), 1 - Math.exp(-dt * 1.1));
      camTgt.lerp(new THREE.Vector3().fromArray(c.tgt), 1 - Math.exp(-dt * 1.1));
      // 아주 느린 공전으로 정지 화면처럼 보이지 않게
      const w = state.t * 0.16;
      camera.position.set(camPos.x + Math.sin(w) * 0.16, camPos.y + Math.sin(w * 0.7) * 0.06,
                          camPos.z + Math.cos(w) * 0.16);
      controls.target.copy(camTgt);
    }
    controls.update();
    composer.render();
  }

  // 컨트롤
  $('play').onclick = () => {
    state.playing = !state.playing;
    $('play').textContent = state.playing ? '일시정지' : '재생';
  };
  $('seek').oninput = (e) => { state.t = (+e.target.value / 1000) * DUR; };
  $('cam').onclick = () => {
    state.camIdx = (state.camIdx + 1) % CAMS.length;
    const c = CAMS[state.camIdx];
    camera.position.set(...c.pos); controls.target.set(...c.tgt);
  };
  $('kiosk').onclick = () => {
    state.kiosk = !state.kiosk;
    document.body.classList.toggle('kiosk', state.kiosk);
    controls.enabled = !state.kiosk;
    if (state.kiosk) {
      state.playing = true;
      $('play').textContent = '일시정지';
      camPos.copy(camera.position); camTgt.copy(controls.target);
    } else setCaption(-1);
  };
  // ?kiosk=1 로 열면 바로 전시 모드 (키오스크 부팅용)
  if (new URLSearchParams(location.search).has('kiosk')) $('kiosk').click();
  $('full').onclick = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen();
  };
  addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
    if (e.key === 'c') $('cam').click();
    if (e.key === 'e') $('kiosk').click();
    if (e.key === 'f') $('full').click();
  });

  window.__state = state;   // 스크린샷/자동화 도구용
  window.__run = run;
  window.__gfx = { renderer, scene, camera, controls, bloom, composer, fly, brain, desk, key, rim, fill };
  $('loading').classList.add('done');
  tick();
}

main().catch((e) => {
  $('loadmsg').innerHTML = '오류: ' + e.message;
  console.error(e);
});
