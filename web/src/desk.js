// 책상 + 모니터 + 자판. 모니터 화면은 CanvasTexture 로 게임을 그린다.
import * as THREE from '../vendor/three.module.js';

export function makeDesk({ screenW = 1024, screenH = 512 } = {}) {
  const g = new THREE.Group();

  const woodMat = new THREE.MeshStandardMaterial({ color: 0x3b352d, roughness: 0.72, metalness: 0.12 });
  const top = new THREE.Mesh(new THREE.BoxGeometry(5.6, 0.10, 2.9), woodMat);
  top.position.set(0, -0.30, -0.45);
  top.receiveShadow = true;
  g.add(top);

  // 모니터
  const caseMat = new THREE.MeshStandardMaterial({ color: 0x26282f, roughness: 0.38, metalness: 0.55 });
  const mon = new THREE.Group();
  mon.position.set(0, 0.62, -1.75);
  const bezel = new THREE.Mesh(new THREE.BoxGeometry(2.55, 1.52, 0.10), caseMat);
  mon.add(bezel);
  // 베젤 테두리에 옅은 빛 — 검은 배경에서 형태가 드러나게
  const edge = new THREE.Mesh(new THREE.PlaneGeometry(2.46, 1.42),
    new THREE.MeshBasicMaterial({ color: 0x4d6a86, transparent: true, opacity: 0.22 }));
  edge.position.z = 0.052; mon.add(edge);
  const canvas = document.createElement('canvas');
  canvas.width = screenW; canvas.height = screenH;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(2.35, 1.28),
    new THREE.MeshBasicMaterial({ map: tex, toneMapped: false }));
  screen.position.z = 0.054;
  mon.add(screen);
  // 받침
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 0.55, 12), caseMat);
  neck.position.set(0, -1.02, 0.02); mon.add(neck);
  const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.44, 0.48, 0.055, 24), caseMat);
  foot.position.set(0, -1.30, 0.06); mon.add(foot);
  // 화면빛
  const glow = new THREE.SpotLight(0x9dc4ea, 3.4, 5.0, 0.72, 0.9, 2.0);
  glow.position.set(0, -0.15, 0.30);
  glow.target.position.set(0, -1.05, 2.4);
  mon.add(glow, glow.target);
  g.add(mon);

  // 자판 — 파리 앞다리가 닿는 곳
  const kb = new THREE.Group();
  kb.position.set(0, -0.228, -0.42);
  kb.rotation.x = -0.06;
  const kbBody = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.05, 0.52),
    new THREE.MeshStandardMaterial({ color: 0x24262b, roughness: 0.6, metalness: 0.25 }));
  kb.add(kbBody);
  const keyMat = new THREE.MeshStandardMaterial({ color: 0x35383f, roughness: 0.55 });
  const keys = [];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 13; c++) {
      const k = new THREE.Mesh(new THREE.BoxGeometry(0.085, 0.03, 0.085), keyMat);
      k.position.set(-0.62 + c * 0.104, 0.038, -0.16 + r * 0.105);
      kb.add(k); keys.push(k);
    }
  }
  // 스페이스바 (도약 키)
  const space = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.032, 0.085),
    new THREE.MeshStandardMaterial({ color: 0x4a4f59, roughness: 0.5, emissive: 0x000000 }));
  space.position.set(0, 0.038, 0.20);
  kb.add(space);
  g.add(kb);

  return { group: g, canvas, tex, screen, monitor: mon, keyboard: kb, spaceKey: space, keys };
}
