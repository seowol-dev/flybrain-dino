// 크롬 공룡게임 2D 렌더러 — 모니터 화면 텍스처용(옆모습)과 파리 시점(1인칭).
const DINO = [
  '..........XXXXXXX',
  '..........XXXXXXXX',
  '..........XXX.XXXX',
  '..........XXXXXXXX',
  '..........XXXXX...',
  '..........XXXXXXX.',
  '.X........XXXX....',
  'XX......XXXXXX....',
  'XXX....XXXXXXXX...',
  'XXXX..XXXXXXXXX...',
  'XXXXXXXXXXXXXXX...',
  '.XXXXXXXXXXXXXX...',
  '..XXXXXXXXXXXX....',
  '...XXXXXXXXXXX....',
  '....XXXXXXXXX.....',
  '.....XXXXXXX......',
  '.....XXX.XXX......',
  '.....XX...XX......',
  '.....XX...XX......',
  '.....XXX..XXX.....',
];
const CACTUS = [
  '..XX..',
  '..XX..',
  'X.XX..',
  'X.XX.X',
  'X.XX.X',
  'XXXXXX',
  '..XX.X',
  '..XX..',
  '..XX..',
  '..XX..',
  '..XX..',
  '..XX..',
];

function blit(ctx, art, x, y, px, color) {
  ctx.fillStyle = color;
  for (let r = 0; r < art.length; r++) {
    const row = art[r];
    for (let c = 0; c < row.length; c++) {
      if (row[c] === 'X') ctx.fillRect(x + c * px, y + r * px, px, px);
    }
  }
}

/** 옆모습(사람이 보는 크롬 공룡게임). frame 은 export_web 의 game.json 한 항목. */
export function drawSideView(ctx, W, H, frame, tSec, { dark = true } = {}) {
  // 크롬 공룡게임의 다크 모드 배색 — 전시 조명에서 흰 화면은 눈을 찌른다
  const BG = dark ? '#16181d' : '#f7f7f7';
  const FG = dark ? '#c9ccd4' : '#535353';
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);
  const px = Math.max(2, Math.round(H / 60));
  const ground = Math.round(H * 0.74);
  // 땅
  ctx.fillStyle = FG;
  ctx.fillRect(0, ground, W, Math.max(1, px * 0.5));
  const off = (tSec * 140) % 40;
  for (let x = -off; x < W; x += 40) ctx.fillRect(x, ground + px * 1.6, px * 3, px * 0.5);
  // 공룡 (x 고정)
  const dinoH = DINO.length * px, dinoX = Math.round(W * 0.14);
  const yOff = (frame.y || 0) * (H * 0.42);
  const dinoY = ground - dinoH - yOff;
  blit(ctx, DINO, dinoX, dinoY, px, FG);
  // 다리 애니메이션 대신 땅에 있을 때만 발 깜빡임
  // 선인장: 거리를 x 로 바꾼다 (0 m = 공룡 위치, 9 m = 화면 오른쪽 끝)
  const cacti = frame.cacti || [];
  for (const [d, hw, h] of cacti) {
    const x = dinoX + (d / 9.0) * (W - dinoX - 20);
    const cpx = Math.max(2, Math.round(px * (h / 0.42)));
    const ch = CACTUS.length * cpx;
    blit(ctx, CACTUS, x, ground - ch, cpx, FG);
  }
  // 점수
  ctx.fillStyle = FG;
  ctx.font = `${Math.round(H * 0.075)}px ui-monospace, monospace`;
  ctx.textAlign = 'right';
  ctx.fillText(String(frame.score).padStart(5, '0'), W - 12, Math.round(H * 0.13));
  ctx.textAlign = 'left';
  if (frame.dead) {
    ctx.fillStyle = FG;
    ctx.font = `bold ${Math.round(H * 0.11)}px ui-monospace, monospace`;
    ctx.fillText('G A M E   O V E R', W * 0.28, H * 0.42);
  }
}

/** 파리 시점 — 겹눈에 들어오는 장면 (1인칭, 루밍). */
export function drawFlyView(ctx, W, H, frame, { hex = true } = {}) {
  const FOVA = 40, FOVE = 25;
  const eye = 0.25 + (frame.y || 0);
  ctx.fillStyle = '#cfe0ef'; ctx.fillRect(0, 0, W, H / 2);
  ctx.fillStyle = '#b6a683'; ctx.fillRect(0, H / 2, W, H / 2);
  const cx = W / 2, cy = H / 2, pxa = W / (2 * FOVA), pxe = H / (2 * FOVE);
  const cacti = (frame.cacti || []).slice().sort((a, b) => b[0] - a[0]);
  ctx.fillStyle = '#2f3a2c';
  for (const [d, hw, h] of cacti) {
    const az = Math.atan(hw / d) * 180 / Math.PI;
    const e0 = Math.atan((0 - eye) / d) * 180 / Math.PI;
    const e1 = Math.atan((h - eye) / d) * 180 / Math.PI;
    ctx.fillRect(cx - az * pxa, cy - e1 * pxe, Math.max(2 * az * pxa, 1), Math.max((e1 - e0) * pxe, 1));
  }
  if (hex) {  // 겹눈 낱눈 격자
    const r = Math.max(4, H / 26);
    ctx.strokeStyle = 'rgba(0,0,0,0.18)'; ctx.lineWidth = 1;
    const dx = r * 1.732, dy = r * 1.5;
    for (let row = -1; row * dy < H + r; row++) {
      for (let col = -1; col * dx < W + dx; col++) {
        const x = col * dx + (row % 2 ? dx / 2 : 0), y = row * dy;
        ctx.beginPath();
        for (let k = 0; k < 6; k++) {
          const a = Math.PI / 6 + k * Math.PI / 3;
          const px = x + r * Math.cos(a), py = y + r * Math.sin(a);
          k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        }
        ctx.closePath(); ctx.stroke();
      }
    }
    const g = ctx.createRadialGradient(cx, cy, H * 0.2, cx, cy, H * 0.72);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(0,0,0,0.55)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  }
}
