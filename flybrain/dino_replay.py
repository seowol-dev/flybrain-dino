"""공룡게임 폐회로 기록을 브라우저에서 재생하는 단일 HTML 파일 생성."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f5f1;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#dcdad3;--accent:#2a6f97}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececef;--muted:#9a9aa2;--card:#1d1d21;--line:#34343a;--accent:#6fb3d9}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,"Apple SD Gothic Neo",system-ui,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:4px 0 2px} .sub{color:var(--muted);margin-bottom:12px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}
@media (max-width:800px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
canvas{width:100%;display:block;border-radius:6px}
.ctl{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:6px 12px;font:inherit;cursor:pointer}
input[type=range]{flex:1;min-width:120px}
.stat{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}
.stat div{background:var(--bg);border-radius:6px;padding:6px}
.stat b{display:block;font-size:16px}
.lg{font-size:12px;color:var(--muted);margin-top:6px}
</style></head><body><div class="wrap">
<h1>__TITLE__</h1>
<div class="sub">__SUB__</div>
<div class="grid">
 <div class="card"><canvas id="eye" height="300"></canvas>
  <div class="lg">초파리 눈에 들어오는 장면(공룡 1인칭). 다가오는 선인장은 각크기가 커지는 루밍 자극이다.</div>
  <div class="ctl"><button id="play">재생</button><input id="seek" type="range" min="0" value="0"><span id="tt">0.00 s</span>
   <select id="spd"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div>
  <div class="stat"><div>점수<b id="s_score">0</b></div><div>각반너비<b id="s_th">0°</b></div>
   <div>도약 높이<b id="s_y">0</b></div><div>거리<b id="s_d">–</b></div></div>
 </div>
 <div class="card"><canvas id="ts" height="420"></canvas>
  <div class="lg">위: 가장 가까운 선인장의 각반너비 θ(도) · 가운데: 루밍 지표(중심−주변 라미나, 순응 후)와 임계선, ▲=도약 · 아래: 중심/주변 풀 발화율(Hz). 세로선 = 현재 시각</div></div>
</div></div>
<script>
const D=__DATA__, TH=__THRESH__;
const eye=document.getElementById('eye'), ts=document.getElementById('ts');
const ec=eye.getContext('2d'), tc=ts.getContext('2d');
let i=0, playing=false, last=0;
const seek=document.getElementById('seek'); seek.max=D.t.length-1;
const FOVA=40, FOVE=25;   // 도
function fit(c){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;
  c.width=r.width*d; c.height=c.height*d; c.getContext('2d').setTransform(d,0,0,d,0,0); return [r.width,c.height/d];}
function drawEye(){
  const [W,H]=fit(eye); const cx=W/2, cy=H/2;
  const px=W/(2*FOVA), py=H/(2*FOVE);
  const horizon=cy;
  ec.fillStyle='#dfe9f2'; ec.fillRect(0,0,W,horizon);
  ec.fillStyle='#b9ad8f'; ec.fillRect(0,horizon,W,H-horizon);
  const eh=D.eye_h[i]+D.y[i];
  const cs=D.cacti[i]||[];
  for(let k=cs.length-1;k>=0;k--){
    const [d,w,h]=cs[k];
    const az=Math.atan(w/d)*180/Math.PI;
    const e0=Math.atan((0-eh)/d)*180/Math.PI, e1=Math.atan((h-eh)/d)*180/Math.PI;
    const x0=cx-az*px, x1=cx+az*px, y0=cy-e1*py, y1=cy-e0*py;
    ec.fillStyle='#2f3a2c'; ec.fillRect(x0,y0,Math.max(x1-x0,1),Math.max(y1-y0,1));
  }
  ec.strokeStyle='rgba(0,0,0,.25)'; ec.beginPath(); ec.moveTo(0,horizon); ec.lineTo(W,horizon); ec.stroke();
  if(D.jump[i]){ec.fillStyle='#d1495b';ec.beginPath();ec.arc(18,18,7,0,7);ec.fill();}
  if(D.dead[i]){ec.fillStyle='rgba(209,73,91,.85)';ec.font='bold 22px sans-serif';ec.fillText('충돌',cx-24,28);}
}
function series(key){return D[key];}
function drawTs(){
  const [W,H]=fit(ts); tc.clearRect(0,0,W,H);
  const n=D.t.length, T=D.t[n-1]||1, pad=34, rowH=(H-3*pad)/3;
  const panes=[{k:['theta_deg'],c:['#2a6f97'],lab:'θ (도)'},
               {k:['loom_sig'],c:['#d1495b'],lab:'루밍 지표'},
               {k:['r_F_pool','r_S_pool'],c:['#3f8f5b','#8a7fb5'],lab:'중심/주변 (Hz)'}];
  panes.forEach((p,pi)=>{
    const y0=pad+pi*(rowH+pad*0.5), y1=y0+rowH;
    let lo=Infinity, hi=-Infinity;
    p.k.forEach(k=>series(k).forEach(v=>{if(v!=null&&isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}}));
    if(pi===1){lo=Math.min(lo,0);hi=Math.max(hi,TH*1.2);}
    if(!isFinite(lo)){lo=0;hi=1;} if(hi-lo<1e-6)hi=lo+1;
    tc.strokeStyle='rgba(128,128,128,.3)'; tc.strokeRect(pad,y0,W-pad-8,rowH);
    tc.fillStyle='#8a8a90'; tc.font='11px sans-serif'; tc.fillText(p.lab,pad,y0-4);
    const X=t=>pad+(t/T)*(W-pad-8), Y=v=>y1-((v-lo)/(hi-lo))*rowH;
    if(pi===1){tc.strokeStyle='rgba(209,73,91,.45)';tc.setLineDash([4,3]);tc.beginPath();
      tc.moveTo(pad,Y(TH));tc.lineTo(W-8,Y(TH));tc.stroke();tc.setLineDash([]);}
    p.k.forEach((k,ki)=>{const s=series(k);tc.strokeStyle=p.c[ki];tc.lineWidth=1.3;tc.beginPath();
      let started=false;
      for(let j=0;j<n;j++){const v=s[j]; if(v==null||!isFinite(v))continue;
        const x=X(D.t[j]),y=Y(v); if(!started){tc.moveTo(x,y);started=true;}else tc.lineTo(x,y);}
      tc.stroke();});
    if(pi===1){tc.fillStyle='#d1495b';
      for(let j=0;j<n;j++) if(D.jump[j]){const x=X(D.t[j]);tc.beginPath();tc.moveTo(x,y1);tc.lineTo(x-4,y1-8);tc.lineTo(x+4,y1-8);tc.closePath();tc.fill();}}
  });
  tc.strokeStyle='rgba(0,0,0,.5)'; tc.beginPath();
  const xx=pad+(D.t[i]/(D.t[n-1]||1))*(W-pad-8); tc.moveTo(xx,10); tc.lineTo(xx,H-10); tc.stroke();
}
function render(){
  drawEye(); drawTs(); seek.value=i;
  document.getElementById('tt').textContent=D.t[i].toFixed(2)+' s';
  document.getElementById('s_score').textContent=D.score[i];
  document.getElementById('s_th').textContent=D.theta_deg[i].toFixed(1)+'°';
  document.getElementById('s_y').textContent=D.y[i].toFixed(2)+' m';
  document.getElementById('s_d').textContent=(D.dist[i]!=null&&isFinite(D.dist[i]))?D.dist[i].toFixed(2)+' m':'–';
}
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'일시정지':'재생';last=performance.now();};
seek.oninput=e=>{i=+e.target.value;render();};
function loop(now){const sp=+document.getElementById('spd').value;
  if(playing&&now-last>(1000/60)/sp){i=(i+1)%D.t.length;last=now;render();}
  requestAnimationFrame(loop);}
window.addEventListener('resize',render); render(); requestAnimationFrame(loop);
</script></body></html>
"""


def write_dino_replay(df: pd.DataFrame, stats: dict, path: str | Path,
                      title: str = "초파리가 하는 크롬 공룡게임", threshold: float = 1.8) -> Path:
    """궤적 DataFrame → 단일 HTML 재생기."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def col(name, default=0.0):
        if name not in df:
            return [default] * len(df)
        v = df[name].to_numpy()
        return [None if (isinstance(x, float) and not np.isfinite(x)) else
                (bool(x) if v.dtype == bool else (float(x) if not isinstance(x, (int, np.integer)) else int(x)))
                for x in v]

    data = {
        "t": col("t"), "y": col("y"), "dist": col("dist"), "theta_deg": col("theta_deg"),
        "score": col("score"), "jump": col("jump", False), "dead": col("dead", False),
        "eye_h": col("eye_h", 0.25), "loom_sig": col("loom_sig"),
        "r_F_pool": col("r_F_pool"), "r_S_pool": col("r_S_pool"),
        "cacti": [list(x) if isinstance(x, (list, tuple)) else [] for x in df.get("cacti", pd.Series([[]] * len(df)))],
    }
    sub = (f"생존 {stats.get('survived_s', 0):.1f}초 · 넘은 선인장 {stats.get('score', 0)}개 · "
           f"도약 {stats.get('jumps', 0)}회 · 프레임 작업 p50 {stats.get('work_p50_ms', 0):.1f} / "
           f"p95 {stats.get('work_p95_ms', 0):.1f} ms (예산 {stats.get('budget_ms', 16.67)} ms) · "
           f"실시간 {'유지' if stats.get('realtime_ok') else '실패'}")
    html = (TEMPLATE.replace("__TITLE__", title).replace("__SUB__", sub)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__THRESH__", repr(float(threshold))))
    path.write_text(html, encoding="utf-8")
    return path
