"""폐회로 실험 결과를 브라우저에서 재생하는 단일 HTML 파일 생성."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .world import World

TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f5f1;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#dcdad3;--accent:#2a6f97}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececef;--muted:#9a9aa2;--card:#1d1d21;--line:#34343a;--accent:#6fb3d9}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,"Apple SD Gothic Neo",system-ui,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:4px 0 2px} .sub{color:var(--muted);margin-bottom:12px}
.grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:14px}
@media (max-width:800px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
canvas{width:100%;display:block}
.ctl{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:6px 12px;font:inherit;cursor:pointer}
input[type=range]{flex:1;min-width:120px}
.stat{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}
.stat div{background:var(--bg);border-radius:6px;padding:6px}
.stat b{display:block;font-size:16px}
.lg{font-size:12px;color:var(--muted);margin-top:4px}
</style></head><body><div class="wrap">
<h1>__TITLE__</h1><div class="sub">FlyWire v783 전뇌 LIF 모델(138,639 뉴런)이 감각을 받아 하행 뉴런으로 몸을 움직인 기록입니다.</div>
<div class="grid">
 <div class="card"><canvas id="arena"></canvas>
  <div class="ctl"><button id="play">재생</button><input id="seek" type="range" min="0" value="0"><span id="tt">0.00 s</span>
  <select id="spd"><option value="1">1×</option><option value="2" selected>2×</option><option value="5">5×</option></select></div>
  <div class="stat"><div>속도<b id="s_speed">0</b></div><div>주둥이<b id="s_prob">–</b></div><div>섭식량<b id="s_en">0</b></div></div>
 </div>
 <div class="card"><canvas id="ts" height="420"></canvas><div class="lg">위: 속도(mm/s) · 가운데: 하행 뉴런 발화율(Hz) · 아래: 섭식/탈출 뉴런(Hz). 세로선 = 현재 시각</div></div>
</div></div>
<script>
const W=__WORLD__, D=__DATA__;
const n=D.t.length, arena=document.getElementById('arena'), ts=document.getElementById('ts');
const seek=document.getElementById('seek'); seek.max=n-1;
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function fit(c,h){const r=c.getBoundingClientRect(), d=devicePixelRatio||1; c.width=r.width*d; c.height=(h||r.width*W.height/W.width)*d; c.getContext('2d').setTransform(d,0,0,d,0,0); return [r.width,(h||r.width*W.height/W.width)]}
const PC={sugar:'#e9b949',bitter:'#7b4fa3',salt_low:'#4f8fc0',water:'#6cc3d5'};
let k=0, playing=false;
function drawArena(){
 const [w,h]=fit(arena), s=w/W.width, g=arena.getContext('2d'); const Y=y=>h-y*s;
 g.clearRect(0,0,w,h); g.strokeStyle=css('--line'); g.strokeRect(0.5,0.5,w-1,h-1);
 for(const z of W.thermal){const gr=g.createRadialGradient(z.x*s,Y(z.y),0,z.x*s,Y(z.y),z.radius*2*s); gr.addColorStop(0,z.delta>0?'rgba(220,70,50,.35)':'rgba(60,120,220,.35)'); gr.addColorStop(1,'rgba(0,0,0,0)'); g.fillStyle=gr; g.fillRect(0,0,w,h)}
 for(const o of W.odors){const gr=g.createRadialGradient(o.x*s,Y(o.y),0,o.x*s,Y(o.y),o.sigma*2*s); gr.addColorStop(0,'rgba(58,157,93,.45)'); gr.addColorStop(1,'rgba(58,157,93,0)'); g.fillStyle=gr; g.fillRect(0,0,w,h); g.fillStyle='#1f6b3a'; g.font='12px sans-serif'; g.fillText('★ '+o.name,o.x*s+4,Y(o.y)-4)}
 for(const p of W.patches){g.globalAlpha=.45; g.fillStyle=PC[p.kind]||'#999'; g.beginPath(); g.arc(p.x*s,Y(p.y),p.radius*s,0,7); g.fill(); g.globalAlpha=1}
 const lab={}; for(const p of W.patches){const key=p.x+','+p.y; (lab[key]=lab[key]||{x:p.x,y:p.y,k:[]}).k.push(p.kind)}
 g.fillStyle=css('--fg'); g.font='12px sans-serif'; g.textAlign='center'; for(const L of Object.values(lab)) g.fillText(L.k.join('+'),L.x*s,Y(L.y)+4); g.textAlign='left';
 g.strokeStyle=css('--muted'); g.lineWidth=1; g.beginPath();
 for(let i=0;i<=k;i++){const x=D.x[i]*s,y=Y(D.y[i]); i?g.lineTo(x,y):g.moveTo(x,y)} g.stroke();
 for(let i=0;i<=k;i++){ if(D.jump[i]){g.fillStyle='#c1121f'; g.fillText('✕',D.x[i]*s-4,Y(D.y[i])+4)} }
 const x=D.x[k]*s,y=Y(D.y[k]),a=-D.heading[k];
 g.save(); g.translate(x,y); g.rotate(a); g.fillStyle=D.proboscis[k]?'#d1495b':css('--fg');
 g.beginPath(); g.ellipse(0,0,7,3.5,0,0,7); g.fill(); g.fillStyle=css('--accent'); g.beginPath(); g.arc(6,0,2.2,0,7); g.fill();
 g.globalAlpha=.35; g.fillStyle=css('--muted'); g.beginPath(); g.ellipse(-1,-5,4,2,-.4,0,7); g.ellipse(-1,5,4,2,.4,0,7); g.fill(); g.restore();
 document.getElementById('tt').textContent=D.t[k].toFixed(2)+' s';
 document.getElementById('s_speed').textContent=D.speed[k].toFixed(1)+' mm/s';
 document.getElementById('s_prob').textContent=D.proboscis[k]?'신전 (먹는 중)':'접힘';
 document.getElementById('s_en').textContent=D.energy[k].toFixed(2);
}
const SER=[[['speed','속도','#2a6f97']],[['r_fwd_L','전진 좌','#588157'],['r_fwd_R','전진 우','#a3b18a'],['r_turn_L','회전 좌','#bc6c25'],['r_turn_R','회전 우','#dda15e'],['r_back','후진(MDN)','#6d597a']],[['r_feed','MN9','#d1495b'],['r_feed_ing','섭식MN','#edae49'],['r_escape','탈출','#00798c'],['r_takeoff','이륙','#30638e']]];
function drawTS(){
 const [w,h]=fit(ts,420), g=ts.getContext('2d'); g.clearRect(0,0,w,h); const ph=h/3, T=D.t[n-1]||1;
 SER.forEach((grp,pi)=>{const y0=pi*ph; let mx=1; for(const [c] of grp) if(D[c]) for(const v of D[c]) mx=Math.max(mx,Math.abs(v));
  g.strokeStyle=css('--line'); g.strokeRect(0.5,y0+4,w-1,ph-8);
  let lx=6; grp.forEach(([c,lab,col])=>{ if(!D[c]) return; g.strokeStyle=col; g.lineWidth=1.3; g.beginPath(); D[c].forEach((v,i)=>{const x=D.t[i]/T*w, y=y0+ph-6-(v/mx)*(ph-14); i?g.lineTo(x,y):g.moveTo(x,y)}); g.stroke(); g.fillStyle=col; g.font='11px sans-serif'; g.fillText(lab,lx,y0+16); lx+=g.measureText(lab).width+12});
  g.fillStyle=css('--muted'); g.fillText('max '+mx.toFixed(0),w-60,y0+16)});
 const x=D.t[k]/T*w; g.strokeStyle=css('--fg'); g.globalAlpha=.5; g.beginPath(); g.moveTo(x,0); g.lineTo(x,h); g.stroke(); g.globalAlpha=1;
}
function draw(){drawArena(); drawTS(); seek.value=k}
let last=0; function loop(ts_){ if(playing){ const sp=+document.getElementById('spd').value; if(!last) last=ts_; const dt=(ts_-last)/1000*sp; last=ts_; const target=D.t[k]+dt; while(k<n-1&&D.t[k]<target)k++; if(k>=n-1){playing=false; document.getElementById('play').textContent='재생'} draw()} requestAnimationFrame(loop)}
document.getElementById('play').onclick=()=>{ if(k>=n-1)k=0; playing=!playing; last=0; document.getElementById('play').textContent=playing?'일시정지':'재생'};
seek.oninput=()=>{k=+seek.value; draw()}; addEventListener('resize',draw); draw(); requestAnimationFrame(loop);
</script></body></html>"""


def write_replay(df: pd.DataFrame, world: World, path: str | Path, title: str = "") -> Path:
    cols = ["t", "x", "y", "heading", "speed", "proboscis", "jump", "energy",
            "r_fwd_L", "r_fwd_R", "r_turn_L", "r_turn_R", "r_back", "r_feed", "r_feed_ing", "r_escape", "r_takeoff"]
    data = {}
    for c in cols:
        if c in df:
            v = df[c]
            data[c] = [bool(x) for x in v] if v.dtype == bool else [round(float(x), 3) for x in v]
    wj = {
        "width": world.width, "height": world.height,
        "patches": [{"x": p.x, "y": p.y, "radius": p.radius, "kind": p.kind} for p in world.patches],
        "odors": [{"x": s.x, "y": s.y, "sigma": s.sigma, "name": s.odorant if isinstance(s.odorant, str) else "odor"} for s in world.odor_sources],
        "thermal": [{"x": z.x, "y": z.y, "radius": z.radius, "delta": z.delta} for z in world.thermal],
    }
    title = title or f"초파리 가상 실험 — {world.name}"
    html = (TEMPLATE.replace("__TITLE__", title)
            .replace("__WORLD__", json.dumps(wj))
            .replace("__DATA__", json.dumps(data)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
