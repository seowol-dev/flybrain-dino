// 기록 데이터 로더. flybrain.export_web 이 만든 파일을 읽는다.
export async function loadRun(base = './data') {
  const j = async (p) => (await fetch(`${base}/${p}`)).json();
  const b = async (p) => (await fetch(`${base}/${p}`)).arrayBuffer();
  const [meta, game] = await Promise.all([j('meta.json'), j('game.json')]);
  const [posBuf, backBuf, actBuf, clsBuf] = await Promise.all([
    b('positions.bin'), b('backdrop.bin'), b('activity.bin'), b('class.bin'),
  ]);
  const N = meta.n_neurons, F = meta.n_frames_act;
  const q = meta.position.quant;
  const toXYZ = (buf, n) => {
    const raw = new Int16Array(buf);
    const out = new Float32Array(n * 3);
    // FlyWire: +x 오른쪽, +y 배쪽, +z 뒤쪽 → three: y 위로, 앞이 -z
    for (let i = 0; i < n; i++) {
      out[i * 3] = raw[i * 3] / q;
      out[i * 3 + 1] = -raw[i * 3 + 1] / q;
      out[i * 3 + 2] = raw[i * 3 + 2] / q;
    }
    return out;
  };
  return {
    meta, game,
    pos: toXYZ(posBuf, N),
    backdrop: toXYZ(backBuf, meta.n_backdrop),
    act: new Uint8Array(actBuf),   // (F, N) 행우선
    cls: new Uint8Array(clsBuf),
    N, F,
  };
}
