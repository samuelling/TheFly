// In-browser version of analysis/dm3_model.py: the FlyWire Dm3 -> TmY orientation circuit.
const Dm3Model = (() => {
  let C = null, nN = 0, nC = 0, X, Y, ffPost, ffCol, ffOn, ffW, dPost, dPre, dW, isType = {};
  const ITERS = 50;
  const TP = Array.from({length: 8}, (_, i) => 2 * Math.PI * i / 8);
  const MP = Array.from({length: 4}, (_, i) => 2 * Math.PI * i / 4 + 0.3);

  function init(circ) {
    C = circ; nN = circ.types.length; nC = circ.columns.length;
    X = Float64Array.from(circ.columns, c => c[0]); Y = Float64Array.from(circ.columns, c => c[1]);
    const ff = circ.ff, dm = circ.dm3;
    ffPost = Int32Array.from(ff, e => e[0]); ffCol = Int32Array.from(ff, e => e[1]);
    ffOn = Uint8Array.from(ff, e => e[2] === 'on' ? 1 : 0); ffW = Float64Array.from(ff, e => e[3]);
    dPost = Int32Array.from(dm, e => e[0]); dPre = Int32Array.from(dm, e => e[1]); dW = Float64Array.from(dm, e => e[2]);
    for (const t of new Set(circ.types)) isType[t] = circ.types.map(x => x === t);
  }

  function grating(theta, c, phase, wl, out) {
    const k = 2 * Math.PI / wl, ct = Math.cos(theta), st = Math.sin(theta);
    for (let i = 0; i < nC; i++) out[i] += c * Math.cos(k * (X[i] * ct + Y[i] * st) + phase);
  }

  function simulate(stim, o) {
    const drive = new Float64Array(nN);
    for (let e = 0; e < ffPost.length; e++) {
      const s = stim[ffCol[e]];
      const v = ffOn[e] ? (s > 0 ? s : 0) : (s < 0 ? -s : 0);
      drive[ffPost[e]] += ffW[e] * v;
    }
    let r = new Float64Array(nN);
    const inh = new Float64Array(nN);
    for (let it = 0; it < ITERS; it++) {
      inh.fill(0);
      if (o.dm3On) for (let e = 0; e < dPost.length; e++) inh[dPost[e]] += dW[e] * r[dPre[e]];
      for (let i = 0; i < nN; i++) {
        const nw = o.mode === 'sub' ? Math.max(drive[i] - o.g * inh[i], 0) : Math.max(drive[i], 0) / (1 + 4 * o.g * inh[i]);
        r[i] = 0.5 * r[i] + 0.5 * nw;
      }
    }
    return r;
  }

  // peak over test phase, averaged over mask phase
  function resp(testTh, testC, maskTh, maskC, o) {
    const out = new Float64Array(nN), stim = new Float64Array(nC);
    const mps = (maskTh === null || maskC <= 0) ? [0] : MP;
    for (const mp of mps) {
      const peak = new Float64Array(nN).fill(-Infinity);
      for (const tp of TP) {
        stim.fill(0);
        grating(testTh, testC, tp, o.wl, stim);
        if (maskTh !== null && maskC > 0) grating(maskTh, maskC, mp, o.wl, stim);
        const r = simulate(stim, o);
        for (let i = 0; i < nN; i++) if (r[i] > peak[i]) peak[i] = r[i];
      }
      for (let i = 0; i < nN; i++) out[i] += peak[i] / mps.length;
    }
    return out;
  }

  const median = a => { const s = Array.from(a).sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
  const pick = (arr, t) => arr.filter((_, i) => isType[t][i]);

  return { init, resp, pick, median, types: () => C.types, centroid: () => C.centroid, isType: t => isType[t] };
})();
