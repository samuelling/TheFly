// TheFly dashboard
const NT_COL = {gaba: '#c0392b', glutamate: '#b9770e', acetylcholine: '#2e7d5b'};
const ntCol = nt => NT_COL[nt] || '#9a988f';
const INH = new Set(['gaba', 'glutamate']);
const LAYOUT = {margin: {l: 55, r: 15, t: 30, b: 45}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#fff',
  font: {family: '-apple-system,Segoe UI,Helvetica,Arial,sans-serif', size: 12, color: '#1d1d1b'},
  xaxis: {gridcolor: '#eee', zeroline: false}, yaxis: {gridcolor: '#eee', zeroline: false}, hovermode: 'closest'};
const CTICKS = {tickvals: [0.03, 0.06, 0.12, 0.25, 0.5, 1], ticktext: ['3%', '6%', '12%', '25%', '50%', '100%']};
const CFG = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d']};
const L = (o) => Object.assign({}, LAYOUT, o, {xaxis: Object.assign({}, LAYOUT.xaxis, o.xaxis || {}), yaxis: Object.assign({}, LAYOUT.yaxis, o.yaxis || {})});
const $ = id => document.getElementById(id);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : (+v).toFixed(d);
const pretty = t => String(t).replace('__perp', '⊥');
const D = {};

async function getJSON(p) { try { const r = await fetch(p); return r.ok ? r.json() : null; } catch (e) { return null; } }

document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#tabs button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.tab').forEach(s => s.classList.toggle('on', s.id === b.dataset.tab));
  window.dispatchEvent(new Event('resize'));
  if (b.dataset.tab === 'dm3' && !D.dm3ran) runDm3();
});

(async () => {
  [D.fw, D.sims, D.circ, D.dsum, D.find, D.wiring, D.graft] = await Promise.all(['data/flywire.json', 'data/sims.json', 'data/dm3_circuit.json',
    'data/dm3_summary.json', 'data/findings.json', 'data/flyvis_wiring.json', 'data/graft.json'].map(getJSON));
  D.byType = Object.fromEntries((D.fw?.types || []).map(t => [t.type, t]));
  renderFindings(); initPools(); initOrient(); initDm3(); initCRF(); initLesion(); initGraft(); renderMethods();
})();

// ---------------------------------------------------------------- overview
function renderFindings() {
  const f = D.find?.cards || [];
  $('findings').innerHTML = f.map(c => `<div class="card"><span class="tag">${c.tag}</span><h3>${c.title}</h3><p>${c.text}</p></div>`).join('');
}

// ---------------------------------------------------------------- hex map of a neuron's inputs
function hexMap(div, typeName, exIdx = 0, title = '') {
  const ex = D.fw.examples[typeName];
  const lat = D.fw.lattice;
  const traces = [{x: lat.map(p => p[0]), y: lat.map(p => p[1]), mode: 'markers', type: 'scatter', hoverinfo: 'skip',
    marker: {symbol: 'hexagon', size: 7, color: '#efede6'}, showlegend: false}];
  if (ex && ex[exIdx]) {
    const cols = ex[exIdx].cols;
    for (const [nt, lab] of [['acetylcholine', 'ACh input'], ['gaba', 'GABA input'], ['glutamate', 'Glu input']]) {
      const c = cols.filter(r => r[2] === nt);
      if (!c.length) continue;
      const mx = Math.max(...cols.map(r => r[3]));
      traces.push({x: c.map(r => r[0]), y: c.map(r => r[1]), mode: 'markers', type: 'scatter', name: lab,
        text: c.map(r => `${r[3]} synapses`), hoverinfo: 'text+name',
        marker: {symbol: 'hexagon', size: c.map(r => 6 + 10 * Math.sqrt(r[3] / mx)), color: ntCol(nt), opacity: 0.8}});
    }
  }
  Plotly.react(div, traces, L({title: {text: title || `${pretty(typeName)} · example neuron ${ex ? (exIdx + 1) + '/' + ex.length : '(no map stored)'}`, font: {size: 13}},
    xaxis: {visible: false, scaleanchor: 'y'}, yaxis: {visible: false}, legend: {orientation: 'h', y: -0.05}, margin: {l: 5, r: 5, t: 30, b: 5}}), CFG);
}

function partnerHTML(t) {
  const p = D.fw.partners[t];
  if (!p) return '';
  const li = a => a.slice(0, 8).map(([k, v]) => `<li>${pretty(k)} <span class="hint">${v.toLocaleString()}</span></li>`).join('');
  return `<div class="partners"><div><b>Top inputs</b><ol>${li(p.inputs)}</ol></div><div><b>Top outputs</b><ol>${li(p.outputs)}</ol></div></div>`;
}

function exampleButtons(id, t, fn) {
  const ex = D.fw.examples[t];
  if (!ex) return '';
  return `<div class="ctrl">example neuron: ${ex.map((e, i) => `<button onclick="${fn}('${t}',${i})">${i + 1}</button>`).join(' ')}</div>`;
}

// ---------------------------------------------------------------- pools
function initPools() {
  if (!D.fw) return;
  const draw = () => {
    const yk = $('poolY').value, inhOnly = $('poolInhOnly').checked;
    const rows = D.fw.types.filter(t => t.all_footprint > 0 && t.out_syn > 2000 && (!inhOnly || INH.has(t.nt)));
    const yv = t => yk === 'out_both' ? (t.frac_to_motion_inputs + t.frac_to_LMC) : t[yk];
    const groups = {};
    rows.forEach(t => (groups[t.nt] = groups[t.nt] || []).push(t));
    const traces = Object.entries(groups).map(([nt, g]) => ({type: 'scatter', mode: 'markers', name: nt || 'unknown',
      x: g.map(t => t.all_footprint), y: g.map(yv), text: g.map(t => pretty(t.type)), customdata: g.map(t => t.type),
      hovertemplate: '<b>%{text}</b><br>footprint %{x} columns<br>%{y:.2f}<extra></extra>',
      marker: {color: ntCol(nt), size: g.map(t => Math.min(28, 6 + 0.9 * Math.sqrt(t.n_cells))), opacity: 0.75, line: {width: 0.5, color: '#fff'}}}));
    const top = rows.filter(t => t.pool_score > 2.5);
    traces.push({type: 'scatter', mode: 'text', x: top.map(t => t.all_footprint), y: top.map(yv), text: top.map(t => pretty(t.type)),
      textposition: 'top center', textfont: {size: 10, color: '#555'}, hoverinfo: 'skip', showlegend: false});
    Plotly.react('poolScatter', traces, L({xaxis: {type: 'log', title: 'input footprint (distinct columns, median per cell)'},
      yaxis: {title: 'fraction of output synapses', range: [-0.03, 1.1]}, legend: {orientation: 'h', y: 1.08}}), CFG);
    $('poolScatter').on('plotly_click', e => { const t = e.points[0].customdata; if (t) poolDetail(t); });
  };
  $('poolY').onchange = draw; $('poolInhOnly').onchange = draw; draw();
  // table
  const cols = [['type', 'type'], ['nt', 'transmitter'], ['n_cells', 'cells'], ['all_footprint', 'footprint'], ['all_radius', 'radius'],
    ['frac_to_motion_inputs', '→ T4/T5 inputs'], ['frac_to_LMC', '→ LMCs'], ['pool_score', 'pool score']];
  let sortK = 'pool_score', asc = false;
  const table = () => {
    const rows = D.fw.types.filter(t => t.pool_score !== null).sort((a, b) => (asc ? 1 : -1) * ((a[sortK] > b[sortK]) - (a[sortK] < b[sortK]))).slice(0, 60);
    $('poolTable').innerHTML = `<table><tr>${cols.map(([k, n]) => `<th data-k="${k}">${n}${k === sortK ? (asc ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr>` +
      rows.map(t => `<tr data-t="${t.type}">${cols.map(([k]) => `<td>${typeof t[k] === 'number' ? fmt(t[k], k === 'n_cells' || k === 'all_footprint' ? 0 : 2) : pretty(t[k] || '')}</td>`).join('')}</tr>`).join('') + '</table>';
    $('poolTable').querySelectorAll('th').forEach(th => th.onclick = () => { asc = sortK === th.dataset.k ? !asc : false; sortK = th.dataset.k; table(); });
    $('poolTable').querySelectorAll('tr[data-t]').forEach(tr => tr.onclick = () => poolDetail(tr.dataset.t));
  };
  table();
  poolDetail('Dm17');
}

function poolDetail(t, ex = 0) {
  const r = D.byType[t]; if (!r) return;
  $('poolDetail').innerHTML = `<h3>${pretty(t)} <span class="hint">${r.nt} · ${r.n_cells} cells (right optic lobe)</span></h3>
    <div class="kv"><span>input footprint</span><span>${fmt(r.all_footprint, 0)} columns (radius ${fmt(r.all_radius, 1)} column spacings)</span>
    <span>output → T4/T5 inputs</span><span>${fmt(100 * r.frac_to_motion_inputs, 0)}%</span>
    <span>output → lamina monopolar cells</span><span>${fmt(100 * r.frac_to_LMC, 0)}%</span>
    <span>output → T4/T5 directly</span><span>${fmt(100 * r.frac_to_T4T5, 0)}%</span>
    <span>pool score</span><span>${fmt(r.pool_score)}</span></div>
    ${exampleButtons('poolDetail', t, 'poolDetail')}<div id="poolHex" class="plot sm"></div>${partnerHTML(t)}`;
  hexMap('poolHex', t, ex);
}

// ---------------------------------------------------------------- orientation
const LAMINA_SET = ['Lawf1', 'Lawf2', 'Lai', 'C2', 'C3', 'T1', 'L1', 'L2', 'L3', 'L4', 'L5'];
const DM3_SET = ['Dm3p', 'Dm3q', 'Dm3v', 'TmY9q', 'TmY9q__perp', 'TmY4'];
function initOrient() {
  if (!D.fw) return;
  const draw = () => {
    const s = $('orSrc').value, mn = +$('orMin').value, hi = $('orHi').value;
    const hs = new Set(hi === 'lamina' ? LAMINA_SET : hi === 'dm3' ? DM3_SET : []);
    const rows = D.fw.types.filter(t => t.n_cells >= mn && t[s + '_elong'] !== null && t[s + '_axis_R'] !== null);
    const groups = {};
    rows.forEach(t => (groups[t.nt] = groups[t.nt] || []).push(t));
    const traces = Object.entries(groups).map(([nt, g]) => ({type: 'scatter', mode: 'markers', name: nt || 'unknown',
      x: g.map(t => t[s + '_elong']), y: g.map(t => t[s + '_axis_R']), text: g.map(t => pretty(t.type)), customdata: g.map(t => t.type),
      hovertemplate: '<b>%{text}</b><br>excess elongation %{x:.2f}<br>axis consistency %{y:.2f}<extra></extra>',
      marker: {color: ntCol(nt), size: g.map(t => Math.min(26, 5 + 0.8 * Math.sqrt(t.n_cells))), opacity: 0.6, line: {width: g.map(t => hs.has(t.type) ? 2 : 0.5), color: g.map(t => hs.has(t.type) ? '#7a4fb5' : '#fff')}}}));
    const lab = rows.filter(t => hs.has(t.type) || (t[s + '_elong'] * t[s + '_axis_R'] > 0.6 && t.n_cells >= 20));
    traces.push({type: 'scatter', mode: 'text', x: lab.map(t => t[s + '_elong']), y: lab.map(t => t[s + '_axis_R']), text: lab.map(t => pretty(t.type)),
      textposition: 'top center', textfont: {size: 10, color: lab.map(t => hs.has(t.type) ? '#7a4fb5' : '#555')}, hoverinfo: 'skip', showlegend: false});
    Plotly.react('orScatter', traces, L({xaxis: {title: 'excess elongation of input pool (median cell)'}, yaxis: {title: 'axis consistency across cells', range: [-0.03, 1.05]},
      legend: {orientation: 'h', y: 1.08}}), CFG);
    $('orScatter').on('plotly_click', e => { const t = e.points[0].customdata; if (t) orDetail(t); });
  };
  ['orSrc', 'orMin', 'orHi'].forEach(k => $(k).onchange = draw); draw();
  orDetail('Dm3p');
  // lamina callout
  const rowsL = LAMINA_SET.map(t => D.byType[t]).filter(Boolean);
  const lw = rowsL.filter(r => ['Lawf1', 'Lawf2', 'Lai', 'C2', 'C3'].includes(r.type));
  $('laminaCallout').innerHTML = `<b>Medulla → lamina, and retina → lamina.</b> ${D.find?.lamina || ''}<br><span class="hint">` +
    lw.map(r => `${r.type}: excess elongation ${fmt(r.exc_elong)} (excitatory input), axis consistency ${fmt(r.exc_axis_R)}, ${fmt(100 * r.frac_to_LMC, 0)}% of output onto LMCs`).join(' · ') + '</span>';
}

function orDetail(t, ex = 0) {
  const r = D.byType[t]; if (!r) return;
  $('orDetail').innerHTML = `<h3>${pretty(t)} <span class="hint">${r.nt} · ${r.n_cells} cells</span></h3>
    <div class="kv"><span>input footprint</span><span>${fmt(r.all_footprint, 0)} columns</span>
    <span>excess elongation (all / exc / inh)</span><span>${fmt(r.all_elong)} / ${fmt(r.exc_elong)} / ${fmt(r.inh_elong)}</span>
    <span>shared axis</span><span>${fmt(r.all_axis, 0)}° (consistency ${fmt(r.all_axis_R)})</span></div>
    <div class="grid2"><div id="orRose" class="plot sm"></div><div id="orHex" class="plot sm"></div></div>
    ${exampleButtons('orDetail', t, 'orDetail')}${partnerHTML(t)}`;
  const h = r.axis_hist;
  if (h) {
    const th = h.map((_, i) => 7.5 + 15 * i);
    Plotly.react('orRose', [{type: 'barpolar', r: h.concat(h), theta: th.concat(th.map(x => x + 180)), marker: {color: ntCol(r.nt), opacity: 0.75}, hoverinfo: 'r+theta'}],
      L({title: {text: 'input-pool axes (cells ≥ 6 columns)', font: {size: 12}}, polar: {angularaxis: {direction: 'counterclockwise', rotation: 0, tickfont: {size: 9}}, radialaxis: {tickfont: {size: 9}}},
        margin: {l: 20, r: 20, t: 40, b: 20}, showlegend: false}), CFG);
  } else $('orRose').innerHTML = '<p class="hint">Too few cells with ≥ 6 input columns for an axis histogram.</p>';
  hexMap('orHex', t, ex, `${pretty(t)} example input pool`);
}

// ---------------------------------------------------------------- Dm3 model
function initDm3() {
  if (!D.circ) { $('mStatus').textContent = 'circuit data missing'; return; }
  Dm3Model.init(D.circ);
  for (const [id, v] of [['mGain', 'mGainV'], ['mMaskC', 'mMaskCV'], ['mWL', 'mWLV']]) $(id).oninput = () => $(v).textContent = $(id).value;
  $('mRun').onclick = runDm3;
  $('dm3Diagram').innerHTML = dm3SVG();
  if (D.dsum) {
    const s = D.dsum, p = t => s[t].ff;
    $('dm3Callout').innerHTML = (D.find?.dm3 || '') + `<br><span class="hint">Preferred grating (wave-vector angle, wiring alone): ` +
      ['Dm3p', 'Dm3q', 'Dm3v', 'TmY9q', 'TmY9q__perp', 'TmY4'].map(t => `${pretty(t)} ${fmt(p(t).pref, 0)}° (R=${fmt(p(t).pref_R)})`).join(' · ') + '</span>';
  }
}

function dm3SVG() {
  const box = (x, y, t, c) => `<g><rect x="${x}" y="${y}" width="86" height="28" rx="5" fill="${c}" opacity=".16" stroke="${c}"/><text x="${x + 43}" y="${y + 18}" text-anchor="middle" font-size="12">${t}</text></g>`;
  const ar = (x1, y1, x2, y2, c, inh) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="1.4" marker-end="url(#${inh ? 'bar' : 'arw'})"/>`;
  return `<svg viewBox="0 0 320 230" width="100%"><defs><marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#2e7d5b"/></marker>
  <marker id="bar" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M5 0V10" stroke="#b9770e" stroke-width="3"/></marker></defs>
  ${box(117, 6, 'Tm1 · Tm2 · L3', '#2e7d5b')}
  ${box(10, 90, 'Dm3p', '#b9770e')}${box(117, 90, 'Dm3q', '#b9770e')}${box(224, 90, 'Dm3v', '#b9770e')}
  ${box(10, 190, 'TmY9q', '#2e7d5b')}${box(117, 190, 'TmY9q⊥', '#2e7d5b')}${box(224, 190, 'TmY4', '#2e7d5b')}
  ${ar(150, 34, 53, 88, '#2e7d5b')}${ar(160, 34, 160, 88, '#2e7d5b')}${ar(170, 34, 267, 88, '#2e7d5b')}
  ${ar(53, 118, 53, 186, '#b9770e', 1)}${ar(160, 118, 160, 186, '#b9770e', 1)}${ar(267, 118, 267, 186, '#b9770e', 1)}
  ${ar(96, 100, 115, 100, '#b9770e', 1)}${ar(203, 108, 222, 108, '#b9770e', 1)}${ar(222, 96, 98, 96, '#b9770e', 1)}
</svg><p class="hint">Each Dm3 subtype sends most of its output to one TmY partner, and the Dm3 subtypes also synapse on each other. Green: excitatory (ACh); amber bars: glutamatergic, modelled as inhibitory.</p>`;
}

function runDm3() {
  if (!D.circ) return;
  D.dm3ran = true;
  const tgt = $('mTarget').value, mode = $('mMode').value, g = +$('mGain').value, mc = +$('mMaskC').value, wl = +$('mWL').value;
  const deg = Math.PI / 180, thetas = Array.from({length: 12}, (_, i) => 15 * i);
  $('mStatus').textContent = 'running…';
  const steps = [];
  const res = {};
  const on = {mode, g, wl, dm3On: true}, off = {mode, g, wl, dm3On: false};
  // 1. tuning with Dm3 off (wiring alone) to get preferences, and with Dm3 on
  steps.push(() => { res.tunOff = thetas.map(t => Dm3Model.resp(t * deg, 0.5, null, 0, off)); });
  steps.push(() => { res.tunOn = thetas.map(t => Dm3Model.resp(t * deg, 0.5, null, 0, on)); });
  steps.push(() => {
    const types = ['Dm3p', 'Dm3q', 'Dm3v', tgt];
    res.tuning = types.map(t => {
      const m = Dm3Model.isType(t);
      const mat = res.tunOn.map(r => Array.from(r).filter((_, i) => m[i]));
      const n = mat[0].length, mx = Array(n).fill(0);
      mat.forEach(row => row.forEach((v, i) => mx[i] = Math.max(mx[i], v)));
      return {t, y: mat.map(row => row.reduce((a, v, i) => a + v / (mx[i] || 1), 0) / n)};
    });
    const m = Dm3Model.isType(tgt);
    const offT = res.tunOff.map(r => Array.from(r).filter((_, i) => m[i]));
    let z = [0, 0];
    offT[0].forEach((_, i) => { let best = 0, bi = 0; offT.forEach((row, k) => { if (row[i] > best) { best = row[i]; bi = k; } }); z[0] += Math.cos(2 * thetas[bi] * deg); z[1] += Math.sin(2 * thetas[bi] * deg); });
    res.pref = ((Math.atan2(z[1], z[0]) / 2 / deg) + 180) % 180;
  });
  const cs = [0.03, 0.06, 0.12, 0.25, 0.5, 1.0];
  steps.push(() => { const m = Dm3Model.isType(tgt); res.crfOn = cs.map(c => Dm3Model.median(Array.from(Dm3Model.resp(res.pref * deg, c, null, 0, on)).filter((_, i) => m[i]))); });
  steps.push(() => { const m = Dm3Model.isType(tgt); res.crfOff = cs.map(c => Dm3Model.median(Array.from(Dm3Model.resp(res.pref * deg, c, null, 0, off)).filter((_, i) => m[i]))); });
  const deltas = [0, 15, 30, 45, 60, 75, 90];
  for (const [k, o] of [['mrOn', on], ['mrOff', off]]) {
    steps.push(() => {
      const m = Dm3Model.isType(tgt), th = res.pref * deg;
      const rt = Array.from(Dm3Model.resp(th, 0.25, null, 0, o)).filter((_, i) => m[i]);
      res[k] = deltas.map(d => {
        const rm = Array.from(Dm3Model.resp(th + d * deg, mc, null, 0, o)).filter((_, i) => m[i]);
        const rb = Array.from(Dm3Model.resp(th, 0.25, th + d * deg, mc, o)).filter((_, i) => m[i]);
        return Dm3Model.median(rb.map((v, i) => (v - rm[i]) / Math.max(rt[i], 1e-6)));
      });
    });
  }
  steps.push(() => { res.map = Dm3Model.resp((res.pref + 90) * deg, 0.5, null, 0, on); });
  let i = 0;
  const next = () => {
    if (i < steps.length) { steps[i++](); $('mStatus').textContent = `running… ${i}/${steps.length}`; setTimeout(next, 0); return; }
    $('mStatus').textContent = `${pretty(tgt)} prefers ${res.pref.toFixed(0)}° gratings (wiring alone).`;
    const colors = {Dm3p: '#b9770e', Dm3q: '#d4a017', Dm3v: '#8e5a0a'};
    Plotly.react('mTuning', res.tuning.map(s => ({x: thetas, y: s.y, name: pretty(s.t), mode: 'lines+markers', line: {color: colors[s.t] || '#2e7d5b', width: s.t === tgt ? 3 : 1.5}})),
      L({title: {text: 'orientation tuning (Dm3 on)', font: {size: 12}}, xaxis: {title: 'grating wave-vector angle (°)', dtick: 30}, yaxis: {title: 'peak / max'}, legend: {orientation: 'h', y: -0.3}}), CFG);
    Plotly.react('mCRF', [{x: cs, y: res.crfOff, name: 'Dm3 removed', mode: 'lines+markers', line: {color: '#9a988f'}}, {x: cs, y: res.crfOn, name: 'Dm3 intact', mode: 'lines+markers', line: {color: '#2e7d5b'}}],
      L({title: {text: `${pretty(tgt)} CRF, preferred grating`, font: {size: 12}}, xaxis: Object.assign({type: 'log', title: 'contrast'}, CTICKS), yaxis: {title: 'response (median cell)'}, legend: {orientation: 'h', y: -0.3}}), CFG);
    Plotly.react('mMask', [{x: deltas, y: res.mrOff, name: 'Dm3 removed', mode: 'lines+markers', line: {color: '#9a988f'}}, {x: deltas, y: res.mrOn, name: 'Dm3 intact', mode: 'lines+markers', line: {color: '#7a4fb5'}}],
      L({title: {text: 'masking ratio vs mask orientation', font: {size: 12}}, xaxis: {title: 'mask − test orientation (°)', dtick: 15}, yaxis: {title: 'test increment on mask / test alone'}, legend: {orientation: 'h', y: -0.3}}), CFG);
    const cen = Dm3Model.centroid(), types = Dm3Model.types();
    const tr = ['Dm3p', 'Dm3q', 'Dm3v'].map((t, k) => { const idx = types.map((x, i) => x === t ? i : -1).filter(i => i >= 0);
      return {x: idx.map(i => cen[i][0]), y: idx.map(i => cen[i][1]), mode: 'markers', type: 'scatter', name: t,
        marker: {symbol: ['triangle-up', 'square', 'circle'][k], size: 6, color: idx.map(i => res.map[i]), colorscale: 'YlOrBr', cmin: 0, showscale: k === 0, colorbar: {thickness: 8, len: 0.6}}}; });
    Plotly.react('mMap', tr, L({title: {text: `Dm3 activity, grating ⟂ to ${pretty(tgt)}'s preference`, font: {size: 12}}, xaxis: {visible: false, scaleanchor: 'y'}, yaxis: {visible: false}, legend: {orientation: 'h', y: -0.05}, margin: {l: 5, r: 5, t: 30, b: 5}}), CFG);
  };
  next();
}

// ---------------------------------------------------------------- flyvis CRFs
const SUR_STYLE = {s0: ['no surround', '#1d1d1b', 'solid'], iso25: ['iso 25%', '#9ecae1', 'solid'], iso50: ['iso 50%', '#4292c6', 'solid'], iso100: ['iso 100%', '#08519c', 'solid'],
  orth50: ['orth 50%', '#fdae6b', 'dash'], orth100: ['orth 100%', '#d94801', 'dash']};
function initCRF() {
  const S = D.sims; if (!S || !S.cells || !Object.keys(S.cells).length) { $('crfPlot').innerHTML = '<p class="hint">Simulation results not available yet.</p>'; return; }
  const types = S.types.filter(t => S.cells[t] && S.cells[t].active);
  $('crfType').innerHTML = types.map(t => `<option ${t === 'T4a' ? 'selected' : ''}>${t}</option>`).join('');
  $('crfType').onchange = () => drawCRF($('crfType').value);
  $('gainSur').onchange = drawGainMap;
  drawCRF('T4a'); drawGainMap();
}
const nrf = (c, p) => p.rmax * c ** p.n / (c ** p.n + p.c50 ** p.n);
function drawCRF(t) {
  const S = D.sims, c = S.cells[t], cs = S.contrasts, tr = [];
  for (const k of S.surround_labels) {
    const [name, col, dash] = SUR_STYLE[k];
    tr.push({x: cs, y: c.mean[k], error_y: {type: 'data', array: c.sem[k], thickness: 1, width: 2, color: col}, name, mode: 'markers', marker: {color: col, size: 7}, legendgroup: k});
    const f = c.fits && c.fits[k];
    if (f) { const xx = Array.from({length: 60}, (_, i) => 0.02 * Math.pow(60, i / 59)); tr.push({x: xx, y: xx.map(v => nrf(v, f)), mode: 'lines', line: {color: col, dash, width: 1.5}, showlegend: false, legendgroup: k, hoverinfo: 'skip'}); }
  }
  Plotly.react('crfPlot', tr, L({title: {text: `${t} · CRF by surround (n=${S.models.length} models)`, font: {size: 13}}, xaxis: Object.assign({type: 'log', title: 'centre contrast'}, CTICKS), yaxis: {title: 'F1 / F1(c=1, no surround)'}}), CFG);
  if (c.size) Plotly.react('sizePlot', [{x: S.radii, y: c.size.mean, error_y: {type: 'data', array: c.size.sem}, mode: 'lines+markers', line: {color: '#2f5d8a'}}],
    L({title: {text: `size tuning (full contrast) · suppression index ${fmt(c.size.size_si)}`, font: {size: 12}}, xaxis: {title: 'grating disk radius (columns)'}, yaxis: {title: 'response / max'}}), CFG);
  else $('sizePlot').innerHTML = '<p class="hint">Size-tuning runs not available.</p>';
}
function drawGainMap() {
  const S = D.sims, k = $('gainSur').value;
  const rows = S.types.filter(t => S.cells[t] && S.cells[t].active && S.cells[t].fits && S.cells[t].fits[k] && S.cells[t].fits.s0).map(t => {
    const f = S.cells[t].fits; return {t, x: Math.log2(Math.min(f[k].c50, 5) / Math.min(f.s0.c50, 5)), y: f[k].rmax / f.s0.rmax, lin: f.s0.c50 > 2};
  });
  Plotly.react('gainMap', [{x: rows.map(r => r.x), y: rows.map(r => r.y), text: rows.map(r => r.t), mode: 'markers+text', textposition: 'top center', textfont: {size: 9, color: '#777'},
    marker: {size: 8, color: rows.map(r => r.lin ? '#bbb' : '#2f5d8a')}, hovertemplate: '<b>%{text}</b><br>log2 c50 ratio %{x:.2f}<br>Rmax ratio %{y:.2f}<extra></extra>'}],
    L({xaxis: {title: 'contrast gain: log2(c50 surround / c50 alone)', zeroline: true}, yaxis: {title: 'response gain: Rmax surround / alone', zeroline: false},
      shapes: [{type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: {color: '#ccc', dash: 'dot'}}, {type: 'line', y0: 1, y1: 1, xref: 'paper', x0: 0, x1: 1, line: {color: '#ccc', dash: 'dot'}}],
      annotations: [{xref: 'paper', yref: 'paper', x: 1, y: 1.06, showarrow: false, text: 'grey = CRF nearly linear (c50 > 2), shifts unreliable', font: {size: 10, color: '#888'}}]}), CFG);
  $('gainMap').on('plotly_click', e => { const t = e.points[0].text; $('crfType').value = t; drawCRF(t); });
  const ss = S.types.filter(t => S.cells[t] && S.cells[t].active).map(t => ({t, iso: S.cells[t].ssi.iso100, orth: S.cells[t].ssi.orth100})).sort((a, b) => b.iso - a.iso);
  Plotly.react('ssiBars', [{x: ss.map(r => r.t), y: ss.map(r => r.iso), type: 'bar', name: 'iso 100%', marker: {color: '#08519c'}}, {x: ss.map(r => r.t), y: ss.map(r => r.orth), type: 'bar', name: 'orth 100%', marker: {color: '#d94801'}}],
    L({title: {text: 'surround suppression index at full contrast (negative = facilitation)', font: {size: 12}}, barmode: 'group', xaxis: {tickangle: -60, tickfont: {size: 9}}, yaxis: {title: 'SSI'}, legend: {orientation: 'h', y: 1.15}, margin: {l: 55, r: 10, t: 40, b: 80}}), CFG);
}

// ---------------------------------------------------------------- lesions
const LES_LABEL = {lateral_inhibition: 'lateral inhibition (all inhibitory cross-column edges)', lateral_excitation: 'lateral excitation (all excitatory cross-column edges)',
  TmY15: 'TmY15 outputs', CT1: 'CT1 outputs', Mi4_Mi9: 'Mi4 + Mi9 outputs', C2_C3: 'C2 + C3 outputs (lamina feedback)', Lawf: 'Lawf1 + Lawf2 outputs', Am: 'Am (lamina amacrine) outputs'};
function initLesion() {
  const S = D.sims; if (!S || !S.lesions || !Object.keys(S.lesions).length) { $('lesBars').innerHTML = '<p class="hint">Lesion runs not available yet.</p>'; return; }
  const names = Object.keys(S.lesions);
  $('lesName').innerHTML = names.map(n => `<option value="${n}">${LES_LABEL[n] || n}</option>`).join('');
  const types = S.types.filter(t => S.cells[t] && S.cells[t].active);
  $('lesType').innerHTML = types.map(t => `<option ${t === 'T4a' ? 'selected' : ''}>${t}</option>`).join('');
  ['lesName', 'lesType', 'lesMetric'].forEach(k => $(k).onchange = drawLesion);
  drawLesion();
  // heatmap
  const z = names.map(n => types.map(t => { const c = S.lesions[n].cells[t]; return (c && c.ssi_lesioned.iso100 !== null) ? c.ssi_lesioned.iso100 - c.ssi_intact.iso100 : null; }));
  Plotly.react('lesHeat', [{type: 'heatmap', z, x: types, y: names.map(n => LES_LABEL[n] || n), colorscale: 'RdBu', zmid: 0, zmin: -0.5, zmax: 0.5, reversescale: true, colorbar: {title: 'ΔSSI'}, hoverongaps: false}],
    L({title: {text: 'change in iso surround suppression after each lesion (red = more suppression; blank = cell silenced by the lesion)', font: {size: 12}}, xaxis: {tickangle: -60, tickfont: {size: 9}}, yaxis: {automargin: true}, margin: {l: 280, r: 10, t: 40, b: 90}}), CFG);
}
function drawLesion() {
  const S = D.sims, n = $('lesName').value, t = $('lesType').value, m = $('lesMetric').value, Ls = S.lesions[n];
  const types = S.types.filter(x => S.cells[x] && S.cells[x].active && Ls.cells[x]);
  const dd = (a, b) => (a === null || b === null) ? null : a - b;
  const val = x => { const c = Ls.cells[x]; return m === 'gain' ? c.gain_change : m === 'ssi_iso' ? dd(c.ssi_lesioned.iso100, c.ssi_intact.iso100) : dd(c.ssi_lesioned.orth100, c.ssi_intact.orth100); };
  const rows = types.map(x => ({x, v: val(x)})).filter(r => r.v !== null).sort((a, b) => b.v - a.v);
  Plotly.react('lesBars', [{x: rows.map(r => r.x), y: rows.map(r => r.v), type: 'bar', marker: {color: rows.map(r => r.x === t ? '#7a4fb5' : '#2f5d8a')}}],
    L({title: {text: `${LES_LABEL[n] || n} · ${$('lesMetric').selectedOptions[0].text} (n=${Ls.models.length} models; silenced cells omitted)`, font: {size: 12}}, xaxis: {tickangle: -60, tickfont: {size: 9}}, yaxis: m === 'gain' ? {type: 'log', title: '× intact'} : {title: 'ΔSSI'}, margin: {l: 55, r: 10, t: 40, b: 80}}), CFG);
  $('lesBars').on('plotly_click', e => { $('lesType').value = e.points[0].x; drawLesion(); });
  const c = Ls.cells[t], cs = S.lesion_contrasts, tr = [];
  for (const k of ['s0', 'iso100', 'orth100']) {
    const [name, col] = SUR_STYLE[k];
    tr.push({x: cs, y: c.intact[k], name: name + ' · intact', mode: 'lines+markers', line: {color: col}});
    tr.push({x: cs, y: c.lesioned[k], name: name + ' · lesioned', mode: 'lines+markers', line: {color: col, dash: 'dot'}, marker: {symbol: 'circle-open'}});
  }
  Plotly.react('lesCRF', tr, L({title: {text: `${t}: intact vs lesioned`, font: {size: 12}}, xaxis: Object.assign({type: 'log', title: 'centre contrast'}, CTICKS), yaxis: {title: 'F1 / intact F1(c=1)'}}), CFG);
}

// ---------------------------------------------------------------- grafted pools
const CFG_LABEL = c => {
  if (c === 'intact') return 'intact model';
  const m = c.match(/^(div|sub)_(all|Dm17|Dm1|Dm15)_g([\d.]+)$/);
  if (m) return `${m[1] === 'div' ? 'divisive' : 'subtractive'} · ${m[2] === 'all' ? 'Dm17 + Dm1 + Dm15' : m[2] + ' only'} · g = ${m[3]}`;
  if (c === 'ft_div_learned') return 'fine-tuned on optic flow · divisive pools on (learned gains)';
  if (c === 'ft_nopool') return 'fine-tuned on optic flow · pools switched off';
  return c;
};
function grCells(c) { return c === 'intact' ? D.graft.intact : D.graft.configs[c].cells; }
function initGraft() {
  const G = D.graft; if (!G || !G.configs || !Object.keys(G.configs).length) { $('grCRF').innerHTML = '<p class="hint">Graft results not available yet.</p>'; return; }
  const types = Object.keys(G.intact);
  $('grType').innerHTML = types.map(t => `<option ${t === 'Tm2' ? 'selected' : ''}>${t}</option>`).join('');
  const cfgs = Object.keys(G.configs).sort((a, b) => (a.startsWith('ft') ? -1 : 0) - (b.startsWith('ft') ? -1 : 0));
  $('grCfg').innerHTML = cfgs.map(c => `<option value="${c}" ${c === 'div_Dm17_g2' ? 'selected' : ''}>${CFG_LABEL(c)}</option>`).join('');
  ['grType', 'grCfg'].forEach(k => $(k).onchange = drawGraft);
  drawGraft(); drawGraftTask();
  $('grCallout').innerHTML = D.find?.graft || '';
}
function drawGraft() {
  const G = D.graft, t = $('grType').value, c = $('grCfg').value, cs = G.contrasts;
  const I = G.intact[t], X = grCells(c)[t];
  const tr = [];
  for (const k of ['s0', 'iso100', 'orth100']) {
    const [name, col] = SUR_STYLE[k];
    tr.push({x: cs, y: I.mean[k], name: name + ' · intact', mode: 'lines+markers', line: {color: col, width: 1.5}, opacity: 0.55});
    if (X) tr.push({x: cs, y: X.mean[k], name: name + ' · grafted', mode: 'lines+markers', line: {color: col, width: 3}, marker: {symbol: 'diamond', size: 8}});
  }
  Plotly.react('grCRF', tr, L({title: {text: `${t}: intact vs ${CFG_LABEL(c)}`, font: {size: 12}}, xaxis: Object.assign({type: 'log', title: 'centre contrast'}, CTICKS), yaxis: {title: 'F1 / intact F1(c=1, no surround)'}}), CFG);
  // gain sweep: SSI vs g for divisive vs subtractive (all pools)
  const sweep = mode => Object.keys(G.configs).filter(k => k.startsWith(mode + '_all_')).map(k => ({g: +k.split('_g')[1], k})).sort((a, b) => a.g - b.g);
  const tr2 = [];
  for (const [mode, col] of [['div', '#7a4fb5'], ['sub', '#b9770e']]) {
    const sw = [{g: 0, k: 'intact'}].concat(sweep(mode));
    tr2.push({x: sw.map(r => r.g), y: sw.map(r => grCells(r.k)[t]?.ssi.iso100), name: `${mode === 'div' ? 'divisive' : 'subtractive'} · iso surround`, mode: 'lines+markers', line: {color: col}});
    tr2.push({x: sw.map(r => r.g), y: sw.map(r => grCells(r.k)[t]?.ssi.orth100), name: `${mode === 'div' ? 'divisive' : 'subtractive'} · orth surround`, mode: 'lines+markers', line: {color: col, dash: 'dot'}});
  }
  Plotly.react('grSweep', tr2, L({title: {text: `${t}: surround suppression vs pool gain (full contrast)`, font: {size: 12}}, xaxis: {title: 'pool gain g', type: 'log', tickvals: [0.5, 2, 8]}, yaxis: {title: 'SSI (positive = suppression)'},
    shapes: [{type: 'line', y0: 0, y1: 0, xref: 'paper', x0: 0, x1: 1, line: {color: '#ccc', dash: 'dot'}}]}), CFG);
  // across types: how the pools change each cell's own CRF (centre alone) vs intact
  const C = grCells(c), rows = Object.keys(C).filter(k => C[k].fits && C[k].fits.s0 && G.intact[k].fits && G.intact[k].fits.s0).map(k => {
    const a = G.intact[k].fits.s0, b = C[k].fits.s0; return {t: k, x: Math.log2(Math.min(b.c50, 5) / Math.min(a.c50, 5)), y: C[k].amp, ssi: C[k].ssi.iso100 - G.intact[k].ssi.iso100};
  });
  Plotly.react('grMap', [{x: rows.map(r => r.x), y: rows.map(r => r.y), text: rows.map(r => r.t), mode: 'markers+text', textposition: 'top center', textfont: {size: 9, color: '#777'},
    marker: {size: 9, color: rows.map(r => r.ssi), colorscale: 'RdBu', reversescale: true, cmid: 0, cmin: -0.4, cmax: 0.4, colorbar: {title: 'ΔSSI', thickness: 10}},
    hovertemplate: '<b>%{text}</b><br>log2 c50 shift %{x:.2f}<br>response at c=1: ×%{y:.2f}<extra></extra>'}],
    L({title: {text: 'centre-only CRF change vs intact (colour: change in iso surround suppression)', font: {size: 12}}, xaxis: {title: 'contrast gain: log2(c50 grafted / intact)'}, yaxis: {title: 'response at full contrast (× intact)', type: 'log'},
      shapes: [{type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: {color: '#ccc', dash: 'dot'}}, {type: 'line', y0: 1, y1: 1, xref: 'paper', x0: 0, x1: 1, line: {color: '#ccc', dash: 'dot'}}]}), CFG);
  $('grMap').on('plotly_click', e => { $('grType').value = e.points[0].text; drawGraft(); });
}
function drawGraftTask() {
  const F = D.graft.finetune || {}, ms = Object.keys(F);
  if (!ms.length) { $('grTask').innerHTML = '<p class="hint">Fine-tuning results not available.</p>'; return; }
  const f = F[ms[0]], g = f.graft, c = f.control;
  const sm = (a, w = 10) => a.map((_, i) => a.slice(Math.max(0, i - w + 1), i + 1).reduce((x, y) => x + y, 0) / Math.min(i + 1, w));
  const tr = [];
  if (g) tr.push({x: g.train_loss.map((_, i) => i + 1), y: sm(g.train_loss), name: 'training · grafted', mode: 'lines', line: {color: '#c9b3e6'}});
  if (c) tr.push({x: c.train_loss.map((_, i) => i + 1), y: sm(c.train_loss), name: 'training · no pools (control)', mode: 'lines', line: {color: '#cfcdc5'}});
  const bars = [['original model', g ? g.val_intact : c.val_intact, '#1d1d1b'], ...(g ? [['+ pools, before fine-tuning', g.val_grafted, '#d94801'], ['+ pools, fine-tuned', g.val_finetuned, '#7a4fb5']] : []),
    ...(c ? [['no pools, same fine-tuning', c.val_finetuned, '#9a988f']] : [])];
  Plotly.react('grTask', [{type: 'bar', x: bars.map(b => b[0]), y: bars.map(b => b[1]), marker: {color: bars.map(b => b[2])}, text: bars.map(b => b[1].toFixed(1)), textposition: 'outside'}],
    L({title: {text: `optic-flow validation error (held-out fold), model ${ms[0]}`, font: {size: 12}}, yaxis: {title: 'L2 flow error', range: [Math.min(...bars.map(b => b[1])) * 0.95, Math.max(...bars.map(b => b[1])) * 1.03]}, showlegend: false, margin: {l: 55, r: 10, t: 40, b: 70}}), CFG);
  if (g) $('grTaskNote').innerHTML = `Fine-tuning: ${g.train_loss.length} iterations on CPU (the published models had 250,000). Pool gains start at ${fmt(g.g_init, 1)} and are learned: ` +
    Object.entries(g.gains_final).map(([k, v]) => `${k} → ${fmt(v)}`).join(', ') + `. Validation fold: ${g.n_val} sequences.`;
}

// ---------------------------------------------------------------- methods
function renderMethods() {
  $('methodsText').innerHTML = D.find?.methods || '';
}
