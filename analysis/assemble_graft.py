"""Assemble grafted-pool experiments into docs/data/graft.json."""
import glob, json, os, re, sys, numpy as np
sys.path.insert(0, 'analysis')
import run_experiments as RX
import graft_experiments as GX
from assemble_sims_lib import fit_family, crf_curves, index

S = json.load(open('docs/data/sims.json'))
types = S['types']
crf_conds = RX.conditions('crf')
LAB = [('s0', 0.0, 0), ('iso50', 0.5, 0), ('iso100', 1.0, 0), ('orth100', 1.0, 90)]
CS = RX.CONTRASTS
files = sorted(glob.glob('analysis/results/graft/*_[0-9][0-9][0-9].npz'))
configs = sorted({re.match(r'.*/(.+)_(\d{3})\.npz', f).group(1) for f in files})
models = sorted({re.match(r'.*_(\d{3})\.npz', f).group(1) for f in files})
out = dict(types=types, contrasts=CS, labels=[l[0] for l in LAB], models=models, configs={}, intact={})


def curves_for(f1, conds, pref):
    return crf_curves(f1, conds, LAB, CS, pref)


base = {}
for m in models:
    f1 = np.load(f'analysis/results/cache/crf_none_{m}.npz')['f1']
    full = [index(crf_conds, 1.0, 0.0, 0, d) for d in RX.DIRS]
    pref = [RX.DIRS[int(np.argmax(f1[full, t]))] for t in range(len(types))]
    base[m] = dict(pref=pref, cur=curves_for(f1, crf_conds, pref), f1=f1, conds=crf_conds)


def dsi(f1, conds, pref, t):
    p = pref[t]; o = (p + 180) % 360
    a = f1[index(conds, 1.0, 0.0, 0, p), t]; b = f1[index(conds, 1.0, 0.0, 0, o), t]
    return float((a - b) / (a + b + 1e-9))


def summarize(get_cur, get_dsi, ms):
    cells = {}
    for ti, t in enumerate(types):
        if not S['cells'][t].get('active'):
            continue
        cur = {k: np.stack([get_cur(m)[k][:, ti] for m in ms]) for k, _, _ in LAB}
        sc = np.stack([base[m]['cur']['s0'][-1, ti] for m in ms])[:, None]; sc[sc <= 1e-9] = 1
        norm = {k: v / sc for k, v in cur.items()}
        mean = {k: v.mean(0) for k, v in norm.items()}
        own = cur['s0'][:, -1:].copy(); own[own <= 1e-9] = 1
        ssi = {k: float(np.mean(1 - cur[k][:, -1] / own[:, 0])) for k in ('iso50', 'iso100', 'orth100')}
        fits = fit_family(CS, {k: mean[k] / max(mean['s0'][-1], 1e-9) for k in mean})
        cells[t] = dict(mean={k: v.round(4).tolist() for k, v in mean.items()}, ssi={k: round(v, 4) for k, v in ssi.items()},
                        fits=fits, amp=round(float(np.mean(cur['s0'][:, -1] / sc[:, 0])), 4),
                        dsi=round(float(np.mean([get_dsi(m, ti) for m in ms])), 3) if t[:2] in ('T4', 'T5') else None)
    return cells


out['intact'] = summarize(lambda m: base[m]['cur'], lambda m, ti: dsi(base[m]['f1'], base[m]['conds'], base[m]['pref'], ti), models)
for c in configs:
    ms = [m for m in models if os.path.exists(f'analysis/results/graft/{c}_{m}.npz')]
    data = {m: np.load(f'analysis/results/graft/{c}_{m}.npz')['f1'] for m in ms}
    cur = {m: curves_for(data[m], GX.CONDS, base[m]['pref']) for m in ms}
    out['configs'][c] = dict(models=ms, cells=summarize(lambda m: cur[m], lambda m, ti: dsi(data[m], GX.CONDS, base[m]['pref'], ti), ms))
ft = {}
for f in glob.glob('analysis/results/graft/finetune_*_*.json'):
    kind = re.match(r'.*/finetune_(graft|control)_\d{3}\.json', f).group(1)
    d = json.load(open(f)); d['train_loss'] = [round(x, 4) for x in d['train_loss']]
    ft.setdefault(d['model'], {})[kind] = d
out['finetune'] = ft
out['calib'] = {m: json.load(open(f'analysis/results/graft/calib_{m}.json')) for m in models if os.path.exists(f'analysis/results/graft/calib_{m}.json')}
out['pools'] = json.load(open('analysis/graft_pools.json'))
json.dump(out, open('docs/data/graft.json', 'w'), separators=(',', ':'), default=lambda o: None)
print('configs', configs, 'models', models)
for c in configs:
    cc = out['configs'][c]['cells']
    ss = np.array([cc[t]['ssi']['iso100'] for t in cc]); amp = np.array([cc[t]['amp'] for t in cc])
    print(f'{c:16s} median SSI iso {np.median(ss):+.3f}  n>0.1: {(ss > 0.1).sum():2d}  median amp {np.median(amp):.2f}',
          ' T4/T5 DSI', np.round(np.mean([cc[t]['dsi'] for t in cc if cc[t]['dsi'] is not None]), 3))
