"""Assemble cached flyvis simulation results into docs/data/sims.json.

Per cell type (central column, preferred drift direction per model):
  CRFs (F1 amplitude) for each surround condition, mean and SEM across models
  Naka-Rushton fits (exponent fixed from the no-surround fit) -> contrast-gain and response-gain shifts
  surround suppression index SSI = 1 - R(c=1, surround) / R(c=1, no surround), iso vs orthogonal
  size tuning and its suppression index; lesion effects on all of the above
"""
import glob, json, os, re, numpy as np
from scipy.optimize import curve_fit
import sys; sys.path.insert(0, 'analysis')
import run_experiments as RX
from model_utils import ranked_models

models, losses = ranked_models()
types = None


def load(exp, lesion):
    out = []
    for m in models:
        f = f'analysis/results/cache/{exp}_{lesion}_{m}.npz'
        if os.path.exists(f):
            out.append((m, np.load(f)['f1']))
    return out


def nr(c, rmax, c50, n):
    return rmax * c ** n / (c ** n + c50 ** n)


def fit_family(cs, curves):
    """curves: dict label -> mean curve. Fit base (no surround) with free n, others share n."""
    cs = np.asarray(cs)
    res = {}
    base = curves['s0']
    try:
        p0, _ = curve_fit(nr, cs, base, p0=[base.max() * 1.2, 0.3, 2], bounds=([1e-6, 0.01, 0.5], [50, 10, 6]), maxfev=20000)
    except Exception:
        return None
    n = p0[2]; res['s0'] = dict(rmax=p0[0], c50=p0[1], n=n)
    for k, y in curves.items():
        if k == 's0':
            continue
        try:
            p, _ = curve_fit(lambda c, a, b: nr(c, a, b, n), cs, y, p0=[p0[0], p0[1]], bounds=([1e-6, 0.01], [50, 10]), maxfev=20000)
            res[k] = dict(rmax=p[0], c50=p[1], n=n)
        except Exception:
            res[k] = None
    return {k: (None if v is None else {a: round(float(b), 4) for a, b in v.items()}) for k, v in res.items()}


crf = load('crf', 'none')
from flyvis import NetworkView
types = list(np.array(np.load(glob.glob('analysis/results/crf_none.npz')[0], allow_pickle=True)['types'])) if os.path.exists('analysis/results/crf_none.npz') else None
if types is None:
    import model_utils
    net = model_utils.load(models[0]); types = list(np.array(net.connectome.unique_cell_types[:]).astype(str))
conds = RX.conditions('crf')
CS = RX.CONTRASTS
SUR = [('s0', 0.0, 0), ('iso25', 0.25, 0), ('iso50', 0.5, 0), ('iso100', 1.0, 0), ('orth50', 0.5, 90), ('orth100', 1.0, 90)]


def index(conds, c, s, o, d):
    for i, k in enumerate(conds):
        if k['c'] == c and k['s'] == s and k.get('o', 0) == o and k['d'] == d:
            return i
    return None


def crf_curves(f1, conds, sur, cs, pref):
    """f1 (conds, types) -> dict label -> (contrasts, types) at each type's pref dir."""
    out = {}
    for lab, s, o in sur:
        arr = np.zeros((len(cs), len(types)))
        for j, c in enumerate(cs):
            for t in range(len(types)):
                arr[j, t] = f1[index(conds, c, s, o, pref[t]), t]
        out[lab] = arr
    return out


prefs = {}
per_model = []
for m, f1 in crf:
    full = [index(conds, 1.0, 0.0, 0, d) for d in RX.DIRS]
    pref = [RX.DIRS[int(np.argmax(f1[full, t]))] for t in range(len(types))]
    prefs[m] = pref
    per_model.append(crf_curves(f1, conds, SUR, CS, pref))

S = dict(models=[m for m, _ in crf], losses=[losses[m] for m, _ in crf], contrasts=CS, types=types,
         surround_labels=[s[0] for s in SUR], dirs=RX.DIRS, radii=RX.RADII, cells={})
if per_model:
    for ti, t in enumerate(types):
        cur = {lab: np.stack([pm[lab][:, ti] for pm in per_model]) for lab, _, _ in SUR}  # (models, contrasts)
        scale = cur['s0'][:, -1:].copy(); scale[scale <= 1e-9] = 1
        norm = {k: v / scale for k, v in cur.items()}
        mean = {k: v.mean(0) for k, v in norm.items()}
        sem = {k: (v.std(0) / np.sqrt(len(v))) for k, v in norm.items()}
        fits = fit_family(CS, mean) if cur['s0'].mean() > 1e-4 else None
        ssi = {k: float(np.mean(1 - norm[k][:, -1])) for k in mean if k != 's0'}
        S['cells'][t] = dict(raw_amp=float(cur['s0'][:, -1].mean()), mean={k: v.round(4).tolist() for k, v in mean.items()},
                             sem={k: v.round(4).tolist() for k, v in sem.items()}, fits=fits,
                             ssi={k: round(v, 4) for k, v in ssi.items()},
                             pref_dirs=[prefs[m][ti] for m, _ in crf])

# size tuning
size = load('size', 'none')
sc = RX.conditions('size')
for ti, t in enumerate(types):
    if t not in S['cells']:
        continue
    curves = []
    for m, f1 in size:
        pd_ = prefs.get(m, [0] * len(types))[ti]
        curves.append([f1[[i for i, k in enumerate(sc) if k['r'] == r and k['d'] == pd_][0], ti] for r in RX.RADII])
    if curves:
        cv = np.array(curves); mx = cv.max(1, keepdims=True); mx[mx <= 1e-9] = 1
        cvn = cv / mx
        S['cells'][t]['size'] = dict(mean=cvn.mean(0).round(4).tolist(), sem=(cvn.std(0) / np.sqrt(len(cvn))).round(4).tolist(),
                                     size_si=round(float(np.mean(1 - cvn[:, -1])), 4))

# lesions
lc = RX.conditions('lesion'); LCS = RX.CONTRASTS[1:]
LSUR = [('s0', 0.0, 0), ('iso100', 1.0, 0), ('orth100', 1.0, 90)]
S['lesions'] = {}
for f in sorted(glob.glob('analysis/results/cache/lesion_*_*.npz')):
    pass
lesion_names = sorted({re.match(r'.*/lesion_(.+)_(\d{3})\.npz', f).group(1) for f in glob.glob('analysis/results/cache/lesion_*_*.npz')})
for ln in lesion_names:
    res = load('lesion', ln)
    if not res:
        continue
    ms = [m for m, _ in res]
    intact = {m: crf_curves(f1, conds, LSUR, LCS, prefs[m]) for m, f1 in crf if m in ms}
    les = {m: crf_curves(f1, lc, LSUR, LCS, prefs[m]) for m, f1 in res if m in prefs}
    ms = [m for m in ms if m in intact and m in les]
    L = dict(models=ms, cells={})
    for ti, t in enumerate(types):
        I = {k: np.stack([intact[m][k][:, ti] for m in ms]) for k, _, _ in LSUR}
        X = {k: np.stack([les[m][k][:, ti] for m in ms]) for k, _, _ in LSUR}
        sc_ = I['s0'][:, -1:].copy(); sc_[sc_ <= 1e-9] = 1
        ssi_i = {k: float(np.mean(1 - I[k][:, -1] / sc_[:, 0])) for k in ('iso100', 'orth100')}
        xs = X['s0'][:, -1:].copy()
        ok = bool(np.all(xs[:, 0] > 0.1 * sc_[:, 0]))  # lesioned cell still responds (>10% of intact)
        xs[xs <= 1e-9] = 1
        ssi_x = {k: (float(np.mean(1 - X[k][:, -1] / xs[:, 0])) if ok else None) for k in ('iso100', 'orth100')}
        L['cells'][t] = dict(intact={k: (v / sc_).mean(0).round(4).tolist() for k, v in I.items()},
                             lesioned={k: (v / sc_).mean(0).round(4).tolist() for k, v in X.items()},
                             gain_change=round(float(np.exp(np.mean(np.log(np.maximum(X['s0'][:, -1], 1e-6) / sc_[:, 0])))), 4),
                             ssi_intact={k: round(v, 4) for k, v in ssi_i.items()},
                             ssi_lesioned={k: (None if v is None else round(v, 4)) for k, v in ssi_x.items()})
    S['lesions'][ln] = L
S['lesion_contrasts'] = LCS
amps = np.array([c['raw_amp'] for c in S['cells'].values()])
thr = 0.05 * float(np.median(amps))
for t, c in S['cells'].items():
    c['active'] = bool(c['raw_amp'] > thr)
S['activity_threshold'] = thr
json.dump(S, open('docs/data/sims.json', 'w'), separators=(',', ':'), allow_nan=False, default=lambda o: None)
print('cells', len(S['cells']), 'lesions', list(S['lesions']), 'models', S['models'])
