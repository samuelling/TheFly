"""Shared helpers for the assemblers (Naka-Rushton fits, CRF extraction)."""
import numpy as np
from scipy.optimize import curve_fit
import sys; sys.path.insert(0, 'analysis')
import run_experiments as RX
import json
types = json.load(open('docs/data/sims.json'))['types']

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



