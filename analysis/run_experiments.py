"""In-silico contrast experiments on the flyvis connectome-constrained ensemble.

Experiments
  crf    centre contrast x surround contrast x direction (iso-oriented surround)
  size   grating-disk radius x direction at full contrast (size tuning)
  lesion crf at surround {0, 1} after cutting a subconnectome
Responses are the central-column cell of each of the 65 cell types; we store the
F1 (stimulus-frequency) amplitude and the DC shift over the last second.
"""
import os, sys, json, argparse, time, itertools, numpy as np, torch
sys.path.insert(0, 'analysis')
from model_utils import ranked_models, load, edge_table, set_lesion, DT
from stimuli import grating_movie
from flyvis.utils.activity_utils import LayerActivity

torch.set_num_threads(2)
TF = 2.0
CONTRASTS = [0.03, 0.06, 0.12, 0.25, 0.5, 1.0]
SURROUNDS = [0.0, 0.25, 0.5, 1.0]
DIRS = [0, 90, 180, 270]
RADII = [1, 2, 3, 4, 6, 8, 11, 15]
T_PRE, T_STIM = 1.0, 2.0
EXCLUDE_LATERAL = {f'R{i}' for i in range(1, 9)} | {f'L{i}' for i in range(1, 6)}


def lesion_mask(net, name):
    st, tt = edge_table(net)
    e = net.connectome.edges
    sign = np.asarray(e.sign[:])
    lateral = (np.asarray(e.du[:]) != 0) | (np.asarray(e.dv[:]) != 0)
    keep = np.ones(len(st))
    groups = {
        'TmY15': ['TmY15'], 'TmY14': ['TmY14'], 'Mi12': ['Mi12'],
        'CT1': ['CT1(M10)', 'CT1(Lo1)'], 'C2_C3': ['C2', 'C3'],
        'Lawf': ['Lawf1', 'Lawf2'], 'Am': ['Am'], 'Mi4_Mi9': ['Mi4', 'Mi9'],
    }
    if name in groups:
        keep[np.isin(st, groups[name])] = 0
    elif name == 'lateral_inhibition':
        keep[(sign < 0) & lateral & ~np.isin(st, list(EXCLUDE_LATERAL))] = 0
    elif name == 'lateral_excitation':
        keep[(sign > 0) & lateral & ~np.isin(st, list(EXCLUDE_LATERAL))] = 0
    else:
        raise ValueError(name)
    return keep, int((keep == 0).sum())


def conditions(exp):
    # o = orientation offset of the surround grating (0 iso, 90 orthogonal)
    if exp == 'crf':
        sur = [(s, 0) for s in SURROUNDS] + [(0.5, 90), (1.0, 90)]
        return [dict(c=c, s=s, o=o, d=d, r=2.0) for c, (s, o), d in itertools.product(CONTRASTS, sur, DIRS)]
    if exp == 'lesion':
        sur = [(0.0, 0), (1.0, 0), (1.0, 90)]
        return [dict(c=c, s=s, o=o, d=d, r=2.0) for c, (s, o), d in itertools.product(CONTRASTS[1:], sur, DIRS)]
    if exp == 'size':
        return [dict(c=1.0, s=0.0, o=0, d=d, r=r) for r, d in itertools.product(RADII, DIRS)]


def movie(cond):
    return grating_movie(cond['c'], cond['s'], cond['d'], r_center=cond['r'], r_gap=cond['r'] + 2 if cond['r'] < 4 else 99,
                         dt=DT, t_pre=T_PRE, t_stim=T_STIM, tf=TF,
                         surround_dir_offset=cond.get('o', 0))[0]


def measure(resp, t):
    """resp: (batch, frames, cells) central responses -> F1 amp and DC shift."""
    win = t >= (T_PRE + T_STIM - 1.0)
    base = (t >= 0.5) & (t < T_PRE)
    ph = np.exp(-2j * np.pi * TF * t[win])
    f1 = 2 * np.abs((resp[:, win, :] * ph[None, :, None]).mean(1))
    dc = resp[:, win, :].mean(1) - resp[:, base, :].mean(1)
    return f1, dc


def run(net, conds, batch=4):
    global TYPES
    t = np.arange(0, T_PRE + T_STIM, DT)
    F1, DC = [], []
    for i in range(0, len(conds), batch):
        x = torch.tensor(np.stack([movie(c) for c in conds[i:i + batch]]))
        with torch.no_grad():
            s = net.fade_in_state(1.0, DT, x[:, 0])
            r = net.simulate(x, DT, initial_state=s).numpy()
        la = LayerActivity(r, net.connectome, keepref=True)
        cen = np.stack([la.central[ct] for ct in TYPES], -1)  # (b, frames, types)
        f1, dc = measure(cen.reshape(cen.shape[0], cen.shape[1], -1), t)
        F1.append(f1); DC.append(dc)
    return np.concatenate(F1), np.concatenate(DC)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('exp'); ap.add_argument('--n_models', type=int, default=10)
    ap.add_argument('--lesions', default='')
    a = ap.parse_args()
    models, losses = ranked_models()
    models = models[:a.n_models]
    lesions = [l for l in a.lesions.split(',') if l] if a.exp == 'lesion' else ['none']
    conds = conditions(a.exp)
    for lname in lesions:
        F1s, DCs, ncut = [], [], 0
        t0 = time.time()
        for m in models:
            net = load(m)
            TYPES = list(np.array(net.connectome.unique_cell_types[:]).astype(str))
            if lname != 'none':
                mask, ncut = lesion_mask(net, lname); set_lesion(net, mask)
            cache = f'analysis/results/cache/{a.exp}_{lname}_{m}.npz'
            if os.path.exists(cache):
                z = np.load(cache); f1, dc = z['f1'], z['dc']
            else:
                f1, dc = run(net, conds)
                np.savez(cache, f1=f1, dc=dc)
            F1s.append(f1); DCs.append(dc)
            print(a.exp, lname, m, f'{time.time()-t0:.0f}s', flush=True)
        np.savez_compressed(f'analysis/results/{a.exp}_{lname}.npz', f1=np.stack(F1s), dc=np.stack(DCs),
                            models=models, losses=[losses[m] for m in models], types=TYPES,
                            conds=json.dumps(conds), n_cut_edges=ncut)
