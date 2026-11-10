"""Part 1b: candidate gain-control pools and orientation signatures in the FlyWire right optic lobe.

Inputs: FlyWire v783 proofread connections (Zenodo 10676866), neuron annotations
(flyconnectome/flywire_annotations, incl. Matsliah et al. optic-lobe types) and the
Matsliah et al. 2024 column assignment of columnar cells (via connectome-interpreter).

For every neuron in the right optic lobe we map the columnar cells that synapse onto it
to their columns, then measure per neuron:
  footprint   number of distinct columns providing input
  radius      synapse-weighted RMS distance of those columns from their centroid (column units)
  elongation  (l1 - l2) / (l1 + l2) of the weighted input covariance, *in excess of* a compact
              disk with the same number of columns at the same place on the (irregular) lattice
  axis        orientation of the excess elongation (deg, eye-map frame)
separately for excitatory (ACh) and inhibitory (GABA, Glu) input. Output fractions onto the
motion-detector inputs and onto lamina monopolar cells identify candidate normalisation pools.
"""
import json, numpy as np, pandas as pd, pyarrow.feather as feather
from connectome_interpreter.external_map import load_dataset

D = '/home/claude/data/flywire/'
OPTIC = {'ME_R', 'LO_R', 'LOP_R', 'LA_R', 'AME_R'}
MOTION_IN = {'Mi1', 'Tm3', 'Mi4', 'Mi9', 'C3', 'CT1', 'TmY15', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'C2', 'Mi10', 'TmY4'}
MOTION = {f'T{a}{b}' for a in '45' for b in 'abcd'}
LMC = {'L1', 'L2', 'L3', 'L4', 'L5'}
MIN_SYN = 3

ann = pd.read_csv(D + 'neuron_annotations.tsv', sep='\t', dtype=str,
                  usecols=['root_id', 'super_class', 'cell_type', 'top_nt', 'side'])
ann['root_id'] = ann.root_id.astype(np.int64)
tmap = ann.set_index('root_id').cell_type
ntmap = ann.set_index('root_id').top_nt

t = feather.read_table(D + 'proofread_connections_783.feather', columns=['pre_pt_root_id', 'post_pt_root_id', 'neuropil', 'syn_count'])
C = t.to_pandas(); del t
C = C[C.neuropil.isin(OPTIC) & (C.syn_count >= MIN_SYN)]
C = C.groupby(['pre_pt_root_id', 'post_pt_root_id'], as_index=False).syn_count.sum()
C.columns = ['pre', 'post', 'n']
C['pre_type'] = C.pre.map(tmap); C['post_type'] = C.post.map(tmap)
C['pre_nt'] = C.pre.map(ntmap)
print('optic-lobe connections', len(C))

col = load_dataset('Matsliah2024')
col = col[col.hemisphere == 'right']
# Matsliah coordinates are doubled-height hex coords: neighbours at (+-1, +-1) and (0, +-2).
# Convert to Euclidean so all six neighbours sit 1 column-spacing apart.
col = col.assign(x=col.x.astype(float) * np.sqrt(3) / 2, y=col.y.astype(float) / 2)
pos = col.set_index('root_id')[['x', 'y']]
lat = col.drop_duplicates('column_id')[['x', 'y']].values.astype(float)
C = C.join(pos, on='pre')


def cov_elong(xy, w):
    w = w / w.sum(); m = (xy * w[:, None]).sum(0); z = xy - m
    S = (z.T * w) @ z
    return S, m


def shape_metrics(xy, w):
    n = len(xy)
    S, m = cov_elong(xy, w)
    # compact-disk null on the actual lattice: the n columns nearest the weighted centroid
    near = lat[np.argsort(((lat - m) ** 2).sum(1))[:n]]
    S0, _ = cov_elong(near, np.ones(n))
    tr, tr0 = np.trace(S), np.trace(S0)
    radius = float(np.sqrt(tr))
    if n < 4 or tr <= 0 or tr0 <= 0:
        return dict(footprint=n, radius=radius, elong=np.nan, axis=np.nan)
    dS = S / tr - S0 / tr0
    ev, evec = np.linalg.eigh(dS)
    elong = float(ev[1] - ev[0])  # excess anisotropy, 0 = as round as a disk on this lattice
    axis = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180)
    return dict(footprint=n, radius=radius, elong=elong, axis=axis)


EXC = {'acetylcholine'}; INH = {'gaba', 'glutamate'}
withpos = C.dropna(subset=['x'])
import os
CACHE = '/home/claude/data/flywire/neuron_shapes.pkl'
rows = []
for post, g in ([] if os.path.exists(CACHE) else withpos.groupby('post')):
    rec = dict(root_id=post, type=tmap.get(post))
    for lab, sel in (('all', g), ('exc', g[g.pre_nt.isin(EXC)]), ('inh', g[g.pre_nt.isin(INH)])):
        if len(sel) == 0:
            continue
        cols = sel.groupby(['x', 'y']).n.sum().reset_index()
        m = shape_metrics(cols[['x', 'y']].values, cols.n.values.astype(float))
        for k, v in m.items():
            rec[f'{lab}_{k}'] = v
    rows.append(rec)
N = pd.read_pickle(CACHE) if os.path.exists(CACHE) else pd.DataFrame(rows)
N.to_pickle(CACHE)
N = N[N.type.notna()]
print('neurons with mapped columnar input', len(N))

# per-type output targets
out = C.groupby(['pre_type']).n.sum()
to_mot = C[C.post_type.isin(MOTION_IN)].groupby('pre_type').n.sum()
to_t45 = C[C.post_type.isin(MOTION)].groupby('pre_type').n.sum()
to_lmc = C[C.post_type.isin(LMC)].groupby('pre_type').n.sum()
nt = ann[ann.side == 'right'].groupby('cell_type').top_nt.agg(lambda s: s.value_counts().index[0] if s.notna().any() else '')


def axis_consistency(a):
    a = np.radians(a.dropna() * 2)
    if len(a) < 3:
        return np.nan, np.nan
    z = np.exp(1j * a).mean()
    return float(np.abs(z)), float(np.degrees(np.angle(z)) / 2 % 180)


T = []
for ty, g in N.groupby('type'):
    if len(g) < 3:
        continue
    r = dict(type=ty, n_cells=int(len(g)), nt=nt.get(ty, ''))
    for lab in ('all', 'exc', 'inh'):
        if f'{lab}_footprint' not in g:
            continue
        r[f'{lab}_footprint'] = float(g[f'{lab}_footprint'].median())
        r[f'{lab}_radius'] = float(g[f'{lab}_radius'].median())
        r[f'{lab}_elong'] = float(g[f'{lab}_elong'].median())
        R, ax = axis_consistency(g[f'{lab}_axis'][g[f'{lab}_footprint'] >= 6])
        r[f'{lab}_axis_R'] = R; r[f'{lab}_axis'] = ax
    tot = out.get(ty, 0)
    r['out_syn'] = float(tot)
    r['frac_to_motion_inputs'] = float(to_mot.get(ty, 0) / tot) if tot else 0.0
    r['frac_to_T4T5'] = float(to_t45.get(ty, 0) / tot) if tot else 0.0
    r['frac_to_LMC'] = float(to_lmc.get(ty, 0) / tot) if tot else 0.0
    T.append(r)
T = pd.DataFrame(T)
inhib = T.nt.isin(INH)
z = lambda s: (s - s.mean()) / s.std()
T['pool_score'] = np.where(inhib, z(np.log1p(T.all_footprint)) + z(T.frac_to_motion_inputs + T.frac_to_LMC), np.nan)
# orientation score: excess elongation x axis consistency across the cells of the type
T['orient_score'] = T.all_elong * T.all_axis_R
T = T.sort_values('pool_score', ascending=False)
print(T[['type', 'nt', 'n_cells', 'all_footprint', 'all_radius', 'all_elong', 'all_axis_R', 'frac_to_motion_inputs', 'frac_to_LMC', 'pool_score']].head(25).to_string())
print(T.sort_values('orient_score', ascending=False)[['type', 'nt', 'n_cells', 'all_footprint', 'all_elong', 'all_axis_R', 'all_axis', 'exc_elong', 'inh_elong', 'orient_score']].head(25).to_string())
T.to_csv('analysis/out_flywire_types.csv', index=False)
N.to_csv('analysis/out_flywire_neurons.csv.gz', index=False)
# store per-type example input maps (mean over cells, aligned on centroid) for the dashboard
maps = {}
top = set(T.dropna(subset=['pool_score']).head(30).type) | set(T.sort_values('orient_score', ascending=False).head(30).type)
for ty in top:
    ids = N[N.type == ty].sort_values('all_footprint', ascending=False).root_id.head(3)
    ex = []
    for rid in ids:
        g = withpos[withpos.post == rid]
        cols = g.groupby(['x', 'y', 'pre_nt']).n.sum().reset_index()
        ex.append(dict(root_id=str(rid), cols=[[round(a, 2), round(b, 2), str(c), int(n)] for a, b, c, n in cols.values]))
    maps[ty] = ex
json.dump(dict(lattice=np.round(lat, 2).tolist(), examples=maps), open('analysis/out_flywire_maps.json', 'w'))
