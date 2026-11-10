"""Extract the Dm3 -> TmY9/TmY4 orientation subconnectome from FlyWire and test it.

Builds a column-level rate model directly from FlyWire synapse counts:
  columnar inputs (ON: Mi1, Tm3, Mi4, C3; OFF: Tm1, Tm2, Tm4, Tm9, L2, L3), mapped to columns
  -> Dm3p / Dm3q / Dm3v (real neurons, real weights, incl. Dm3 <-> Dm3 synapses)
  -> TmY9q, TmY9q__perp, TmY4 (real neurons; columnar input + Dm3 input)
Signs follow the presynaptic transmitter (ACh +, GABA/Glu -). Glutamate is assumed
inhibitory (GluCl), which is the main unverified assumption.
Exports the sparse circuit as JSON for the in-browser model and a precomputed summary.
"""
import json, numpy as np, pandas as pd, pyarrow.feather as feather
from connectome_interpreter.external_map import load_dataset

D = '/home/claude/data/flywire/'
ON = {'Mi1', 'Tm3', 'Mi4', 'C3'}
OFF = {'Tm1', 'Tm2', 'Tm4', 'Tm9', 'L2', 'L3'}
DM3 = ['Dm3p', 'Dm3q', 'Dm3v']
TMY = ['TmY9q', 'TmY9q__perp', 'TmY4']

ann = pd.read_csv(D + 'neuron_annotations.tsv', sep='\t', dtype=str, usecols=['root_id', 'cell_type', 'top_nt', 'side'])
ann['root_id'] = ann.root_id.astype(np.int64)
tmap = ann.set_index('root_id').cell_type; ntmap = ann.set_index('root_id').top_nt
C = feather.read_table(D + 'proofread_connections_783.feather', columns=['pre_pt_root_id', 'post_pt_root_id', 'neuropil', 'syn_count']).to_pandas()
C = C[C.neuropil.isin({'ME_R', 'LO_R', 'LOP_R', 'AME_R'}) & (C.syn_count >= 3)]
C = C.groupby(['pre_pt_root_id', 'post_pt_root_id'], as_index=False).syn_count.sum()
C.columns = ['pre', 'post', 'n']
C['pre_t'] = C.pre.map(tmap); C['post_t'] = C.post.map(tmap); C['pre_nt'] = C.pre.map(ntmap)

col = load_dataset('Matsliah2024'); col = col[col.hemisphere == 'right']
col = col.assign(x=col.x.astype(float) * np.sqrt(3) / 2, y=col.y.astype(float) / 2)
cols = col.drop_duplicates('column_id').sort_values('column_id')
cid = {c: i for i, c in enumerate(cols.column_id)}
xy = cols[['x', 'y']].values
root2col = dict(zip(col.root_id, col.column_id.map(cid)))
C['pre_col'] = C.pre.map(root2col)

sign = lambda nt: 1.0 if nt == 'acetylcholine' else (-1.0 if nt in ('gaba', 'glutamate') else 0.0)
neurons = {}
for t in DM3 + TMY:
    ids = sorted(ann[(ann.cell_type == t)].root_id)
    ids = [i for i in ids if i in set(C.post)]
    neurons[t] = ids
all_ids = [i for t in DM3 + TMY for i in neurons[t]]
idx = {r: k for k, r in enumerate(all_ids)}
types = [t for t in DM3 + TMY for _ in neurons[t]]

# column -> neuron weights (ON / OFF channels, signed)
colw = []  # (post_idx, col, channel, w)
rec = []   # (post_idx, pre_idx, w) for Dm3 -> {Dm3, TmY}
for r in all_ids:
    g = C[C.post == r]
    for ch, S in (('on', ON), ('off', OFF)):
        h = g[g.pre_t.isin(S) & g.pre_col.notna()]
        for (c_, nt), n in h.groupby(['pre_col', 'pre_nt']).n.sum().items():
            colw.append((idx[r], int(c_), ch, float(n) * sign(nt)))
    h = g[g.pre.isin(idx) & g.pre_t.isin(DM3)]
    for pre, n in h.groupby('pre').n.sum().items():
        rec.append((idx[r], idx[pre], float(n)))
print({t: len(neurons[t]) for t in neurons}, 'column edges', len(colw), 'Dm3 edges', len(rec))

# normalise: each neuron's |feedforward| weights sum to 1, Dm3 input weights sum to 1
ff = pd.DataFrame(colw, columns=['post', 'col', 'ch', 'w'])
ff['w'] /= ff.groupby('post').w.transform(lambda s: s.abs().sum())
rr = pd.DataFrame(rec, columns=['post', 'pre', 'w'])
rr['w'] /= rr.groupby('post').w.transform('sum')
centroid = np.zeros((len(all_ids), 2))
for p, g in ff[ff.w > 0].groupby('post'):
    w = g.w.values; centroid[p] = (xy[g.col.values] * w[:, None]).sum(0) / w.sum()

circuit = dict(columns=np.round(xy, 3).tolist(), types=types, root_ids=[str(i) for i in all_ids],
               centroid=np.round(centroid, 2).tolist(),
               ff=[[int(a), int(b), c, round(float(d), 5)] for a, b, c, d in ff.values],
               dm3=[[int(a), int(b), round(float(c), 5)] for a, b, c in rr.values])
json.dump(circuit, open('docs/data/dm3_circuit.json', 'w'), separators=(',', ':'))


