"""Map FlyWire wiring of candidate normalization pools onto flyvis cell types.

For each pool (Dm17, Dm1, Dm15) in the right optic lobe:
  inputs   signed fraction of the pool's input synapses coming from each flyvis cell type
  outputs  fraction of the pool's output synapses onto each flyvis cell type
  radius   synapse-weighted RMS radius of the pool's columnar input (column spacings, from flywire_pools)
Writes analysis/graft_pools.json.
"""
import json, numpy as np, pandas as pd, pyarrow.feather as feather
D = '/home/claude/data/flywire/'
POOLS = ['Dm17', 'Dm1', 'Dm15']
FLYVIS = json.load(open('analysis/out_flyvis_wiring.json'))['types']
fv_types = {t['type'] for t in FLYVIS}
ann = pd.read_csv(D + 'neuron_annotations.tsv', sep='\t', dtype=str, usecols=['root_id', 'cell_type', 'top_nt'])
ann['root_id'] = ann.root_id.astype(np.int64)
tmap = ann.set_index('root_id').cell_type; ntmap = ann.set_index('root_id').top_nt
C = feather.read_table(D + 'proofread_connections_783.feather', columns=['pre_pt_root_id', 'post_pt_root_id', 'neuropil', 'syn_count']).to_pandas()
C = C[C.neuropil.isin({'ME_R', 'LO_R', 'LOP_R', 'LA_R', 'AME_R'}) & (C.syn_count >= 3)]
C['pre_t'] = C.pre_pt_root_id.map(tmap); C['post_t'] = C.post_pt_root_id.map(tmap); C['pre_nt'] = C.pre_pt_root_id.map(ntmap)


def fv(t, neuropil):
    if t == 'CT1':
        return 'CT1(M10)' if neuropil == 'ME_R' else 'CT1(Lo1)'
    return t if t in fv_types else None


T = pd.read_csv('analysis/out_flywire_types.csv').set_index('type')
out = {}
for p in POOLS:
    inc = C[C.post_t == p].copy(); inc['fv'] = [fv(t, n) for t, n in zip(inc.pre_t, inc.neuropil)]
    sign = inc.pre_nt.map(lambda n: 1.0 if n == 'acetylcholine' else (-1.0 if n in ('gaba', 'glutamate') else 0.0))
    inc['w'] = inc.syn_count * sign
    tot_in = inc.syn_count.sum()
    win = (inc.dropna(subset=['fv']).groupby('fv').w.sum() / tot_in)
    og = C[C.pre_t == p].copy(); og['fv'] = [fv(t, n) for t, n in zip(og.post_t, og.neuropil)]
    tot_out = og.syn_count.sum()
    wout = og.dropna(subset=['fv']).groupby('fv').syn_count.sum() / tot_out
    out[p] = dict(nt=str(T.loc[p, 'nt']), radius=float(T.loc[p, 'all_radius']), footprint=float(T.loc[p, 'all_footprint']),
                  inputs={k: round(float(v), 4) for k, v in win.sort_values(key=abs, ascending=False).items() if abs(v) >= 0.005},
                  outputs={k: round(float(v), 4) for k, v in wout.sort_values(ascending=False).items() if v >= 0.005},
                  frac_in_mapped=round(float(inc.dropna(subset=['fv']).syn_count.sum() / tot_in), 3),
                  frac_out_mapped=round(float(og.dropna(subset=['fv']).syn_count.sum() / tot_out), 3))
    print(p, out[p]['nt'], 'r', out[p]['radius'], 'in mapped', out[p]['frac_in_mapped'], 'out mapped', out[p]['frac_out_mapped'])
    print('  IN ', list(out[p]['inputs'].items())[:8]); print('  OUT', list(out[p]['outputs'].items())[:8])
json.dump(out, open('analysis/graft_pools.json', 'w'), indent=1)
