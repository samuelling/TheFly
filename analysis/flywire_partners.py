"""Top input/output partner types (right optic lobe, >=3 synapses) for selected types -> JSON."""
import json, sys, numpy as np, pandas as pd, pyarrow.feather as feather
D = '/home/claude/data/flywire/'
ann = pd.read_csv(D + 'neuron_annotations.tsv', sep='\t', dtype=str, usecols=['root_id', 'cell_type', 'top_nt'])
ann['root_id'] = ann.root_id.astype(np.int64); tmap = ann.set_index('root_id').cell_type
C = feather.read_table(D + 'proofread_connections_783.feather', columns=['pre_pt_root_id', 'post_pt_root_id', 'neuropil', 'syn_count']).to_pandas()
C = C[C.neuropil.isin({'ME_R', 'LO_R', 'LOP_R', 'LA_R', 'AME_R'}) & (C.syn_count >= 3)]
C['pre_t'] = C.pre_pt_root_id.map(tmap); C['post_t'] = C.post_pt_root_id.map(tmap)
T = pd.read_csv('analysis/out_flywire_types.csv')
sel = set(T.dropna(subset=['pool_score']).sort_values('pool_score', ascending=False).head(30).type) | \
      set(T.sort_values('orient_score', ascending=False).head(30).type) | {'Lawf1', 'Lawf2', 'Lai', 'C2', 'C3', 'CT1', 'Am'}
out = {}
for t in sel:
    o = C[C.pre_t == t].groupby('post_t').syn_count.sum().sort_values(ascending=False)
    i = C[C.post_t == t].groupby('pre_t').syn_count.sum().sort_values(ascending=False)
    out[t] = dict(outputs=[[k, int(v)] for k, v in o.head(10).items()], inputs=[[k, int(v)] for k, v in i.head(10).items()],
                  out_total=int(o.sum()), in_total=int(i.sum()))
json.dump(out, open('analysis/out_flywire_partners.json', 'w'))
for t in sys.argv[1:]:
    if t in out:
        print(t, 'IN', out[t]['inputs'][:6]); print(t, 'OUT', out[t]['outputs'][:6])
