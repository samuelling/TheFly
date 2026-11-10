"""Pack FlyWire analysis outputs into compact JSON for the dashboard (docs/data)."""
import json, numpy as np, pandas as pd
T = pd.read_csv('analysis/out_flywire_types.csv')
N = pd.read_csv('analysis/out_flywire_neurons.csv.gz')
keep = ['type', 'nt', 'n_cells', 'all_footprint', 'all_radius', 'all_elong', 'all_axis_R', 'all_axis',
        'exc_footprint', 'exc_elong', 'exc_axis_R', 'exc_axis', 'inh_footprint', 'inh_elong', 'inh_axis_R', 'inh_axis',
        'out_syn', 'frac_to_motion_inputs', 'frac_to_T4T5', 'frac_to_LMC', 'pool_score', 'orient_score']
T = T[keep]
hist = {}
for ty, g in N.groupby('type'):
    a = g.loc[g.all_footprint >= 6, 'all_axis'].dropna()
    if len(a) >= 3:
        hist[ty] = np.histogram(a, bins=np.arange(0, 181, 15))[0].tolist()
T['axis_hist'] = T.type.map(hist)
recs = json.loads(T.round(4).to_json(orient='records'))
maps = json.load(open('analysis/out_flywire_maps.json'))
partners = json.load(open('analysis/out_flywire_partners.json'))
json.dump(dict(types=recs, lattice=maps['lattice'], examples=maps['examples'], partners=partners),
          open('docs/data/flywire.json', 'w'), separators=(',', ':'))
w = json.load(open('analysis/out_flyvis_wiring.json'))
json.dump(w, open('docs/data/flyvis_wiring.json', 'w'), separators=(',', ':'))
print(len(recs), 'types;', len(maps['examples']), 'example maps')
