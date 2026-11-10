"""Part 1a: wiring signatures of candidate normalisation pools in the flyvis connectome.

For each cell type we measure, from the edges onto its central cell:
  - sign of its output synapses (from the connectome's predicted transmitters)
  - input footprint: number of distinct columns that feed it, and the
    synapse-weighted radius of that pool (in column spacings)
  - how much of its output lands on the inputs to T4/T5 (the motion detectors)
"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
from model_utils import load

net = load('000')
c = net.connectome
E = c.edges.to_df() if hasattr(c.edges, 'to_df') else None
e = c.edges
df = pd.DataFrame(dict(st=np.array(e.source_type[:]).astype(str), tt=np.array(e.target_type[:]).astype(str),
                       su=e.source_u[:], sv=e.source_v[:], tu=e.target_u[:], tv=e.target_v[:],
                       n=e.n_syn[:], sign=e.sign[:]))
central = df[(df.tu == 0) & (df.tv == 0)]
types = list(np.array(c.unique_cell_types[:]).astype(str))

def hexdist(du, dv):
    return (np.abs(du) + np.abs(dv) + np.abs(du + dv)) / 2

motion = [t for t in types if t[:2] in ('T4', 'T5')]
mo_in = central[central.tt.isin(motion)].groupby('st').n.sum()
motion_inputs = sorted(set(mo_in.index) - set(motion))

rows = []
out_tot = df.groupby('st').n.sum()
for t in types:
    inc = central[central.tt == t]
    outg = df[df.st == t]
    sign = float(np.sign(outg.sign.mean())) if len(outg) else 0.0
    cols = inc[['su', 'sv']].drop_duplicates()
    d = hexdist(inc.su, inc.sv)
    radius = float((d * inc.n).sum() / max(inc.n.sum(), 1e-9)) if len(inc) else 0.0
    to_mi = outg[outg.tt.isin(motion_inputs + motion)].n.sum()
    top_out = outg.groupby('tt').n.sum().sort_values(ascending=False).head(6)
    top_in = inc.groupby('st').n.sum().sort_values(ascending=False).head(6)
    rows.append(dict(type=t, sign=sign, n_in_columns=int(len(cols)), pool_radius=round(radius, 2),
                     n_in_syn=float(inc.n.sum()), frac_out_to_motion_circuit=round(float(to_mi / max(out_tot.get(t, 0), 1e-9)), 3),
                     top_outputs={k: round(float(v), 1) for k, v in top_out.items()},
                     top_inputs={k: round(float(v), 1) for k, v in top_in.items()}))
R = pd.DataFrame(rows)
# normalisation-pool score: inhibitory, spatially wide input pool, output onto motion circuit
z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
R['pool_score'] = np.where(R.sign < 0, z(np.log1p(R.n_in_columns)) + z(R.pool_radius) + z(R.frac_out_to_motion_circuit), np.nan)
R = R.sort_values('pool_score', ascending=False)
print(R[['type', 'sign', 'n_in_columns', 'pool_radius', 'frac_out_to_motion_circuit', 'pool_score']].head(25).to_string())
print('motion inputs:', motion_inputs)
json.dump(dict(motion_inputs=motion_inputs, types=R.replace({np.nan: None}).to_dict('records')),
          open('analysis/out_flyvis_wiring.json', 'w'), indent=1)
# edge list between types (for subconnectome graph), synapses per central target cell
g = central.groupby(['st', 'tt']).agg(n=('n', 'sum'), sign=('sign', 'mean')).reset_index()
g.to_csv('analysis/out_type_graph.csv', index=False)
