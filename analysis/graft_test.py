import sys, time; sys.path.insert(0, "analysis")
import numpy as np, torch
from model_utils import load, DT
from graft import Graft
import run_experiments as RX
torch.set_num_threads(2)
net = load('000'); RX.TYPES = list(np.array(net.connectome.unique_cell_types[:]).astype(str))
gr = Graft(net); gr.calibrate()
fi = net.fade_in_state
def fade(*a, **k):
    gr.P = None; return fi(*a, **k)
net.fade_in_state = fade
print('targets', gr.targets, 'Pref', {p: round(v, 4) for p, v in gr.Pref.items()})
# grey check
x = torch.full((1, 50, 1, 721), 0.5)
for g in (0, 8):
    gr.set({'Dm17': g, 'Dm1': g, 'Dm15': g}, 'div')
    with torch.no_grad():
        s = net.fade_in_state(1.0, DT, x[:, 0]); r = net.simulate(x, DT, initial_state=s)
    print('grey g', g, float(r[0, -1].abs().mean()))
conds = [dict(c=1.0, s=s_, o=o, d=d, r=2.0) for s_, o in ((0, 0), (1, 0), (1, 90)) for d in RX.DIRS]
for g, mode in ((0, 'div'), (1, 'div'), (4, 'div'), (4, 'sub')):
    gr.set({'Dm17': g, 'Dm1': g, 'Dm15': g}, mode)
    t0 = time.time(); f1, dc = RX.run(net, conds)
    f1 = f1.reshape(3, 4, -1)
    pref = f1[0].argmax(0); R = f1[:, pref, np.arange(f1.shape[2])]
    ssi = 1 - R[1] / np.maximum(R[0], 1e-9); so = 1 - R[2] / np.maximum(R[0], 1e-9)
    pick = [RX.TYPES.index(t) for t in ['L2', 'Mi1', 'Tm1', 'Tm2', 'Tm3', 'Mi4', 'Mi9', 'T4a', 'T4c', 'T5a', 'T5c', 'Tm9']]
    print(mode, g, f'{time.time()-t0:.0f}s', ' '.join(f'{RX.TYPES[i]}:{ssi[i]:+.2f}/{so[i]:+.2f}' for i in pick))
