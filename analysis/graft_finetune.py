"""Fine-tune a flyvis network with grafted divisive pools on the original optic-flow task.

The published models were trained for 250k iterations on GPUs; here we can afford a few hundred
on CPU, so this tests whether the trained network *accommodates* the pools (task error recovers)
and what the pools do to CRFs afterwards, not training from scratch.

Usage: python graft_finetune.py MODEL N_ITERS INIT_GAIN
Writes analysis/results/graft/finetune_<model>.json and a state dict.
"""
import sys, json, time, numpy as np, torch
sys.path.insert(0, 'analysis')
from torch.utils.data import DataLoader, SubsetRandomSampler
from flyvis.datasets.sintel import MultiTaskSintel
from flyvis.task.objectives import l2norm
from flyvis import NetworkView
from model_utils import ENSEMBLE, DT
from graft import Graft
torch.set_num_threads(2)

m, n_iters, g_init = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
view = NetworkView(ENSEMBLE / m)
cfg = view.dir.config.task.dataset
kw = {k: v for k, v in cfg.items() if k not in ('type',)}
ds = MultiTaskSintel(**kw)
train_idx, val_idx = ds.get_random_data_split(fold=1, n_folds=4, shuffle=True, seed=0)


def evaluate(net, dec, n=None):
    """Validation loss on the held-out fold (no augmentation), batch of 4."""
    idx = list(val_idx)[:n] if n else list(val_idx)
    losses = []
    with ds.augmentation(False), torch.no_grad():
        for i in range(0, len(idx), 4):
            data = [ds[j] for j in idx[i:i + 4]]
            lum = torch.stack([d['lum'] for d in data]); flow = torch.stack([d['flow'] for d in data])
            s = net.fade_in_state(1.0, DT, lum[:, 0])
            r = net.simulate(lum, DT, initial_state=s)
            losses.append(float(l2norm(dec(r), flow)) * len(data))
    return sum(losses) / len(idx)


net = view.init_network(); net.eval(); dec = view.init_decoder()['flow']; dec.eval()
log = dict(model=m, g_init=g_init, n_val=len(val_idx))
t0 = time.time()
log['val_intact'] = evaluate(net, dec); print('intact', log['val_intact'], f'{time.time()-t0:.0f}s', flush=True)
CONTROL = g_init == 0  # control run: same fine-tuning, no pools
if not CONTROL:
    gr = Graft(net); gr.calibrate()
    fi = net.fade_in_state
    def fade(*a, **k):
        gr.P = None; return fi(*a, **k)
    net.fade_in_state = fade
    gr.set({p: g_init for p in gr.pools}, 'div')
    log['val_grafted'] = evaluate(net, dec); print('grafted', log['val_grafted'], flush=True)
    gparams = gr.make_learnable({p: g_init for p in gr.pools})
else:
    class _N: P = None; log_g = {}
    gr = _N(); gparams = []
    log['val_grafted'] = log['val_intact']
opt = torch.optim.Adam([{'params': [p for p in net.parameters() if p.requires_grad], 'lr': 1e-4},
                        {'params': list(dec.parameters()), 'lr': 1e-4},
                        ] + ([{'params': gparams, 'lr': 1e-2}] if gparams else []))
loader = DataLoader(ds, batch_size=4, sampler=SubsetRandomSampler(train_idx))
net.train(); dec.train()
hist = []
it = 0
with ds.augmentation(True):
    while it < n_iters:
        for data in loader:
            lum, flow = data['lum'], data['flow']
            with torch.no_grad():
                s = net.fade_in_state(0.5, DT, lum[:, 0])
            gr.P = None
            net.stimulus.zero(*lum.shape[:2]); net.stimulus.add_input(lum)
            opt.zero_grad()
            act = net(net.stimulus(), DT, state=s)
            loss = l2norm(dec(act), flow)
            loss.backward(); opt.step()
            hist.append(float(loss)); it += 1
            if it % 10 == 0:
                print(it, np.mean(hist[-10:]).round(4), {p: round(float(torch.exp(v)), 3) for p, v in gr.log_g.items()}, f'{time.time()-t0:.0f}s', flush=True)
            if it >= n_iters:
                break
net.eval(); dec.eval()
log['train_loss'] = hist
log['gains_final'] = {p: float(torch.exp(v)) for p, v in gr.log_g.items()}
log['val_finetuned'] = evaluate(net, dec); print('finetuned', log['val_finetuned'], flush=True)
tag = 'control' if CONTROL else 'graft'
json.dump(log, open(f'analysis/results/graft/finetune_{tag}_{m}.json', 'w'))
torch.save(dict(net=net.state_dict(), log_g={p: float(v) for p, v in gr.log_g.items()}), f'/home/claude/data/finetuned_{tag}_{m}.pt')
