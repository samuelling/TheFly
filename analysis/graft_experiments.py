"""CRF x surround experiments on flyvis networks with grafted FlyWire normalization pools.

Configs: divisive pools at several gains, subtractive at matched gains, and each pool alone.
Stimuli and response measure are identical to run_experiments.py (crf conditions subset),
so g = 0 equals the intact model (taken from the crf cache).
"""
import os, sys, json, time, itertools, numpy as np, torch
sys.path.insert(0, 'analysis')
import run_experiments as RX
from model_utils import load, DT
from graft import Graft
torch.set_num_threads(2)

SUR = [(0.0, 0), (0.5, 0), (1.0, 0), (1.0, 90)]
CONDS = [dict(c=c, s=s, o=o, d=d, r=2.0) for c, (s, o), d in itertools.product(RX.CONTRASTS, SUR, RX.DIRS)]
ALL = ('Dm17', 'Dm1', 'Dm15')
CONFIGS = [('div', {p: g for p in ALL}, f'div_all_g{g:g}') for g in (0.5, 2, 8)] + \
          [('sub', {p: g for p in ALL}, f'sub_all_g{g:g}') for g in (0.5, 2)] + \
          [('div', {p: 2.0}, f'div_{p}_g2') for p in ALL]

if __name__ == '__main__' and not (len(sys.argv) > 2 and sys.argv[2] == 'ft'):
    models = sys.argv[1].split(',')
    os.makedirs('analysis/results/graft', exist_ok=True)
    for m in models:
        net = load(m); RX.TYPES = list(np.array(net.connectome.unique_cell_types[:]).astype(str))
        gr = Graft(net); gr.calibrate()
        fi = net.fade_in_state
        def fade(*a, **k):
            gr.P = None; return fi(*a, **k)
        net.fade_in_state = fade
        json.dump(dict(Pref=gr.Pref, targets=gr.targets, a=gr.a, win=gr.win), open(f'analysis/results/graft/calib_{m}.json', 'w'))
        for mode, gains, name in CONFIGS:
            f = f'analysis/results/graft/{name}_{m}.npz'
            if os.path.exists(f):
                continue
            t0 = time.time()
            gr.set(gains, mode)
            f1, dc = RX.run(net, CONDS)
            np.savez(f, f1=f1, dc=dc)
            print(m, name, f'{time.time()-t0:.0f}s', flush=True)


def run_finetuned(m):
    """CRFs of the fine-tuned grafted network, pools on (learned gains) and pools off."""
    ck = torch.load(f'/home/claude/data/finetuned_{m}.pt') if os.path.exists(f'/home/claude/data/finetuned_{m}.pt') else torch.load(f'/home/claude/data/finetuned_graft_{m}.pt')
    net = load(m); net.load_state_dict(ck['net']); net.eval()
    RX.TYPES = list(np.array(net.connectome.unique_cell_types[:]).astype(str))
    gr = Graft(net); gr.calibrate()
    fi = net.fade_in_state
    def fade(*a, **k):
        gr.P = None; return fi(*a, **k)
    net.fade_in_state = fade
    gains = {p: float(np.exp(v)) for p, v in ck['log_g'].items()}
    for name, g in (('ft_div_learned', gains), ('ft_nopool', {p: 0.0 for p in gains})):
        f = f'analysis/results/graft/{name}_{m}.npz'
        if os.path.exists(f):
            continue
        gr.set(g, 'div'); t0 = time.time()
        f1, dc = RX.run(net, CONDS); np.savez(f, f1=f1, dc=dc)
        print(m, name, gains, f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__' and len(sys.argv) > 2 and sys.argv[2] == 'ft':
    run_finetuned(sys.argv[1])
