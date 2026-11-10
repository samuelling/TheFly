"""Helpers for loading the flyvis ensemble and lesioning edges."""
import h5py, numpy as np, torch, flyvis
from flyvis import NetworkView

ENSEMBLE = flyvis.results_dir / "flow/0000"
DT = 0.02  # integration step used in training


def ranked_models():
    """Model ids ordered by validation loss (best first)."""
    out = []
    for d in sorted(ENSEMBLE.iterdir()):
        if d.name.isnumeric():
            with h5py.File(d / "validation_loss.h5") as h:
                out.append((float(h["data"][()]), d.name))
    return [m for _, m in sorted(out)], dict((m, l) for l, m in out)


def load(model_id):
    net = NetworkView(ENSEMBLE / model_id).init_network()
    net.eval()
    return net


def edge_table(net):
    e = net.connectome.edges
    st = np.array(e.source_type[:]).astype(str)
    tt = np.array(e.target_type[:]).astype(str)
    return st, tt


def set_lesion(net, mask):
    """Multiply every edge weight by mask (1 keep, 0 cut). mask=None restores."""
    dyn = net.dynamics
    if not hasattr(dyn, "_orig_derived"):
        dyn._orig_derived = dyn.write_derived_params
    if mask is None:
        dyn.write_derived_params = dyn._orig_derived
        return
    m = torch.tensor(mask, dtype=torch.float32)

    def patched(params, **kw):
        dyn._orig_derived(params, **kw)
        params.edges.weight = params.edges.weight * m

    dyn.write_derived_params = patched
