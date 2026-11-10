"""Graft FlyWire normalization pools (Dm17, Dm1, Dm15) onto a trained flyvis network.

Each pool p is a smooth field over the 721 columns:
    tau_p dP_p/dt = -P_p + relu( K_p * d_p ),   d_p(col) = sum_s w_ps relu(V_s(col))
with input weights w_ps (signed synapse fractions from FlyWire) and a Gaussian kernel K_p whose
RMS radius equals the pool's measured FlyWire input radius. Pool output reaches target type t
with relative weight a_pt (output synapse fraction, normalized to the pool's main target).

Pools are driven by their inputs' depolarization relative to the grey operating point V0, so they
are silent on a uniform grey screen and the trained network's grey state is untouched. With
G = sum_p g_p a_pt P_p / Pref_p (Pref = pool level for a full-field 100% grating, so g = 1 means
"divide by 2 at full contrast" for the main target), the pool acts on the target's net drive
D = bias + synaptic input + stimulus as
    divisive    D' = D0 + (D - D0) / (1 + G)     shunting toward the operating point: a gain change
    subtractive D' = D - sigma_t * G            hyperpolarizing current, sigma_t = SD of D - D0
                                                 under the full-field grating (one modulation unit)
"""
import json, numpy as np, torch
from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel

POOLS = json.load(open('analysis/graft_pools.json'))


class Graft:
    def __init__(self, net, pools=('Dm17', 'Dm1', 'Dm15'), tau=0.05):
        self.net, self.pools, self.tau = net, list(pools), tau
        n = net.connectome.nodes
        types = np.array(n.type[:]).astype(str); u = np.array(n.u[:]); v = np.array(n.v[:])
        hu, hv = get_hex_coords(15)
        col = {(a, b): i for i, (a, b) in enumerate(zip(hu, hv))}
        self.nC = len(hu)
        x, y = hex_to_pixel(hu, hv); xy = np.c_[x, y]
        xy /= np.min(np.linalg.norm(xy[1:] - xy[0], axis=1)[np.linalg.norm(xy[1:] - xy[0], axis=1) > 0])
        d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
        self.type_idx = {}
        for t in np.unique(types):
            m = np.where(types == t)[0]
            order = np.array([col[(a, b)] for a, b in zip(u[m], v[m])])
            idx = np.full(self.nC, -1); idx[order] = m
            self.type_idx[t] = torch.tensor(idx)
        self.K, self.win, self.a = {}, {}, {}
        for p in self.pools:
            sig = POOLS[p]['radius'] / np.sqrt(2)  # RMS radius of a 2-D Gaussian = sigma * sqrt(2)
            K = np.exp(-d2 / (2 * sig ** 2)); K /= K.sum(1, keepdims=True)
            self.K[p] = torch.tensor(K, dtype=torch.float32)
            self.win[p] = {s: w for s, w in POOLS[p]['inputs'].items() if s in self.type_idx}
            outs = {t: f for t, f in POOLS[p]['outputs'].items() if t in self.type_idx}
            mx = max(outs.values())
            self.a[p] = {t: f / mx for t, f in outs.items()}
        self.targets = sorted({t for p in self.pools for t in self.a[p]})
        self.g = {p: 0.0 for p in self.pools}; self.mode = 'div'
        self.calibrated = False
        self._orig = net.dynamics.write_state_velocity
        net.dynamics.write_state_velocity = self._velocity
        self.P = None

    # -- pool drive from current activity
    def _drive(self, act, p):
        # synaptic output change relative to the grey operating point (half-wave rectified)
        d = 0
        for s, w in self.win[p].items():
            i = self.type_idx[s]
            d = d + w * torch.relu(act[:, i] - self.V0[:, i])
        return torch.relu(d @ self.K[p].T)

    def reset(self, batch):
        self.P = {p: self.P0[p].expand(batch, -1).clone() for p in self.pools} if self.calibrated else None

    def gain(self, p):
        return torch.exp(self.log_g[p]) if getattr(self, 'log_g', None) is not None else self.g[p]

    def make_learnable(self, init):
        """Pool gains become trainable parameters (log scale)."""
        self.log_g = torch.nn.ParameterDict({p: torch.nn.Parameter(torch.tensor(float(np.log(init[p])))) for p in self.pools})
        self.g = {p: 1.0 for p in self.pools}  # flags 'active'
        return list(self.log_g.values())

    def _G(self, P, t):
        G = 0
        for p in self.pools:
            if t in self.a[p] and self.g[p] > 0:
                G = G + self.gain(p) * self.a[p][t] * P[p] / self.Pref[p]
        return G

    def _velocity(self, vel, state, params, target_sum, x_t, dt, **kw):
        act = state.nodes.activity
        D = params.nodes.bias + target_sum(params.edges.weight * self.net.dynamics.activation(state.sources.activity)) + x_t
        if self.calibrated and any(v > 0 for v in self.g.values()):
            if self.P is None or self.P[self.pools[0]].shape[0] != act.shape[0]:
                self.reset(act.shape[0])
            D = D.clone()
            for t in self.targets:
                G = self._G(self.P, t)
                if isinstance(G, int):
                    continue
                idx = self.type_idx[t]
                if self.mode == 'div':
                    D0 = self.D0[:, idx]
                    D[:, idx] = D0 + (D[:, idx] - D0) / (1 + G)
                else:
                    D[:, idx] = D[:, idx] - self.sigma[t] * G
            for p in self.pools:
                self.P[p] = self.P[p] + dt / self.tau * (-self.P[p] + self._drive(act, p))
        self._lastD = D
        if getattr(self, '_rec', None) is not None:
            self._rec.append(D.detach().clone())
        vel.nodes.activity = (-act + D) / torch.max(params.nodes.time_const, torch.tensor(dt).float())

    def calibrate(self, dt=0.02):
        """1) grey screen, pools passive: operating point V0 and target drives D0.
        2) full-field drifting grating at 100% contrast: pool level Pref that defines g = 1."""
        from stimuli import grating_movie
        g, self.g = self.g, {p: 0.0 for p in self.pools}
        x = torch.full((1, int(3.0 / dt), 1, 721), 0.5)
        with torch.no_grad():
            s = self.net.fade_in_state(1.0, dt, x[:, 0])
            r = self.net.simulate(x, dt, initial_state=s)
        self.V0 = r[:, -1].clone()
        self.D0 = self._lastD.clone()
        mov = torch.tensor(np.stack([grating_movie(1.0, 1.0, d, r_center=20, r_gap=99, dt=dt)[0] for d in (0, 90)]))
        with torch.no_grad():
            s = self.net.fade_in_state(1.0, dt, mov[:, 0])
            self._rec = []
            r = self.net.simulate(mov, dt, initial_state=s)
            Drec = torch.stack(self._rec[-50:], 1); self._rec = None  # (b, t, nodes)
        self.sigma = {t: (Drec[:, :, self.type_idx[t]] - self.D0[:, None, self.type_idx[t]]).std((0, 1))[None] for t in self.targets}
        win = r[:, -50:]  # last second
        self.Pref = {}
        for p in self.pools:
            P = torch.stack([self._drive(win[:, k], p) for k in range(win.shape[1])], 1)  # (b, t, cols)
            self.Pref[p] = float(P.mean()) + 1e-6
        self.P0 = {p: torch.zeros(1, self.nC) for p in self.pools}
        self.calibrated = True
        self.g = g
        self._refresh_G0()

    def _refresh_G0(self):
        self.G0 = {t: self._G(self.P0, t) if not isinstance(self._G(self.P0, t), int) else torch.zeros(1, self.nC) for t in self.targets}

    def set(self, gains, mode='div'):
        self.g = {p: float(gains.get(p, 0.0)) for p in self.pools}
        self.mode = mode
        self._refresh_G0()
        self.P = None
