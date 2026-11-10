"""Reference implementation of the FlyWire Dm3 -> TmY orientation circuit model (mirrors docs/js/dm3.js).

Reads docs/data/dm3_circuit.json (built by dm3_circuit.py) and writes docs/data/dm3_summary.json:
  1. orientation tuning of every Dm3 / TmY neuron, which emerges from the wiring alone
  2. CRFs of each TmY type to its preferred grating, Dm3 intact vs removed, with
     subtractive vs divisive (shunting) Dm3 inhibition
  3. orientation-tuned gain control: summation index of a preferred test plus a mask
     rotated by 0-90 deg, SI = R(test+mask) / (R(test) + R(mask)); SI < 1 = suppression.
     The Dm3 contribution is SI(with Dm3) - SI(without Dm3).
Responses are the peak over test-grating phase (the optimally placed bar), averaged over mask phase.
"""
import json, numpy as np, scipy.sparse as sp

circ = json.load(open('docs/data/dm3_circuit.json'))
xy = np.array(circ['columns']); types = np.array(circ['types'])
nN, nC = len(types), len(xy)
ff = np.array([[e[0], e[1], e[2] == 'on', e[3]] for e in circ['ff']], dtype=float)
W_on = sp.csr_matrix((ff[ff[:, 2] == 1, 3], (ff[ff[:, 2] == 1, 0].astype(int), ff[ff[:, 2] == 1, 1].astype(int))), shape=(nN, nC))
W_off = sp.csr_matrix((ff[ff[:, 2] == 0, 3], (ff[ff[:, 2] == 0, 0].astype(int), ff[ff[:, 2] == 0, 1].astype(int))), shape=(nN, nC))
dm = np.array(circ['dm3'])
W_dm = sp.csr_matrix((dm[:, 2], (dm[:, 0].astype(int), dm[:, 1].astype(int))), shape=(nN, nN))
DM3 = ['Dm3p', 'Dm3q', 'Dm3v']; TMY = ['TmY9q', 'TmY9q__perp', 'TmY4']
WL = 4.0; ITERS = 50  # grating wavelength in column spacings


def gratings(theta, c, phases):
    k = 2 * np.pi / WL
    proj = xy[:, 0] * np.cos(theta) + xy[:, 1] * np.sin(theta)
    return c * np.cos(k * proj[:, None] + np.asarray(phases)[None, :])  # (columns, batch)


def simulate(stim, g=1.0, mode='div', dm3_on=True):
    """stim (columns, batch) contrast -> steady-state rates (neurons, batch)."""
    drive = W_on @ np.maximum(stim, 0) + W_off @ np.maximum(-stim, 0)
    r = np.zeros_like(drive)
    for _ in range(ITERS):
        inh = W_dm @ r if dm3_on else 0.0
        new = np.maximum(drive - g * inh, 0) if mode == 'sub' else np.maximum(drive, 0) / (1 + 4 * g * inh)
        r = 0.5 * r + 0.5 * new
    return r


TP = np.linspace(0, 2 * np.pi, 8, endpoint=False)
MP = np.linspace(0, 2 * np.pi, 4, endpoint=False) + 0.3
PT, PM = [a.ravel() for a in np.meshgrid(TP, MP)]


def resp(test_th, test_c, mask_th=None, mask_c=0.0, **kw):
    """Peak response over test-grating phase (optimally placed bars), averaged over mask phase."""
    s = gratings(test_th, test_c, PT)
    if mask_th is not None and mask_c > 0:
        s = s + gratings(mask_th, mask_c, PM)
    r = simulate(s, **kw).reshape(nN, len(MP), len(TP))
    return r.max(2).mean(1)


if __name__ == '__main__':
    S = dict(wavelength=WL)
    thetas = np.radians(np.arange(0, 180, 15)); S['thetas'] = np.degrees(thetas).tolist()
    # tuning from the feedforward wiring alone (Dm3 inhibition off), then with Dm3 on (divisive)
    tun = {k: np.stack([resp(th, 0.5, dm3_on=k == 'dm3') for th in thetas]) for k in ('ff', 'dm3')}
    for t in DM3 + TMY:
        m = types == t
        S[t] = dict(n=int(m.sum()))
        for k, tuning in tun.items():
            pref = np.degrees(thetas[np.argmax(tuning, 0)])
            tc = tuning[:, m] / (tuning[:, m].max(0, keepdims=True) + 1e-9)
            z = np.exp(2j * np.radians(pref[m])).mean()
            # orientation selectivity index per neuron (circular variance based)
            osi = np.abs((tuning[:, m] * np.exp(2j * thetas)[:, None]).sum(0)) / (tuning[:, m].sum(0) + 1e-9)
            S[t][k] = dict(tuning=tc.mean(1).round(3).tolist(), pref_R=round(float(abs(z)), 3),
                           pref=round(float(np.degrees(np.angle(z)) / 2 % 180), 1), osi=round(float(np.median(osi)), 3),
                           pref_hist=np.histogram(pref[m], bins=np.arange(-7.5, 180, 15))[0].tolist())
        S[t]['pref'] = S[t]['ff']['pref']
        print(t, 'ff', S[t]['ff']['pref'], S[t]['ff']['pref_R'], S[t]['ff']['osi'], '| dm3', S[t]['dm3']['pref'], S[t]['dm3']['pref_R'], S[t]['dm3']['osi'])
    cs = [0.03, 0.06, 0.12, 0.25, 0.5, 1.0]; S['crf_contrasts'] = cs
    deltas = [0, 15, 30, 45, 60, 75, 90]; S['mask_deltas'] = deltas; S['mask_contrast'] = 0.25; S['test_contrast'] = 0.25
    for t in TMY:
        m = types == t; th = np.radians(S[t]['pref'])
        crf = {}
        for mode in ('sub', 'div'):
            for on in (True, False):
                crf[f'{mode}_{"dm3" if on else "nodm3"}'] = [round(float(np.median(resp(th, c, mode=mode, dm3_on=on)[m])), 4) for c in cs]
        S[t]['crf'] = crf
        si, raw = {}, {}
        for mode, g in (('sub', 1.0), ('div', 1.0), ('div', 3.0), ('div', 10.0), ('sub', 3.0)):
            for on in ((True, False) if g == 1.0 else (True,)):
                key = f'{mode}_g{g:g}_{"dm3" if on else "nodm3"}'
                rt = resp(th, 0.25, mode=mode, g=g, dm3_on=on)[m]
                RT, RM, RB = [], [], []
                for d in deltas:
                    mth = th + np.radians(d)
                    RM.append(resp(mth, 0.25, mode=mode, g=g, dm3_on=on)[m])  # mask alone, at its own peak phase
                    RB.append(resp(th, 0.25, mth, 0.25, mode=mode, g=g, dm3_on=on)[m])
                RM, RB = np.array(RM), np.array(RB)
                # masking ratio: test-driven increment on the mask background / test alone (1 = no masking)
                mr = np.median((RB - RM) / np.maximum(rt, 1e-6)[None], 1)
                sidx = np.median(RB / np.maximum(rt[None] + RM, 1e-6), 1)
                si[key] = dict(masking_ratio=mr.round(4).tolist(), summation_index=sidx.round(4).tolist())
        S[t]['masking'] = si
        for k, v in si.items():
            print(t, k, 'MR', v['masking_ratio'])
    json.dump(S, open('docs/data/dm3_summary.json', 'w'))
