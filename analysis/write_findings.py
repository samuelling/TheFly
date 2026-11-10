"""Write docs/data/findings.json: overview cards, callouts and methods, filled from the computed results."""
import json, numpy as np
fw = json.load(open('docs/data/flywire.json')); T = {t['type']: t for t in fw['types']}
ds = json.load(open('docs/data/dm3_summary.json'))
S = json.load(open('docs/data/sims.json'))
wir = json.load(open('docs/data/flyvis_wiring.json'))
f = lambda v, d=2: '–' if v is None else f'{v:.{d}f}'
pools = sorted([t for t in fw['types'] if t['pool_score'] is not None], key=lambda t: -t['pool_score'])[:8]
pool_txt = ', '.join(f"{t['type']} ({t['nt'][:4]}, ~{t['all_footprint']:.0f} columns)" for t in pools)
maxrad = max(t['pool_radius'] for t in wir['types'])
cells = S.get('cells', {})
act = [t for t in cells if cells[t].get('active', cells[t]['raw_amp'] > 1e-3)]
ssi = np.array([cells[t]['ssi']['iso100'] for t in act]) if act else np.array([0])
n_supp = int((ssi > 0.1).sum()); n_fac = int((ssi < -0.1).sum()); med = float(np.median(ssi))
fac = ', '.join(sorted([t for t in act if cells[t]['ssi']['iso100'] < -0.1], key=lambda t: cells[t]['ssi']['iso100'])[:4])
d3 = {t: ds[t]['ff'] for t in ['Dm3p', 'Dm3q', 'Dm3v', 'TmY9q', 'TmY9q__perp', 'TmY4']}
dang = lambda a, b: abs(((a - b) + 90) % 180 - 90)
pairs = [('TmY9q', 'Dm3p'), ('TmY9q__perp', 'Dm3q'), ('TmY4', 'Dm3v')]
orth = ', '.join(f"{a.replace('__perp', '⊥')}–{b} {dang(d3[a]['pref'], d3[b]['pref']):.0f}°" for a, b in pairs)
mr = {t: ds[t]['masking'] for t in ['TmY9q', 'TmY9q__perp', 'TmY4']}
def dm3eff(t, k='div_g1'):
    on, off = np.array(mr[t][f'{k}_dm3']['masking_ratio']), np.array(mr[t][f'{k}_nodm3']['masking_ratio'])
    return on - off
eff = {t: dm3eff(t) for t in mr}
iso_vs_orth = np.mean([eff[t][0] for t in eff]), np.mean([eff[t][-1] for t in eff])
les = S.get('lesions', {})
def les_line(name):
    if name not in les: return ''
    L = les[name]['cells']
    d = {t: L[t]['ssi_lesioned']['iso100'] - L[t]['ssi_intact']['iso100'] for t in act if t in L}
    top = sorted(d.items(), key=lambda kv: -abs(kv[1]))[:3]
    return ', '.join(f'{t} {v:+.2f}' for t, v in top)
cards = [
 dict(tag='Connectome', title='Candidate normalization pools exist, outside the model',
      text=f"Ranking FlyWire optic-lobe types by inhibitory transmitter, input footprint and output onto the T4/T5 inputs or lamina monopolar cells gives: {pool_txt}. "
           f"Most are Dm/Pm amacrine-like cells driven by L2/L5/Tm1 and feeding back onto the same pathways: the wiring a divisive denominator needs."),
 dict(tag='Simulation', title='The flyvis model has almost no surround normalization',
      text=f"In the connectome-constrained ensemble, a full-contrast iso surround (≥4 columns out) changes the centre response by a median of {abs(med)*100:.1f}% across {len(act)} responsive cell types. "
           f"{n_supp} types are suppressed by more than 10%; {n_fac} are instead facilitated (e.g. {fac}): in this model an iso surround adds drive rather than dividing it. "
           f"Its widest direct input pool spans ~{maxrad:.1f} column spacings, and the wide-field Dm/Pm types above are not in it. So the model lacks the surround normalization (the denominator) that Drews et al. measured."),
 dict(tag='Orientation', title='The Dm3 triad: three orientation channels in the wiring',
      text=f"Measuring input-pool shape cell by cell on the column lattice singles out Dm3p, Dm3q and Dm3v: elongated pools with axes "
           f"{T['Dm3p']['all_axis']:.0f}°, {T['Dm3q']['all_axis']:.0f}° and {T['Dm3v']['all_axis']:.0f}° (about 60° apart), each shared across hundreds of cells "
           f"(axis consistency ≥ {min(T[x]['all_axis_R'] for x in ('Dm3p','Dm3q','Dm3v')):.2f}). Each subtype sends most of its output to one TmY partner. This matches the hex-axis naming of Matsliah et al., recovered here from wiring alone."),
 dict(tag='Gain control', title='Wired for cross-orientation inhibition, acts mostly untuned',
      text=f"Simulating the real Dm3 → TmY neurons, each TmY's preferred orientation sits close to orthogonal to the Dm3 subtype that inhibits it ({orth}). "
           f"Yet with divisive inhibition Dm3 adds more masking for a parallel mask ({iso_vs_orth[0]:+.2f} masking ratio) than an orthogonal one ({iso_vs_orth[1]:+.2f}), "
           "because individual Dm3 tuning is broad (OSI ≈ 0.16) and contrast energy dominates. Orientation-tuned gain control would need sharper Dm3 tuning or nonlinear dendrites."),
]
if les:
    def stat(name, key='iso100'):
        L = les[name]['cells']
        d = {t: L[t]['ssi_lesioned'][key] - L[t]['ssi_intact'][key] for t in act if t in L and L[t]['ssi_lesioned'][key] is not None}
        g = float(np.median([L[t]['gain_change'] for t in act if t in L]))
        return d, g
    li, gli = stat('lateral_inhibition'); le, gle = stat('lateral_excitation'); mm, gmm = stat('Mi4_Mi9')
    top = lambda d: ', '.join(f'{t} {v:+.2f}' for t, v in sorted(d.items(), key=lambda kv: -abs(kv[1]))[:3])
    cards.append(dict(tag='Lesions', title="The model's surround effects come from a few cross-column paths",
      text=f"Cutting every inhibitory cross-column edge shifts iso surround suppression most in ({top(li)}) and lowers responses overall (median ×{gli:.2f}). "
           f"Cutting cross-column excitation acts on a different set ({top(le)}). Removing Mi4 + Mi9 destabilizes the network (responses ×{gmm:.1f}), "
           "so T4's flanking inhibition is also what holds the trained model's gain in check. Single-type lesions are in the Lesions tab."))
import os
if os.path.exists('docs/data/graft.json'):
    G = json.load(open('docs/data/graft.json')); GI = G['intact']
    def dssi(cfg, t, k='iso100'):
        return G['configs'][cfg]['cells'][t]['ssi'][k] - GI[t]['ssi'][k]
    c17 = 'div_Dm17_g2'
    top17 = sorted(G['configs'][c17]['cells'], key=lambda t: -dssi(c17, t))[:4]
    ft = G['finetune'].get('000', {})
    gft, cft = ft.get('graft'), ft.get('control')
    tm1 = G['configs']['div_Dm15_g2']['cells']['Tm1']['mean']['s0']; tm1i = GI['Tm1']['mean']['s0']
    t45 = [t for t in GI if GI[t]['dsi'] is not None]
    dsi_i = np.mean([GI[t]['dsi'] for t in t45]); dsi_f = np.mean([G['configs']['ft_div_learned']['cells'][t]['dsi'] for t in t45]) if 'ft_div_learned' in G['configs'] else None
    t45d = np.median([dssi(c17, t) for t in t45])
    graft_txt = (f"Grafting Dm17 alone (wide pool, σ ≈ {POOLS17:.1f} columns) produces surround suppression where Drews et al. found it: in the T4/T5 inputs. " if False else '')
    cards.append(dict(tag='Graft', title='Adding Dm17 gives surround normalization, but only upstream',
      text=f"Grafted as a divisive pool with FlyWire wiring, Dm17 makes its targets surround-suppressed ({', '.join(f'{t} {dssi(c17, t):+.2f}' for t in top17)} in iso SSI) and leaves their centre responses nearly intact. "
           f"The local pools act on the centre instead: Dm15 compresses Tm1's CRF (100% contrast gives {tm1[-1]/tm1i[-1]:.2f}× the intact response, 50% gives {tm1[-2]/tm1i[-2]:.2f}×). "
           f"T4/T5 do not inherit the surround effect (median ΔSSI {t45d:+.2f}), and the subtractive version is less specific and can amplify (Mi1 ×{G['configs']['sub_all_g2']['cells']['Mi1']['amp']:.1f})."
           + (f" After {len(gft['train_loss'])} iterations of fine-tuning on optic flow, the grafted network's task error is {gft['val_finetuned']:.1f} versus {cft['val_finetuned']:.1f} for the same fine-tuning without pools (original {gft['val_intact']:.1f}). "
              f"The learned gains stay well above zero ({', '.join(f'{k} {v:.2f}' for k, v in gft['gains_final'].items())}) and T4/T5 direction selectivity is kept (DSI {dsi_i:.2f} → {dsi_f:.2f})." if gft and cft else '')))
    graft_callout = ("<b>Reading this tab.</b> Dividing and subtracting were matched to first order, so differences between them come from the form of the interaction alone. "
        "Dm17 pools over a wide area and so is driven by the surround. Dm1 and Dm15 pool over only 1–2.5 columns, so they are driven by the centre stimulus itself. They therefore act more like self-normalization (CRF compression) than surround suppression. "
        "The fine-tuning is short (one model, one seed, CPU), so the task comparison is suggestive, not conclusive: a proper test retrains the ensemble with the pools from the start.")
else:
    graft_callout = ''
lam = T.get('Lawf1'), T.get('Lawf2'), T.get('C2'), T.get('C3')
lamina = ("The lamina is incomplete in the FAFB volume and R1–6 are not column-mapped, so retina → lamina convergence can't be measured here (neural superposition pools six photoreceptors that view one point, which is round by construction). "
          f"For medulla → lamina feedback, individual Lawf1 and Lawf2 cells do pool excitatory input over elongated patches (excess elongation {f(lam[0]['exc_elong'])} and {f(lam[1]['exc_elong'])}), "
          f"but their axes are not shared across the type (consistency {f(lam[0]['exc_axis_R'])}, {f(lam[1]['exc_axis_R'])}). C2 and C3 pool only ~{lam[2]['all_footprint']:.0f} and ~{lam[3]['all_footprint']:.0f} columns. "
          "So there is no type-wide orientation signal in the feedback to the lamina. The orientation structure appears one step later, in the medulla, in Dm3 and TmY9.")
dm3 = ("<b>Reading the model.</b> Orientation tuning of Dm3 and TmY emerges from the measured wiring alone: no parameters were fit. Divisive inhibition compresses the CRF at high contrast (a normalization-like shape), "
       "while subtractive inhibition scales it. The main unverified assumption is that Dm3's glutamate is inhibitory (GluCl). Try raising the Dm3 gain, switching to subtractive, or shortening the wavelength.")
methods = """
<h2>Data</h2><ul>
<li><b>FlyWire v783</b> proofread connections (Zenodo 10676866), restricted to right optic-lobe neuropils (ME, LO, LOP, LA, AME), connections ≥ 3 synapses. Cell types and transmitter predictions from flyconnectome/flywire_annotations (Schlegel et al. 2024; Matsliah et al. 2024 optic-lobe types; Eckstein et al. 2024 transmitters).</li>
<li><b>Column map</b>: the Matsliah et al. 2024 assignment of 22,684 columnar cells to 796 columns (via connectome-interpreter), converted from doubled-height hex coordinates to Euclidean so all six neighbours are 1 column-spacing apart.</li>
<li><b>flyvis</b> (Lappalainen et al. 2024): 50-model ensemble trained on optic flow; 65 cell types on a 721-column lattice; passive point neurons, threshold-linear graded synapses.</li></ul>
<h2>Connectome measures</h2><ul>
<li><b>Footprint</b>: distinct columns providing columnar input to a neuron. <b>Radius</b>: synapse-weighted RMS distance from the input centroid.</li>
<li><b>Excess elongation</b>: λ₁−λ₂ of the trace-normalized weighted input covariance minus that of a compact disk with the same number of columns at the same place on the real (irregular) lattice. <b>Axis consistency</b>: resultant length of doubled axis angles across cells with ≥ 6 input columns.</li>
<li><b>Pool score</b> (inhibitory types only): z(log footprint) + z(fraction of output onto T4/T5 inputs + lamina monopolar cells).</li></ul>
<h2>Simulations</h2><ul>
<li>Drifting sinusoidal gratings (λ = 8 columns, 2 Hz) in a centre disk (r = 2) with an annulus from r = 4 to the edge, iso or orthogonal, at dt = 20 ms (the training step). Response = F1 of the central cell over the last second, at each type's preferred of 4 directions. Best 5 models by validation loss for CRFs and size tuning; best 3 for lesions.</li>
<li>Naka–Rushton fits with the exponent fixed by the no-surround fit; c50 is capped at 5 for the gain map, and types with c50 > 2 are flagged as near-linear.</li>
<li>Lesions zero the weights of the chosen edges in trained networks, with no retraining.</li>
<li><b>Grafted pools</b> (<code>graft.py</code>): Dm17, Dm1 and Dm15 are added as pool fields over the 721 columns (τ = 50 ms). Inputs are the signed FlyWire synapse fractions from flyvis cell types, half-wave rectified relative to the grey operating point. The kernel is Gaussian, with RMS radius equal to the FlyWire input radius (7.2, 2.5 and 1.3 columns). Output weights are the FlyWire output fractions onto flyvis types. Divisive: D' = D₀ + (D − D₀)/(1 + G). Subtractive: D' = D − σ·G, with σ the drive SD under a full-field grating. g = 1 halves the main target's modulation at full contrast. CRF runs use the 2 best models. Fine-tuning: Adam (network 1e-4, decoder 1e-4, log-gains 1e-2), batch 4, original augmentation, fold-1 validation split; 120 iterations for the grafted network and for a no-pool control.</li>
<li><b>Dm3 model</b>: every Dm3p/q/v and TmY9q/TmY9q⊥/TmY4 neuron, FlyWire weights normalized per neuron. Columnar inputs carry rectified ON/OFF contrast; ACh +, GABA/Glu −. Rate dynamics run to steady state; responses are peaks over test-grating phase, averaged over mask phase. Masking ratio = (R(test+mask) − R(mask)) / R(test).</li></ul>
<h2>Caveats</h2><ul>
<li>Transmitter → sign is a prediction. Glutamate is treated as inhibitory throughout, and that is unverified for Dm3, Dm15 and Dm17.</li>
<li>FlyWire has no gap junctions or neuromodulators, and synapse count is a noisy proxy for weight.</li>
<li>The flyvis model omits most wide-field cells, so its lack of surround normalization is a statement about the model as much as the fly. The natural next step is to graft the candidate pools onto it and retrain with conductance-based (shunting) synapses.</li>
<li>The Dm3 model is static (no temporal dynamics) and treats columnar inputs as ideal ON/OFF rectifiers.</li></ul>
<p class="hint">Code: <code>analysis/</code> in this repo. Reproduce with <code>flywire_pools.py</code>, <code>dm3_circuit.py</code>, <code>dm3_model.py</code>, <code>run_experiments.py</code>, <code>assemble_*.py</code>, <code>write_findings.py</code>.</p>
"""
json.dump(dict(cards=cards, lamina=lamina, dm3=dm3, methods=methods, graft=graft_callout), open('docs/data/findings.json', 'w'), indent=1)
print(json.dumps(cards, indent=1)[:3000])
