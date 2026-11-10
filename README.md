# TheFly

Looking for the circuits behind contrast gain control in the *Drosophila* optic-lobe connectome, and what they do to the contrast response function (CRF). Everything here is computed from published data, with no new experiments.

**Dashboard:** https://samuelling.github.io/TheFly/ (served from `docs/`)

## What's in it

| Tab | Question | Data |
|---|---|---|
| Normalization pools | Which inhibitory cell types pool many columns and feed back onto the inputs to T4/T5 or the lamina? | FlyWire v783, right optic lobe |
| Orientation | Which types have elongated input pools that share an axis across cells (the wiring signature of orientation tuning)? | FlyWire + Matsliah et al. 2024 column map |
| Dm3 circuit model | Does the Dm3 → TmY9/TmY4 subconnectome produce orientation-tuned gain control? | Every Dm3/TmY neuron and its synapses; runs in the browser |
| In-silico CRFs | How do surround contrast (iso vs orthogonal) and stimulus size change CRFs in a connectome-constrained model? | flyvis ensemble (Lappalainen et al. 2024) |
| Lesions | Which subconnectomes carry the surround effects? | flyvis, edges cut in trained networks |

## Reproduce

```bash
uv venv -p python3.12 venv && source venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install flyvis connectome-interpreter pyarrow playwright
flyvis download-pretrained --skip_large_files
# FlyWire connections (852 MB) + annotations
curl -L -o data/proofread_connections_783.feather "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"
curl -L -o data/neuron_annotations.tsv https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv

python analysis/flyvis_wiring.py          # pool radii in the flyvis connectome
python analysis/flywire_pools.py          # footprints, elongation, axes, pool scores
python analysis/flywire_partners.py       # partner lists
python analysis/dm3_circuit.py            # export the Dm3 -> TmY circuit
python analysis/dm3_model.py              # reference run of the Dm3 model
python analysis/run_experiments.py crf --n_models 5
python analysis/run_experiments.py size --n_models 5
python analysis/run_experiments.py lesion --n_models 3 --lesions lateral_inhibition,lateral_excitation,TmY15,CT1,Mi4_Mi9,C2_C3,Lawf,Am
python analysis/assemble_flywire.py && python analysis/assemble_sims.py && python analysis/write_findings.py
```

The FlyWire paths in the scripts point to `/home/claude/data/flywire/`; edit `D = ...` at the top of each FlyWire script.

## Key caveats

- Transmitter → sign is a prediction. Glutamate is treated as inhibitory (GluCl).
- There are no gap junctions or neuromodulators, and synapse count is a proxy for synaptic weight.
- The flyvis model contains none of the wide-field Dm/Pm candidates, so its weak surround effects are partly a property of the model.

## Sources

Dorkenwald et al. 2024; Schlegel et al. 2024; Matsliah et al. 2024 (Nature); Lappalainen et al. 2024 (Nature); Drews et al. 2020 (Curr Biol); Eckstein et al. 2024 (Cell).
