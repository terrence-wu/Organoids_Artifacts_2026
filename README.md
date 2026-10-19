# inSilicoSimRx

Simulation and reanalysis code for:

> Tsou and Wu. **"Identifying Measurement Artifacts Driving Variability in Organoid Drug Screening."** *SLAS Discovery* 

Patient-derived organoid (PDO) drug screens usually read ATP luminescence once, at a
fixed time after dosing, and divide each well by an untreated control. This repository
contains the code used to show how three properties of that measurement bias apparent
potency with organoid size, and to apply the corresponding corrections to a published
size-stratified screen:

| artifact | what it is |
|---|---|
| **A1**, volume weighting | a pooled signal counts each organoid in proportion to its volume |
| **A2**, growth normalization | each well is divided by an untreated control that grew during the assay |
| **A3**, readout nonlinearity | luminescence rises faster than ATP (RLU ∝ ATP^1.186) |

It also implements the attainability criterion, *E*max·*G* ≤ 0.5, which states when a
reported IC₅₀ reflects net cell kill rather than the control's growth.

---

## Repository contents

Scripts are grouped by the section of the paper they support.

| paper section | scripts | main outputs |
|---|---|---|
| §2.1–2.3 Simulation model | `pdo_model.py` | (imported by the scripts below) |
| §2.4, §3.1 Factorial decomposition (Fig. 1) | `sweep_sep_rebuild.py` | [[output file]] |
| SM2 Dose-grid convergence | `ladder_sizes.py`, `ladder_217.py`, `sweep217.py`, `analyse217.py` | `sweep217_results.txt` |
| SM6 Swapped proliferation dependence | `factorial_swapped_fprolif.py`, `analyse_swapped_fprolif.py` | `matched_fprolif_results.txt` |
| §2.5, §3.2 Attainability (Fig. 2) | `attainability.py`, `attainability_duration.py`, `attainability_sensitivity.py` | `attainability.json`, `attainability_duration.json`, `attainability_sensitivity.json` |
| §2.6, §3.3 Reanalysis of Zieger et al. (Fig. 3, Table 1) | `zieger_correction.py` | `zieger_cascade.json` |
| Table S1 Estimator dependence | `estimator_table.py` | `estimator_table.json` |
| SM4 Bootstrap uncertainty | `zieger_bootstrap.py` | `zieger_bootstrap.json` |
| §2.7, SM5 Morphology contrast | `morphology_bootstrap.py` | printed to console |
| Figures | `fig1_factorial_v4.py`, `fig2_attainability_v3.py`, `fig3_cascade.py`, `fig4_decision_rule_v2.py`; Supplementary Fig. S1: `fig3d_morphology.py` | PNG and PDF |

---

## Requirements

- Python 3.12
- NumPy, SciPy, Matplotlib, openpyxl

```bash
pip install -r requirements.txt
```


---

## Data

The reanalysis uses the supplementary tables of:

> Zieger V, Woehr E, Traichel J, Brummer T, Zengerle R, Kartmann S, Zimmermann S.
> Impact of non-genetic heterogeneity of BRAF-mutant colon cancer organoids on growth
> kinetics, drug sensitivity and Wnt dynamics. *Int J Cancer* (2026).
> doi:10.1002/ijc.70453

Download the supplementary file `ijc70453-sup-0002-tables.xlsx` from the article page and
place it in `data/`. [[State whether the file is redistributed here; check the article's
license first.]] No new experimental data were generated.

---

## Reproducing the results

Run from the repository root. Scripts that read the Zieger data take the spreadsheet path
as their first argument.

```bash
# Simulation (§3.1, §3.2)
python sweep_sep_rebuild.py                 # [[arguments, runtime]]
python sweep217.py && python analyse217.py
python factorial_swapped_fprolif.py && python analyse_swapped_fprolif.py
python attainability.py
python attainability_duration.py
python attainability_sensitivity.py

# Reanalysis (§3.3)
python zieger_correction.py data/ijc70453-sup-0002-tables.xlsx    # [[confirm]]
python estimator_table.py   data/ijc70453-sup-0002-tables.xlsx
python zieger_bootstrap.py  data/ijc70453-sup-0002-tables.xlsx    # [[confirm]]
python morphology_bootstrap.py data/ijc70453-sup-0002-tables.xlsx

# Figures
python fig1_factorial_v4.py   # etc.
```

Bootstrap analyses use 10,000 resamples with seed 20260928.

---

## Model conventions

- **Units.** The solver works in cm and s. Diameters and radii in function arguments are
  in µm.
- **Diameter vs radius.** `simulate(drug, R_um, ...)` takes the organoid **diameter**,
  despite the argument name. [[Confirm, and note any other function whose `R_um` is a
  radius, e.g. `oxygen_profile`.]]
- **Viability.** `simulate()` returns the volume-averaged cell-number multiplier from
  dosing to readout (1 at dosing). Its value at zero dose is the control fold-growth *G*.
- **Surface condition.** `BC_FREE = True` sets the organoid surface to the dome's *free*
  concentration (*C*gel/*K*eff). `False` reproduces an earlier, incorrect convention and
  is kept only for comparison.
- **Oxygen.** `PHI_FROM_RADIUS = True` builds the Thiele modulus from the radius, as in
  Shi et al. (2026). `False` reproduces an earlier error and is kept only for comparison.
- **Fixed geometry.** Organoid diameter is held at its dosing value for the whole assay.
- **Agents.** Simulations use cisplatin and cetuximab. 



---

## Citation

If you use this code, please cite the paper above. 

## License

GPL-3.0 License

## Contact


