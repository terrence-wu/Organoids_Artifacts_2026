"""Bootstrap of the shared-dose (10, 20 nM) viability contrast, cystic vs solid,
COCM arm, 20 organoids/well. Canonical estimator: count-corrected luminescence,
ratio of pooled treated mean to pooled same-replicate control mean
(luminescence scale). Wells resampled within (arm, dose, replicate) and controls
within (arm, replicate); B = 10 000. Also reports each biological replicate alone."""
import sys, collections, numpy as np, openpyxl
XLSX = sys.argv[1] if len(sys.argv) > 1 else "ijc70453-sup-0002-tables.xlsx"
C = "Trametinib concentration [nM]"; CORR = "Organoid number corrected luminescence intensity"
wb = openpyxl.load_workbook(XLSX, data_only=True); ws = wb["S2_CellTiter Glo Organoid Data"]
hdr = [c.value for c in ws[1]]
rows = [dict(zip(hdr, [c.value for c in r])) for r in ws.iter_rows(min_row=2)]
cells = collections.defaultdict(list)
for r in rows:
    if r["Medium composition"] == "COCM" and r["Seeding density"] == 20:
        cells[(r["Organoid morphology"], float(r[C]), r["Biological replicate"])].append(float(r[CORR]))
cells = {k: np.array(v) for k, v in cells.items()}
def via(cl, arm, dose, reps=(1, 2)):
    t = np.concatenate([cl[(arm, dose, r)] for r in reps])
    c = np.concatenate([cl[(arm, 0.0, r)] for r in reps])
    return t.mean() / c.mean()
rng = np.random.default_rng(20260928); B = 10_000
for dose in (10.0, 20.0):
    pt = {a: via(cells, a, dose) for a in ("Cystic", "Solid")}
    bs = []
    for _ in range(B):
        cb = {k: rng.choice(v, v.size) for k, v in cells.items()}
        bs.append((via(cb, "Cystic", dose), via(cb, "Solid", dose)))
    bs = np.array(bs); ratio = bs[:, 0] / bs[:, 1]
    print(f"{dose:g} nM  cystic {pt['Cystic']:.3f} {np.percentile(bs[:,0],[2.5,97.5]).round(3)}  "
          f"solid {pt['Solid']:.3f} {np.percentile(bs[:,1],[2.5,97.5]).round(3)}  "
          f"ratio {pt['Cystic']/pt['Solid']:.2f} {np.percentile(ratio,[2.5,97.5]).round(2)}  "
          f"P(cystic>solid) {np.mean(ratio>1):.4f}")
    for rp in (1, 2):
        print(f"     rep{rp} alone: cystic {via(cells,'Cystic',dose,(rp,)):.3f}  solid {via(cells,'Solid',dose,(rp,)):.3f}")
