"""estimator_table.py (v2) -- Table S1: the Zieger cascade under all eight
viability-estimator combinations (column x aggregate x order) and four
growth statistics: median of per-well ratios [canonical], ratio of means,
mean of ratios, and block-matched (median of per-well ratios over the control
wells of the biological replicates in which each dose was acquired, so G is
per dose block). Uses the canonical ic50 (running minimum) of zieger_correction.py.
v2 2026-10-02: adds block-matched G (Methods M6; HANDOFF v3 decision 4).
"""
import sys, json, collections, itertools, numpy as np, openpyxl
from scipy.optimize import brentq
XLSX = sys.argv[1] if len(sys.argv) > 1 else "ijc70453-sup-0002-tables.xlsx"
BINS = ["50-100", "100-150", "150-200"]; H = 1.1859; MSR_LO = 2.46
RAW, CORR = "Luminescence intensity", "Organoid number corrected luminescence intensity"
A1C, A8C = "Total organoid area per well at day 1 [µm²]", "Total organoid area per well at day 8 [µm²]"
DIA, CONC = "Initial organoid diameter [µm]", "Trametinib concentration [nM]"
wb = openpyxl.load_workbook(XLSX, data_only=True)
def sheet(n):
    ws = wb[n]; h = [c.value for c in ws[1]]
    return [dict(zip(h, [c.value for c in r])) for r in ws.iter_rows(min_row=2)]
S1, S2 = sheet("S1_Organoid Growth Data"), sheet("S2_CellTiter Glo Organoid Data")
strat = lambda rs, b: [r for r in rs if r[DIA] == b and r["Seeding density"] == 20 and r["Medium composition"] == "COCM+GF"]
def ic50(d, v):
    v = np.minimum.accumulate(np.asarray(v, float)); lv = np.log10(d)
    if not (v[0] >= 0.5 >= v[-1]): return np.nan
    return 10 ** brentq(lambda t: np.interp(t, lv, v) - 0.5, lv[0], lv[-1])
def curve(b, col, agg, order):
    byc = collections.defaultdict(list)
    for r in strat(S2, b): byc[float(r[CONC])].append(r)
    ctrl = collections.defaultdict(list)
    for r in byc[0.0]: ctrl[r["Biological replicate"]].append(float(r[col]))
    A = np.mean if agg == "mean" else np.median
    d = np.array(sorted(c for c in byc if c > 0)); v = []
    for c in d:
        tr = byc[c]; reps = {t["Biological replicate"] for t in tr}
        if order == "ratio of aggregates":
            v.append(A([float(t[col]) for t in tr]) / A([q for rp in reps for q in ctrl[rp]]))
        else:
            v.append(A([float(t[col]) / A(ctrl[t["Biological replicate"]]) for t in tr]))
    return d, np.array(v)
Gs = {}
for b in BINS:
    rs = strat(S1, b); A1 = np.array([float(r[A1C]) for r in rs]); A8 = np.array([float(r[A8C]) for r in rs])
    Gs[b] = {"median of ratios": float(np.median(A8 / A1)), "ratio of means": float(A8.mean() / A1.mean()),
             "mean of ratios": float(np.mean(A8 / A1))}
# block-matched G: per stratum, per dose, median A8/A1 over S1 control wells from the
# biological replicates in which that dose was acquired in S2.
def growth_ratios_by_rep(b):
    out = collections.defaultdict(list)
    for r in strat(S1, b): out[r["Biological replicate"]].append(float(r[A8C]) / float(r[A1C]))
    return out
GBM = {}
for b in BINS:
    gr = growth_ratios_by_rep(b); reps_at = collections.defaultdict(set)
    for r in strat(S2, b): reps_at[float(r[CONC])].add(r["Biological replicate"])
    d = sorted(c for c in reps_at if c > 0)
    GBM[b] = np.array([np.median([x for rp in sorted(reps_at[c]) for x in gr[rp]]) for c in d])
    Gs[b]["block-matched"] = {f"{c:g} nM": float(g) for c, g in zip(d, GBM[b])}
GSTATS = ("median of ratios", "ratio of means", "mean of ratios", "block-matched")
rows = []
for col, agg, order in itertools.product(("count-corrected", "raw"), ("mean", "median"), ("ratio of aggregates", "aggregate of ratios")):
    cc = CORR if col == "count-corrected" else RAW
    cv = {b: curve(b, cc, agg, order) for b in BINS}
    ic0 = np.array([ic50(*cv[b]) for b in BINS])
    ic1 = np.array([ic50(cv[b][0], cv[b][1] ** (1 / H)) for b in BINS])
    for gname in GSTATS:
        gv = (lambda b: GBM[b]) if gname == "block-matched" else (lambda b: Gs[b][gname])
        v3 = {b: cv[b][1] ** (1 / H) * gv(b) for b in BINS}
        ic3 = np.array([ic50(cv[b][0], v3[b]) for b in BINS])
        emg = [float(np.min(v3[b])) for b in BINS]; top = [float(v3[b][-1]) for b in BINS]
        rows.append(dict(column=col, aggregate=agg, order=order, G=gname,
                         ic0=ic0.tolist(), ic1=ic1.tolist(), ic3=ic3.tolist(),
                         s0=float(ic0.max() / ic0.min()), s1=float(ic1.max() / ic1.min()),
                         s3=float(np.nanmax(ic3) / np.nanmin(ic3)), across3=float(ic3[2] / ic3[0]),
                         EmaxG_min=emg, EmaxG_top=top,
                         argmin_dose=[float(cv[b][0][int(np.argmin(v3[b]))]) for b in BINS],
                         n_strata=int(np.sum(np.isfinite(ic3))),
                         canonical=(col, agg, order, gname) == ("count-corrected", "mean", "ratio of aggregates", "median of ratios")))
json.dump(dict(G=Gs, rows=rows), open("estimator_table.json", "w"), indent=1)
print("G by statistic:", {b: {k: (round(v, 2) if not isinstance(v, dict) else {kk: round(vv, 2) for kk, vv in v.items()}) for k, v in g.items()} for b, g in Gs.items()})
print(f"{'column':16s}{'agg':7s}{'order':21s}{'G stat':18s}{'s0':>6s}{'s1':>6s}{'s3':>6s}{'<MSR':>5s}  ic3                 EmaxG(min)")
for r in rows:
    print(f"{r['column']:16s}{r['aggregate']:7s}{r['order']:21s}{r['G']:18s}{r['s0']:6.2f}{r['s1']:6.2f}{r['s3']:6.2f}{'yes' if r['s3']<MSR_LO else 'NO':>5s}  "
          f"{np.round(r['ic3'],1)!s:20s}{np.round(r['EmaxG_min'],2)!s}{'  <- canonical' if r['canonical'] else ''}")
s3 = [r["s3"] for r in rows]; s0 = [r["s0"] for r in rows]
two = [r for r in rows if r["n_strata"] < 3]
print(f"two-strata variants: {len(two)}/{len(rows)}; all-three variants below 2.46: "
      f"{sum(r['s3'] < MSR_LO for r in rows if r['n_strata'] == 3)}/{sum(r['n_strata'] == 3 for r in rows)}")
print(f"\nstage 0 range {min(s0):.2f}-{max(s0):.2f}; stage 3 range {min(s3):.2f}-{max(s3):.2f}; above 2.46 in {sum(x>=MSR_LO for x in s3)}/{len(s3)}")
