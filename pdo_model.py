"""
pdo_model.py -- canonical PDO assay-geometry model.
===================================================

Supersedes stage1_pde.py, o2_kinetics_test.py and the model half of
morris_screen.py, which had drifted apart (NR = 32 vs 16, Krogh vs first-order
oxygen, different tolerances). USE THIS for all recomputation.

Compartments
    medium   well-mixed, V_med
    dome     well-mixed, capacity K_eff*V_gel, finite transfer time tau_gel
    organoid 1-D radial finite-volume PDE, Dirichlet C(R) = C_gel
    oxygen   quasi-steady, precomputed per condition; consumption is a PACKED
             TISSUE property so the profile depends on RADIUS ALONE
    cells    N is a cell-NUMBER MULTIPLIER (1.0 at dosing), no logistic cap

Endpoints returned by `endpoints()`
    GR50, GRmax   growth-rate normalised (Hafner 2016). Return NaN when the
                  control does not grow -- this is STRUCTURAL, not numerical:
                  log2(x_ctrl/x0) is the denominator. How often it happens is
                  itself a headline result (83% of prior at 500 um).
    IC50, Emax    raw relative viability. Defined whenever the control survives.

Numerical lessons already paid for -- do not undo:
  * clip C and N non-negative inside the RHS (non-integer Hill on a negative
    number returns NaN and kills the solve)
  * smooth the quiescence switch (hard switches wreck the BDF Jacobian)
  * surface face spans dr/2, interior faces dr
  * no logistic cap on N (it makes every shell saturate and erases the O2 effect)
  * NEVER hard-code NR. The O2 boundary layer does not scale with organoid size, so a fixed
    node count silently under-resolves large organoids. simulate() defaults to NR_for(diameter);
    pass NR only to override that deliberately.
"""

import numpy as np
from scipy.integrate import solve_ivp

# ------------------------------------------------------------------ constants
V_MED = 100.0                 # uL, 96-well
T_ASSAY = 72.0 * 3600.0       # s
P_SURF = 1.9e-7               # mol/cm^3, ~190 uM at air saturation
P_QUI = 0.50 * P_SURF         # Shi 2026: quiescence at 100 uM of 200 uM bulk
P_LETH = 0.025 * P_SURF       # Shi 2026: lethality at 5 uM
P_KM = 0.15 * P_SURF

# Krogh (zeroth-order) alternative.  The profile depends on q and D ONLY through kappa = q/D,
# so kappa is the single calibratable parameter.  KROGH_QD is settable; see krogh_calibrate.py.
D_O2_KROGH = 2.0e-5           # cm^2/s   -- ABOVE both cited values (Matrigel 1.01e-5, collagen 1.46e-5)
Q_O2_KROGH = 6.0e-8           # mol/(cm^3 s) -- ABOVE the measured consumption range
KROGH_QD = Q_O2_KROGH / D_O2_KROGH        # cm^-2, the only quantity the profile uses

# Shi et al. APL Bioeng 2026, Table: fitted (k/D)_O2 in um^-2, mean +/- SD.
#   PANC-1 cell line spheroids         1.82 +/- 0.67 e-3   (n = 26)   MEASURED
#   3R-4/3R-5-1 patient-derived        9.12 +/- 3.30 e-3   (n = 56)   MEASURED
#   HepG2                              4.29e-3, NO SD -- Magliaro et al. 2019, a DIFFERENT study.
#     Dropped from all robustness claims 2026-08-29: it is a literature point estimate with no
#     dispersion, so it cannot carry an uncertainty band. Key retained only so legacy code runs.
#
# NOMINAL IS NOW THE PATIENT-DERIVED LINE. This project is about patient-derived organoids;
# PANC-1 is an immortalised cell line grown as a spheroid and is retained as a CONTRAST case.
# The +/-1SD windows do not overlap (1.15-2.49 vs 5.82-12.42), so the two are genuinely distinct
# systems, not two draws from one population.
KD_O2 = {"PDO": 9.12e-3, "PDO_lo": 5.82e-3, "PDO_hi": 12.42e-3,
         "PANC1": 1.82e-3, "PANC1_lo": 1.15e-3, "PANC1_hi": 2.49e-3,
         "PDO_pancreatic": 9.12e-3,       # legacy alias
         "HepG2": 4.29e-3}                # deprecated, see above
KD_O2_SD = {"PDO": 3.30e-3, "PANC1": 0.67e-3}
KD_O2_N = {"PDO": 56, "PANC1": 26}

# ------------------------------------------------------- gel/organoid interface convention
# C_gel is a TOTAL concentration: K_eff is total-to-free, and both the transfer flux J and the
# retardation in D_eff assume it. C inside the organoid is FREE -- it is scored against EC50 and
# carries no partition coefficient of its own. The free concentration is therefore what must be
# continuous across the interface.
#
# BC_FREE = False reproduces the pre-2026-08 behaviour (Cg[-1] = C_gel), under which drug
# equilibrates to K_eff x the medium concentration inside a compartment with no binding capacity
# to hold it. Retained only so the two conventions can be compared; do not use it for results.
# See test_interface_equilibrium.py::T1.
BC_FREE = True

# RESOLVED 2026-08-29 from Shi et al.'s own text: they model the spheroid as a "spherical catalytic
# particle of radius (R)", and state that a critical radius R_crit = 72 um corresponds to a
# spheroid DIAMETER of about 150 um. Their R is unambiguously the radius, so Phi = R*sqrt(k/D) must
# be built from the radius. The historical code passed the diameter, solving the oxygen problem for
# an organoid 2x its labelled size. False reproduces that behaviour for provenance only.
PHI_FROM_RADIUS = True

DRUGS = {
    "cisplatin": dict(MW=300.0, D_aq=7.5e-6, hind=0.85, K_gel=1.3, k_up=3e-5,
                      hill=1.3, EC50=6.0, f_prolif=0.60, unit="uM"),
    "docetaxel": dict(MW=808.0, D_aq=3.4e-6, hind=0.62, K_gel=24.0, k_up=3e-4,
                      hill=1.3, EC50=0.003, f_prolif=0.95, unit="uM"),
    "cetuximab": dict(MW=152000.0, D_aq=4.2e-7, hind=0.15, K_gel=2.2, k_up=3e-4,
                      hill=1.0, EC50=1.0, f_prolif=0.80, unit="ug/mL"),
}


def D_aq_from_MW(MW):
    return 7.5e-6 * (300.0 / MW) ** 0.40


# ------------------------------------------------------------------ radial mesh
DR_TARGET_UM = 2.6            # ~1/4 of the SHORTEST measured O2 decay length (10 um, PDO)


def NR_for(D_um, dr_um=DR_TARGET_UM, nr_min=16, nr_max=256):
    """Radial nodes needed to resolve the O2 boundary layer at this DIAMETER.

    The O2 decay length sqrt(D/k) is 10-23 um (MEASURED, Shi 2026) and does NOT scale with
    organoid size, so a FIXED NR under-resolves large organoids: NR = 24 gives dr = 10.4 um at
    500 um, one cell across the whole shell, and up to 19% error in Emax(500). Hold dr fixed
    instead. NR_for(150) = 29, NR_for(500) = 97, NR_for(600) = 116.

    ⚠️ ARGUMENT IS A DIAMETER (it halves internally), as everywhere else in this module. It was
    named R_um until 2026-08-30, which is the same radius/diameter ambiguity that produced the
    Thiele defect; renamed so the next reader cannot repeat it. All in-tree callers pass
    positionally, so the rename is source-compatible.

    nr_min = 16 is a floor for the DRUG PDE at small radii, not for the O2 layer -- below ~125 um
    diameter the dr target is already met. Dropping the small arm (43 um) from the historical
    NR = 24 to NR_for = 16 moves S_mx and Emax by <= 1e-5 across the whole panel, i.e. three
    orders below the smallest tier-1 effect. Measured 2026-08-30, do not re-litigate.
    """
    return int(min(nr_max, max(nr_min, np.ceil((D_um / 2.0) / dr_um))))


def hindrance_from_MW(MW):
    return 1.0 / (1.0 + (MW / 8000.0) ** 0.60)


# ------------------------------------------------------------------ oxygen
def oxygen_profile(r_um, R_um, kinetics="firstorder", kD=KD_O2["PDO"]):
    """Quasi-steady O2. Krogh over-predicts oxygenation; first-order (Shi)
    over-predicts hypoxia. Reality (Michaelis-Menten) lies between -- run both."""
    if kinetics == "krogh":
        R, r = R_um * 1e-4, np.asarray(r_um) * 1e-4
        return np.clip(P_SURF - (KROGH_QD / 6.0) * (R**2 - r**2), 0.0, P_SURF)
    Phi = R_um * np.sqrt(kD)
    rhat = np.clip(np.asarray(r_um) / R_um, 1e-6, 1.0)
    if Phi > 300:
        frac = np.exp(-Phi * (1.0 - rhat))
    else:
        frac = np.sinh(np.clip(Phi * rhat, 0, 700)) / (rhat * np.sinh(Phi))
    return P_SURF * np.clip(frac, 0.0, 1.0)


def cell_rates(R_um, NR, rho_max, K_nec, kinetics="firstorder", kD=KD_O2["PDO"]):
    """Precompute rho(r) and k_nec(r). Sigmoid switches keep the RHS smooth.

    ⚠️ PHI_FROM_RADIUS.  R_um is a DIAMETER everywhere in this module (simulate does
    R = R_um/2). The sphere Thiele modulus Phi = R*sqrt(k/D) takes the RADIUS. Passing the
    diameter makes Phi 2x too large, i.e. solves the oxygen problem for an organoid twice its
    labelled size. Evidence: Shi et al.'s published 90 um transport/hypoxia crossover is
    reproduced only by the radius convention (centre at 0.575 of surface O2, just above
    quiescence); the diameter convention puts a 90 um organoid at 0.165, already deeply
    quiescent. Signature: prolif_frac(150 um, as coded) = 0.3460 = prolif_frac(300 um, radius).
    Default is left at the historical behaviour so nothing changes silently -- see
    THIELE_CONVENTION_MEMO.md. Flipping this rescales every diameter in the project by 2x.
    """
    R_o2 = (R_um / 2.0) if PHI_FROM_RADIUS else R_um
    r = (np.arange(NR) + 0.5) / NR * R_o2
    P = oxygen_profile(r, R_o2, kinetics, kD)
    w_pro = 1.0 / (1.0 + np.exp(-(P - P_QUI) / (0.10 * P_QUI)))
    w_via = 1.0 / (1.0 + np.exp(-(P - P_LETH) / (0.10 * P_LETH)))
    rho = rho_max * (P / (P + P_KM)) * w_pro
    prolif_frac = float(np.average(w_pro, weights=(np.arange(NR) + 0.5) ** 2))
    return rho, K_nec * (1.0 - w_via), prolif_frac


# ------------------------------------------------------------------ transport
def dome_params(p, V_gel, frac):
    """Returns (K_eff, k_transfer). Theta = K_eff*V_gel/V_med;  Da = tau/t_assay."""
    K_eff = 1.0 + (p["K_gel"] - 1.0) * frac
    D_eff = p["D_aq"] * (p["hind"] ** (frac / 0.5)) / K_eff
    R_dome = (3.0 * V_gel * 1e-3 / (2.0 * np.pi)) ** (1.0 / 3.0)   # cm, hemisphere
    return K_eff, 6.0 * D_eff / R_dome ** 2


def theta_Da(p, V_gel, frac, t_assay=T_ASSAY):
    K_eff, k_tr = dome_params(p, V_gel, frac)
    return K_eff * V_gel / V_MED, (1.0 / k_tr) / t_assay


# ------------------------------------------------------------------ RHS
def _rhs(t, y, p, R, dr, D_org, K_eff, k_tr, V_gel, rho, k_nec, E_max, rho_max):
    C_gel = max(y[1], 0.0)
    C = np.maximum(y[2:2 + len(rho)], 0.0)
    N = np.clip(y[2 + len(rho):], 0.0, 1e3)
    NR = len(rho)

    Cg = np.empty(NR + 2)
    # no-flux centre; Dirichlet surface. See BC_FREE above for why the surface value is the FREE
    # concentration C_gel / K_eff rather than the total C_gel.
    Cg[1:-1] = C; Cg[0] = C[0]; Cg[-1] = C_gel / K_eff if BC_FREE else C_gel
    rf = np.linspace(0.0, R, NR + 1)
    sp = np.full(NR + 1, dr); sp[0] = sp[-1] = 0.5 * dr
    flux = D_org * (Cg[1:] - Cg[:-1]) / sp * rf ** 2
    vol = (rf[1:] ** 3 - rf[:-1] ** 3) / 3.0
    dC = (flux[1:] - flux[:-1]) / vol - p["k_up"] * C * N

    J = k_tr * (max(y[0], 0.0) - C_gel / K_eff) * V_gel * 1e-3
    Cp = C ** p["hill"]
    E = E_max * Cp / (p["EC50"] ** p["hill"] + Cp)
    pf = p["f_prolif"] * (rho / rho_max) + (1.0 - p["f_prolif"])
    dN = (rho - k_nec) * N - E * pf * N
    return np.concatenate(([-J / (V_MED * 1e-3), J / (V_gel * 1e-3)], dC, dN))


def simulate(drug, R_um, V_gel, frac, dose, NR=None, T_double=30.0, E_ratio=3.0,
             K_nec_rel=0.25, kinetics="firstorder", kD=KD_O2["PDO"], override=None):
    """Mean cell-number multiplier at t_assay. Returns (value, prolif_fraction).

    NR = None (the default) resolves the mesh from the diameter via NR_for, i.e. fixed
    dr = 2.6 um. This is the settled convention (PROJECT_STATE v6 sec.3). It was declared settled
    on 2026-08-29 but NOT wired in: the signature still read NR = 24 and no caller passed NR_for,
    so the convention held only where a caller happened to remember it. Fixed 2026-08-30.

    Why it mattered: at 500 um, NR = 24 gives dr = 10.4 um and Emax = 0.4944, against a converged
    0.5267 -- a -6.1% error that STRADDLES THE Emax = 0.5 BOUNDARY, so the coarse default said
    IC50 exists where the converged mesh says it is gone. Nothing published was affected
    (ladder_sweep passed NR explicitly, and at the line-indexed arms 43/124 um the old default was
    already at or below the dr target), but the trap was live and silent -- wrong number, no error.

    Pass NR explicitly only to override the convention, e.g. for a convergence study.
    """
    p = dict(DRUGS[drug]); p.update(override or {})
    NR = NR_for(R_um) if NR is None else int(NR)
    rho_max = np.log(2) / (T_double * 3600.0)
    rho, k_nec, pf = cell_rates(R_um, NR, rho_max, K_nec_rel * rho_max, kinetics, kD)
    R = (R_um / 2.0) * 1e-4
    dr = R / NR
    K_eff, k_tr = dome_params(p, V_gel, frac)
    y0 = np.concatenate(([dose, 0.0], np.zeros(NR), np.ones(NR)))
    sol = solve_ivp(_rhs, (0.0, T_ASSAY), y0, method="BDF",
                    args=(p, R, dr, p["D_aq"] * p["hind"] * 0.35, K_eff, k_tr,
                          V_gel, rho, k_nec, E_ratio * rho_max, rho_max),
                    rtol=1e-6, atol=1e-9, t_eval=[0.0, T_ASSAY])
    if not sol.success:
        return np.nan, pf
    rf = np.linspace(0.0, R, NR + 1)
    vol = (rf[1:] ** 3 - rf[:-1] ** 3) / 3.0
    return float(np.sum(sol.y[2 + NR:, -1] * vol) / np.sum(vol)), pf


def endpoints(drug, R_um, V_gel, frac, n_doses=10, span=300.0, **kw):
    """GR50/GRmax (NaN when the control does not grow) and IC50/Emax (robust)."""
    p = DRUGS[drug]
    doses = np.logspace(np.log10(p["EC50"] / span), np.log10(p["EC50"] * span), n_doses)
    xc, pf = simulate(drug, R_um, V_gel, frac, 0.0, **kw)
    xs = np.array([simulate(drug, R_um, V_gel, frac, d, **kw)[0] for d in doses])
    res = dict(doses=doses, ctrl=xc, prolif_frac=pf,
               GR50=np.nan, GRmax=np.nan, IC50=np.nan, Emax=np.nan, gr=None, rel=None)
    if not np.isfinite(xc) or xc <= 0 or not np.all(np.isfinite(xs)):
        return res

    def cross(y, tgt):
        return 10 ** np.interp(-tgt, -y, np.log10(doses)) if y[0] >= tgt >= y[-1] else np.nan

    rel = xs / xc
    res.update(rel=rel, Emax=float(rel[-1]), IC50=cross(rel, 0.5))
    if xc <= 1.05:
        return res      # GR structurally undefined -- the control did not grow
    gr = 2.0 ** (np.log2(np.maximum(xs, 1e-9)) / np.log2(xc)) - 1.0
    res.update(gr=gr, GRmax=float(gr[-1]), GR50=cross(gr, 0.5))
    return res


if __name__ == "__main__":
    print("canonical model self-test (first-order O2, patient-derived k/D)\n")
    print(f"{'drug':<11}{'cond':<18}{'prolif%':>8}{'ctrl':>7}"
          f"{'GR50':>10}{'GRmax':>8}{'IC50':>10}{'Emax':>7}")
    for drug in DRUGS:
        for lab, R, V, f in [("150um lo-mx", 150., 5., .25),
                             ("150um hi-mx", 150., 20., .75),
                             ("500um lo-mx", 500., 5., .25)]:
            e = endpoints(drug, R, V, f)
            g = f"{e['GR50']:.3g}" if np.isfinite(e["GR50"]) else "undef"
            i = f"{e['IC50']:.3g}" if np.isfinite(e["IC50"]) else "n.r."
            print(f"{drug:<11}{lab:<18}{100*e['prolif_frac']:>7.0f}%{e['ctrl']:>7.2f}"
                  f"{g:>10}{e['GRmax']:>8.2f}{i:>10}{e['Emax']:>7.2f}")
        th, da = theta_Da(DRUGS[drug], 20.0, 0.75)
        print(f"{'':<11}{'20uL/75%: ':<18}Theta={th:.2f}  Da={da:.2f}\n")
