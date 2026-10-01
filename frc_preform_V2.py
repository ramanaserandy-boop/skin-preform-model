"""
frc_preform.py  (v2)
====================
Collagen fiber preform generator for the artificial skin project
(Ecoflex 00-31 matrix poured over a pre-built fiber network).

The preform is a SCALED-UP MACROSCOPIC ANALOG of the dermis. Only
dimensionless quantities are matched to the literature (orientation
distribution, two-family geometry, tortuosity / recruitment, volume
fraction, connectivity). Absolute lengths (mm) are fabrication choices.

Every parameter carries a provenance status, shown in the
"Parameter provenance" tab:
    VERIFIED            value and citation confirmed during the parameter review
    NEEDS VERIFICATION  from literature or partner table, not yet confirmed
    NEEDS VALUE         concept is published, numeric value still to extract
    DESIGN              fabrication choice, no literature value applies
    PROJECT SPEC        set by this project (e.g. 2.54 mm sample thickness)
    LIMITATION          known modelling simplification

Changes from v1 (kept as frc_preform_v1_original.py):
    - Inter-layer rotation (composite laminate convention) replaced by two
      fiber families at +/- theta around a mean direction (Langer line analog)
    - N identical layers replaced by papillary (thin, loose) + reticular
      (thick, two families) layers inside the real 2.54 mm thickness
    - Gaussian "variance"/"max" sliders (actually SDs) replaced by pi-periodic
      von Mises sampling set by circular SD, separately in-plane / out-of-plane
    - Unitless crimp amplitude/frequency replaced by a per-fiber tortuosity
      distribution (straightening strain) + crimp wavelength in mm
    - Fiber count replaced by fiber volume fraction; finite fiber length
    - Fixed random seed, physical units (mm) everywhere
    - Clipping artifact fixed (polyline split where it leaves the window)
    - Validation: nematic order parameter (same math as the Hough pipeline),
      achieved GOH kappa vs 0.1404 target, achieved volume fraction,
      crossing count / segment-length ratio
    - Design check: affine recruitment estimate of the J-curve
    - Export: fiber table (CSV) and 1:1 scale ply templates (PDF)

Run:  streamlit run frc_preform.py
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# PARAMETER PROVENANCE
# ----------------------------------------------------------------------
PROVENANCE = [
    {
        "Parameter": "Target GOH dispersion kappa (per reticular family)",
        "Default": "0.1404",
        "Status": "VERIFIED",
        "Source": "Ni Annaidh et al. 2012, Ann Biomed Eng, doi:10.1007/s10439-012-0542-3",
        "Note": "Human dermis, histology. 0 = aligned, 1/3 = isotropic. Used as a "
                "validation target (Validation tab), not as a direct input, because "
                "GOH kappa is rotationally symmetric and mixes in-plane and out-of-plane spread.",
    },
    {
        "Parameter": "In-plane spread sigma_ip",
        "Default": "23 deg",
        "Status": "NEEDS VERIFICATION",
        "Source": "Partner table cites 'Yu et al. 2015, Sci Rep 5:17635'. That article is "
                  "Bancelin et al. 2015 (mouse skin). Source of 23 deg unresolved.",
        "Note": "23 deg is consistent with b ~ 2.1, which is what Ni Annaidh's kappa = 0.1404 "
                "implies. Partner's b = 0.793 was NOT used: it implies ~40 deg and "
                "contradicts both other values.",
    },
    {
        "Parameter": "Out-of-plane spread sigma_op",
        "Default": "5 deg",
        "Status": "NEEDS VALUE",
        "Source": "Alberini et al. 2024, Sci Rep 14, doi:10.1038/s41598-024-51550-5 "
                  "(3D in/out-of-plane dispersion, human and mouse skin SHG)",
        "Note": "Literature: skin out-of-plane concentration is higher than in-plane, so "
                "sigma_op < sigma_ip. Extract the numeric value from Alberini 2024.",
    },
    {
        "Parameter": "Two-family half-angle +/- theta",
        "Default": "41 deg",
        "Status": "NEEDS VERIFICATION",
        "Source": "Ni Annaidh et al. 2012 (ABME and/or JMBBM 5:139)",
        "Note": "Two symmetric families is the standard dermis model. The 41 deg value "
                "has not been confirmed in the paper's results tables.",
    },
    {
        "Parameter": "Mean direction theta0",
        "Default": "0 deg (sample long axis)",
        "Status": "DESIGN",
        "Source": "Concept: Ni Annaidh et al. 2012, JMBBM 5:139 (fiber orientation follows Langer lines)",
        "Note": "Set to the angle between the tensile axis and the intended Langer line.",
    },
    {
        "Parameter": "Papillary vs reticular structure",
        "Default": "thin loose fibers on top, thick two-family bundles below",
        "Status": "VERIFIED (qualitative)",
        "Source": "Bancelin et al. 2015, Sci Rep 5:17635",
        "Note": "Thickness fraction, diameters and papillary isotropy are DESIGN choices.",
    },
    {
        "Parameter": "Sample thickness",
        "Default": "2.54 mm",
        "Status": "PROJECT SPEC",
        "Source": "Current Ecoflex 00-31 casting mold",
        "Note": "",
    },
    {
        "Parameter": "Straightening strain distribution (tortuosity - 1)",
        "Default": "mean 0.25, SD 0.10 (lognormal)",
        "Status": "DESIGN (calibrate)",
        "Source": "Mechanism: Comninou & Yannas 1976, J Biomech 9:427 (not re-checked this session). "
                  "Upper bound: human failure strain 54% +/- 17%, Ni Annaidh et al. 2012 JMBBM.",
        "Note": "Main lever for the J-curve. A SPREAD is required for a smooth J; equal crimp "
                "gives a sharp knee. Calibrate against MTS curves of real skin.",
    },
    {
        "Parameter": "Crimp wavelength",
        "Default": "1.0 mm",
        "Status": "DESIGN (scaled)",
        "Source": "Native crimp wavelength 10-200 um: 'Microcrimped Collagen Fiber-Elastin "
                  "Composites' (PMC3213053)",
        "Note": "Native scale is not reproducible by hand. Only tortuosity is matched; wavelength "
                "is chosen for fabrication. Crimp is planar (v1's helical 0.4 factor removed).",
    },
    {
        "Parameter": "Segment / fiber length ratio target",
        "Default": "~0.3",
        "Status": "NEEDS VERIFICATION",
        "Source": "Derived from partner values L_fiber 40 um, segment 10-15 um, attributed to "
                  "Witt et al. 2022, J Biomech Eng 144:041008, Table 3 (paper verified, table not)",
        "Note": "Mouse skin simulation (RVE), not tissue measurement. Only meaningful if "
                "crossings are physically bonded in the preform.",
    },
    {
        "Parameter": "Pre-strain during cure",
        "Default": "0",
        "Status": "DESIGN",
        "Source": "Concept: Alexander & Cook 1977, J Invest Dermatol 69:310 (not re-checked this session)",
        "Note": "Thread tension while the matrix cures. Reduces effective slack.",
    },
    {
        "Parameter": "Fiber volume fraction",
        "Default": "reticular 0.05, papillary 0.02",
        "Status": "DESIGN",
        "Source": "Woessner et al. 2021, Front Bioeng Biotechnol 9:642866",
        "Note": "Native dermis is far denser. Woessner shows organization and volume fraction "
                "depend on measurement scale, so state which scale you match.",
    },
    {
        "Parameter": "Fiber / matrix stiffness ratio",
        "Default": "1e4",
        "Status": "DESIGN (measure)",
        "Source": "Measure silk thread and Ecoflex 00-31 moduli on the MTS",
        "Note": "Only affects the J-curve estimate, not the geometry.",
    },
    {
        "Parameter": "Fiber kinematics in J-curve estimate",
        "Default": "affine",
        "Status": "LIMITATION",
        "Source": "Woessner et al. 2021 (affine overestimates realignment); Chandran & Barocas "
                  "2006, J Biomech Eng 128:259 (network reorientation creates a toe region)",
        "Note": "Screening estimate only. No fiber reorientation, sliding or failure modelled.",
    },
    {
        "Parameter": "Partner value DV = 0.55",
        "Default": "not used",
        "Status": "NEEDS VERIFICATION",
        "Source": "Woessner et al. 2021",
        "Note": "Directional variance has several definitions. Not used until the formula is known.",
    },
]

MAX_FIBERS_PER_PLY = 1500

# ----------------------------------------------------------------------
# MATH HELPERS
# ----------------------------------------------------------------------
_PSI = np.linspace(-np.pi, np.pi, 8001)


def concentration_from_sd(sd_deg):
    """
    Concentration a of a pi-periodic von Mises density p(phi) ~ exp(a cos 2phi)
    whose circular SD (for the axial angle phi) equals sd_deg.
    Doubled angle psi = 2 phi is von Mises(0, a); circular SD of phi is
    sqrt(-2 ln R) / 2 with R = I1(a)/I0(a), so R must equal exp(-2 sd^2).
    Returns np.inf for sd <= 0 and 0.0 when the spread is effectively isotropic.
    """
    if sd_deg <= 0:
        return np.inf
    target = np.exp(-2.0 * np.radians(sd_deg) ** 2)
    if target < 1e-3:
        return 0.0

    def mean_resultant(a):
        w = np.exp(a * (np.cos(_PSI) - 1.0))
        return float((w * np.cos(_PSI)).sum() / w.sum())

    lo, hi = 0.0, 3000.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_resultant(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sample_axial(rng, a, n):
    """Sample axial angles (rad, in (-pi/2, pi/2]) from exp(a cos 2phi)."""
    if np.isinf(a):
        return np.zeros(n)
    return rng.vonmises(0.0, a, n) / 2.0


# Tortuosity of y = A sin(2 pi x / lambda): tau = mean(sqrt(1 + k^2 cos^2 t)),
# k = 2 pi A / lambda. Precompute and invert by interpolation.
_K = np.linspace(0.0, 40.0, 4001)
_T = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
_TAU = np.sqrt(1.0 + (_K[:, None] * np.cos(_T[None, :])) ** 2).mean(axis=1)


def amplitude_from_tortuosity(tau, wavelength):
    k = np.interp(tau, _TAU, _K)
    return k * wavelength / (2 * np.pi)


def lognormal_strains(rng, mean, sd, n):
    """Positive, right-skewed straightening strains with given mean and SD."""
    if mean <= 0:
        return np.zeros(n)
    if sd <= 0:
        return np.full(n, mean)
    s2 = np.log(1.0 + (sd / mean) ** 2)
    return rng.lognormal(np.log(mean) - s2 / 2.0, np.sqrt(s2), n)


# ----------------------------------------------------------------------
# GEOMETRY GENERATION
# ----------------------------------------------------------------------
def build_plies(p):
    """Ply stack: reticular plies at the bottom, papillary ply on top."""
    T = p["T"]
    t_pap = p["f_pap"] * T
    t_ret = T - t_pap
    plies = []
    n_r = int(p["n_ret"])
    for k in range(n_r):
        z0 = k * t_ret / n_r
        if p["arrangement"] == "interwoven":
            fams = ["+", "-"]
        else:
            fams = ["+"] if k % 2 == 0 else ["-"]
        plies.append(dict(name=f"Reticular ply {k + 1}", kind="reticular",
                          z0=z0, z1=z0 + t_ret / n_r, families=fams))
    if t_pap > 0:
        plies.append(dict(name="Papillary ply", kind="papillary",
                          z0=t_ret, z1=T, families=["p"]))
    return plies


def generate_preform(p):
    """
    Returns (fibers DataFrame, runs list, plies list, warnings list).
    runs[i] is a list of (n, 3) arrays: the in-window pieces of fiber i.
    """
    rng = np.random.default_rng(int(p["seed"]))
    W, L = p["W"], p["L"]
    half_ext = W / 2 + L / 2          # centers placed in an extended window
    area_ext = (2 * half_ext) ** 2

    a_ip = concentration_from_sd(p["sd_ip"])
    a_op = concentration_from_sd(p["sd_op"])
    a_pap = 0.0 if p["pap_iso"] else concentration_from_sd(p["pap_sd"])
    th0 = np.radians(p["theta0"])
    half = np.radians(p["half_angle"])

    eps_mean = p["eps_mean"]
    mean_tau = 1.0 + eps_mean
    rows, runs, warnings = [], [], []
    plies = build_plies(p)

    for pi_, ply in enumerate(plies):
        is_ret = ply["kind"] == "reticular"
        d = p["d_ret"] if is_ret else p["d_pap"]
        vf = p["vf_ret"] if is_ret else p["vf_pap"]
        t_ply = ply["z1"] - ply["z0"]
        fiber_vol = np.pi * d ** 2 / 4 * L * mean_tau
        n = int(round(vf * area_ext * t_ply / fiber_vol)) if fiber_vol > 0 else 0
        if n > MAX_FIBERS_PER_PLY:
            warnings.append(f"{ply['name']}: {n} fibers requested, capped at "
                            f"{MAX_FIBERS_PER_PLY}. Achieved volume fraction will be lower.")
            n = MAX_FIBERS_PER_PLY
        if n == 0:
            continue

        # family assignment
        if is_ret and len(ply["families"]) == 2:
            fam = np.where(rng.random(n) < p["frac_plus"], "+", "-")
        else:
            fam = np.full(n, ply["families"][0])

        # orientation
        if is_ret:
            mean_dir = th0 + np.where(fam == "+", half, -half)
            phi = mean_dir + sample_axial(rng, a_ip, n)
        else:
            mean_dir = np.full(n, th0)
            phi = th0 + (rng.uniform(-np.pi / 2, np.pi / 2, n) if p["pap_iso"]
                         else sample_axial(rng, a_pap, n))
        elev = sample_axial(rng, a_op, n)

        # crimp / recruitment
        eps = lognormal_strains(rng, eps_mean, p["eps_sd"], n)
        tau = 1.0 + eps
        tau_eff = np.maximum(tau / (1.0 + p["prestrain"]), 1.0)  # geometry after pre-tension
        amp = amplitude_from_tortuosity(tau_eff, p["crimp_wl"])
        phase = rng.uniform(0, 2 * np.pi, n)

        cx = rng.uniform(-half_ext, half_ext, n)
        cy = rng.uniform(-half_ext, half_ext, n)
        cz = rng.uniform(ply["z0"] + d / 2, max(ply["z1"] - d / 2, ply["z0"] + d / 2), n)

        n_pts = int(np.clip(L / p["crimp_wl"] * 24, 40, 400))
        s = np.linspace(-L / 2, L / 2, n_pts)

        for i in range(n):
            dvec = np.array([np.cos(elev[i]) * np.cos(phi[i]),
                             np.cos(elev[i]) * np.sin(phi[i]),
                             np.sin(elev[i])])
            nvec = np.array([-np.sin(phi[i]), np.cos(phi[i]), 0.0])
            u = amp[i] * np.sin(2 * np.pi * s / p["crimp_wl"] + phase[i])
            P = (np.array([cx[i], cy[i], cz[i]])[None, :]
                 + s[:, None] * dvec[None, :] + u[:, None] * nvec[None, :])

            inside = ((np.abs(P[:, 0]) <= W / 2) & (np.abs(P[:, 1]) <= W / 2)
                      & (P[:, 2] >= 0) & (P[:, 2] <= p["T"]))
            pieces = []
            if inside.any():
                edges = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
                for a0, a1 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                    if a1 - a0 >= 2:
                        pieces.append(P[a0:a1])
            arc_in = sum(np.linalg.norm(np.diff(q, axis=0), axis=1).sum() for q in pieces)

            rows.append(dict(
                fiber_id=len(rows), ply=pi_, ply_name=ply["name"], layer=ply["kind"],
                family=fam[i], cx_mm=cx[i], cy_mm=cy[i], cz_mm=cz[i],
                inplane_deg=np.degrees(phi[i]), elevation_deg=np.degrees(elev[i]),
                family_mean_deg=np.degrees(mean_dir[i]),
                chord_len_mm=L, tortuosity=tau[i], tortuosity_after_prestrain=tau_eff[i],
                crimp_amp_mm=amp[i], crimp_wavelength_mm=p["crimp_wl"],
                crimp_phase_rad=phase[i], diameter_mm=d, arc_in_window_mm=arc_in,
                dx=dvec[0], dy=dvec[1], dz=dvec[2],
            ))
            runs.append(pieces)

    return pd.DataFrame(rows), runs, plies, warnings


# ----------------------------------------------------------------------
# VALIDATION METRICS
# ----------------------------------------------------------------------
def nematic_order(runs_subset):
    """
    Length-weighted 2D nematic order parameter of the top-view projection,
    same definition as the Hough pipeline: S = |sum l e^{2i phi}| / sum l.
    Returns (S, dominant angle in deg [0, 180)).
    """
    num, den = 0j, 0.0
    for pieces in runs_subset:
        for q in pieces:
            v = np.diff(q[:, :2], axis=0)
            l = np.hypot(v[:, 0], v[:, 1])
            ang = np.arctan2(v[:, 1], v[:, 0])
            num += (l * np.exp(2j * ang)).sum()
            den += l.sum()
    if den == 0:
        return np.nan, np.nan
    return abs(num) / den, (np.degrees(np.angle(num)) / 2) % 180


def goh_kappa_from_b(b):
    """GOH kappa = <sin^2 Theta>/2 for the 3D density rho ~ exp(2b cos^2 Theta)."""
    th = np.linspace(0.0, np.pi, 20001)
    w = np.exp(2 * b * (np.cos(th) ** 2 - 1.0)) * np.sin(th)
    return float((w * np.sin(th) ** 2).sum() / w.sum() / 2)


def inplane_concentration(dev_rad):
    """Fit b of a planar pi-periodic von Mises to in-plane deviations (rad)."""
    if len(dev_rad) < 2:
        return np.nan
    R = abs(np.mean(np.exp(2j * dev_rad)))
    if R <= 1e-6:
        return 0.0
    sd_deg = np.degrees(np.sqrt(-2 * np.log(min(R, 1 - 1e-12))) / 2)
    return concentration_from_sd(sd_deg)


def achieved_goh_kappa(df_family):
    """
    Two estimates of GOH kappa for one fiber family.
    'inplane': fit b to the in-plane angles around the family mean, then
               kappa(b) from the GOH formula. This mirrors a histology-based
               workflow (2D sections) and is the comparison used against 0.1404.
               ASSUMPTION (NEEDS VERIFICATION): that Ni Annaidh 2012 derived
               kappa this way; check their Methods.
    '3d':      <sin^2 Theta>/2 of the actual 3D chords (true GOH definition).
    """
    if df_family.empty:
        return np.nan, np.nan
    m = np.radians(df_family["family_mean_deg"].to_numpy())
    phi = np.radians(df_family["inplane_deg"].to_numpy())
    dev = np.angle(np.exp(2j * (phi - m))) / 2
    b = inplane_concentration(dev)
    k_inplane = goh_kappa_from_b(b) if np.isfinite(b) else 0.0
    cos_t = np.abs(df_family["dx"] * np.cos(m) + df_family["dy"] * np.sin(m))
    k_3d = float(np.mean(1 - cos_t ** 2) / 2)
    return k_inplane, k_3d


def contact_counts(df):
    """
    Contacts per fiber: chords crossing in XY whose z separation at the
    crossing is below the mean of the two diameters.
    """
    n = len(df)
    if n < 2:
        return np.zeros(n, dtype=int)
    C = df[["cx_mm", "cy_mm", "cz_mm"]].to_numpy()
    D = df[["dx", "dy", "dz"]].to_numpy()
    Lh = df["chord_len_mm"].to_numpy()[:, None] / 2
    P1, P2 = C - Lh * D, C + Lh * D
    r = P2 - P1
    diam = df["diameter_mm"].to_numpy()
    counts = np.zeros(n, dtype=int)
    idx_all = np.arange(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        for s in range(0, n, 256):
            sl = slice(s, min(s + 256, n))
            ri, rj = r[sl, None, :2], r[None, :, :2]
            qp = P1[None, :, :2] - P1[sl, None, :2]
            den = ri[..., 0] * rj[..., 1] - ri[..., 1] * rj[..., 0]
            t = (qp[..., 0] * rj[..., 1] - qp[..., 1] * rj[..., 0]) / den
            u = (qp[..., 0] * ri[..., 1] - qp[..., 1] * ri[..., 0]) / den
            ok = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
            zi = P1[sl, 2][:, None] + t * r[sl, 2][:, None]
            zj = P1[:, 2][None, :] + u * r[:, 2][None, :]
            ok &= np.abs(zi - zj) <= (diam[sl][:, None] + diam[None, :]) / 2
            ok &= idx_all[None, :] != idx_all[sl][:, None]
            counts[sl] = ok.sum(axis=1)
    return counts


# ----------------------------------------------------------------------
# DESIGN CHECK: AFFINE RECRUITMENT J-CURVE ESTIMATE
# ----------------------------------------------------------------------
def jcurve_estimate(df, p, n_steps=250):
    """
    Uniaxial stretch along x, incompressible affine kinematics
    F = diag(lam, lam^-1/2, lam^-1/2). A fiber engages when its chord stretch
    exceeds its tortuosity (after pre-strain). Linear elastic fiber after
    engagement, no compression, no reorientation, no failure.
    Normalized Cauchy stress sigma_xx / E_matrix:
        matrix: neo-Hookean (lam^2 - 1/lam) / 3
        fibers: sum_i w_i (E_f/E_m) e_i (lam dx_i)^2 / lam_f_i^2
    with w_i = in-window fiber volume / sample volume.
    """
    strain = np.linspace(0.0, p["strain_max"], n_steps)
    lam = 1.0 + strain
    df = df[df["arc_in_window_mm"] > 0]
    sample_vol = p["W"] ** 2 * p["T"]
    w = (df["arc_in_window_mm"] * np.pi * df["diameter_mm"] ** 2 / 4).to_numpy() / sample_vol
    dx, dy, dz = df["dx"].to_numpy(), df["dy"].to_numpy(), df["dz"].to_numpy()
    lam_eng = df["tortuosity"].to_numpy() / (1.0 + p["prestrain"])

    lf = np.sqrt(lam[:, None] ** 2 * dx ** 2 + (dy ** 2 + dz ** 2) / lam[:, None])
    e = np.clip(lf / lam_eng - 1.0, 0.0, None)
    sig_f = (w * p["stiff_ratio"] * e * (lam[:, None] * dx) ** 2 / lf ** 2).sum(axis=1)
    sig_m = (lam ** 2 - 1.0 / lam) / 3.0
    recruited = ((e > 0) * w).sum(axis=1) / max(w.sum(), 1e-12)
    return pd.DataFrame(dict(strain=strain, stress=sig_m + sig_f,
                             matrix=sig_m, fibers=sig_f, recruited=recruited))


def phase_bounds(jc, r1, r2):
    """
    Toe ends when recruitment passes r1, heel ends at r2 (DESIGN definitions).
    Returns None for a boundary not reached within the applied strain range.
    """
    s = jc["strain"].to_numpy()
    rec = jc["recruited"].to_numpy()
    toe_end = float(s[np.argmax(rec >= r1)]) if (rec >= r1).any() else None
    heel_end = float(s[np.argmax(rec >= r2)]) if (rec >= r2).any() else None
    return toe_end, heel_end


# ----------------------------------------------------------------------
# EXPORT: 1:1 PLY TEMPLATES
# ----------------------------------------------------------------------
def ply_templates_pdf(df, runs, plies, W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    margin = 25.0                       # mm
    size = W + 2 * margin
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for k, ply in enumerate(plies):
            fig = plt.figure(figsize=(size / 25.4, size / 25.4))
            ax = fig.add_axes([margin / size, margin / size, W / size, W / size])
            ax.set_xlim(-W / 2, W / 2)
            ax.set_ylim(-W / 2, W / 2)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            sub = df[df["ply"] == k]
            for i in sub.index:
                lw = sub.at[i, "diameter_mm"] * 72 / 25.4
                for q in runs[i]:
                    ax.plot(q[:, 0], q[:, 1], color="black", lw=lw, solid_capstyle="butt")
            ax.plot([-W / 2, -W / 2 + 5], [-W / 2 - 4, -W / 2 - 4], color="black", lw=1.5,
                    clip_on=False)
            ax.text(-W / 2, -W / 2 - 6, "5 mm\nprint at 100%, no scaling", fontsize=6,
                    va="top", clip_on=False)
            ax.text(-W / 2, W / 2 + 3,
                    f"{ply['name']}, z = {ply['z0']:.2f} to {ply['z1']:.2f} mm\n"
                    f"{len(sub)} fibers, x = sample long axis",
                    fontsize=6, va="bottom", clip_on=False)
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()


# ----------------------------------------------------------------------
# STREAMLIT APP
# ----------------------------------------------------------------------
PLY_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf",
              "#7f7f7f", "#bcbd22", "#e377c2"]
FAMILY_SHADE = {"+": 1.0, "-": 0.6, "p": 1.0}


@st.cache_data(show_spinner="Generating fiber network")
def cached_generate(p_items):
    p = dict(p_items)
    df, runs, plies, warnings = generate_preform(p)
    counts = contact_counts(df[df["arc_in_window_mm"] > 0]) if len(df) else np.array([])
    return df, runs, plies, warnings, counts


def polyline_trace(pieces_list, z_scale, color, name, mode3d=True):
    xs, ys, zs = [], [], []
    for pieces in pieces_list:
        for q in pieces:
            xs.extend(q[:, 0].tolist() + [None])
            ys.extend(q[:, 1].tolist() + [None])
            zs.extend((q[:, 2] * z_scale).tolist() + [None])
    if mode3d:
        return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name,
                            line=dict(color=color, width=3), hoverinfo="skip")
    return go.Scatter(x=xs, y=ys, mode="lines", name=name,
                      line=dict(color=color, width=1.5), hoverinfo="skip")


def main():
    st.set_page_config(layout="wide", page_title="Collagen Preform Generator v2")
    st.title("Collagen fiber preform generator")
    st.caption("Scaled-up macroscopic analog of the dermis. All lengths in mm. "
               "Check the Parameter provenance tab before using any value in a paper.")

    sb = st.sidebar
    sb.header("Reproducibility")
    seed = sb.number_input("Random seed", 0, 1_000_000, 42,
                           help="Same seed + same parameters = identical layout.")

    sb.header("Sample")
    W = sb.slider("Window size W (mm, square)", 5.0, 50.0, 20.0, 1.0)
    T = sb.number_input("Thickness T (mm)", 0.5, 10.0, 2.54, 0.01, help="PROJECT SPEC")

    sb.header("Layers")
    f_pap = sb.slider("Papillary fraction of thickness", 0.0, 0.5, 0.2, 0.05, help="DESIGN")
    n_ret = sb.number_input("Reticular plies", 1, 8, 3, help="DESIGN: fabrication plies")
    arrangement = sb.radio("Reticular family arrangement",
                           ["interwoven", "alternating"],
                           format_func=lambda s: {"interwoven": "Both families in every ply",
                                                  "alternating": "Alternate +theta / -theta plies"}[s])

    sb.header("Reticular orientation")
    theta0 = sb.slider("Mean direction theta0 (deg from long axis)", -90, 90, 0, help="DESIGN")
    half_angle = sb.slider("Family half-angle +/- theta (deg)", 0, 90, 41,
                           help="NEEDS VERIFICATION (Ni Annaidh 2012)")
    frac_plus = sb.slider("Fraction in +theta family", 0.0, 1.0, 0.5, 0.05, help="DESIGN")
    sd_ip = sb.slider("In-plane spread sigma_ip (circular SD, deg)", 2.0, 60.0, 23.0, 0.5,
                      help="NEEDS VERIFICATION. Compare achieved GOH kappa to 0.1404 in Validation.")
    sd_op = sb.slider("Out-of-plane spread sigma_op (circular SD, deg)", 0.0, 30.0, 5.0, 0.5,
                      help="NEEDS VALUE (Alberini 2024)")

    sb.header("Papillary orientation")
    pap_iso = sb.checkbox("Isotropic in-plane", True, help="DESIGN")
    pap_sd = sb.slider("Papillary in-plane spread (deg)", 2.0, 60.0, 40.0, 0.5, disabled=pap_iso)

    sb.header("Fibers")
    L = sb.slider("Fiber chord length (mm)", 1.0, 40.0, 6.0, 0.5, help="DESIGN")
    d_ret = sb.number_input("Reticular fiber diameter (mm)", 0.01, 1.0, 0.15, 0.01, help="DESIGN")
    d_pap = sb.number_input("Papillary fiber diameter (mm)", 0.01, 1.0, 0.05, 0.01,
                            help="DESIGN (current silk: 0.05 mm)")
    vf_ret = sb.slider("Reticular volume fraction", 0.0, 0.3, 0.05, 0.005, help="DESIGN")
    vf_pap = sb.slider("Papillary volume fraction", 0.0, 0.3, 0.02, 0.005, help="DESIGN")

    sb.header("Crimp and recruitment")
    eps_mean = sb.slider("Mean straightening strain (tortuosity - 1)", 0.0, 1.0, 0.25, 0.01,
                         help="DESIGN, calibrate to the real skin J-curve")
    eps_sd = sb.slider("Straightening strain SD", 0.0, 0.5, 0.10, 0.01,
                       help="Spread = gradual recruitment = smooth J")
    crimp_wl = sb.slider("Crimp wavelength (mm)", 0.2, 5.0, 1.0, 0.1, help="DESIGN (scaled)")
    prestrain = sb.slider("Pre-strain during cure", 0.0, 0.3, 0.0, 0.01, help="DESIGN")

    sb.header("Connectivity")
    bonded = sb.checkbox("Crossings are bonded in the physical preform", False,
                         help="Segment-length metrics only apply if crossings are glued/knotted.")

    sb.header("J-curve design check")
    stiff_ratio = sb.number_input("Fiber / matrix stiffness ratio", 1.0, 1e7, 1e4, format="%.0f",
                                  help="DESIGN: measure both on the MTS")
    strain_max = sb.slider("Max applied strain", 0.1, 1.5, 0.8, 0.05)
    r1 = sb.slider("Toe ends at recruited fraction", 0.01, 0.5, 0.05, 0.01, help="DESIGN definition")
    r2 = sb.slider("Heel ends at recruited fraction", 0.5, 0.99, 0.90, 0.01, help="DESIGN definition")

    sb.header("Display")
    z_scale = sb.slider("Z exaggeration (3D view only)", 1.0, 10.0, 3.0, 0.5)

    p = dict(seed=seed, W=W, T=T, f_pap=f_pap, n_ret=n_ret, arrangement=arrangement,
             theta0=theta0, half_angle=half_angle, frac_plus=frac_plus, sd_ip=sd_ip,
             sd_op=sd_op, pap_iso=pap_iso, pap_sd=pap_sd, L=L, d_ret=d_ret, d_pap=d_pap,
             vf_ret=vf_ret, vf_pap=vf_pap, eps_mean=eps_mean, eps_sd=eps_sd,
             crimp_wl=crimp_wl, prestrain=prestrain)
    gen_keys = tuple(sorted(p.items()))
    df, runs, plies, warnings, counts = cached_generate(gen_keys)
    p.update(stiff_ratio=stiff_ratio, strain_max=strain_max)

    for wmsg in warnings:
        st.warning(wmsg)
    if df.empty:
        st.info("No fibers generated. Increase a volume fraction or the window size.")
        return

    tabs = st.tabs(["3D preform", "Ply templates", "J-curve design check",
                    "Validation", "Parameter provenance", "Export"])

    # 3D view
    with tabs[0]:
        fig = go.Figure()
        for k, ply in enumerate(plies):
            sub = df[df["ply"] == k]
            for fam in sub["family"].unique():
                ids = sub.index[sub["family"] == fam]
                color = PLY_COLORS[k % len(PLY_COLORS)] if fam != "-" else "#F2A900"
                if ply["kind"] == "papillary":
                    color = "#d62728"
                label = f"{ply['name']} ({'+theta' if fam == '+' else '-theta' if fam == '-' else 'random'})"
                fig.add_trace(polyline_trace([runs[i] for i in ids], z_scale, color, label))
        fig.update_layout(
            scene=dict(xaxis=dict(range=[-W / 2, W / 2], title="x, long axis (mm)"),
                       yaxis=dict(range=[-W / 2, W / 2], title="y (mm)"),
                       zaxis=dict(range=[0, T * z_scale], title=f"z (mm x{z_scale:g})"),
                       aspectmode="manual",
                       aspectratio=dict(x=1, y=1, z=max(T * z_scale / W, 0.05)),
                       bgcolor="rgb(20,20,20)"),
            margin=dict(l=0, r=0, b=0, t=0), height=700, legend=dict(itemsizing="constant"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Blue/green/purple: +theta family per ply. Gold: -theta family. Red: papillary.")

    # Ply templates
    with tabs[1]:
        names = [pl["name"] for pl in plies]
        pick = st.selectbox("Ply", range(len(plies)), format_func=lambda k: names[k])
        sub = df[df["ply"] == pick]
        fig2 = go.Figure()
        for fam in sub["family"].unique():
            ids = sub.index[sub["family"] == fam]
            fig2.add_trace(polyline_trace([runs[i] for i in ids], 1.0,
                                          "#F2A900" if fam == "-" else "#1f77b4",
                                          {"+": "+theta", "-": "-theta", "p": "random"}[fam],
                                          mode3d=False))
        fig2.update_layout(xaxis=dict(range=[-W / 2, W / 2], title="x, long axis (mm)",
                                      constrain="domain"),
                           yaxis=dict(range=[-W / 2, W / 2], title="y (mm)",
                                      scaleanchor="x", scaleratio=1),
                           height=650, margin=dict(l=40, r=10, t=10, b=40))
        st.plotly_chart(fig2, width="stretch")
        ply = plies[pick]
        st.caption(f"z = {ply['z0']:.2f} to {ply['z1']:.2f} mm, {len(sub)} fibers. "
                   "1:1 printable templates for every ply are in the Export tab.")

    # J-curve
    with tabs[2]:
        jc = jcurve_estimate(df, p)
        toe_end, heel_end = phase_bounds(jc, r1, r2)
        st.warning("Screening estimate only. Affine kinematics overestimate fiber realignment "
                   "(Woessner et al. 2021). No reorientation, sliding or failure is modelled. "
                   "Use it to compare designs, then validate on the MTS.")
        fig3 = go.Figure()
        t_end = toe_end if toe_end is not None else strain_max
        h_end = heel_end if heel_end is not None else strain_max
        fig3.add_vrect(x0=0, x1=t_end, fillcolor="#2ca02c", opacity=0.08, line_width=0,
                       annotation_text="toe", annotation_position="top left")
        if toe_end is not None:
            fig3.add_vrect(x0=t_end, x1=h_end, fillcolor="#F2A900", opacity=0.10, line_width=0,
                           annotation_text="heel", annotation_position="top left")
        if heel_end is not None:
            fig3.add_vrect(x0=h_end, x1=strain_max, fillcolor="#d62728", opacity=0.06,
                           line_width=0, annotation_text="linear", annotation_position="top left")
        fig3.add_trace(go.Scatter(x=jc["strain"], y=jc["stress"], name="total stress / E_matrix"))
        fig3.add_trace(go.Scatter(x=jc["strain"], y=jc["matrix"], name="matrix only",
                                  line=dict(dash="dot")))
        fig3.add_trace(go.Scatter(x=jc["strain"], y=jc["recruited"], name="recruited fraction",
                                  yaxis="y2", line=dict(dash="dash")))
        fig3.update_layout(xaxis_title="Engineering strain along x",
                           yaxis_title="Normalized Cauchy stress (sigma / E_matrix)",
                           yaxis2=dict(title="Recruited fraction", overlaying="y", side="right",
                                       range=[0, 1.05]),
                           height=550, margin=dict(t=30))
        st.plotly_chart(fig3, width="stretch")
        c1, c2, c3 = st.columns(3)
        c1.metric("Toe ends at strain", f"{toe_end:.3f}" if toe_end is not None else "not reached")
        c2.metric("Heel ends at strain", f"{heel_end:.3f}" if heel_end is not None else "not reached")
        c3.metric("Recruited at max strain", f"{jc['recruited'].iloc[-1]:.2f}")
        if heel_end is None:
            st.info("Recruitment never reaches the heel threshold. Under uniaxial stretch, fibers "
                    "at large angles to x are shortened by lateral contraction and engage late or "
                    "never. Lower the half-angle, raise max strain, or lower the threshold.")
        st.caption("Reference: human back skin failure strain 54% +/- 17% "
                   "(Ni Annaidh et al. 2012, JMBBM). The toe and heel should end well before that.")

    # Validation
    with tabs[3]:
        in_win = df["arc_in_window_mm"] > 0
        st.subheader("Orientation (top-view projection)")
        st.caption("Length-weighted nematic order parameter, same definition as the Hough "
                   "pipeline. Run the Hough script on a photo of the built ply and compare.")
        rows = []
        for k, ply in enumerate(plies):
            ids = df.index[(df["ply"] == k) & in_win]
            S, ang = nematic_order([runs[i] for i in ids])
            vf_ach = (df.loc[ids, "arc_in_window_mm"] * np.pi * df.loc[ids, "diameter_mm"] ** 2 / 4
                      ).sum() / (W * W * (ply["z1"] - ply["z0"]))
            target_vf = vf_ret if ply["kind"] == "reticular" else vf_pap
            rows.append({"Ply": ply["name"], "Fibers in window": len(ids),
                         "Order parameter S": round(S, 3),
                         "Dominant angle (deg)": round(ang, 1),
                         "Volume fraction (target)": target_vf,
                         "Volume fraction (achieved)": round(vf_ach, 4)})
        S_all, ang_all = nematic_order([runs[i] for i in df.index[in_win]])
        rows.append({"Ply": "All plies", "Fibers in window": int(in_win.sum()),
                     "Order parameter S": round(S_all, 3),
                     "Dominant angle (deg)": round(ang_all, 1),
                     "Volume fraction (target)": None, "Volume fraction (achieved)": None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.subheader("Dispersion vs literature")
        ret = df[(df["layer"] == "reticular") & in_win]
        kp_ip, kp_3d = achieved_goh_kappa(ret[ret["family"] == "+"])
        km_ip, km_3d = achieved_goh_kappa(ret[ret["family"] == "-"])
        c1, c2, c3 = st.columns(3)
        c1.metric("+theta family, kappa (in-plane fit)", f"{kp_ip:.4f}",
                  f"{kp_ip - 0.1404:+.4f} vs 0.1404", delta_color="off")
        c2.metric("-theta family, kappa (in-plane fit)", f"{km_ip:.4f}",
                  f"{km_ip - 0.1404:+.4f} vs 0.1404", delta_color="off")
        c3.metric("Literature target", "0.1404", "Ni Annaidh 2012, human dermis",
                  delta_color="off")
        st.caption(f"True 3D <sin^2>/2 of the generated chords: +theta {kp_3d:.4f}, "
                   f"-theta {km_3d:.4f}. These differ from the in-plane-fit values because the "
                   "GOH formula assumes a rotationally symmetric 3D spread, while skin (and this "
                   "generator) is much tighter out of plane. The in-plane fit is the comparison "
                   "that matches histology-based measurement; confirm Ni Annaidh's exact method.")
        st.caption("Note: two families at +/- theta partly cancel in the 2D order parameter. "
                   "At 41 deg, S can be near 0 even for a highly structured network, so S alone "
                   "cannot tell a two-family dermis from a random one. Check the angle histogram "
                   "for two peaks.")

        st.subheader("Connectivity")
        if bonded:
            k_mean = counts.mean() if len(counts) else np.nan
            ratio = float(np.mean(1.0 / (counts + 1))) if len(counts) else np.nan
            c1, c2 = st.columns(2)
            c1.metric("Mean contacts per fiber", f"{k_mean:.2f}")
            c2.metric("Mean segment / fiber length", f"{ratio:.3f}", f"{ratio - 0.3:+.3f} vs ~0.3",
                      delta_color="off")
            st.caption("Target ~0.3 is NEEDS VERIFICATION (derived from partner's Witt 2022 values, "
                       "a mouse skin simulation). Contacts use straight chords.")
        else:
            st.info("Crossings are not bonded, so the network cannot transmit load fiber to fiber "
                    "and connectivity targets do not apply. Enable the checkbox if you plan to "
                    "glue or knot crossings.")

    # Provenance
    with tabs[4]:
        st.dataframe(pd.DataFrame(PROVENANCE), hide_index=True, width="stretch", height=620)

    # Export
    with tabs[5]:
        export = df.drop(columns=["dx", "dy", "dz"]).copy()
        export.insert(0, "seed", seed)
        st.download_button("Download fiber table (CSV)", export.to_csv(index=False).encode(),
                           file_name=f"preform_fibers_seed{seed}.csv", mime="text/csv")
        if st.button("Build 1:1 ply templates (PDF)"):
            st.session_state["pdf"] = ply_templates_pdf(df, runs, plies, W)
        if "pdf" in st.session_state:
            st.download_button("Download ply templates (PDF)", st.session_state["pdf"],
                               file_name=f"preform_templates_seed{seed}.pdf",
                               mime="application/pdf")
        st.caption("One page per ply, line width = fiber diameter, 5 mm scale bar. "
                   "Print at 100% and check the scale bar with a ruler before laying fibers.")
        st.json({k: (v if not isinstance(v, np.generic) else v.item()) for k, v in p.items()})


if __name__ == "__main__":
    main()
