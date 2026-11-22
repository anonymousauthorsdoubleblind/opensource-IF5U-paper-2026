"""Rebuild every data table of the 5-fluorouracil test from the released files.

Reads   data/            (run logs, structures, Hessian force files, IR tables, MANIFEST.csv)
Writes  tables/*.csv     one CSV per table shown on tables.html

Usage (from the repository root, i.e. the folder that holds data/ and code/):
    pip install numpy
    python code/analysis/make_tables.py

Nothing here is fitted or tuned. Every number is read from a released file or
computed from one with the formula named in the "source" column. The only
numbers typed in by hand are marked "from the paper" (the starting energies of
Runs 1 to 3, whose start files are not stored) and the static tables at the end.
"""
import csv
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)

SCALE_SOURCE = "in-sample fit on the same 25 bands"


def write(name, header, rows):
    with open(OUT / name, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    print(f"{name:34s} {len(rows):4d} rows")


def jload(p):
    with open(p) as fh:
        return json.load(fh)


def read_xyz(p):
    lines = Path(p).read_text().splitlines()
    n = int(lines[0])
    el, xyz = [], []
    for ln in lines[2:2 + n]:
        t = ln.split()
        el.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    return el, np.array(xyz)


def kabsch(P, Q):
    """Align P onto Q (rigid, no reflection). Returns aligned P."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    return Pc @ R + Q.mean(0)


def rmsd(P, Q):
    A = kabsch(P, Q)
    return float(np.sqrt(((A - Q) ** 2).sum(1).mean()))


def bond(x, i, j):
    return float(np.linalg.norm(x[i] - x[j]))


def angle(x, i, j, k):
    a, b = x[i] - x[j], x[k] - x[j]
    return float(np.degrees(np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))))


# ---------------------------------------------------------------- runs
RUN_DIRS = sorted(glob.glob(str(DATA / "module1" / "run*")))
RUN_NAMES = ["Run 1", "Run 2", "Run 3"]
RUN_START = ["standard start", "bent start", "very bent start"]
PAPER_START_EV = ["0.11", "3.11", "6.38"]  # from the paper; start files are not stored
summ = [jload(d + "/summary.json") for d in RUN_DIRS]
meta = [jload(d + "/run_meta.json") for d in RUN_DIRS]
steps = [jload(d + "/steps.json") for d in RUN_DIRS]
finals = [read_xyz(d + "/final.xyz") for d in RUN_DIRS]
el = finals[0][0]
X = [f[1] for f in finals]
m2el, m2x = read_xyz(DATA / "module2_T1" / "opt_final.xyz")

rm = [0.0] + [rmsd(X[k], X[0]) for k in (1, 2)]
dE = [(s["final_energy_eV"] - summ[0]["final_energy_eV"]) * 1000 for s in summ]

rows = []
def row(metric, vals, unit, src):
    rows.append([metric] + list(vals) + [unit, src])

row("Starting structure", RUN_START, "", "run folder name")
row("Start above final energy (from the paper)", PAPER_START_EV, "eV", "paper; start files are not stored")
row("Converged", [s["converged"] for s in summ], "", "summary.json: converged")
row("Final energy", [f"{s['final_energy_eV']:.6f}" for s in summ], "eV", "summary.json: final_energy_eV")
row("Final energy minus Run 1", [f"{v:+.2f}" for v in dE], "meV", "computed from the line above")
row("Final structure vs Run 1 (RMSD after alignment)", [f"{v:.4f}" for v in rm], "A", "computed from final.xyz, Kabsch alignment, 12 atoms")
row("Gradient evaluations in the optimization", [s["gradients_optimization"] for s in summ], "", "summary.json: gradients_optimization")
row("Gradient evaluations for checks", [s["gradients_evaluation"] for s in summ], "", "summary.json: gradients_evaluation")
row("Gradient evaluations, total", [s["gradients_total"] for s in summ], "", "summary.json: gradients_total")
row("Optimizer steps", [s["steps_total"] for s in summ], "", "summary.json: steps_total")
row("Passes with steps", [s["passes_with_steps"] for s in summ], "", "summary.json: passes_with_steps")
row("Gradient budget", [m["gradient_budget"] for m in meta], "", "run_meta.json: gradient_budget")
row("Starting step cap", [m["schedule"][0]["maxstep"] for m in meta], "A", "run_meta.json: schedule[0].maxstep")
row("Final force (Module 1 units)", [f"{s['final_fmax']:.4f}" for s in summ], "Module 1 units", "summary.json: final_fmax (about 1.89 times smaller than true eV/A)")
row("Force target (Module 1 units)", [m["endpoint_fmax"] for m in meta], "Module 1 units", "run_meta.json: endpoint_fmax")
row("Optimization time", [s["minutes_optimization"] for s in summ], "min", "summary.json: minutes_optimization")
row("Wall time, total", [s["minutes_total_wall"] for s in summ], "min", "summary.json: minutes_total_wall")
row("Mean time per gradient", [s["mean_seconds_per_gradient"] if isinstance(s["mean_seconds_per_gradient"], (int, float)) else json.dumps(s["mean_seconds_per_gradient"]) for s in summ], "s", "summary.json: mean_seconds_per_gradient")
row("Bond graph intact at the end", [s["final_structure"]["bond_graph_ok"] for s in summ], "", "summary.json: final_structure.bond_graph_ok")
row("Ring out-of-plane RMS at the end", [s["final_structure"]["plane_rms_A"] for s in summ], "A", "summary.json: final_structure.plane_rms_A")
row("Smallest curvature at the end", [min(s["curvature_kappa_eV_A2"]) for s in summ], "eV/A^2", "summary.json: curvature_kappa_eV_A2 (all positive)")
row("Any negative curvature", [s["curvature_any_negative"] for s in summ], "", "summary.json: curvature_any_negative")
row("Switched to another energy valley", [s["switched_basin"] for s in summ], "", "summary.json: switched_basin")
row("SCF steps that did not converge", [s["scf_not_converged_steps"] for s in summ], "", "summary.json: scf_not_converged_steps")
write("t02_runs.csv", ["metric", "Run 1", "Run 2", "Run 3", "unit", "source"], rows)

# first table: headline numbers
thermo = jload(DATA / "module2_T1" / "thermo.json")
vib = thermo["frequencies_cm1"]
inten = thermo["intensities_km_mol"]
comp = []
with open(DATA / "comparison" / "5fu_T1_ir_vs_argon_matrix.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        try:
            comp.append((r["matrix_mode"], float(r["matrix_cm1"]), float(r["matrix_rel_intensity"]),
                         float(r["skala_cm1_unscaled"]), float(r["skala_km_per_mol"])))
        except (ValueError, TypeError, KeyError):
            continue
exp = [c[1] for c in comp]
calc = [c[3] for c in comp]
s = sum(e * c for e, c in zip(exp, calc)) / sum(c * c for c in calc)
sc = [c * s for c in calc]
err_sc = [a - b for a, b in zip(sc, exp)]
err_un = [a - b for a, b in zip(calc, exp)]

def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return rk

def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5

rho = pearson(ranks([c[2] for c in comp]), ranks([c[4] for c in comp]))
mae = lambda v: sum(abs(a) for a in v) / len(v)
low = [i for i, e in enumerate(exp) if e < 1000]
high = [i for i, e in enumerate(exp) if e >= 1000]
co = [i for i, c in enumerate(comp) if c[0] in ("Q32", "Q33")]

write("t01_headline.csv", ["quantity", "value", "unit", "table / source"], [
    ["Starting shapes tried (Module 1)", "3 (0.11, 3.11 and 6.38 eV above the final energy)", "", "Table 2"],
    ["Run 2 final structure vs Run 1", f"{rm[1]:.4f} A and {abs(dE[1]):.2f} meV", "", "Table 2"],
    ["Run 3 final structure vs Run 1", f"{rm[2]:.4f} A and {abs(dE[2]):.2f} meV", "", "Table 2"],
    ["Gradient evaluations in the three runs", " / ".join(str(x["gradients_optimization"]) for x in summ), "", "Table 2"],
    ["Imaginary frequencies at the final shape (Module 2)", thermo["n_imaginary"], "", "Table 8"],
    ["Real vibrational frequencies", len(vib), "", "Table 10"],
    ["Lowest real vibration", f"{thermo['lowest_real_vib_cm1']:.1f}", "cm-1", "Table 10"],
    ["Bands compared with the argon-matrix spectrum", len(comp), "", "Table 13"],
    ["Scale factor (fitted on the same bands)", f"{s:.4f}", "", "Table 14"],
    ["Typical error after scaling (mean absolute)", f"{mae(err_sc):.1f}", "cm-1", "Table 14"],
    ["Largest error after scaling", f"{max(abs(e) for e in err_sc):.1f}", "cm-1", "Table 14"],
    ["Both C=O bands after scaling", f"within {max(abs(err_sc[i]) for i in co):.1f}", "cm-1", "Table 14"],
    ["Rank correlation of band intensities", f"{rho:.2f}", "", "Table 14"],
    ["Frequencies rebuilt from the 73 saved force files", "see Table 11", "", "Table 11"],
])

# ---------------------------------------------------------------- steps
rows = []
for k, st in enumerate(steps):
    fin = summ[k]["final_energy_eV"]
    for r in st:
        rows.append([RUN_NAMES[k], r["pass_id"], r["step"], f"{r['energy_eV']:.5f}", f"{(r['energy_eV'] - fin) * 1000:.1f}",
                     f"{r['fmax']:.4f}", f"{r['max_move']:.4f}", r["maxstep"], r["gradients"], f"{r['dE_eV']:.5f}",
                     r["scf_cycles"], f"{r['t_opt_s']:.1f}"])
write("t03_steps.csv", ["run", "pass", "step", "energy (eV)", "energy minus final (meV)", "force (Module 1 units)", "largest move (A)",
                        "step cap (A)", "gradients so far", "energy change (eV)", "SCF cycles", "optimization time (s)"], rows)

# ---------------------------------------------------------------- geometry
rows = []
labels = RUN_NAMES + ["Module 2 (def2-TZVP)"]
allx = X + [m2x]
for lab, x in zip(labels, allx):
    for i, (e, p) in enumerate(zip(el, x)):
        rows.append([lab, i, e, f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])
write("t06_final_coordinates.csv", ["structure", "atom", "element", "x (A)", "y (A)", "z (A)"], rows)

BONDS = [("O0=C1", 0, 1), ("C1-N2", 1, 2), ("N2-C3", 2, 3), ("C3=C4", 3, 4), ("C4-F5", 4, 5), ("C4-C6", 4, 6),
         ("C6=O7", 6, 7), ("C6-N8", 6, 8), ("N8-C1", 8, 1), ("N2-H9", 2, 9), ("C3-H10", 3, 10), ("N8-H11", 8, 11)]
ANGLES = [("O0-C1-N2", 0, 1, 2), ("O0-C1-N8", 0, 1, 8), ("C1-N2-C3", 1, 2, 3), ("N2-C3-C4", 2, 3, 4), ("C3-C4-C6", 3, 4, 6),
          ("C4-C6-N8", 4, 6, 8), ("C6-N8-C1", 6, 8, 1), ("N8-C1-N2", 8, 1, 2), ("F5-C4-C3", 5, 4, 3), ("F5-C4-C6", 5, 4, 6),
          ("O7-C6-C4", 7, 6, 4), ("O7-C6-N8", 7, 6, 8)]
rows = []
for name, i, j in BONDS:
    rows.append(["bond", name] + [f"{bond(x, i, j):.4f}" for x in allx] + ["A"])
for name, i, j, k in ANGLES:
    rows.append(["angle", name] + [f"{angle(x, i, j, k):.2f}" for x in allx] + ["degrees"])
write("t04_bonds_angles.csv", ["type", "atoms", "Run 1", "Run 2", "Run 3", "Module 2 (def2-TZVP)", "unit"], rows)

rows = []
d2 = np.linalg.norm(kabsch(X[1], X[0]) - X[0], axis=1)
d3 = np.linalg.norm(kabsch(X[2], X[0]) - X[0], axis=1)
d4 = np.linalg.norm(kabsch(m2x, X[1]) - X[1], axis=1)
for i, e in enumerate(el):
    rows.append([i, e, f"{d2[i]:.4f}", f"{d3[i]:.4f}", f"{d4[i]:.4f}"])
rows.append(["", "RMSD", f"{rmsd(X[1], X[0]):.4f}", f"{rmsd(X[2], X[0]):.4f}", f"{rmsd(m2x, X[1]):.4f}"])
write("t05_atom_displacements.csv", ["atom", "element", "Run 2 vs Run 1 (A)", "Run 3 vs Run 1 (A)", "Module 2 vs Run 2 (A)"], rows)

# ---------------------------------------------------------------- module 2
opt = jload(DATA / "module2_T1" / "opt_summary.json")
rows = [[p["pass_id"], p["converged"], p["steps"], f"{p['fmax_eV_A']:.5f}"] for p in opt["passes"]]
rows.append(["total", opt["converged"], sum(p["steps"] for p in opt["passes"]), f"{opt['fmax_eV_A']:.5f}"])
write("t07_module2_optimization.csv", ["pass", "converged", "steps", "largest force (eV/A, true units)"], rows)

env = jload(DATA / "module2_T1" / "environment.json")
rows = [["input", opt["start"]], ["basis", opt["basis"]], ["grid level", opt["grid_level"]],
        ["energy at the final shape (eV)", f"{opt['E_eV']:.6f}"], ["energy at the final shape (Eh)", f"{opt['E_Eh']:.8f}"],
        ["force target (eV/A)", opt["target_fmax_eV_A"]], ["gradient evaluations", opt["gradients"]], ["minutes", opt["minutes"]],
        ["GPU", env["gpu"]], ["Python", env["python"]], ["run started", env["started"]]]
rows += [[f"setting: {k}", v] for k, v in env["settings"].items()]
rows += [[f"package: {k}", v] for k, v in env["packages"].items()]
write("t09_module2_setup.csv", ["item", "value"], rows)

rows = [["Energy at the final shape", f"{thermo['E_eV']:.6f}", "eV"], ["Energy at the final shape", f"{thermo['E_Eh']:.8f}", "Eh"],
        ["Zero-point energy", f"{thermo['ZPE_eV']:.4f}", "eV"], ["Enthalpy H(298.15 K)", f"{thermo['H298_eV']:.4f}", "eV"],
        ["Free energy G(298.15 K)", f"{thermo['G298_eV']:.4f}", "eV"], ["Temperature", thermo["T_K"], "K"], ["Pressure", thermo["P_Pa"], "Pa"],
        ["Symmetry number", thermo["symmetry_number"], ""], ["Model", thermo["model"], ""],
        ["Imaginary frequencies", thermo["n_imaginary"], ""], ["Lowest real vibration", f"{thermo['lowest_real_vib_cm1']:.3f}", "cm-1"]]
for i, v in enumerate(thermo["tr_residual_cm1"], 1):
    rows.append([f"Translation/rotation mode {i} (should be near 0)", f"{v:.2f}", "cm-1"])
write("t08_thermo.csv", ["quantity", "value", "unit"], rows)

def region(f):
    if f < 700: return "ring motions, N-H wags"
    if f < 1000: return "C-H, N-H out-of-plane wags"
    if f < 1500: return "in-plane bends, C-F band"
    if f < 2000: return "C=O, C=C stretches"
    return "C-H, N-H stretches"
mx = max(inten)
rows = [[i + 1, f"{f:.3f}", f"{f * s:.1f}", f"{a:.2f}", f"{100 * a / mx:.1f}", region(f)] for i, (f, a) in enumerate(zip(vib, inten))]
write("t10_vibrations.csv", ["mode", "frequency, unscaled (cm-1)", f"frequency x {s:.4f} (cm-1)", "IR intensity (km/mol)", "intensity relative to strongest (%)", "region"], rows)

# ---------------------------------------------------------------- IR comparison
rows = []
with open(DATA / "comparison" / "5fu_T1_ir_vs_argon_matrix.csv", newline="") as fh:
    rd = csv.DictReader(fh)
    for r in rd:
        try:
            float(r["matrix_cm1"]); float(r["skala_cm1_unscaled"])
        except (ValueError, TypeError, KeyError):
            continue
        rows.append([r["matrix_mode"], r["matrix_assignment_TED"], r["matrix_cm1"], r["matrix_rel_intensity"], r["skala_cm1_unscaled"],
                     r["skala_km_per_mol"], r["diff_unscaled_cm1"], f"{float(r['skala_cm1_unscaled']) * s:.1f}",
                     f"{float(r['skala_cm1_unscaled']) * s - float(r['matrix_cm1']):+.1f}"])
write("t13_ir_comparison.csv", ["band", "assignment (Ivanov 2017)", "measured (cm-1)", "measured relative intensity", "computed, unscaled (cm-1)",
                                "computed intensity (km/mol)", "unscaled minus measured (cm-1)", f"computed x {s:.4f} (cm-1)",
                                "scaled minus measured (cm-1)"], rows)

def stat(name, idx):
    u = [err_un[i] for i in idx]; c = [err_sc[i] for i in idx]
    return [name, len(idx), f"{sum(u) / len(u):+.1f}", f"{mae(u):.1f}", f"{sum(c) / len(c):+.1f}", f"{mae(c):.1f}", f"{max(abs(v) for v in c):.1f}"]
allidx = list(range(len(comp)))
write("t14_ir_stats.csv", ["subset of bands", "bands", "mean signed error, unscaled (cm-1)", "mean absolute error, unscaled (cm-1)",
                           "mean signed error, scaled (cm-1)", "mean absolute error, scaled (cm-1)", "largest absolute error, scaled (cm-1)"], [
    stat("all bands", allidx), stat("below 1000 cm-1", low), stat("1000 cm-1 and above", high), stat("the two C=O bands", co),
    ["scale factor", "", "", "", "", "", f"{s:.4f} ({SCALE_SOURCE})"],
    ["rank correlation of intensities", "", "", "", "", "", f"{rho:.2f}"]])

with open(DATA / "ir_reference" / "5fu_T1_Ar_matrix_bands.csv", newline="") as fh:
    rows = [[r["mode"], r["nu_Ar_cm1"], r["I_rel_integrated"], r["assignment_TED_as_printed"], r["note"]] for r in csv.DictReader(fh)]
write("t15_reference_bands.csv", ["band", "measured (cm-1)", "relative intensity", "assignment as printed", "note"], rows)

# ---------------------------------------------------------------- Hessian from the saved forces
cache = DATA / "module2_T1" / "ir_cache"
def arr(o):
    shape, _, flat = o["__ndarray__"]
    return np.array(flat, float).reshape(shape)

delta = env["settings"]["delta"]
mass = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998}
M = np.repeat([mass[e] for e in m2el], 3)
H = np.zeros((36, 36))
files = []
for i in range(12):
    for k, ax in enumerate("xyz"):
        fp = jload(cache / f"cache.{i}{ax}+.json")
        fm = jload(cache / f"cache.{i}{ax}-.json")
        p, m_ = arr(fp["forces"]).ravel(), arr(fm["forces"]).ravel()
        H[3 * i + k] = -(p - m_) / (2 * delta)
        for sign, f in (("+", fp), ("-", fm)):
            d = arr(f["dipole"])
            files.append([f"cache.{i}{ax}{sign}.json", i, m2el[i], ax, sign, f"{np.abs(arr(f['forces'])).max():.4f}",
                          f"{d[0]:.4f}", f"{d[1]:.4f}", f"{d[2]:.4f}", f"{np.linalg.norm(d):.4f}"])
eqf = jload(cache / "cache.eq.json")
d = arr(eqf["dipole"])
files.insert(0, ["cache.eq.json", "", "", "equilibrium", "", f"{np.abs(arr(eqf['forces'])).max():.4f}",
                 f"{d[0]:.4f}", f"{d[1]:.4f}", f"{d[2]:.4f}", f"{np.linalg.norm(d):.4f}"])
write("t12_displacement_files.csv", ["file", "atom", "element", "axis", "sign", "largest force component (eV/A)",
                                     "dipole x", "dipole y", "dipole z", "dipole length"], files)

Hs = (H + H.T) / 2
w = np.linalg.eigvalsh(Hs / np.sqrt(np.outer(M, M)))
f_re = np.sign(w) * np.sqrt(np.abs(w)) * 521.47
published = sorted(list(thermo["tr_residual_cm1"]) + list(vib))
rows = []
for i, (a, b) in enumerate(zip(f_re, published), 1):
    kind = "translation or rotation" if i <= 6 else "vibration"
    diff = f"{a - b:+.2f}"
    rows.append([i, kind, f"{a:.2f}", f"{b:.2f}", "0.00" if abs(a - b) < 0.005 else diff])
rows.append(["", "largest absolute difference", "", "", f"{max(abs(a - b) for a, b in zip(f_re, published)):.2f}"])
rows.append(["", "Hessian asymmetry, largest |H - H^T| (eV/A^2)", "", "", f"{np.abs(H - H.T).max():.3f}"])
rows.append(["", "Hessian, largest entry (eV/A^2)", "", "", f"{np.abs(H).max():.2f}"])
rows.append(["", "finite-difference step (A)", "", "", delta])
rows.append(["", "largest force at the equilibrium geometry (eV/A)", "", "", f"{np.abs(arr(eqf['forces'])).max():.4f}"])
write("t11_hessian_recompute.csv", ["mode", "kind", "rebuilt from the force files (cm-1)", "published (cm-1)", "difference (cm-1)"], rows)

# ---------------------------------------------------------------- manifest
man = []
with open(DATA / "MANIFEST.csv", newline="") as fh:
    man = list(csv.DictReader(fh))
agg = {}
for r in man:
    k = (r["kind"], r["run_id"] or "")
    a = agg.setdefault(k, [0, 0])
    a[0] += 1
    a[1] += int(r["bytes"])
write("t18_manifest_summary.csv", ["kind", "run or group", "files", "bytes"],
      [[k[0], k[1], v[0], v[1]] for k, v in sorted(agg.items())] + [["all", "", len(man), sum(int(r["bytes"]) for r in man)]])

rows = []
bad = 0
for r in man:
    if "/ir_cache/" in r["path"]:
        continue
    p = ROOT / r["path"]
    ok = ""
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        ok = "yes" if h == r["sha256"] else "NO"
        bad += ok == "NO"
    rows.append([r["path"], r["bytes"], r["sha256"], r["kind"], r["description"], ok])
write("t19_checksums.csv", ["path", "bytes", "SHA-256", "kind", "description", "checksum matches"], rows)
print("checksum mismatches:", bad)

# ---------------------------------------------------------------- static tables
write("t16_rerun_expectations.csv", ["quantity", "what to expect on a rerun", "why"], [
    ["Final structure of each run", "same shape, within a few thousandths of an angstrom", "the minimum is a property of the functional, not of the path taken"],
    ["Final energy", "same within about 1 meV", "same reason; small differences from grid and convergence thresholds"],
    ["Number of gradient evaluations", "may differ by several", "the optimizer path need not be identical from run to run"],
    ["Frequencies and intensities", "same within about 1 cm-1 and a few percent", "same structure, same Hessian recipe; GPU arithmetic is not bit-for-bit repeatable"],
    ["Imaginary frequencies", "none", "a true minimum has none; a rerun that finds one has not converged"],
    ["Scale factor, errors, rank correlation", "identical for the same input table", "pure arithmetic on a fixed table"],
    ["File checksums in MANIFEST.csv", "identical for the released files", "the released files are fixed; new runs write new files"],
])
write("t17_claims_ledger.csv", ["claim", "number", "table", "status"], [
    ["Computed IR is consistent with the argon-matrix spectrum", "11 cm-1 after one scale factor (in-sample)", "Table 14", "supported"],
    ["Both C=O bands are within 4 cm-1 after scaling", "1764.5 vs 1761.2 and 1740.3 vs 1742.0", "Table 13", "supported"],
    ["Strong and weak peaks come in the same order", "rank correlation 0.86", "Table 14", "supported"],
    ["Zero imaginary frequencies (T1, def2-TZVP, Run 2 structure)", "0 of 30", "Table 8, Table 11", "supported"],
    ["IR computed from a 3.1 eV start", "Module 2 from the Run 2 structure", "Table 9", "supported"],
    ["Starts up to 6.4 eV reach one structure", "within 0.0072 A and 0.51 meV (Run 2); 0.0051 A and 0.29 meV (Run 3)", "Table 2, Table 5", "supported (geometry only)"],
    ["IR computed from the 6.4 eV start", "none", "", "not claimed"],
    ["Skala-1.1 more accurate or faster than other methods", "none", "", "not claimed"],
])
