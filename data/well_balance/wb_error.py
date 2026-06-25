#!/usr/bin/env python3
# =====================================================================
#  wb_error.py  --  errori L1 / Linf per il test di WELL-BALANCING
#
#  Soluzione stazionaria (lake at rest): h + Z = eq_level  ->  h_ref = eq_level - Zp
#  A riposo: hu = hv = 0.
#  Errori per particella:
#     err_h  = | hp - (eq_level - Zp) |
#     err_hu = | hp * vpx |
#     err_hv = | hp * vpy |
#
#  Uso:
#     python3 wb_error.py particles_finale.csv                 (eq_level=10 default)
#     python3 wb_error.py particles_finale.csv --eq 10
#     python3 wb_error.py cpu.csv gpu.csv                      (confronta due output)
#
#  Riproduce le norme della Tabella 1 del paper (Fois et al. 2024):
#     L_inf = max |err|         L1 (area-pesata) = sum |err|*Ap
#  Stampa anche la media |err| come riferimento aggiuntivo.
# =====================================================================
import sys, csv

def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    # header: nomi tra virgolette ed eventuali spazi -> pulisco
    header = [c.strip().strip('"').strip() for c in rows[0]]
    idx = {name: i for i, name in enumerate(header)}
    for needed in ("hp", "vpx", "vpy", "Zp", "Ap"):
        if needed not in idx:
            sys.exit(f"Colonna '{needed}' assente in {path}. Header letto: {header}")
    data = []
    for r in rows[1:]:
        if not r or all(c.strip() == "" for c in r):
            continue
        data.append(r)
    return idx, data

def errors(path, eq_level):
    idx, data = load(path)
    n = len(data)
    sum_h = sum_hu = sum_hv = 0.0      # L1 area-pesata
    max_h = max_hu = max_hv = 0.0      # Linf
    abs_h = abs_hu = abs_hv = 0.0      # per la media
    area_tot = 0.0
    for r in data:
        hp  = float(r[idx["hp"]])
        vpx = float(r[idx["vpx"]])
        vpy = float(r[idx["vpy"]])
        Zp  = float(r[idx["Zp"]])
        Ap  = float(r[idx["Ap"]])
        eh  = abs(hp - (eq_level - Zp))
        ehu = abs(hp * vpx)
        ehv = abs(hp * vpy)
        sum_h += eh*Ap;  sum_hu += ehu*Ap;  sum_hv += ehv*Ap;  area_tot += Ap
        abs_h += eh;     abs_hu += ehu;      abs_hv += ehv
        max_h = max(max_h, eh); max_hu = max(max_hu, ehu); max_hv = max(max_hv, ehv)
    return {
        "n": n,
        "L1":   (sum_h, sum_hu, sum_hv),
        "Linf": (max_h, max_hu, max_hv),
        "mean": (abs_h/n, abs_hu/n, abs_hv/n),
    }

def report(path, eq_level):
    e = errors(path, eq_level)
    print(f"\n=== {path}   (particelle: {e['n']},  eq_level={eq_level}) ===")
    print(f"{'':10s}{'L1 (area)':>16s}{'Linf':>16s}{'media|.|':>16s}")
    for k, lab in enumerate(("h", "hu", "hv")):
        print(f"{lab:10s}{e['L1'][k]:16.3e}{e['Linf'][k]:16.3e}{e['mean'][k]:16.3e}")
    worst = max(e["Linf"])
    if worst < 1e-10:
        verdict = "PRECISIONE MACCHINA -> well-balancing VALIDATO"
    elif worst < 1e-5:
        verdict = "~1e-7: sospetto residuo +1e-6 nei termini H (rivedi Edit A/B/C sulla CPU)"
    else:
        verdict = "ERRORE GROSSO: problema vero (BC, init, eq_level nel JSON, fix-massa)"
    print(f"  -> max Linf = {worst:.3e}   {verdict}")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    eq = 10.0
    if "--eq" in args:
        i = args.index("--eq"); eq = float(args[i+1]); del args[i:i+2]
    if not args:
        sys.exit("Uso: python3 wb_error.py file1.csv [file2.csv] [--eq 10]")
    for p in args:
        report(p, eq)