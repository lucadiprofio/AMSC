#!/usr/bin/env python3
# =====================================================================
#  conservation_check.py  --  verifica la conservazione (Eq.44) nel tempo
#
#  Legge conservation_errors.csv (header: time,err_mass,err_mom).
#  err_mass / err_mom sono errori RELATIVI tra particelle e nodi.
#  Devono restare a precisione macchina (~1e-14) a OGNI passo.
#
#  Uso:
#     python3 conservation_check.py conservation_errors.csv
#     python3 conservation_check.py cpu.csv gpu.csv     (confronta due run)
# =====================================================================
import sys, csv

def load(path):
    t, em, eo = [], [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if not row or row[0].strip() == "":
                continue
            t.append(float(row[0])); em.append(float(row[1])); eo.append(float(row[2]))
    return t, em, eo

def report(path):
    t, em, eo = load(path)
    n = len(t)
    max_m, max_o = max(em), max(eo)
    # passo in cui si raggiunge il massimo (per capire SE e QUANDO degrada)
    im = em.index(max_m); io = eo.index(max_o)
    print(f"\n=== {path}   ({n} passi, t finale = {t[-1]:.4g}) ===")
    print(f"  err_mass : max {max_m:.3e} (a t={t[im]:.4g}) | finale {em[-1]:.3e}")
    print(f"  err_mom  : max {max_o:.3e} (a t={t[io]:.4g}) | finale {eo[-1]:.3e}")
    worst = max(max_m, max_o)
    if worst < 1e-10:
        print(f"  -> max {worst:.3e}: conservazione a PRECISIONE MACCHINA a ogni passo. OK (Eq.44).")
    elif worst < 1e-6:
        print(f"  -> max {worst:.3e}: piccolo ma sopra macchina. Guarda se cresce nel tempo (degrado P2G?).")
    else:
        print(f"  -> max {worst:.3e}: la conservazione NON regge. Bug nel trasferimento P2G/G2P o NaN.")
    # segnale di degrado: l'errore cresce monotono?
    growth = eo[-1] / (eo[0] + 1e-300)
    if eo[0] > 0 and growth > 100:
        print(f"  ATTENZIONE: err_mom cresce ~{growth:.0e}x dall'inizio alla fine (possibile accumulo).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Uso: python3 conservation_check.py file1.csv [file2.csv]")
    for p in sys.argv[1:]:
        report(p)