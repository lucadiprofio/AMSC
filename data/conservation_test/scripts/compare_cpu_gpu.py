#!/usr/bin/env python3
# =====================================================================
#  compare_cpu_gpu.py  --  scarto CPU vs GPU particella per particella
#
#  Non esiste soluzione esatta per questo test: si confrontano le due
#  implementazioni indipendenti tra loro. Se restano vicine su un caso
#  DINAMICO, e' forte evidenza che advezione + update h + reologia sono
#  coerenti. Si allinea sulla colonna 'label' (ID particella stabile);
#  se assente, si usa l'ordine di riga.
#
#  Uso:
#     # un singolo step (stesso numero nei due file):
#     python3 compare_cpu_gpu.py cpu/nc_particles_490.csv gpu/nc_particles_490.csv
#
#     # tutti gli step comuni in due cartelle (mostra la CRESCITA dello scarto):
#     python3 compare_cpu_gpu.py cpu_dir/ gpu_dir/
# =====================================================================
import sys, os, csv, glob, re

FIELDS = ["hp", "vpx", "vpy", "mom_px", "mom_py"]

def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = [c.strip().strip('"').strip() for c in rows[0]]
    idx = {name: i for i, name in enumerate(header)}
    recs = []
    for r in rows[1:]:
        if not r or all(c.strip() == "" for c in r):
            continue
        recs.append(r)
    return idx, recs

def keyed(idx, recs):
    # dizionario label -> riga, se 'label' esiste; altrimenti per indice
    if "label" in idx:
        d = {}
        for r in recs:
            d[r[idx["label"]].strip()] = r
        return d, True
    return {str(i): r for i, r in enumerate(recs)}, False

def compare_one(fa, fb, verbose=True):
    ia, ra = load(fa)
    ib, rb = load(fb)
    da, by_label = keyed(ia, ra)
    db, _ = keyed(ib, rb)
    common = set(da) & set(db)
    if verbose and (len(da) != len(db) or len(common) != len(da)):
        print(f"  NOTA: A={len(da)} part, B={len(db)} part, comuni={len(common)} "
              f"({'per label' if by_label else 'per riga'})")
    stats = {f: [0.0, 0.0] for f in FIELDS}  # [max_abs, sum_abs]
    for k in common:
        for f in FIELDS:
            if f in ia and f in ib:
                va = float(da[k][ia[f]]); vb = float(db[k][ib[f]])
                d = abs(va - vb)
                if d > stats[f][0]:
                    stats[f][0] = d
                stats[f][1] += d
    n = max(len(common), 1)
    return stats, n

def step_num(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Uso: python3 compare_cpu_gpu.py <cpu> <gpu>  (due file CSV o due cartelle)")
    a, b = sys.argv[1], sys.argv[2]

    if os.path.isdir(a) and os.path.isdir(b):
        fa = {step_num(p): p for p in glob.glob(os.path.join(a, "*nc_particles_*.csv"))}
        fb = {step_num(p): p for p in glob.glob(os.path.join(b, "*nc_particles_*.csv"))}
        steps = sorted(set(fa) & set(fb))
        if not steps:
            sys.exit("Nessuno step comune trovato nelle due cartelle.")
        print(f"Step comuni: {len(steps)} (da {steps[0]} a {steps[-1]}). "
              f"Scarto max |CPU-GPU| nel tempo:")
        print(f"{'step':>8s}" + "".join(f"{f:>13s}" for f in FIELDS))
        for s in steps:
            st, n = compare_one(fa[s], fb[s], verbose=False)
            print(f"{s:>8d}" + "".join(f"{st[f][0]:13.3e}" for f in FIELDS))
        # verdetto sull'ultimo step
        st, n = compare_one(fa[steps[-1]], fb[steps[-1]], verbose=False)
        worst = max(st[f][0] for f in FIELDS)
        print(f"\n  ultimo step: max scarto = {worst:.3e}")
    else:
        st, n = compare_one(a, b)
        print(f"\n=== CPU vs GPU  ({n} particelle confrontate) ===")
        print(f"{'campo':>8s}{'max |diff|':>15s}{'media |diff|':>15s}")
        for f in FIELDS:
            print(f"{f:>8s}{st[f][0]:15.3e}{st[f][1]/n:15.3e}")
        worst = max(st[f][0] for f in FIELDS)
        print(f"\n  max scarto = {worst:.3e}")

    if worst < 1e-9:
        print("  -> identiche entro l'aritmetica FP: implementazioni coerenti.")
    elif worst < 1e-4:
        print("  -> piccolo scarto da riordino somme FP su caso dinamico: NORMALE e atteso.")
    else:
        print("  -> scarto GRANDE: una delle due ha un bug (controlla se cresce nel tempo).")