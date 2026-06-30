#!/usr/bin/env python3
# =====================================================================
#  clamp_analyze.py  --  quando e quanto scattano i tre safeguard
#
#  Legge clamp_log.csv:
#     time,n_vcap,max_vmag,n_scfloor,min_sc,n_hpfloor,min_hp
#
#  Per ogni clamp dice: se scatta mai, quante volte, QUANDO la prima
#  volta (rispetto all'impatto), e quanto la grandezza va fuori soglia.
#  Verdetto: MORTO / guardia post-impatto / CEROTTO precoce.
#
#  Uso:  python3 clamp_analyze.py clamp_log.csv --impact 3.0
# =====================================================================
import sys, csv

def main(path, t_impact):
    t, nv, mv, ns, msc, nh, mhp = [], [], [], [], [], [], []
    with open(path, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if not row or row[0].strip() == "":
                continue
            t.append(float(row[0]))
            nv.append(int(row[1]));  mv.append(float(row[2]))
            ns.append(int(row[3]));  msc.append(float(row[4]))
            nh.append(int(row[5]));  mhp.append(float(row[6]))
    nsteps = len(t)

    def verdict(counts, name, soglia_desc, extreme):
        fired = [(t[i], counts[i]) for i in range(nsteps) if counts[i] > 0]
        print(f"\n--- {name} ---")
        if not fired:
            print(f"  MAI scattato in {nsteps} step. -> MORTO: rimuovilo da entrambe.")
            return
        first_t = fired[0][0]; last_t = fired[-1][0]
        n_steps_fired = len(fired); total = sum(c for _, c in fired)
        peak = max(c for _, c in fired)
        before = sum(1 for tt, _ in fired if tt < t_impact)
        print(f"  scatta in {n_steps_fired}/{nsteps} step | scatti totali {total} | max in 1 step {peak}")
        print(f"  prima volta t={first_t:.3g} | ultima t={last_t:.3g} | {soglia_desc}: {extreme:.3e}")
        if before > 0 and first_t < t_impact:
            print(f"  -> CEROTTO: scatta gia' PRIMA dell'impatto (t<{t_impact}) in {before} step. "
                  f"Sintomo di dt troppo grande / instabilita', non guardia innocua.")
        else:
            print(f"  -> guardia POST-impatto (t>={t_impact}): regime gia' caotico, accettabile se documentato.")

    print(f"=== {path}  ({nsteps} step, t finale {t[-1]:.3g}, impatto assunto a t={t_impact}) ===")
    # vcap
    peak_v = max(mv); peak_v_t = t[mv.index(peak_v)]
    verdict(nv, "vcap (clamp velocita' 50 m/s)",
            f"picco velocita' raggiunto (a t={peak_v_t:.3g})", peak_v)
    if peak_v <= 50.0:
        print(f"  NB: picco velocita' = {peak_v:.2f} m/s <= 50: il cap non taglia mai davvero.")
    else:
        print(f"  NB: picco velocita' = {peak_v:.2f} m/s > 50: queste velocita' sono plausibili o numeriche?")
    # sc floor
    verdict(ns, "sc floor (0.1)", "sc minimo raggiunto (pre-floor)", min(msc))
    # hp floor
    verdict(nh, "hp floor (1e-2)", "hp minimo naturale (pre-floor)", min(mhp))
    print()

if __name__ == "__main__":
    args = sys.argv[1:]
    t_imp = 3.0
    if "--impact" in args:
        i = args.index("--impact"); t_imp = float(args[i+1]); del args[i:i+2]
    if not args:
        sys.exit("Uso: python3 clamp_analyze.py clamp_log.csv [--impact 3.0]")
    main(args[0], t_imp)