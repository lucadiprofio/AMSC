#ifndef MERGE_SPLIT_OPS_CMES_H
#define MERGE_SPLIT_OPS_CMES_H

// =============================================================================
//  Adaptive merge/split for depth-averaged (shallow-water) MPM.
//
//  Criterion: LOCAL and deformation-based (CMES 2009, Ma/Zhang/Lian/Zhou),
//  adapted to the depth-averaged variables (Ap, hp, Vp). No ELFS / no BFS.
//
//  Characteristic length of a column:   r_i = sqrt(Ap_i)
//  Note: Ap evolves as  Ap *= (1 + dt*div(v))  every step, so Ap already
//  integrates the in-plane stretch history -> r_i is the accumulated-stretch
//  indicator (no reference configuration needed, unlike a solid).
//
//    - SPLIT  if  r_i > alpha * min_h        (column over-stretched/thinned)
//             or  gamma_dot*dt > shear_split (high deviatoric shear, optional)
//             AND hp_i >= split_hp_min       (do NOT refine negligibly-thin mass)
//    - MERGE  if  r_i < beta  * min_h        (column over-compressed/thick)
//
//  GENERIC HEADER: behaviour is set entirely through ms_config. Specialize a
//  case by tuning the parameters; the optional triggers (shear_split) and the
//  split_hp_min gate are DISABLED by default so the default is fully generic.
//
//  Consistency rules (matched to the time-step update hp /= (1+dt*div),
//  Vp = const, Ap = Vp/hp):
//    - Mp, Vp conserved (sum on merge, halve on split).
//    - hp is the PRIMARY field; Ap is DERIVED as Ap = Vp/hp.
//    - on MERGE: sum Vp and Ap, then DERIVE hp = Vp/Ap (do NOT average hp).
//    - on SPLIT: 1 -> 2 along the principal stretch direction.
//
//  Conservation:
//    - mass:      exact (sum / halving).
//    - momentum:  exact (mass-weighted velocity on merge; vp copied on split).
//    - energy:    merge is inelastic (consistent with Bingham dissipation);
//                 only co-moving particles are merged (|dv| < max_dv).
// =============================================================================

#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include <particles.h>
#include <quadgrid_cpp.h>

struct ms_config {
  // --- split / merge thresholds (fractions of the cell size) ---
  double alpha = 1.3;  ///< split if sqrt(Ap) > alpha*min_h   (over-stretched)
  double beta  = 0.30; ///< merge if sqrt(Ap) < beta *min_h   (over-compressed)

  // --- gate: do NOT split material thinner than this (negligible mass) ---
  ///  0.0 => disabled (generic). Set e.g. 0.2 to stop refining the vanishing
  ///  rarefaction tail and reclaim particles without losing the (massive) front.
  double split_hp_min = 0.2;

  // --- optional shear-driven refinement (correct sign: high shear -> split) ---
  double shear_split = 0.05; ///< split if gamma_dot*dt > shear_split.
                             ///< default huge => disabled. e.g. 0.05 to enable.

  // --- guards ---
  int    min_level = -2; ///< do not split finer than this
  int    max_level =  2; ///< do not merge coarser than this
  double hp_min    = 0.05;  ///< do not merge if the column is thinner than this
  double max_dv    = 0.01;  ///< merge only particles with |dv| < max_dv

  int    min_particles_per_cell = 2; ///< never empty a cell below this
  int    max_ops        = 1000000000; ///< cap on split+merge ops per call (huge => off)
  int    call_interval  = 10; ///< execute every N steps (used by caller)
};

enum action_t : int {
  KEEP          = 0,
  SPLIT         = 1, // 1 -> 2
  MERGE_PRIMARY = 2, // survivor of a 2 -> 1
  MERGE_SECONDARY = 3 // absorbed
};

// Extensive = scales with the amount of material (summed on merge, halved on
// split). Everything else is intensive (copied on split, mass-averaged on
// merge). hp is intensive BUT is overridden (derived) on merge - see below.
static const std::set<std::string> extensive_props = {"Mp", "Vp", "Ap",
                                                       "mom_px", "mom_py"};

inline bool is_extensive(const std::string &name) {
  return extensive_props.count(name) > 0;
}

// ----------------------------------------------------------------------------
//  Conservation diagnostics
// ----------------------------------------------------------------------------
struct conservation_check {
  double total_mass, total_momx, total_momy;
  particles_t::idx_t num_particles;
};

inline conservation_check compute_conservation(const particles_t &ptcls) {
  conservation_check cc{0.0, 0.0, 0.0, ptcls.num_particles};
  const particles_t::idx_t N = ptcls.num_particles;
  const double *Mp  = ptcls.dprops.at("Mp").data();
  const double *mpx = ptcls.dprops.at("mom_px").data();
  const double *mpy = ptcls.dprops.at("mom_py").data();
  double t_m = 0.0, t_x = 0.0, t_y = 0.0;
  for (particles_t::idx_t i = 0; i < N; ++i) {
    t_m += Mp[i];
    t_x += mpx[i];
    t_y += mpy[i];
  }
  cc.total_mass = t_m;
  cc.total_momx = t_x;
  cc.total_momy = t_y;
  return cc;
}

inline void print_conservation(const std::string &label,
                               const conservation_check &cc) {
  std::cerr << "[" << label << "] N=" << cc.num_particles
            << "  mass=" << cc.total_mass << "  momx=" << cc.total_momx
            << "  momy=" << cc.total_momy << std::endl;
}

// ----------------------------------------------------------------------------
//  Principal stretch direction from the in-plane strain-rate tensor.
//  Returns a unit vector along the eigenvector of the largest eigenvalue.
// ----------------------------------------------------------------------------
inline void principal_stretch_dir(double exx, double eyy, double exy,
                                  double &ux, double &uy) {
  const double tr   = exx + eyy;
  const double diff = 0.5 * (exx - eyy);
  const double disc = std::sqrt(diff * diff + exy * exy);
  const double l1   = 0.5 * tr + disc; // largest eigenvalue
  // eigenvector for l1:  (l1 - eyy, exy)
  double vx = l1 - eyy;
  double vy = exy;
  double n  = std::sqrt(vx * vx + vy * vy);
  if (n < 1e-12) { // (near-)isotropic: no preferred direction
    ux = 1.0;
    uy = 0.0;
  } else {
    ux = vx / n;
    uy = vy / n;
  }
}

// ----------------------------------------------------------------------------
//  decide_actions: per-particle marking (local, GPU-offloadable) + in-cell
//  pairing of merge candidates (host).
// ----------------------------------------------------------------------------
template <typename idx_t>
inline void decide_actions(const particles_t &ptcls, const ms_config &cfg,
                           double dt, std::vector<action_t> &actions,
                           std::vector<int> &merge_partner) {
  const idx_t N = ptcls.num_particles;
  actions.assign(N, KEEP);
  merge_partner.assign(N, -1);

  const double min_h = std::min(ptcls.grid.hx(), ptcls.grid.hy());
  const double r_split = cfg.alpha * min_h; // sqrt(Ap) above -> split
  const double r_merge = cfg.beta  * min_h; // sqrt(Ap) below -> merge
  const double shear_split  = cfg.shear_split;
  const double split_hp_min = cfg.split_hp_min;
  const double hp_min = cfg.hp_min;
  const int min_level = cfg.min_level;
  const int max_level = cfg.max_level;

  const double *Ap   = ptcls.dprops.at("Ap").data();
  const double *hp   = ptcls.dprops.at("hp").data();
  const double *exx_ = ptcls.dprops.at("vpx_dx").data();
  const double *eyy_ = ptcls.dprops.at("vpy_dy").data();
  const double *exy1 = ptcls.dprops.at("vpx_dy").data();
  const double *exy2 = ptcls.dprops.at("vpy_dx").data();
  const idx_t  *lvl  = ptcls.iprops.at("level").data();
  action_t *act = actions.data();

  // ---- per-particle decision (purely local -> safe on the device) ----
  #pragma omp target teams distribute parallel for \
      map(to: Ap[0:N], hp[0:N], exx_[0:N], eyy_[0:N], exy1[0:N], exy2[0:N], lvl[0:N]) \
      map(tofrom: act[0:N])
  for (idx_t ip = 0; ip < N; ++ip) {
    const double r_i = std::sqrt(Ap[ip]);
    const double exx = exx_[ip];
    const double eyy = eyy_[ip];
    const double exy = 0.5 * (exy1[ip] + exy2[ip]);
    // scalar shear rate |D| = sqrt(2 D:D)
    const double gdot =
        std::sqrt(2.0 * (exx * exx + eyy * eyy + 2.0 * exy * exy));

    const bool stretched = (r_i > r_split);
    const bool sheared   = (gdot * dt > shear_split);
    const bool thick     = (r_i < r_merge);
    // gate: skip refining negligibly-thin (low-mass) material
    const bool thin_skip = (hp[ip] < split_hp_min);

    if ((stretched || sheared) && !thin_skip &&
        (static_cast<int>(lvl[ip]) > min_level)) {
      act[ip] = SPLIT;
    } else if (thick && hp[ip] > hp_min &&
               static_cast<int>(lvl[ip]) < max_level) {
      act[ip] = MERGE_PRIMARY;
    }
  }

  // ---- pair MERGE candidates within each cell (host) ----
  const double *vpx = ptcls.dprops.at("vpx").data();
  const double *vpy = ptcls.dprops.at("vpy").data();
  const double max_dv2 = cfg.max_dv * cfg.max_dv;

  int n_split = 0, n_merge = 0;
  for (auto const &[cell, plist] : ptcls.grd_to_ptcl) {

    // cap splits
    for (auto p : plist) {
      if (actions[p] == SPLIT) {
        if (n_split < cfg.max_ops) ++n_split;
        else actions[p] = KEEP;
      }
    }

    // collect merge candidates
    std::vector<idx_t> cand;
    for (auto p : plist)
      if (actions[p] == MERGE_PRIMARY) cand.push_back(p);

    std::vector<char> paired(cand.size(), 0);
    for (std::size_t i = 0; i < cand.size() && n_merge < cfg.max_ops; ++i) {
      if (paired[i]) continue;
      idx_t i1 = cand[i];
      double best = std::numeric_limits<double>::max();
      std::size_t bj = cand.size();
      for (std::size_t j = i + 1; j < cand.size(); ++j) {
        if (paired[j]) continue;
        idx_t i2 = cand[j];
        const double dx = ptcls.x[i1] - ptcls.x[i2];
        const double dy = ptcls.y[i1] - ptcls.y[i2];
        const double d2 = dx * dx + dy * dy;
        const double dvx = vpx[i1] - vpx[i2];
        const double dvy = vpy[i1] - vpy[i2];
        // only merge co-moving particles (energy + do not smear yield surface)
        if (d2 < best && (dvx * dvx + dvy * dvy) < max_dv2) {
          best = d2;
          bj = j;
        }
      }
      if (bj < cand.size()) {
        actions[i1] = MERGE_PRIMARY;
        actions[cand[bj]] = MERGE_SECONDARY;
        merge_partner[i1] = static_cast<int>(cand[bj]);
        paired[i] = 1;
        paired[bj] = 1;
        ++n_merge;
      } else {
        actions[i1] = KEEP; // no partner
      }
    }
    for (std::size_t i = 0; i < cand.size(); ++i)
      if (!paired[i]) actions[cand[i]] = KEEP;

    // never drop a cell below min_particles_per_cell: undo merges first
    int projected = 0;
    for (auto p : plist) {
      if (actions[p] == KEEP) projected += 1;
      else if (actions[p] == SPLIT) projected += 2;       // 1 -> 2
      else if (actions[p] == MERGE_PRIMARY) projected += 1; // 2 -> 1
      // MERGE_SECONDARY -> 0
    }
    while (projected < cfg.min_particles_per_cell) {
      bool undone = false;
      for (auto p : plist) {
        if (actions[p] == MERGE_PRIMARY) {
          int q = merge_partner[p];
          if (q >= 0) actions[q] = KEEP;
          merge_partner[p] = -1;
          actions[p] = KEEP;
          --n_merge;
          projected += 1;
          undone = true;
          break;
        }
      }
      if (!undone) break;
    }
  }

  std::cerr << "  [decide] splits=" << n_split << "  merges=" << n_merge
            << std::endl;
}

// ----------------------------------------------------------------------------
//  execute_merge_split: build the new particle arrays.
// ----------------------------------------------------------------------------
inline void execute_merge_split(particles_t &ptcls,
                                const std::vector<action_t> &actions,
                                const std::vector<int> &merge_partner,
                                const ms_config & /*cfg*/) {
  using idx_t = particles_t::idx_t;
  const idx_t N = ptcls.num_particles;

  print_conservation("BEFORE merge/split", compute_conservation(ptcls));

  // prefix-sum: output offsets and final size
  const int weights[] = {1, 2, 1, 0}; // KEEP, SPLIT(1->2), MERGE_PRIMARY, SECONDARY
  std::vector<int> offset(N, 0);
  int Nnew = 0;
  for (idx_t i = 0; i < N; ++i) {
    offset[i] = Nnew;
    Nnew += weights[static_cast<int>(actions[i])];
  }

  std::vector<double> nx(Nnew), ny(Nnew);

  std::vector<std::string> dpk, ipk;
  for (auto const &[k, v] : ptcls.dprops) dpk.push_back(k);
  for (auto const &[k, v] : ptcls.iprops) ipk.push_back(k);

  const int ndp = (int)dpk.size(), nip = (int)ipk.size();
  std::vector<std::vector<double>> ndp_v(ndp, std::vector<double>(Nnew));
  std::vector<std::vector<idx_t>>  nip_v(nip, std::vector<idx_t>(Nnew));

  std::vector<const double *> in_dp(ndp);
  std::vector<double *>       out_dp(ndp);
  std::vector<int>            ext(ndp, 0);
  std::vector<const idx_t *>  in_ip(nip);
  std::vector<idx_t *>        out_ip(nip);

  // named-property indices (avoid map lookups in the hot loop)
  int iMp=-1,ivpx=-1,ivpy=-1,ixp=-1,iyp=-1,impx=-1,impy=-1,iAp=-1,iVp=-1,ihp=-1;
  int iexx=-1,ieyy=-1,iexy1=-1,iexy2=-1;
  for (int k = 0; k < ndp; ++k) {
    in_dp[k]  = ptcls.dprops.at(dpk[k]).data();
    out_dp[k] = ndp_v[k].data();
    ext[k]    = is_extensive(dpk[k]) ? 1 : 0;
    const std::string &n = dpk[k];
    if (n=="Mp") iMp=k; else if (n=="vpx") ivpx=k; else if (n=="vpy") ivpy=k;
    else if (n=="xp") ixp=k; else if (n=="yp") iyp=k;
    else if (n=="mom_px") impx=k; else if (n=="mom_py") impy=k;
    else if (n=="Ap") iAp=k; else if (n=="Vp") iVp=k; else if (n=="hp") ihp=k;
    else if (n=="vpx_dx") iexx=k; else if (n=="vpy_dy") ieyy=k;
    else if (n=="vpx_dy") iexy1=k; else if (n=="vpy_dx") iexy2=k;
  }
  int ilevel=-1, ilabel=-1;
  for (int k = 0; k < nip; ++k) {
    in_ip[k]  = ptcls.iprops.at(ipk[k]).data();
    out_ip[k] = nip_v[k].data();
    if (ipk[k]=="level") ilevel=k; else if (ipk[k]=="label") ilabel=k;
  }

  const double hx = ptcls.grid.hx(), hy = ptcls.grid.hy();
  const double Lx = ptcls.grid.num_cols() * hx;
  const double Ly = ptcls.grid.num_rows() * hy;
  const double eps = 1e-10;
  auto clampx = [&](double v){ return v<eps?eps:(v>Lx-eps?Lx-eps:v); };
  auto clampy = [&](double v){ return v<eps?eps:(v>Ly-eps?Ly-eps:v); };

  #pragma omp parallel for schedule(guided)
  for (idx_t i = 0; i < N; ++i) {
    const int a = static_cast<int>(actions[i]);

    if (a == KEEP) {
      const int o = offset[i];
      nx[o] = ptcls.x[i];
      ny[o] = ptcls.y[i];
      for (int k = 0; k < ndp; ++k) out_dp[k][o] = in_dp[k][i];
      for (int k = 0; k < nip; ++k) out_ip[k][o] = in_ip[k][i];

    } else if (a == SPLIT) {
      const int o1 = offset[i], o2 = offset[i] + 1;
      const idx_t level = (ilevel>=0) ? in_ip[ilevel][i] : 0;

      // principal stretch direction
      double ux, uy;
      const double exx  = (iexx >=0)? in_dp[iexx ][i] : 0.0;
      const double eyy  = (ieyy >=0)? in_dp[ieyy ][i] : 0.0;
      const double exy  = 0.5 * (((iexy1>=0)? in_dp[iexy1][i]:0.0) +
                                 ((iexy2>=0)? in_dp[iexy2][i]:0.0));
      principal_stretch_dir(exx, eyy, exy, ux, uy);

      const double r_i = (iAp>=0)? std::sqrt(in_dp[iAp][i]) : 0.0;
      const double off = 0.25 * r_i; // children stay inside parent footprint
      const double cx  = ptcls.x[i],  cy = ptcls.y[i];

      nx[o1] = clampx(cx + off*ux);  ny[o1] = clampy(cy + off*uy);
      nx[o2] = clampx(cx - off*ux);  ny[o2] = clampy(cy - off*uy);

      // generic: extensive halved, intensive copied
      for (int k = 0; k < ndp; ++k) {
        const double val = in_dp[k][i] / (ext[k] ? 2.0 : 1.0);
        out_dp[k][o1] = val;
        out_dp[k][o2] = val;
      }
      for (int k = 0; k < nip; ++k) {
        out_ip[k][o1] = in_ip[k][i];
        out_ip[k][o2] = in_ip[k][i];
      }

      // enforce the convention: hp primary (copied), Ap = Vp/hp
      if (iAp>=0 && iVp>=0 && ihp>=0) {
        const double h = out_dp[ihp][o1]; // == parent hp
        out_dp[iAp][o1] = out_dp[iVp][o1] / h;
        out_dp[iAp][o2] = out_dp[iVp][o2] / h;
      }
      // positions
      if (ixp>=0){ out_dp[ixp][o1]=nx[o1]; out_dp[ixp][o2]=nx[o2]; }
      if (iyp>=0){ out_dp[iyp][o1]=ny[o1]; out_dp[iyp][o2]=ny[o2]; }
      // momentum = vp * Mp  (vp copied -> exactly conserves total momentum)
      if (impx>=0 && iMp>=0 && ivpx>=0) {
        out_dp[impx][o1] = out_dp[iMp][o1]*out_dp[ivpx][o1];
        out_dp[impx][o2] = out_dp[iMp][o2]*out_dp[ivpx][o2];
      }
      if (impy>=0 && iMp>=0 && ivpy>=0) {
        out_dp[impy][o1] = out_dp[iMp][o1]*out_dp[ivpy][o1];
        out_dp[impy][o2] = out_dp[iMp][o2]*out_dp[ivpy][o2];
      }
      if (ilevel>=0){ out_ip[ilevel][o1]=level-1; out_ip[ilevel][o2]=level-1; }
      if (ilabel>=0){ out_ip[ilabel][o1]=-1;      out_ip[ilabel][o2]=-1; }

    } else if (a == MERGE_PRIMARY) {
      const int j = merge_partner[i];
      const int o = offset[i];
      const idx_t level = (ilevel>=0) ? in_ip[ilevel][i] : 0;

      const double M1 = (iMp>=0)? in_dp[iMp][i] : 1.0;
      const double M2 = (iMp>=0)? in_dp[iMp][j] : 1.0;
      const double Mt = M1 + M2;

      nx[o] = (M1*ptcls.x[i] + M2*ptcls.x[j]) / Mt; // centre of mass
      ny[o] = (M1*ptcls.y[i] + M2*ptcls.y[j]) / Mt;

      // generic: extensive summed, intensive mass-averaged
      for (int k = 0; k < ndp; ++k) {
        out_dp[k][o] = ext[k] ? (in_dp[k][i] + in_dp[k][j])
                              : (M1*in_dp[k][i] + M2*in_dp[k][j]) / Mt;
      }
      for (int k = 0; k < nip; ++k) out_ip[k][o] = in_ip[k][i];

      // KEY FIX: hp is DERIVED, not averaged.  Vp,Ap already summed above.
      if (ihp>=0 && iVp>=0 && iAp>=0)
        out_dp[ihp][o] = out_dp[iVp][o] / out_dp[iAp][o];

      if (ixp>=0) out_dp[ixp][o] = nx[o];
      if (iyp>=0) out_dp[iyp][o] = ny[o];
      // vpx/vpy were mass-averaged above (== momentum-conserving velocity);
      // rebuild momentum consistently.
      if (impx>=0 && iMp>=0 && ivpx>=0)
        out_dp[impx][o] = out_dp[iMp][o]*out_dp[ivpx][o];
      if (impy>=0 && iMp>=0 && ivpy>=0)
        out_dp[impy][o] = out_dp[iMp][o]*out_dp[ivpy][o];
      if (ilevel>=0) out_ip[ilevel][o] = level + 1;
      if (ilabel>=0) out_ip[ilabel][o] = -1;
    }
    // MERGE_SECONDARY: absorbed, writes nothing.
  }

  ptcls.x = std::move(nx);
  ptcls.y = std::move(ny);
  for (int k = 0; k < ndp; ++k) ptcls.dprops[dpk[k]] = std::move(ndp_v[k]);
  for (int k = 0; k < nip; ++k) ptcls.iprops[ipk[k]] = std::move(nip_v[k]);
  ptcls.num_particles = Nnew;

  print_conservation("AFTER  merge/split", compute_conservation(ptcls));
}

// ----------------------------------------------------------------------------
//  Public entry point.  NOTE the signature: dt is required.
// ----------------------------------------------------------------------------
template <typename idx_t>
inline void adaptive_merge_split(particles_t &ptcls, const ms_config &cfg,
                                 double dt) {
  std::vector<action_t> actions;
  std::vector<int> merge_partner;
  decide_actions<idx_t>(ptcls, cfg, dt, actions, merge_partner);

  bool any = false;
  for (auto a : actions)
    if (a != KEEP) { any = true; break; }
  if (!any) {
    print_conservation("AFTER  merge/split (no-op)", compute_conservation(ptcls));
    return;
  }
  execute_merge_split(ptcls, actions, merge_partner, cfg);
}

#endif // MERGE_SPLIT_OPS_CMES_H