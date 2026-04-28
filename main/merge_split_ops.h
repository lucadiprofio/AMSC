#ifndef MERGE_SPLIT_OPS_H
#define MERGE_SPLIT_OPS_H

#include <particles.h>
#include <quadgrid_cpp.h>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <set>

// Configuration
struct merge_split_config {
  int    min_per_cell  = 2;
  int    max_per_cell  = 8;
  int    call_interval = 50;
  double split_offset  = 0.05;
  int    max_ops       = 10;
};



static const std::set<std::string> extensive_props = {
  "Mp", "Vp", "Ap", "mom_px", "mom_py"
};

inline bool is_extensive (const std::string & name) {
  return extensive_props.count (name) > 0;
}

struct conservation_check {
  double total_mass, total_momx, total_momy;
  int num_particles;
};

inline conservation_check
compute_conservation (const particles_t & ptcls)
{
  conservation_check cc{0.0, 0.0, 0.0, static_cast<int>(ptcls.num_particles)};
  for (particles_t::idx_t i = 0; i < ptcls.num_particles; ++i) {
    cc.total_mass += ptcls.dprops.at ("Mp")[i];
    cc.total_momx += ptcls.dprops.at ("mom_px")[i];
    cc.total_momy += ptcls.dprops.at ("mom_py")[i];
  }
  return cc;
}

inline void
print_conservation (const std::string & label, const conservation_check & cc)
{
  std::cerr << "[" << label << "] N=" << cc.num_particles
            << "  mass=" << cc.total_mass
            << "  momx=" << cc.total_momx
            << "  momy=" << cc.total_momy << std::endl;
}


// Struct for a single particle
struct particle_data {
  double px, py;
  std::map<std::string, double> dp;
  std::map<std::string, particles_t::idx_t> ip;
};

inline particle_data
extract_particle (const particles_t & ptcls, particles_t::idx_t i)
{
  particle_data p;
  p.px = ptcls.x[i];
  p.py = ptcls.y[i];
  for (auto & [key, vec] : ptcls.dprops)
    p.dp[key] = vec[i];
  for (auto & [key, vec] : ptcls.iprops)
    p.ip[key] = vec[i];
  return p;
}



// It builds a NEW set of particles.
// For each original particle, it decides:
// - KEEP: Copy it into the new set
// - SPLIT: Create 2 children in the new set
// - MERGE: Combine it with its partner in the new set
// Finally, replace everything in ptcls.

inline void
merge_split (particles_t & ptcls, const merge_split_config & cfg)
{
  using idx_t = particles_t::idx_t;
  idx_t N = ptcls.num_particles;

  auto cc_before = compute_conservation (ptcls);
  print_conservation ("BEFORE merge/split", cc_before);

 //Decide what to do for each particle

  enum action_t { KEEP, SPLIT, MERGE_PRIMARY, MERGE_SECONDARY };
  std::vector<action_t> action (N, KEEP);
  std::vector<idx_t> merge_partner (N, 0);

  int n_splits = 0, n_merges = 0;

  for (auto & [cell_idx, ptcl_list] : ptcls.grd_to_ptcl) {
    int np = static_cast<int> (ptcl_list.size ());

    //if n_particles < n_minimo then SPLIT
    if (np > 0 && np < cfg.min_per_cell) {
      for (auto ip : ptcl_list) {
        if (n_splits < cfg.max_ops) {
          action[ip] = SPLIT;
          ++n_splits;
        }
      }
    }

    //if n_particles > n_massimo then MERGE
    if (np > cfg.max_per_cell) {
      int excess = np - cfg.max_per_cell;
      for (int k = 0; k + 1 < 2 * excess && k + 1 < np; k += 2) {
        if (n_merges < cfg.max_ops) {
          idx_t i1 = ptcl_list[k];
          idx_t i2 = ptcl_list[k + 1];
          if (action[i1] == KEEP && action[i2] == KEEP) {
            action[i1] = MERGE_PRIMARY;
            action[i2] = MERGE_SECONDARY;
            merge_partner[i1] = i2;
            ++n_merges;
          }
        }
      }
    }
  }

  //If no splits or merge to do, return
  if (n_splits == 0 && n_merges == 0) {
    print_conservation ("AFTER  merge/split (no-op)", cc_before);
    return;
  }

  std::cerr << "  splits: " << n_splits << "  merges: " << n_merges << std::endl;

  //Build a new set of particle

  double offset = cfg.split_offset * ptcls.grid.hx ();
  double xmin = 0.0;
  double xmax = ptcls.grid.num_cols () * ptcls.grid.hx ();
  double ymin = 0.0;
  double ymax = ptcls.grid.num_rows () * ptcls.grid.hy ();

  idx_t est_size = N + n_splits - n_merges;
  std::vector<double> new_x, new_y;
  new_x.reserve (est_size);
  new_y.reserve (est_size);

  // Collect existing keys
  std::vector<std::string> dp_keys, ip_keys;
  for (auto & [key, vec] : ptcls.dprops) dp_keys.push_back (key);
  for (auto & [key, vec] : ptcls.iprops) ip_keys.push_back (key);

  std::vector<std::vector<double>> new_dp (dp_keys.size ());
  std::vector<std::vector<idx_t>> new_ip (ip_keys.size ());
  for (std::size_t k = 0; k < dp_keys.size (); ++k)
    new_dp[k].reserve (est_size);
  for (std::size_t k = 0; k < ip_keys.size (); ++k)
    new_ip[k].reserve (est_size);

  auto add_particle = [&] (const particle_data & p) {
    new_x.push_back (p.px);
    new_y.push_back (p.py);
    for (std::size_t k = 0; k < dp_keys.size (); ++k)
      new_dp[k].push_back (p.dp.at (dp_keys[k]));
    for (std::size_t k = 0; k < ip_keys.size (); ++k)
      new_ip[k].push_back (p.ip.at (ip_keys[k]));
  };

  for (idx_t i = 0; i < N; ++i) {

    switch (action[i]) {

      case KEEP: {
        particle_data p = extract_particle (ptcls, i);
        add_particle (p);
        break;
      }

      case SPLIT: {
        particle_data mother = extract_particle (ptcls, i);

        if (mother.ip["level"] <= -2) {
          add_particle (mother);
          break;
        }

        double x1 = mother.px - offset, x2 = mother.px + offset;
        double y1 = mother.py,          y2 = mother.py;

        if (x1 < xmin + 1e-10 || x2 > xmax - 1e-10) {
          x1 = mother.px;  x2 = mother.px;
          y1 = mother.py - offset;  y2 = mother.py + offset;
          if (y1 < ymin + 1e-10 || y2 > ymax - 1e-10) {
            add_particle (mother);
            break;
          }
        }

        // Daughter 1
        particle_data d1 = mother;
        d1.px = x1;  d1.py = y1;
        d1.dp["xp"] = x1;  d1.dp["yp"] = y1;
        for (auto & key : extensive_props)
          d1.dp[key] = mother.dp.at (key) / 2.0;
        d1.dp["mom_px"] = d1.dp["Mp"] * d1.dp["vpx"];
        d1.dp["mom_py"] = d1.dp["Mp"] * d1.dp["vpy"];
        d1.ip["label"] = -1;
        d1.ip["level"] = mother.ip["level"] - 1;
        add_particle (d1);

        // Daughter 2
        particle_data d2 = mother;
        d2.px = x2;  d2.py = y2;
        d2.dp["xp"] = x2;  d2.dp["yp"] = y2;
        for (auto & key : extensive_props)
          d2.dp[key] = mother.dp.at (key) / 2.0;
        d2.dp["mom_px"] = d2.dp["Mp"] * d2.dp["vpx"];
        d2.dp["mom_py"] = d2.dp["Mp"] * d2.dp["vpy"];
        d2.ip["label"] = -1;
        d2.ip["level"] = mother.ip["level"] - 1;
        add_particle (d2);

        break;
      }

      case MERGE_PRIMARY: {
        idx_t j = merge_partner[i];
        particle_data p1 = extract_particle (ptcls, i);
        particle_data p2 = extract_particle (ptcls, j);

        double M1 = p1.dp["Mp"];
        double M2 = p2.dp["Mp"];
        double Mtot = M1 + M2;
        if (Mtot < 1e-15) {
          add_particle (p1);
          break;
        }

        particle_data merged;
        merged.px = (M1 * p1.px + M2 * p2.px) / Mtot;
        merged.py = (M1 * p1.py + M2 * p2.py) / Mtot;

        for (auto & key : dp_keys) {
          if (is_extensive (key))
            merged.dp[key] = p1.dp[key] + p2.dp[key];
          else
            merged.dp[key] = (M1 * p1.dp[key] + M2 * p2.dp[key]) / Mtot;
        }
        merged.dp["xp"] = merged.px;
        merged.dp["yp"] = merged.py;
        merged.dp["mom_px"] = merged.dp["Mp"] * merged.dp["vpx"];
        merged.dp["mom_py"] = merged.dp["Mp"] * merged.dp["vpy"];
        merged.ip["level"] = p1.ip["level"] + 1;

        for (auto & key : ip_keys)
          merged.ip[key] = p1.ip[key];

        // Non mergere sopra livello +2
        if (p1.ip["level"] >= 2) {
          add_particle (p1);
          particle_data p2_keep = extract_particle (ptcls, j);
          add_particle (p2_keep);
          break;
        }
          
        add_particle (merged);
        break;
      }

      case MERGE_SECONDARY:
        break;
    }
  }

  // Replace in ptcls

  ptcls.x = std::move (new_x);
  ptcls.y = std::move (new_y);

  for (std::size_t k = 0; k < dp_keys.size (); ++k)
    ptcls.dprops[dp_keys[k]] = std::move (new_dp[k]);

  for (std::size_t k = 0; k < ip_keys.size (); ++k){
    ptcls.iprops[ip_keys[k]] = std::move (new_ip[k]);
  }

  ptcls.num_particles = ptcls.x.size ();

  auto cc_after = compute_conservation (ptcls);
  print_conservation ("AFTER  merge/split", cc_after);

  double mass_err = std::abs (cc_after.total_mass - cc_before.total_mass);
  if (mass_err > 1e-10)
    std::cerr << "WARNING: mass conservation error = " << mass_err << std::endl;
}

#endif

