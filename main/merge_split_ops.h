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

// Helper: find index of a key in dp_keys/ip_keys
inline std::size_t
find_key (const std::vector<std::string> & keys, const std::string & name)
{
  for (std::size_t k = 0; k < keys.size (); ++k)
    if (keys[k] == name) return k;
  return keys.size ();  // not found
}

// copy particle i directly from ptcls to new arrays
inline void
copy_particle (const particles_t & ptcls,
               particles_t::idx_t i,
               std::vector<double> & nx, std::vector<double> & ny,
               std::vector<std::vector<double>> & ndp,
               std::vector<std::vector<particles_t::idx_t>> & nip,
               const std::vector<std::string> & dp_keys,
               const std::vector<std::string> & ip_keys)
{
  nx.push_back (ptcls.x[i]);
  ny.push_back (ptcls.y[i]);
  for (std::size_t k = 0; k < dp_keys.size (); ++k)
    ndp[k].push_back (ptcls.dprops.at (dp_keys[k])[i]);
  for (std::size_t k = 0; k < ip_keys.size (); ++k)
    nip[k].push_back (ptcls.iprops.at (ip_keys[k])[i]);
}


// add a split daughter from mother at index i
// dx, dy = offset from mother position

inline void
add_split_daughter (const particles_t & ptcls,
                    particles_t::idx_t i,
                    double dx, double dy,
                    std::vector<double> & nx, std::vector<double> & ny,
                    std::vector<std::vector<double>> & ndp,
                    std::vector<std::vector<particles_t::idx_t>> & nip,
                    const std::vector<std::string> & dp_keys,
                    const std::vector<std::string> & ip_keys,
                    std::size_t idx_xp, std::size_t idx_yp,
                    std::size_t idx_Mp, std::size_t idx_vpx, std::size_t idx_vpy,
                    std::size_t idx_momx, std::size_t idx_momy,
                    std::size_t idx_label, std::size_t idx_level)
{
  double new_px = ptcls.x[i] + dx;
  double new_py = ptcls.y[i] + dy;

  nx.push_back (new_px);
  ny.push_back (new_py);

  // Copy all dprops: halve extensive, inherit intensive
  for (std::size_t k = 0; k < dp_keys.size (); ++k) {
    double val = ptcls.dprops.at (dp_keys[k])[i];
    ndp[k].push_back (is_extensive (dp_keys[k]) ? val / 2.0 : val);
  }

  // Copy all iprops
  for (std::size_t k = 0; k < ip_keys.size (); ++k)
    nip[k].push_back (ptcls.iprops.at (ip_keys[k])[i]);

  // Fix xp, yp, momentum, label, level on the last inserted element
  std::size_t last = nx.size () - 1;
  ndp[idx_xp][last] = new_px;
  ndp[idx_yp][last] = new_py;
  double mp = ndp[idx_Mp][last];
  ndp[idx_momx][last] = mp * ndp[idx_vpx][last];
  ndp[idx_momy][last] = mp * ndp[idx_vpy][last];
  nip[idx_label][last] = -1;
  nip[idx_level][last] = ptcls.iprops.at ("level")[i] - 1;
}

//add a merged particle from particles i1 and i2
inline void
add_merged_particle (const particles_t & ptcls,
                     particles_t::idx_t i1, particles_t::idx_t i2,
                     std::vector<double> & nx, std::vector<double> & ny,
                     std::vector<std::vector<double>> & ndp,
                     std::vector<std::vector<particles_t::idx_t>> & nip,
                     const std::vector<std::string> & dp_keys,
                     const std::vector<std::string> & ip_keys,
                     std::size_t idx_xp, std::size_t idx_yp,
                     std::size_t idx_Mp, std::size_t idx_vpx, std::size_t idx_vpy,
                     std::size_t idx_momx, std::size_t idx_momy,
                     std::size_t idx_level)
{
  double M1 = ptcls.dprops.at ("Mp")[i1];
  double M2 = ptcls.dprops.at ("Mp")[i2];
  double Mtot = M1 + M2;

  double new_px = (M1 * ptcls.x[i1] + M2 * ptcls.x[i2]) / Mtot;
  double new_py = (M1 * ptcls.y[i1] + M2 * ptcls.y[i2]) / Mtot;

  nx.push_back (new_px);
  ny.push_back (new_py);

  // dprops: sum extensive, mass-weighted average intensive
  for (std::size_t k = 0; k < dp_keys.size (); ++k) {
    double v1 = ptcls.dprops.at (dp_keys[k])[i1];
    double v2 = ptcls.dprops.at (dp_keys[k])[i2];
    if (is_extensive (dp_keys[k]))
      ndp[k].push_back (v1 + v2);
    else
      ndp[k].push_back ((M1 * v1 + M2 * v2) / Mtot);
  }

  // iprops: inherit from first
  for (std::size_t k = 0; k < ip_keys.size (); ++k)
    nip[k].push_back (ptcls.iprops.at (ip_keys[k])[i1]);

  // Fix xp, yp, momentum, level
  std::size_t last = nx.size () - 1;
  ndp[idx_xp][last] = new_px;
  ndp[idx_yp][last] = new_py;
  double mp = ndp[idx_Mp][last];
  ndp[idx_momx][last] = mp * ndp[idx_vpx][last];
  ndp[idx_momy][last] = mp * ndp[idx_vpy][last];
  nip[idx_level][last] = ptcls.iprops.at ("level")[i1] + 1;
}

// MAIN FUNCTION
inline void
merge_split (particles_t & ptcls, const merge_split_config & cfg)
{
  using idx_t = particles_t::idx_t;
  idx_t N = ptcls.num_particles;

  auto cc_before = compute_conservation (ptcls);
  print_conservation ("BEFORE merge/split", cc_before);

  //  PHASE 1: Decide actions

  enum action_t { KEEP, SPLIT, MERGE_PRIMARY, MERGE_SECONDARY };
  std::vector<action_t> action (N, KEEP);
  std::vector<idx_t> merge_partner (N, 0);

  int n_splits = 0, n_merges = 0;

  for (auto & [cell_idx, ptcl_list] : ptcls.grd_to_ptcl) {
    int np = static_cast<int> (ptcl_list.size ());

    // Determine if this cell is on the material border
    idx_t row = cell_idx / ptcls.grid.num_cols ();
    idx_t col = cell_idx % ptcls.grid.num_cols ();

    bool is_border = false;
    idx_t ncols = ptcls.grid.num_cols ();
    idx_t nrows = ptcls.grid.num_rows ();

    // Check 4 neighbors: up, down, left, right
    if (row > 0) {
      idx_t nc = (row - 1) * ncols + col;
      if (ptcls.grd_to_ptcl.count (nc) == 0 || ptcls.grd_to_ptcl.at (nc).empty ())
        is_border = true;
    } else {
      is_border = true; 
    }
    if (!is_border && row < nrows - 1) {
      idx_t nc = (row + 1) * ncols + col;
      if (ptcls.grd_to_ptcl.count (nc) == 0 || ptcls.grd_to_ptcl.at (nc).empty ())
        is_border = true;
    } else if (row >= nrows - 1) {
      is_border = true;
    }
    if (!is_border && col > 0) {
      idx_t nc = row * ncols + (col - 1);
      if (ptcls.grd_to_ptcl.count (nc) == 0 || ptcls.grd_to_ptcl.at (nc).empty ())
        is_border = true;
    } else if (col <= 0) {
      is_border = true;
    }
    if (!is_border && col < ncols - 1) {
      idx_t nc = row * ncols + (col + 1);
      if (ptcls.grd_to_ptcl.count (nc) == 0 || ptcls.grd_to_ptcl.at (nc).empty ())
        is_border = true;
    } else if (col >= ncols - 1) {
      is_border = true;
    }

    // Dynamic thresholds
    int local_min = is_border ? cfg.min_per_cell + 1 : cfg.min_per_cell - 1;
    int local_max = is_border ? cfg.max_per_cell + 4 : cfg.max_per_cell - 4;
    if (local_min < 1) local_min = 1;
    if (local_max < 2) local_max = 2;

    // SPLIT
    if (np > 0 && np < local_min) {
      for (auto ip : ptcl_list) {
        if (n_splits < cfg.max_ops) {
          action[ip] = SPLIT;
          ++n_splits;
        }
      }
    }

    // MERGE
    if (np > local_max) {
      int excess = np - local_max;
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

  if (n_splits == 0 && n_merges == 0) {
    print_conservation ("AFTER  merge/split (no-op)", cc_before);
    return;
  }

  std::cerr << "  splits: " << n_splits << "  merges: " << n_merges << std::endl;

  //PHASE 2: Prepare new arrays

  double offset = cfg.split_offset * ptcls.grid.hx ();
  double xmin = 0.0;
  double xmax = ptcls.grid.num_cols () * ptcls.grid.hx ();
  double ymin = 0.0;
  double ymax = ptcls.grid.num_rows () * ptcls.grid.hy ();

  // Collect keys
  std::vector<std::string> dp_keys, ip_keys;
  for (auto & [key, vec] : ptcls.dprops) dp_keys.push_back (key);
  for (auto & [key, vec] : ptcls.iprops) ip_keys.push_back (key);

  // Pre-compute indices for frequently accessed fields
  std::size_t idx_xp    = find_key (dp_keys, "xp");
  std::size_t idx_yp    = find_key (dp_keys, "yp");
  std::size_t idx_Mp    = find_key (dp_keys, "Mp");
  std::size_t idx_vpx   = find_key (dp_keys, "vpx");
  std::size_t idx_vpy   = find_key (dp_keys, "vpy");
  std::size_t idx_momx  = find_key (dp_keys, "mom_px");
  std::size_t idx_momy  = find_key (dp_keys, "mom_py");
  std::size_t idx_label = find_key (ip_keys, "label");
  std::size_t idx_level = find_key (ip_keys, "level");

  // Allocate new arrays
  idx_t est = N + n_splits - n_merges;
  std::vector<double> new_x, new_y;
  new_x.reserve (est);
  new_y.reserve (est);

  std::vector<std::vector<double>> new_dp (dp_keys.size ());
  std::vector<std::vector<idx_t>> new_ip (ip_keys.size ());
  for (std::size_t k = 0; k < dp_keys.size (); ++k) new_dp[k].reserve (est);
  for (std::size_t k = 0; k < ip_keys.size (); ++k) new_ip[k].reserve (est);

  // PHASE 3: Build new particle set

  for (idx_t i = 0; i < N; ++i) {

    switch (action[i]) {

      case KEEP: {
        copy_particle (ptcls, i, new_x, new_y, new_dp, new_ip, dp_keys, ip_keys);
        break;
      }

      case SPLIT: {
        // Level check: don't split below -2
        if (ptcls.iprops.at ("level")[i] <= -2) {
          copy_particle (ptcls, i, new_x, new_y, new_dp, new_ip, dp_keys, ip_keys);
          break;
        }

        // Compute daughter positions
        double dx = offset, dy = 0.0;
        double x1 = ptcls.x[i] - dx, x2 = ptcls.x[i] + dx;

        // If out of domain along x, try along y
        if (x1 < xmin + 1e-10 || x2 > xmax - 1e-10) {
          dx = 0.0;  dy = offset;
          double y1 = ptcls.y[i] - dy, y2 = ptcls.y[i] + dy;
          if (y1 < ymin + 1e-10 || y2 > ymax - 1e-10) {
            copy_particle (ptcls, i, new_x, new_y, new_dp, new_ip, dp_keys, ip_keys);
            break;
          }
        }

        // Add two daughters
        add_split_daughter (ptcls, i, -dx, -dy,
                            new_x, new_y, new_dp, new_ip,
                            dp_keys, ip_keys,
                            idx_xp, idx_yp, idx_Mp, idx_vpx, idx_vpy,
                            idx_momx, idx_momy, idx_label, idx_level);

        add_split_daughter (ptcls, i, +dx, +dy,
                            new_x, new_y, new_dp, new_ip,
                            dp_keys, ip_keys,
                            idx_xp, idx_yp, idx_Mp, idx_vpx, idx_vpy,
                            idx_momx, idx_momy, idx_label, idx_level);
        break;
      }

      case MERGE_PRIMARY: {
        idx_t j = merge_partner[i];

        // Level check: don't merge above +2
        if (ptcls.iprops.at ("level")[i] >= 2) {
          copy_particle (ptcls, i, new_x, new_y, new_dp, new_ip, dp_keys, ip_keys);
          copy_particle (ptcls, j, new_x, new_y, new_dp, new_ip, dp_keys, ip_keys);
          break;
        }

        double Mtot = ptcls.dprops.at ("Mp")[i] + ptcls.dprops.at ("Mp")[j];
        if (Mtot < 1e-15) {
          copy_particle (ptcls, i, new_x, new_y, new_dp, new_ip, dp_keys, ip_keys);
          break;
        }

        add_merged_particle (ptcls, i, j,
                             new_x, new_y, new_dp, new_ip,
                             dp_keys, ip_keys,
                             idx_xp, idx_yp, idx_Mp, idx_vpx, idx_vpy,
                             idx_momx, idx_momy, idx_level);
        break;
      }

      case MERGE_SECONDARY:
        // Already handled by MERGE_PRIMARY
        break;
    }
  }

  // PHASE 4: Replace in ptcls

  ptcls.x = std::move (new_x);
  ptcls.y = std::move (new_y);
  for (std::size_t k = 0; k < dp_keys.size (); ++k)
    ptcls.dprops[dp_keys[k]] = std::move (new_dp[k]);
  for (std::size_t k = 0; k < ip_keys.size (); ++k)
    ptcls.iprops[ip_keys[k]] = std::move (new_ip[k]);
  ptcls.num_particles = ptcls.x.size ();

  auto cc_after = compute_conservation (ptcls);
  print_conservation ("AFTER  merge/split", cc_after);

  double mass_err = std::abs (cc_after.total_mass - cc_before.total_mass);
  if (mass_err > 1e-10)
    std::cerr << "WARNING: mass conservation error = " << mass_err << std::endl;
}

#endif