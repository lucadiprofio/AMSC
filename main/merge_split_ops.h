#ifndef MERGE_SPLIT_OPS_H
#define MERGE_SPLIT_OPS_H

#include <algorithm> // clamp()
#include <queue>
#include <set>
#include <vector>

#include <particles.h>
#include <quadgrid_cpp.h>

/// @brief Configuration for ELFS-based adaptive merge/split.
///   - split  if  elfs(x_i)  <  alpha * r_i   (near boundary → refine)
///   - merge  if  elfs(x_i)  >  beta  * r_i   (far from boundary → coarsen)
///
/// where r_i = sqrt(Ap_i) is the particle characteristic length.
struct ms_config {
  double alpha = 2.0; /// Split threshold
  double beta = 3.0;  /// Merge threshold
  // double h = 2.5;     /// Inter-particle spacing parameter
  int max_ops = 50; /// Safety cap on total split+merge operations per call
  int call_interval = 10; /// Execute merge/split every N time steps
  int min_level = -2;     /// Don't split below this level (finer bound)
  int max_level = 2;      /// Don't merge above this level (coarser bound)
};

enum action_t : int {
  KEEP = 0,
  SPLIT = 1,
  MERGE_PRIMARY = 2,
  MERGE_SECONDARY = 3
};

static const std::set<std::string> extensive_props = {"Mp", "Vp", "Ap",
                                                      "mom_px", "mom_py"};

inline bool is_extensive(const std::string &name) {
  return extensive_props.count(name) > 0;
}

struct conservation_check {
  double total_mass, total_momx, total_momy;
  int num_particles;
};

inline conservation_check compute_conservation(const particles_t &ptcls) {
  conservation_check cc{0.0, 0.0, 0.0, static_cast<int>(ptcls.num_particles)};
  for (particles_t::idx_t i = 0; i < ptcls.num_particles; ++i) {
    cc.total_mass += ptcls.dprops.at("Mp")[i];
    cc.total_momx += ptcls.dprops.at("mom_px")[i];
    cc.total_momy += ptcls.dprops.at("mom_py")[i];
  }
  return cc;
}

inline void print_conservation(const std::string &label,
                               const conservation_check &cc) {
  std::cerr << "[" << label << "] N=" << cc.num_particles
            << "  mass=" << cc.total_mass << "  momx=" << cc.total_momx
            << "  momy=" << cc.total_momy << std::endl;
}

// Single-particle data (AoS view, used only in cold-path
// rebuild; not in hot compute kernels)
struct particle_data {
  double px, py;
  std::map<std::string, double> dp;
  std::map<std::string, particles_t::idx_t> ip;
};

inline particle_data extract_particle(const particles_t &ptcls,
                                      particles_t::idx_t i) {
  particle_data p;
  p.px = ptcls.x[i];
  p.py = ptcls.y[i];
  for (auto &[key, vec] : ptcls.dprops)
    p.dp[key] = vec[i];
  for (auto &[key, vec] : ptcls.iprops)
    p.ip[key] = vec[i];
  return p;
}

/// @brief Mark cells that are genuinely exterior to the fluid.
///
/// Criteria: only empty cells reachable from the grid edge
/// through a path of other empty cells are marked exterior.
/// Internal holes (empty cells surrounded by fluid) are NOT marked,
/// preventing false boundary detection.
inline void mark_exterior_cells(const particles_t &ptcls,
                                std::vector<bool> &is_exterior) {

  using idx_t = particles_t::idx_t;
  const idx_t nrows = ptcls.grid.num_rows();
  const idx_t ncols = ptcls.grid.num_cols();
  const int ncells = nrows * ncols;

  auto flat_idx = [&](idx_t r, idx_t c) -> int { return r + nrows * c; };

  // TODO: generalize the double cycle following
  // auto explore_cells

  // identify cells with particles
  std::vector<bool> has_particles(ncells, false);
  for (auto const &[cell_idx, ptcl_list] : ptcls.grd_to_ptcl)
    if (!ptcl_list.empty())
      has_particles[cell_idx] = true;

  // flood-fill bfs from grid boundary through empty cells
  is_exterior.assign(ncells, false);
  std::queue<int> q;

  // Seed: empty cells on the grid boundary
  for (idx_t c = 0; c < ncols; ++c) {
    for (idx_t r = 0; r < nrows; ++r) {
      if (r == 0 || r == nrows - 1 || c == 0 || c == ncols - 1) {
        int idx = flat_idx(r, c);
        if (!has_particles[idx] && !is_exterior[idx]) {
          is_exterior[idx] = true;
          q.push(idx);
        }
      }
    }
  }

  // TODO: here maybe is sufficient to consider only 4 cells and not 8
  // propagate only through empty cells
  const int dr[] = {-1, 0, 1, -1, 1, -1, 0, 1};
  const int dc[] = {-1, -1, -1, 0, 0, 1, 1, 1};

  while (!q.empty()) {
    int cidx = q.front();
    q.pop();
    idx_t r = cidx % nrows;
    idx_t c = cidx / nrows;

    // "around cell"
    for (int k = 0; k < 8; ++k) {
      int nr = static_cast<int>(r) + dr[k];
      int nc = static_cast<int>(c) + dc[k];
      if (nr < 0 || nr >= static_cast<int>(nrows) || nc < 0 ||
          nc >= static_cast<int>(ncols))
        continue;
      int nidx = flat_idx(nr, nc);
      if (!has_particles[nidx] && !is_exterior[nidx]) {
        is_exterior[nidx] = true;
        q.push(nidx);
      }
    }
  }
}

/// @brief Compute per-cell distance to the fluid boundary.
///
/// The fluid boundary is the interface between wet cells (with
/// particles) and exterior cells (identified by mark_exterior_cells).
inline void compute_boundary_distance(const particles_t &ptcls,
                                      const std::vector<bool> &is_exterior,
                                      std::vector<double> &cell_dist) {

  using idx_t = particles_t::idx_t;
  const idx_t nrows = ptcls.grid.num_rows();
  const idx_t ncols = ptcls.grid.num_cols();
  const double h = std::min(ptcls.grid.hx(), ptcls.grid.hy());
  const int ncells = nrows * ncols;

  // infinity via standard library
  constexpr double INF = std::numeric_limits<double>::max();
  cell_dist.assign(ncells, INF);

  auto flat_idx = [&](idx_t r, idx_t c) -> int { return r + nrows * c; };

  // Identify cells with particles
  std::vector<bool> has_particles(ncells, false);
  for (auto const &[cell_idx, ptcl_list] : ptcls.grd_to_ptcl)
    if (!ptcl_list.empty())
      has_particles[cell_idx] = true;

  // (cell_index, distance_in_cells)
  std::queue<std::pair<int, int>> bfs;

  // wet cells adjacent to an exterior cell
  for (idx_t c = 0; c < ncols; ++c) {
    for (idx_t r = 0; r < nrows; ++r) {
      int idx = flat_idx(r, c);
      if (!has_particles[idx])
        continue; // only seed from wet cells

      // checks whether the wet cell is surrounded by any exterior cell
      bool on_fluid_boundary = false;
      if (r > 0 && is_exterior[flat_idx(r - 1, c)])
        on_fluid_boundary = true;
      if (r < nrows - 1 && is_exterior[flat_idx(r + 1, c)])
        on_fluid_boundary = true;
      if (c > 0 && is_exterior[flat_idx(r, c - 1)])
        on_fluid_boundary = true;
      if (c < ncols - 1 && is_exterior[flat_idx(r, c + 1)])
        on_fluid_boundary = true;

      if (on_fluid_boundary) {
        cell_dist[idx] = 0.0;
        // queue filled by wet cells that are on boundary
        bfs.push({idx, 0});
      }
    }
  }

  // Flood-fill inward through wet cells
  const int dr[] = {-1, 1, 0, 0};
  const int dc[] = {0, 0, -1, 1};

  while (!bfs.empty()) {
    auto [cidx, d] = bfs.front();
    bfs.pop();

    idx_t r = cidx % nrows;
    idx_t c = cidx / nrows;

    for (int k = 0; k < 4; ++k) {
      int nr = static_cast<int>(r) + dr[k];
      int nc = static_cast<int>(c) + dc[k];
      if (nr < 0 || nr >= static_cast<int>(nrows) || nc < 0 ||
          nc >= static_cast<int>(ncols))
        continue;

      int nidx = flat_idx(nr, nc);
      if (!has_particles[nidx])
        continue; // don't propagate into empty cells

      double new_dist = (d + 1) * h;
      if (new_dist < cell_dist[nidx]) {
        cell_dist[nidx] = new_dist;
        bfs.push({nidx, d + 1});
      }
    }
  }
}

// Uses the precomputed cell-level distance as a first-order
// approximation. For particles near the boundary (at most one
// cell far from it), refines with the actual Euclidean distance from
// the particle position to the nearest exterior cell edge.
inline void compute_elfs(const particles_t &ptcls,
                         const std::vector<double> &cell_dist,
                         const std::vector<bool> &is_exterior,
                         std::vector<double> &elfs_values) {

  using idx_t = particles_t::idx_t;
  const idx_t nrows = ptcls.grid.num_rows();
  const idx_t ncols = ptcls.grid.num_cols();
  const double hx = ptcls.grid.hx();
  const double hy = ptcls.grid.hy();
  const double refine_threshold = std::max(hx, hy);

  elfs_values.resize(ptcls.num_particles);

  for (idx_t ip = 0; ip < ptcls.num_particles; ++ip) {
    int cell = ptcls.ptcl_to_grd[ip];
    double base = cell_dist[cell];

    // If far at least by one cell from boundary,
    // we assume that cell-level approx is good enough
    if (base > refine_threshold) {
      elfs_values[ip] = base;
      continue;
    }

    // refine: min Euclidean distance to any exterior cell edge
    double px = ptcls.x[ip];
    double py = ptcls.y[ip];
    double min_dist = base;

    idx_t r0 = cell % nrows;
    idx_t c0 = cell / nrows;

    // search in a small neighborhood around the particle's cell
    constexpr int radius = 2;
    for (int dr = -radius; dr <= radius; ++dr) {
      for (int dc = -radius; dc <= radius; ++dc) {
        int r = static_cast<int>(r0) + dr;
        int c = static_cast<int>(c0) + dc;
        if (r < 0 || r >= static_cast<int>(nrows) || c < 0 ||
            c >= static_cast<int>(ncols))
          continue;

        int nidx = r + nrows * c;
        if (!is_exterior[nidx])
          continue;

        // closest point on this exterior cell's bounding box
        double cx_lo = c * hx, cx_hi = (c + 1) * hx;
        double cy_lo = r * hy, cy_hi = (r + 1) * hy;

        // clamp px and py, of the wet cell, to the exterior cell bounding box
        double nearest_x = std::clamp(px, cx_lo, cx_hi);
        double nearest_y = std::clamp(py, cy_lo, cy_hi);

        double dx = px - nearest_x;
        double dy = py - nearest_y;
        double d = std::sqrt(dx * dx + dy * dy);

        min_dist = std::min(min_dist, d);
      }
    }

    elfs_values[ip] = min_dist;
  }
}

//  - split  if  elfs(x_i) < alpha * r_i   (near boundary)
//  - merge  if  elfs(x_i) > beta  * r_i   (far from boundary)
//  - keep   otherwise
//
// where r_i = sqrt(Ap_i) is the particle characteristic length.
template <typename idx_t>
inline void decide_actions(const particles_t &ptcls,
                           const std::vector<double> &elfs_values,
                           const ms_config &cfg, std::vector<action_t> &actions,
                           std::vector<idx_t> &merge_partner) {

  const idx_t N_p = ptcls.num_particles;

  actions.assign(N_p, KEEP);
  merge_partner.assign(N_p, -1);

  const std::vector<double> &Ap_vec = ptcls.dprops.at("Ap");
  for (idx_t ip = 0; ip < N_p; ++ip) {
    const double r_i = std::sqrt(Ap_vec[ip]);
    const double elfs_i = elfs_values[ip];

    if (elfs_i < cfg.alpha * r_i) {
      actions[ip] = SPLIT;
    } else if (elfs_i > cfg.beta * r_i) {
      actions[ip] = MERGE_PRIMARY;
    }
    // else: KEEP
  }

  // pair merge candidates within each cell
  // using nearest-neighbor distance
  int n_splits = 0, n_merges = 0;
  for (auto const &[cell_idx, ptcl_list] : ptcls.grd_to_ptcl) {

    // Enforce max_ops cap on splits
    for (auto pidx : ptcl_list) {
      if (actions[pidx] == SPLIT) {
        if (n_splits < cfg.max_ops)
          ++n_splits;
        else
          actions[pidx] = KEEP; // cap reached
      }
    }

    // Collect merge candidates in this cell
    std::vector<idx_t> candidates;
    for (auto pidx : ptcl_list) {
      if (actions[pidx] == MERGE_PRIMARY)
        candidates.push_back(pidx);
    }

    // Greedy nearest-neighbor pairing
    std::vector<bool> paired(candidates.size(), false);

    for (std::size_t i = 0; i < candidates.size() && n_merges < cfg.max_ops;
         ++i) {
      if (paired[i])
        continue;

      idx_t i1 = candidates[i];
      double best_dist = std::numeric_limits<double>::max();
      std::size_t best_j = candidates.size(); // invalid sentinel

      // Find nearest unpaired candidate
      for (std::size_t j = i + 1; j < candidates.size(); ++j) {
        if (paired[j])
          continue;
        idx_t i2 = candidates[j];
        double dx = ptcls.x[i1] - ptcls.x[i2];
        double dy = ptcls.y[i1] - ptcls.y[i2];
        double d2 = dx * dx + dy * dy;
        if (d2 < best_dist) {
          best_dist = d2;
          best_j = j;
        }
      }

      if (best_j < candidates.size()) {
        idx_t i2 = candidates[best_j];
        actions[i1] = MERGE_PRIMARY;
        actions[i2] = MERGE_SECONDARY;
        merge_partner[i1] = i2;
        paired[i] = true;
        paired[best_j] = true;
        ++n_merges;
      } else {
        actions[i1] = KEEP; // no partner found
      }
    }

    // Remaining unpaired candidates → demote to KEEP
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (!paired[i])
        actions[candidates[i]] = KEEP;
    }
  }

  std::cerr << "  [decide] splits=" << n_splits << "  merges=" << n_merges
            << std::endl;
}

inline void execute_merge_split(particles_t &ptcls,
                                const std::vector<action_t> &actions,
                                const std::vector<int> &merge_partner,
                                const ms_config &cfg) {

  using idx_t = particles_t::idx_t;
  const idx_t N_p = ptcls.num_particles;

  auto cc_before = compute_conservation(ptcls);
  print_conservation("BEFORE merge/split", cc_before);

  // Estimate output size
  int est = 0;
  for (idx_t i = 0; i < N_p; ++i) {
    if (actions[i] == SPLIT)
      est += 2;
    else if (actions[i] == MERGE_SECONDARY)
      est += 0;
    else
      est += 1;
  }

  // start ops
  std::vector<double> new_x, new_y;
  new_x.reserve(est);
  new_y.reserve(est);

  std::vector<std::string> dp_keys, ip_keys;
  for (auto &[key, vec] : ptcls.dprops)
    dp_keys.push_back(key);
  for (auto &[key, vec] : ptcls.iprops)
    ip_keys.push_back(key);
  std::vector<std::vector<double>> new_dp(dp_keys.size());
  std::vector<std::vector<idx_t>> new_ip(ip_keys.size());
  for (std::size_t k = 0; k < dp_keys.size(); ++k)
    new_dp[k].reserve(est);
  for (std::size_t k = 0; k < ip_keys.size(); ++k)
    new_ip[k].reserve(est);

  // lambda: push a particle_data into the new arrays
  auto add_particle = [&](const particle_data &p) {
    new_x.push_back(p.px);
    new_y.push_back(p.py);
    for (std::size_t k = 0; k < dp_keys.size(); ++k)
      new_dp[k].push_back(p.dp.at(dp_keys[k]));
    for (std::size_t k = 0; k < ip_keys.size(); ++k)
      new_ip[k].push_back(p.ip.at(ip_keys[k]));
  };

  for (idx_t i = 0; i < N_p; ++i) {

    switch (actions[i]) {

    case KEEP: {
      add_particle(extract_particle(ptcls, i));
      break;
    }

    case SPLIT: {
      particle_data mother = extract_particle(ptcls, i);

      if (mother.ip.at("level") <= cfg.min_level) {
        add_particle(mother); // keep unchanged
        break;
      }

      // Per-particle split distance: d = r_i / (2h)
      const double r_i = std::sqrt(mother.dp.at("Ap"));
      const double offset =
          r_i / (2.0 * std::min(ptcls.grid.hx(), ptcls.grid.hy()));

      double x1 = mother.px - offset, x2 = mother.px + offset;
      double y1 = mother.py, y2 = mother.py;

      // If horizontal split goes out of bounds, try vertical
      const double xmin = 0.0;
      const double xmax = ptcls.grid.num_cols() * ptcls.grid.hx();
      if (x1 < xmin + 1e-10 || x2 > xmax - 1e-10) {
        x1 = mother.px;
        x2 = mother.px;
        y1 = mother.py - offset;
        y2 = mother.py + offset;

        const double ymin = 0.0;
        const double ymax = ptcls.grid.num_rows() * ptcls.grid.hy();
        if (y1 < ymin + 1e-10 || y2 > ymax - 1e-10) {
          add_particle(mother); // give up, keep unchanged since no enough space
          break;
        }
      }

      idx_t daughter_level = mother.ip.at("level") - 1;

      particle_data d1 = mother;
      d1.px = x1;
      d1.py = y1;
      d1.dp["xp"] = x1;
      d1.dp["yp"] = y1;
      for (auto &key : extensive_props)
        d1.dp[key] = mother.dp.at(key) / 2.0;
      d1.dp["mom_px"] = d1.dp["Mp"] * d1.dp["vpx"];
      d1.dp["mom_py"] = d1.dp["Mp"] * d1.dp["vpy"];
      d1.ip["level"] = daughter_level;
      d1.ip["label"] = -1;
      add_particle(d1);

      particle_data d2 = mother;
      d2.px = x2;
      d2.py = y2;
      d2.dp["xp"] = x2;
      d2.dp["yp"] = y2;
      for (auto &key : extensive_props)
        d2.dp[key] = mother.dp.at(key) / 2.0;
      d2.dp["mom_px"] = d2.dp["Mp"] * d2.dp["vpx"];
      d2.dp["mom_py"] = d2.dp["Mp"] * d2.dp["vpy"];
      d2.ip["level"] = daughter_level;
      d2.ip["label"] = -1;
      add_particle(d2);

      break;
    }

    case MERGE_PRIMARY: {
      idx_t j = merge_partner[i];
      particle_data p1 = extract_particle(ptcls, i);
      particle_data p2 = extract_particle(ptcls, j);

      if (p1.ip.at("level") >= cfg.max_level) {
        add_particle(p1); // keep both unchanged
        add_particle(p2);
        break;
      }

      double M1 = p1.dp["Mp"];
      double M2 = p2.dp["Mp"];
      double Mtot = M1 + M2;
      particle_data merged;
      merged.px = (M1 * p1.px + M2 * p2.px) / Mtot;
      merged.py = (M1 * p1.py + M2 * p2.py) / Mtot;

      for (auto &key : dp_keys) {
        if (is_extensive(key))
          merged.dp[key] = p1.dp[key] + p2.dp[key];
        else
          merged.dp[key] = (M1 * p1.dp[key] + M2 * p2.dp[key]) / Mtot;
      }
      merged.dp["xp"] = merged.px;
      merged.dp["yp"] = merged.py;
      merged.dp["mom_px"] = merged.dp["Mp"] * merged.dp["vpx"];
      merged.dp["mom_py"] = merged.dp["Mp"] * merged.dp["vpy"];

      for (auto &key : ip_keys)
        merged.ip[key] = p1.ip[key];

      merged.ip["level"] = p1.ip.at("level") + 1;

      add_particle(merged);
      break;
    }

    case MERGE_SECONDARY:
      break;
    }
  }

  // update in ptcls
  ptcls.x = std::move(new_x);
  ptcls.y = std::move(new_y);

  for (std::size_t k = 0; k < dp_keys.size(); ++k)
    ptcls.dprops[dp_keys[k]] = std::move(new_dp[k]);
  for (std::size_t k = 0; k < ip_keys.size(); ++k)
    ptcls.iprops[ip_keys[k]] = std::move(new_ip[k]);

  ptcls.num_particles = static_cast<idx_t>(ptcls.x.size());

  // check
  auto cc_after = compute_conservation(ptcls);
  print_conservation("AFTER  merge/split", cc_after);
}

template <typename idx_t>
inline void adaptive_merge_split(particles_t &ptcls, const ms_config &cfg) {

  std::vector<bool> is_exterior;
  mark_exterior_cells(ptcls, is_exterior);

  // BFS distance to fluid boundary
  std::vector<double> cell_dist;
  compute_boundary_distance(ptcls, is_exterior, cell_dist);

  // per-particle ELFS
  std::vector<double> elfs_values;
  compute_elfs(ptcls, cell_dist, is_exterior, elfs_values);

  // decide actions
  std::vector<action_t> actions;
  std::vector<idx_t> merge_partner;
  decide_actions<idx_t>(ptcls, elfs_values, cfg, actions, merge_partner);

  // quick exit if nothing to do
  bool any_action = false;
  for (auto a : actions) {
    if (a != KEEP) {
      any_action = true;
      break;
    }
  }
  if (!any_action) {
    auto cc = compute_conservation(ptcls);
    print_conservation("AFTER  merge/split (no-op)", cc);
    return;
  }

  execute_merge_split(ptcls, actions, merge_partner, cfg);
}

#endif
