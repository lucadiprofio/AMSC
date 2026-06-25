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
  int min_particles_per_cell = 2;
  double stretch_thresh = 10000000; /// [NUOVO] Trigger cinematico (Divergenza massima tollerata)

  double hp_min = 0.05;    /// Trigger to prevent numerical fractures
  double max_dv = 0.01;    /// Velocity tolerance to conserve energy
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
  particles_t::idx_t num_particles;
};

inline conservation_check compute_conservation(const particles_t &ptcls) {
  conservation_check cc{0.0, 0.0, 0.0, ptcls.num_particles};
  particles_t::idx_t N = ptcls.num_particles;
  double t_mass = 0.0, t_momx = 0.0, t_momy = 0.0;

  const double* Mps = ptcls.dprops.at("Mp").data();
  const double* mom_pxs = ptcls.dprops.at("mom_px").data();
  const double* mom_pys = ptcls.dprops.at("mom_py").data();
  #pragma omp target teams distribute parallel for \
    map(to: Mps[0:N], mom_pxs[0:N], mom_pys[0:N]) \
    reduction(+: t_mass, t_momx, t_momy)
  for (particles_t::idx_t i = 0; i < N; ++i) {
    t_mass += Mps[i];
    t_momx += mom_pxs[i];
    t_momy += mom_pys[i];
  }
  cc.total_mass = t_mass;
  cc.total_momx = t_momx;
  cc.total_momy = t_momy;
  return cc;
}

inline void print_conservation(const std::string &label,
                               const conservation_check &cc) {
  std::cerr << "[" << label << "] N=" << cc.num_particles
            << "  mass=" << cc.total_mass << "  momx=" << cc.total_momx
            << "  momy=" << cc.total_momy << std::endl;
}

/// @brief Mark cells that are genuinely exterior to the fluid.
///
/// Criteria: only empty cells reachable from the grid edge
/// through a path of other empty cells are marked exterior.
/// Internal holes (empty cells surrounded by fluid) are NOT marked,
/// preventing false boundary detection.
inline void mark_exterior_cells(const particles_t &ptcls,
                                std::vector<char> &is_exterior) {
  using idx_t = particles_t::idx_t;

  const idx_t nrows = ptcls.grid.num_rows();
  const idx_t ncols = ptcls.grid.num_cols();
  const int ncells = nrows * ncols;

  auto flat_idx = [&](idx_t r, idx_t c) -> int { return r + nrows * c; };

  // identify cells with particles
  std::vector<char> has_particles(ncells, 0);
  for (auto const &[cell_idx, ptcl_list] : ptcls.grd_to_ptcl)
    if (!ptcl_list.empty())
      has_particles[cell_idx] = 1;

  // flood-fill bfs from grid boundary through empty cells
  is_exterior.assign(ncells, 0);
  std::queue<int> q;

  // Seed: empty cells on the grid boundary
  for (idx_t c = 0; c < ncols; ++c) {
    for (idx_t r : {0, nrows - 1}) {
      int idx = flat_idx(r, c);
      if (!has_particles[idx] && !is_exterior[idx]) {
        is_exterior[idx] = 1;
        q.push(idx);
      }
    }
  }
  for (idx_t r = 0; r < nrows; ++r) {
    for (idx_t c : {0, ncols - 1}) {
      int idx = flat_idx(r, c);
      if (!has_particles[idx] && !is_exterior[idx]) {
        is_exterior[idx] = 1;
        q.push(idx);
      }
    }
  }
  // rewriting the previous loop
  /*
  for (idx_t c = 0; c < ncols; ++c) {
    for (idx_t r = 0; r < nrows; ++r) {
      if (r == 0 || r == nrows - 1 || c == 0 || c == ncols - 1) {
        int idx = flat_idx(r, c);
        if (!has_particles[idx] && !is_exterior[idx]) {
          is_exterior[idx] = 1;
          q.push(idx);
        }
      }
    }
  }
  */

  // propagate only through empty cells
  const int dr[] = {-1, 1, 0, 0};
  const int dc[] = {0, 0, -1, 1};

  while (!q.empty()) {
    int cidx = q.front();
    q.pop();
    idx_t r = cidx % nrows;
    idx_t c = cidx / nrows;

    // "around cell"
    for (int k = 0; k < 4; ++k) {
      int nr = static_cast<int>(r) + dr[k];
      int nc = static_cast<int>(c) + dc[k];
      if (nr < 0 || nr >= static_cast<int>(nrows) || nc < 0 ||
          nc >= static_cast<int>(ncols))
        continue;
      int nidx = flat_idx(nr, nc);
      if (!has_particles[nidx] && !is_exterior[nidx]) {
        is_exterior[nidx] = 1;
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
                                      const std::vector<char> &is_exterior,
                                      std::vector<double> &cell_dist) {
  using idx_t = particles_t::idx_t;

  const idx_t nrows = ptcls.grid.num_rows();
  const idx_t ncols = ptcls.grid.num_cols();
  const int ncells = nrows * ncols;
  const double h = std::min(ptcls.grid.hx(), ptcls.grid.hy());

  // infinity via standard library
  constexpr double INF = std::numeric_limits<double>::max();
  cell_dist.assign(ncells, INF);

  auto flat_idx = [&](idx_t r, idx_t c) -> int { return r + nrows * c; };

  // Identify cells with particles
  std::vector<char> has_particles(ncells, 0);
  for (auto const &[cell_idx, ptcl_list] : ptcls.grd_to_ptcl)
    if (!ptcl_list.empty())
      has_particles[cell_idx] = 1;

  // (cell_index, distance_in_cells)
  std::queue<std::pair<int, int>> bfs;

  // wet cells adjacent to an exterior cell
  for (idx_t c = 0; c < ncols; ++c) {
    for (idx_t r = 0; r < nrows; ++r) {
      int idx = flat_idx(r, c);
      if (!has_particles[idx])
        continue; // only seed from wet cells

      // checks whether the wet cell is surrounded by any exterior cell
      /*bool on_fluid_boundary = false;
      if (r > 0 && is_exterior[flat_idx(r - 1, c)])
        on_fluid_boundary = true;
      if (r < nrows - 1 && is_exterior[flat_idx(r + 1, c)])
        on_fluid_boundary = true;
      if (c > 0 && is_exterior[flat_idx(r, c - 1)])
        on_fluid_boundary = true;
      if (c < ncols - 1 && is_exterior[flat_idx(r, c + 1)])
        on_fluid_boundary = true;
      if (r == 0 || r == nrows - 1 || c == 0 || c == ncols - 1) 
        on_fluid_boundary = true;*/

      bool is_domain_edge = (r == 0 || r == nrows - 1 || c == 0 || c == ncols - 1);

      bool on_fluid_boundary = false;
      if (!is_domain_edge) {
          if (c > 0         && is_exterior[flat_idx(r, c - 1)]) on_fluid_boundary = true;
          if (c < ncols - 1 && is_exterior[flat_idx(r, c + 1)]) on_fluid_boundary = true;
      }

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
                         const std::vector<char> &is_exterior,
                         std::vector<double> &elfs_values) {
  using idx_t = particles_t::idx_t;

  const idx_t nrows = ptcls.grid.num_rows();
  const idx_t ncols = ptcls.grid.num_cols();
  const int ncells = nrows * ncols;

  const double hx = ptcls.grid.hx();
  const double hy = ptcls.grid.hy();
  const double refine_threshold = std::max(hx, hy);

  const idx_t N_p = ptcls.num_particles;
  elfs_values.resize(N_p);
  
  const auto* ptcl_cell = ptcls.ptcl_to_grd.data();
  const double* dist = cell_dist.data();
  const char* ext = is_exterior.data();
  double* elfs = elfs_values.data();
  
  const double* px_ptr = ptcls.x.data();
  const double* py_ptr = ptcls.y.data();
  #pragma omp target teams distribute parallel for \
    map(to: px_ptr[0:N_p], py_ptr[0:N_p], ptcl_cell[0:N_p], dist[0:ncells], ext[0:ncells]) \
    map(tofrom: elfs[0:N_p])
  for (idx_t ip = 0; ip < N_p; ++ip) {
    int cell = ptcl_cell[ip];
    double base = dist[cell];

    if (base > refine_threshold) {
      elfs[ip] = base;
      continue;
    }

    double px = px_ptr[ip];
    double py = py_ptr[ip];
    double min_dist = base;

    idx_t r0 = cell % nrows;
    idx_t c0 = cell / nrows;

    constexpr int radius = 2;
    //for (int dr = -radius; dr <= radius; ++dr) {
    int dr=0;
      for (int dc = -radius; dc <= radius; ++dc) {
        int r = static_cast<int>(r0) + dr;
        int c = static_cast<int>(c0) + dc;
        if (r < 0 || r >= static_cast<int>(nrows) || c < 0 ||
            c >= static_cast<int>(ncols))
          continue;

        int nidx = r + nrows * c;
        if (!ext[nidx])
          continue;

          // closest point on this exterior cell's bounding box
        double cx_lo = c * hx, cx_hi = (c + 1) * hx;
        double cy_lo = r * hy, cy_hi = (r + 1) * hy;

        // clamp px and py, of the wet cell, to the exterior cell bounding box
        // removing usage of std::clamp for now (don't know if gpu supports it)
        double nearest_x = px < cx_lo ? cx_lo : (px > cx_hi ? cx_hi : px);
        double nearest_y = py < cy_lo ? cy_lo : (py > cy_hi ? cy_hi : py);

        double dx = px - nearest_x;
        double dy = py - nearest_y;
        double d = std::sqrt(dx * dx + dy * dy);

        // same siutation as std::clamp case
        if (d < min_dist) min_dist = d;
      }
    //}
    elfs[ip] = min_dist;
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
                           std::vector<int> &merge_partner) {
  // comodity variables
  double alpha = cfg.alpha;
  double beta = cfg.beta;
  int min_level = cfg.min_level;
  int max_level = cfg.max_level;
  
  const double hx = ptcls.grid.hx();
  const double hy = ptcls.grid.hy();
  const double min_h = std::min(hx, hy);
  const double hp_min = cfg.hp_min;
  const int ncols = ptcls.grid.num_cols();
  const int nrows = ptcls.grid.num_rows();

  const idx_t N_p = ptcls.num_particles;
  actions.assign(N_p, KEEP);
  merge_partner.assign(N_p, -1);


  // pointers to exploit openmp offloading.
  // This is the best way to handle large amount of data on the gpu
  const double* Ap_vec = ptcls.dprops.at("Ap").data();
  const double* elfs = elfs_values.data();
  action_t* act = actions.data();
  const idx_t* level_vec = ptcls.iprops.at("level").data();
  const double* px = ptcls.x.data();
  const double* py = ptcls.y.data();
  const double* vpx_vec = ptcls.dprops.at("vpx").data();
  const double* vpy_vec = ptcls.dprops.at("vpy").data();
  const double* hp_vec = ptcls.dprops.at("hp").data();
  const double* vpx_dx_vec = ptcls.dprops.at("vpx_dx").data();
  const double* vpy_dy_vec = ptcls.dprops.at("vpy_dy").data();
  #pragma omp target teams distribute parallel for \
    map(to: Ap_vec[0:N_p], elfs[0:N_p], level_vec[0:N_p], px[0:N_p], py[0:N_p], hp_vec[0:N_p], vpx_dx_vec[0:N_p], vpy_dy_vec[0:N_p]) \
    map(tofrom: act[0:N_p])
  for (idx_t ip = 0; ip < N_p; ++ip) {
    const double r_i = std::sqrt(Ap_vec[ip]);
    const double elfs_i = elfs[ip];
    const double hp_i = hp_vec[ip];

    const double divergence = vpx_dx_vec[ip] + vpy_dy_vec[ip];

    if ((elfs_i < alpha * r_i ) || (divergence > cfg.stretch_thresh)) {
      if (level_vec[ip] > min_level) act[ip] = SPLIT;
    } else if (elfs_i > beta * r_i && hp_i > hp_min) {
      if (level_vec[ip] < max_level) act[ip] = MERGE_PRIMARY;
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
        double dvx = vpx_vec[i1] - vpx_vec[i2];
        double dvy = vpy_vec[i1] - vpy_vec[i2];
        double dv2 = dvx * dvx + dvy * dvy;
        if (d2 < best_dist && dv2 < cfg.max_dv * cfg.max_dv) {
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
      if (!paired[i]) actions[candidates[i]] = KEEP;
    }

    // Enforce minimum particles per cell
    int projected = 0;
    for (auto pidx : ptcl_list) {
      if (actions[pidx] == KEEP) projected += 1;
      else if (actions[pidx] == SPLIT) projected += 4;   // 1→4 split
      else if (actions[pidx] == MERGE_PRIMARY) projected += 1;  // 2→1
      // MERGE_SECONDARY contributes 0 (absorbed)
    }
    while (projected < 2) {
      bool found = false;
      for (auto pidx : ptcl_list) {
        if (actions[pidx] == MERGE_PRIMARY) {
          idx_t partner = merge_partner[pidx];
          if (partner >= 0) actions[partner] = KEEP;
          merge_partner[pidx] = -1;
          actions[pidx] = KEEP;
          n_merges--;
          projected += 1;
          found = true;
          break;
        }
      }
      if (!found) break;
    }

  }

  // // 4. Sanitize: Ensure no MERGE_PRIMARY is left without a partner!
  // // This prevents SegFaults if a particle was marked MERGE_PRIMARY by the GPU
  // // but wasn't processed by the CPU (e.g., if it's out of grid bounds / not in grd_to_ptcl).
  // #pragma omp parallel for schedule(static)
  // for (idx_t i = 0; i < N_p; ++i) {
  //   if (actions[i] == MERGE_PRIMARY && merge_partner[i] < 0) {
  //     actions[i] = KEEP;
  //   }
  // }

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

  // estimate the new size and save offsets to parallelize the loop
  std::vector<int> offsets(N_p, 0);
  const int weights[] = {1, 4, 1, 0};
  
  int exact_est = 0;
  // rearrange the arithmetic to avoid branch in the loop
  // using prefix-sum (scan) technique also for subsequent offloading...
  for (idx_t i = 0; i < N_p; ++i) {
    offsets[i] = exact_est;
    exact_est += weights[static_cast<int>(actions[i])];
  }
  // thanks to this loop we know exaclty the final size of the arrays and the positions of each particle in the new arrays

  // allocate new arrays
  std::vector<double> new_x(exact_est), new_y(exact_est);

  std::vector<std::string> dp_keys, ip_keys;
  // parallelizing these loops is not necessary since there are only very few properties
  for (auto const &[key, vec] : ptcls.dprops) dp_keys.push_back(key);
  for (auto const &[key, vec] : ptcls.iprops) ip_keys.push_back(key);

  std::vector<std::vector<double>> new_dp(dp_keys.size());
  for (auto &property : new_dp) property.resize(exact_est);
  std::vector<std::vector<idx_t>> new_ip(ip_keys.size());
  for (auto &property : new_ip) property.resize(exact_est);

  // prepare data pointers for openmp offloading
  int num_dp = dp_keys.size();
  std::vector<const double*> in_dp_ptrs(num_dp);
  std::vector<double*> out_dp_ptrs(num_dp);

  // the function already defined is not gpu friendly
  std::vector<int> is_ext(num_dp, 0);

  // mapping of dprops to avoid the lookup in the map (slow)
  enum DP_ID { e_Mp = 0, e_vpx, e_vpy, e_xp, e_yp, e_mom_px, e_mom_py, e_Ap, e_DP_NUM };
  const std::string dp_names[] = {"Mp", "vpx", "vpy", "xp", "yp", "mom_px", "mom_py", "Ap"};
  int dp_idx[e_DP_NUM] = {-1, -1, -1, -1, -1, -1, -1, -1};

  // for each property, store its pointer
  for (int k = 0; k < num_dp; ++k) {
    // few iterations here, map lookup not a bottleneck
    in_dp_ptrs[k] = ptcls.dprops.at(dp_keys[k]).data();
    out_dp_ptrs[k] = new_dp[k].data();
    if (is_extensive(dp_keys[k])) is_ext[k] = 1;
    

    for (int e = 0; e < e_DP_NUM; ++e) {
      if (dp_keys[k] == dp_names[e]) dp_idx[e] = k;
    }
  }

  int num_ip = ip_keys.size();
  std::vector<const idx_t*> in_ip_ptrs(num_ip);
  std::vector<idx_t*> out_ip_ptrs(num_ip);
  
  // mapping of iprops
  enum IP_ID { e_level = 0, e_label, e_IP_NUM };
  const std::string ip_names[] = {"level", "label"};
  int ip_idx[e_IP_NUM] = {-1, -1};

  for (int k = 0; k < num_ip; ++k) {
    in_ip_ptrs[k] = ptcls.iprops.at(ip_keys[k]).data();
    out_ip_ptrs[k] = new_ip[k].data();
    

    for (int e = 0; e < e_IP_NUM; ++e) {
      if (ip_keys[k] == ip_names[e]) ip_idx[e] = k;
    }
  }

  // start ops
  const double hx = ptcls.grid.hx();
  const double hy = ptcls.grid.hy();
  const double min_h = std::min(hx, hy);
  const int ncols = ptcls.grid.num_cols();
  const int nrows = ptcls.grid.num_rows();

#pragma omp parallel for schedule(guided)
  for (idx_t i = 0; i < N_p; ++i) {
    if (actions[i] == KEEP) {
      int out = offsets[i];

      new_x[out] = ptcls.x[i];
      new_y[out] = ptcls.y[i];
      for (int k = 0; k < num_dp; ++k) out_dp_ptrs[k][out] = in_dp_ptrs[k][i];
      for (int k = 0; k < num_ip; ++k) out_ip_ptrs[k][out] = in_ip_ptrs[k][i];
    } else if (actions[i] == SPLIT) {
      int out1 = offsets[i];
      int out2 = offsets[i] + 1;
      int out3 = offsets[i] + 2;
      int out4 = offsets[i] + 3;
      idx_t level = in_ip_ptrs[ip_idx[e_level]][i];
      
      double px = ptcls.x[i];
      double py = ptcls.y[i];
      double r_i = std::sqrt(in_dp_ptrs[dp_idx[e_Ap]][i]);
      double offset_dist = r_i / (2.0);

      double x[4] = {px + offset_dist, px + offset_dist, px - offset_dist, px - offset_dist};
      double y[4] = {py + offset_dist, py - offset_dist, py + offset_dist, py - offset_dist};

      // If horizontal split goes out of bounds, try vertical
      for (int k = 0; k < 4; ++k) {
        if (x[k] < 1e-10 || x[k] > ncols * hx - 1e-10)
          x[k] = std::max(1e-10, std::min(ncols * hx - 1e-10, x[k]));
        if (y[k] < 1e-10 || y[k] > nrows * hy - 1e-10)
          y[k] = std::max(1e-10, std::min(nrows * hy - 1e-10, y[k]));
      }

      new_x[out1] = x[0]; new_y[out1] = y[0];
      new_x[out2] = x[1]; new_y[out2] = y[1];
      new_x[out3] = x[2]; new_y[out3] = y[2];
      new_x[out4] = x[3]; new_y[out4] = y[3];

      // Copy all properties to children (halving extensive ones)
      for (int k = 0; k < num_dp; ++k) {
        double val = in_dp_ptrs[k][i] / (is_ext[k] ? 4.0 : 1.0);
        out_dp_ptrs[k][out1] = val;
        out_dp_ptrs[k][out2] = val;
        out_dp_ptrs[k][out3] = val;
        out_dp_ptrs[k][out4] = val;
      }
      for (int k = 0; k < num_ip; ++k) {
        out_ip_ptrs[k][out1] = in_ip_ptrs[k][i];
        out_ip_ptrs[k][out2] = in_ip_ptrs[k][i];
        out_ip_ptrs[k][out3] = in_ip_ptrs[k][i];
        out_ip_ptrs[k][out4] = in_ip_ptrs[k][i];
      }

      // helper functions to update specific physical properties
      auto set_dp = [&](int e, double v1, double v2, double v3, double v4) { if (dp_idx[e] >= 0) { out_dp_ptrs[dp_idx[e]][out1] = v1; out_dp_ptrs[dp_idx[e]][out2] = v2; out_dp_ptrs[dp_idx[e]][out3] = v3; out_dp_ptrs[dp_idx[e]][out4] = v4; } };
      auto set_ip = [&](int e, idx_t v1, idx_t v2, idx_t v3, idx_t v4)   { if (ip_idx[e] >= 0) { out_ip_ptrs[ip_idx[e]][out1] = v1; out_ip_ptrs[ip_idx[e]][out2] = v2; out_ip_ptrs[ip_idx[e]][out3] = v3; out_ip_ptrs[ip_idx[e]][out4] = v4; } };

      set_dp(e_xp, x[0], x[1], x[2], x[3]);
      set_dp(e_yp, y[0], y[1], y[2], y[3]);
      
      if (dp_idx[e_mom_px] >= 0) {
        double p1 = out_dp_ptrs[dp_idx[e_Mp]][out1] * out_dp_ptrs[dp_idx[e_vpx]][out1];
        double p2 = out_dp_ptrs[dp_idx[e_Mp]][out2] * out_dp_ptrs[dp_idx[e_vpx]][out2];
        double p3 = out_dp_ptrs[dp_idx[e_Mp]][out3] * out_dp_ptrs[dp_idx[e_vpx]][out3];
        double p4 = out_dp_ptrs[dp_idx[e_Mp]][out4] * out_dp_ptrs[dp_idx[e_vpx]][out4];
        set_dp(e_mom_px, p1, p2, p3, p4);
      }
      if (dp_idx[e_mom_py] >= 0) {
        double p1 = out_dp_ptrs[dp_idx[e_Mp]][out1] * out_dp_ptrs[dp_idx[e_vpy]][out1];
        double p2 = out_dp_ptrs[dp_idx[e_Mp]][out2] * out_dp_ptrs[dp_idx[e_vpy]][out2];
        double p3 = out_dp_ptrs[dp_idx[e_Mp]][out3] * out_dp_ptrs[dp_idx[e_vpy]][out3];
        double p4 = out_dp_ptrs[dp_idx[e_Mp]][out4] * out_dp_ptrs[dp_idx[e_vpy]][out4];
        set_dp(e_mom_py, p1, p2, p3, p4);
      }

      set_ip(e_level, level - 1, level - 1, level -1, level -1);
      set_ip(e_label, -1, -1, -1, -1);

    } else if (actions[i] == MERGE_PRIMARY) {
      int j = merge_partner[i];
      int out = offsets[i];
      idx_t level = in_ip_ptrs[ip_idx[e_level]][i];
      
      double M1 = in_dp_ptrs[dp_idx[e_Mp]][i];
      double M2 = in_dp_ptrs[dp_idx[e_Mp]][j];
      double Mtot = M1 + M2;

      new_x[out] = (M1 * ptcls.x[i] + M2 * ptcls.x[j]) / Mtot;
      new_y[out] = (M1 * ptcls.y[i] + M2 * ptcls.y[j]) / Mtot;

      for (int k = 0; k < num_dp; ++k) {
        if (is_ext[k]) {
          out_dp_ptrs[k][out] = in_dp_ptrs[k][i] + in_dp_ptrs[k][j];
        } else {
          out_dp_ptrs[k][out] = (M1 * in_dp_ptrs[k][i] + M2 * in_dp_ptrs[k][j]) / Mtot;
        }
      }
      for (int k = 0; k < num_ip; ++k) {
        out_ip_ptrs[k][out] = in_ip_ptrs[k][i];
      }

      if (dp_idx[e_xp] >= 0) out_dp_ptrs[dp_idx[e_xp]][out] = new_x[out];
      if (dp_idx[e_yp] >= 0) out_dp_ptrs[dp_idx[e_yp]][out] = new_y[out];
      if (dp_idx[e_mom_px] >= 0) out_dp_ptrs[dp_idx[e_mom_px]][out] = out_dp_ptrs[dp_idx[e_Mp]][out] * out_dp_ptrs[dp_idx[e_vpx]][out];
      if (dp_idx[e_mom_py] >= 0) out_dp_ptrs[dp_idx[e_mom_py]][out] = out_dp_ptrs[dp_idx[e_Mp]][out] * out_dp_ptrs[dp_idx[e_vpy]][out];
      if (ip_idx[e_level] >= 0) out_ip_ptrs[ip_idx[e_level]][out] = level + 1;
    }
  }

  ptcls.x = std::move(new_x);
  ptcls.y = std::move(new_y);
  for (int k = 0; k < num_dp; ++k) ptcls.dprops[dp_keys[k]] = std::move(new_dp[k]);
  for (int k = 0; k < num_ip; ++k) ptcls.iprops[ip_keys[k]] = std::move(new_ip[k]);
  ptcls.num_particles = exact_est;

  auto cc_after = compute_conservation(ptcls);
  print_conservation("AFTER  merge/split", cc_after);
}

template <typename idx_t>
inline void adaptive_merge_split(particles_t &ptcls, const ms_config &cfg) {

  std::vector<char> is_exterior;
  mark_exterior_cells(ptcls, is_exterior);

  // BFS distance to fluid boundary
  std::vector<double> cell_dist;
  compute_boundary_distance(ptcls, is_exterior, cell_dist);

  // per-particle ELFS
  std::vector<double> elfs_values;
  compute_elfs(ptcls, cell_dist, is_exterior, elfs_values);

  // decide actions
  std::vector<action_t> actions;
  std::vector<int> merge_partner;
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
