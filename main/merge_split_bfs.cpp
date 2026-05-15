#include <queue>
#include <vector>
#include <particles.h>
#include <quadgrid_cpp.h>

/// @brief Mark cells that are genuinely exterior to the fluid.
///
/// Criteria: only empty cells reachable from the grid edge
/// through a path of other empty cells are marked exterior.
/// Internal holes (empty cells surrounded by fluid) are NOT marked,
/// preventing false boundary detection.
void mark_exterior_cells(const particles_t &ptcls,
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
void compute_boundary_distance(const particles_t &ptcls,
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