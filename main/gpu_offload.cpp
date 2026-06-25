#include "merge_split_ops.h"
#include "mpm_data.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <particles.h>
#include <quadgrid_cpp.h>
#include <timer.h>

#define gind2row quadgrid_t<std::vector<double>>::gind2row
#define gind2col quadgrid_t<std::vector<double>>::gind2col
#define shp quadgrid_t<std::vector<double>>::shp
#define shg quadgrid_t<std::vector<double>>::shg
#define gt quadgrid_t<std::vector<double>>::gt

using idx_t = quadgrid_t<std::vector<double>>::idx_t;
cdf::timer::timer_t my_timer{};

int main() {
  DATA data("DATA.json");
  bool WRITE_OUTPUT = true;

  std::ofstream err_file("conservation_errors_gpu.csv");
    err_file << "time,err_mass,err_mom\n";

  quadgrid_t<std::vector<double>> grid;
  grid.set_sizes(data.Ney, data.Nex, data.hx, data.hy);

  idx_t num_particles = data.x.size();
  particles_t ptcls(num_particles, {"label", "level"},
                    {"Mp",     "Ap",     "vpx",    "vpy",      "mom_px",
                     "mom_py", "hp",     "Vp",     "F_ext_px", "F_ext_py",
                     "apx",    "apy",    "F_11",   "F_12",     "F_21",
                     "F_22",   "vpx_dx", "vpx_dy", "vpy_dx",   "vpy_dy",
                     "Fb_x",   "Fb_y",   "hpZ",    "dZxp",     "dZyp",
                     "Zp",     "xp",     "yp",     "Fric_px",  "Fric_py",
                    "Fpx", "Fpy", "vpxL", "vpyL", "H"},
                    grid, data.x, data.y);
  ptcls.dprops["Mp"] = data.Mp;
  ptcls.dprops["Ap"] = data.Ap;
  ptcls.dprops["vpx"] = data.vpx;
  ptcls.dprops["vpy"] = data.vpy;
  ptcls.dprops["mom_px"] = data.mom_px;
  ptcls.dprops["mom_py"] = data.mom_py;
  ptcls.dprops["hp"] = data.hp;
  ptcls.dprops["Vp"] = data.Vp;
  ptcls.dprops["xp"] = data.x;
  ptcls.dprops["yp"] = data.y;

#pragma omp parallel for schedule(static)
  for (idx_t ip = 0; ip < num_particles; ++ip) {
    ptcls.dprops["vpx_dx"][ip] = 0.0;
    ptcls.dprops["vpx_dy"][ip] = 0.0;
    ptcls.dprops["vpy_dx"][ip] = 0.0;
    ptcls.dprops["vpy_dy"][ip] = 0.0;
    ptcls.dprops["apx"][ip] = 0.0;
    ptcls.dprops["apy"][ip] = 0.0;
    ptcls.dprops["Fb_x"][ip] = 0.0;
    ptcls.dprops["Fb_y"][ip] = 0.0;
    ptcls.dprops["dZxp"][ip] = 0.0;
    ptcls.dprops["dZyp"][ip] = 0.0;
    ptcls.dprops["Zp"][ip] = 0.0;
    ptcls.dprops["hpZ"][ip] = 0.0;
    ptcls.dprops["Fpx"][ip] = 0.0;
    ptcls.dprops["Fpy"][ip] = 0.0;
    ptcls.dprops["vpxL"][ip] = 0.0;
    ptcls.dprops["vpyL"][ip] = 0.0;
  }

  std::iota(ptcls.iprops["label"].begin(), ptcls.iprops["label"].end(), 0);
  std::fill(ptcls.iprops["level"].begin(), ptcls.iprops["level"].end(), 0);

  double t = 0.0;
  double dt;
  double cel; // max velocity
  double fric_ang = data.phi * M_PI / 180.; // 37. * M_PI / 180.;

  // for each property we have a vector with the values associated to each node
  // of the grid
  std::map<std::string, std::vector<double>> vars{
      {"Mv", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"mom_vx", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"mom_vy", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"F_ext_vx", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"F_ext_vy", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"F_int_vx", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"F_int_vy", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"Fric_x", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"Fric_y", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"div_v", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"avx", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"avy", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"vvx", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"vvy", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"Z", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"dZdx", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"dZdy", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"Ftot_vx", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"Ftot_vy", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"HV", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"FPxv", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"FPyv", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"vvxL", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"vvyL", std::vector<double>(grid.num_global_nodes(), 0.0)}},

      Plotvars{{"rho_v", std::vector<double>(grid.num_global_nodes(), 0.0)},
               {"avx", std::vector<double>(grid.num_global_nodes(), 0.0)},
               {"avy", std::vector<double>(grid.num_global_nodes(), 0.0)},
               {"vvx", std::vector<double>(grid.num_global_nodes(), 0.0)},
               {"vvy", std::vector<double>(grid.num_global_nodes(), 0.0)}};

  vars["Z"] = data.Z;
  vars["dZdx"] = data.dZdx;
  vars["dZdy"] = data.dZdy;

  ptcls.g2p(vars, std::vector<std::string>{"Z"},
            std::vector<std::string>{"Zp"});
  ptcls.g2p(vars, std::vector<std::string>{"dZdx", "dZdy"},
            std::vector<std::string>{"dZxp", "dZyp"});

  double eq_level = data.eq_level;  // letto da DATA.json se disponibile
  for (idx_t ip = 0; ip < num_particles; ++ip) {
      double diff = eq_level - ptcls.dprops["Zp"][ip];
      ptcls.dprops["H"][ip] = (eq_level > 0 && diff > 0) ? diff : 0.0;
  }

#pragma omp parallel for schedule(static)
  for (idx_t ip = 0; ip < num_particles; ++ip) {
    ptcls.dprops["hpZ"][ip] = ptcls.dprops["hp"][ip] + ptcls.dprops["Zp"][ip];

    double hp_val = ptcls.dprops["hp"][ip];
    double H_val = ptcls.dprops["H"][ip];
    double h_corr = hp_val > 1e-10 ? hp_val - H_val*H_val/hp_val : 0.0;
    // init forces
    ptcls.dprops["F_11"][ip] = 0.5 * data.rho * data.g * h_corr;
    ptcls.dprops["F_12"][ip] = 0.0;
    ptcls.dprops["F_21"][ip] = 0.0;
    ptcls.dprops["F_22"][ip] = 0.5 * data.rho * data.g * h_corr;    
  }

  int it = 0;
  ptcls.build_mass();
  grid.vtk_export("GRID_forZ.vts", vars);

  dt = 1.0e-3;

  int np = ptcls.num_particles;

  const int nn = grid.num_global_nodes();
  const int nrows = grid.num_rows();
  const int ncols = grid.num_cols();
  const double hx = grid.hx();
  const double hy = grid.hy();

  const double rho = data.rho; // density
  const double g_c = data.g;   // gravity (?)

  // EXTRACT RAW POINTERS
  double *d_x = ptcls.x.data();
  double *d_y = ptcls.y.data();
  double *d_vpx = ptcls.dprops["vpx"].data();
  double *d_vpy = ptcls.dprops["vpy"].data();
  double *d_apx = ptcls.dprops["apx"].data();
  double *d_apy = ptcls.dprops["apy"].data();
  double *d_hp = ptcls.dprops["hp"].data();
  double *d_Ap = ptcls.dprops["Ap"].data();
  double *d_Vp = ptcls.dprops["Vp"].data();
  double *d_Mp = ptcls.dprops["Mp"].data();
  double *d_mom_px = ptcls.dprops["mom_px"].data();
  double *d_mom_py = ptcls.dprops["mom_py"].data();
  double *d_F11 = ptcls.dprops["F_11"].data();
  double *d_F12 = ptcls.dprops["F_12"].data();
  double *d_F21 = ptcls.dprops["F_21"].data();
  double *d_F22 = ptcls.dprops["F_22"].data();
  double *d_vpx_dx = ptcls.dprops["vpx_dx"].data();
  double *d_vpx_dy = ptcls.dprops["vpx_dy"].data();
  double *d_vpy_dx = ptcls.dprops["vpy_dx"].data();
  double *d_vpy_dy = ptcls.dprops["vpy_dy"].data();
  double *d_Fb_x = ptcls.dprops["Fb_x"].data();
  double *d_Fb_y = ptcls.dprops["Fb_y"].data();
  double *d_Fric_px = ptcls.dprops["Fric_px"].data();
  double *d_Fric_py = ptcls.dprops["Fric_py"].data();
  double *d_dZxp = ptcls.dprops["dZxp"].data();
  double *d_dZyp = ptcls.dprops["dZyp"].data();
  double *d_Zp = ptcls.dprops["Zp"].data();
  double *d_hpZ = ptcls.dprops["hpZ"].data();
  int *d_p2g = ptcls.ptcl_to_grd.data();

  double *d_Mv = vars["Mv"].data();
  double *d_mom_vx = vars["mom_vx"].data();
  double *d_mom_vy = vars["mom_vy"].data();
  double *d_F_ext_vx = vars["F_ext_vx"].data();
  double *d_F_ext_vy = vars["F_ext_vy"].data();
  double *d_F_int_vx = vars["F_int_vx"].data();
  double *d_F_int_vy = vars["F_int_vy"].data();
  double *d_Ftot_vx = vars["Ftot_vx"].data();
  double *d_Ftot_vy = vars["Ftot_vy"].data();
  double *d_avx = vars["avx"].data();
  double *d_avy = vars["avy"].data();
  double *d_vvx = vars["vvx"].data();
  double *d_vvy = vars["vvy"].data();
  double *d_Fric_x = vars["Fric_x"].data();
  double *d_Fric_y = vars["Fric_y"].data();
  double *d_dZdx = vars["dZdx"].data();
  double *d_dZdy = vars["dZdy"].data();
  double *d_Z = vars["Z"].data();
  double* d_Fpx = ptcls.dprops["Fpx"].data();
  double* d_Fpy = ptcls.dprops["Fpy"].data();
  double* d_vpxL = ptcls.dprops["vpxL"].data();
  double* d_vpyL = ptcls.dprops["vpyL"].data();
  double* d_FPxv = vars["FPxv"].data();
  double* d_FPyv = vars["FPyv"].data();
  double* d_vvxL = vars["vvxL"].data();
  double* d_vvyL = vars["vvyL"].data();
  double* d_H = ptcls.dprops["H"].data();
  double* d_HV = vars["HV"].data();
  double* d_pv_rho = Plotvars["rho_v"].data();
  double* d_pv_vvx = Plotvars["vvx"].data();
  double* d_pv_vvy = Plotvars["vvy"].data();
  double* d_pv_avx = Plotvars["avx"].data();
  double* d_pv_avy = Plotvars["avy"].data();


  const double A_coeff = 3.0 / 2.0;
  const double C_coeff = 65.0 / 32.0;

  double mu = data.mu;
  double tau_Y = data.tauy;

  const double cc = data.BINGHAM_ON;
  const double fric_on = data.FRICTION_ON;
  const double xi_coeff = data.xi;
  const int bc_flag = data.BC_FLAG;
  const int ms_on = data.MERGE_SPLIT_ON;
  const double tan_fa = std::tan(fric_ang);

  // Color index arrays (rebuilt each iteration)
  std::vector<int> color_indices(np);
  int ncells = nrows * ncols;
  std::vector<int> cell_start(ncells + 1, 0);
  std::vector<int> cell_ptcls(np);
  std::vector<int> color_cell_indices(ncells);
  int color_cell_offsets[5] = {0};

  std::ofstream clamp_file("clamp_log.csv");
  clamp_file << "time,n_vcap,max_vmag,n_scfloor,min_sc,n_hpfloor,min_hp\n";

  // TIME LOOP
  while (t < data.T) {
    // timer
    my_timer.tic("update dt");
    double max_vx = *std::max_element(d_vpx, d_vpx + np);
    double min_vx = *std::min_element(d_vpx, d_vpx + np);
    max_vx = std::max(std::abs(max_vx), std::abs(min_vx));
    double max_vy = *std::max_element(d_vpy, d_vpy + np);
    double min_vy = *std::min_element(d_vpy, d_vpy + np);
    max_vy = std::max(std::abs(max_vy), std::abs(min_vy));
    double hmax = *std::max_element(d_hp, d_hp + np);
    double max_vel = std::max(std::sqrt(g_c * hmax) + max_vx,
                              std::sqrt(g_c * hmax) + max_vy);
    cel = std::abs(max_vel);

    if (it > 0)
      dt = data.CFL * std::min(data.hx, data.hy) / (1e-2 + cel); // to avoid cell-crossing
    std::cout << "time = " << t << "  dt = " << dt << std::endl;
    my_timer.toc("update dt");

    // csv
    it++;
    if(WRITE_OUTPUT==true){
      my_timer.tic("save csv");
      std::string filename = "nc_particles_" + std::to_string(it) + ".csv";
      if (t >= 0.0) {
         if(it%10==0) {
          std::ofstream OF(filename.c_str());
          ptcls.print<particles_t::output_format::csv>(OF);
          OF.close();
         }
      }
      my_timer.toc("save csv");
    }

    my_timer.tic("reorder");
    ptcls.init_particle_mesh(); // ref -> src/particles.cpp
    d_p2g = ptcls.ptcl_to_grd.data();
    my_timer.toc("reorder");

    // merge-split
    /*ms_config ms_cfg;
    if (ms_on && it % ms_cfg.call_interval == 0 && it > 0) {
      my_timer.tic("merge_split");
      adaptive_merge_split<idx_t>(ptcls, ms_cfg);

      np = ptcls.num_particles;

      d_x = ptcls.x.data();
      d_y = ptcls.y.data();
      d_vpx = ptcls.dprops["vpx"].data();
      d_vpy = ptcls.dprops["vpy"].data();
      d_apx = ptcls.dprops["apx"].data();
      d_apy = ptcls.dprops["apy"].data();
      d_hp = ptcls.dprops["hp"].data();
      d_Ap = ptcls.dprops["Ap"].data();
      d_Vp = ptcls.dprops["Vp"].data();
      d_Mp = ptcls.dprops["Mp"].data();
      d_mom_px = ptcls.dprops["mom_px"].data();
      d_mom_py = ptcls.dprops["mom_py"].data();
      d_F11 = ptcls.dprops["F_11"].data();
      d_F12 = ptcls.dprops["F_12"].data();
      d_F21 = ptcls.dprops["F_21"].data();
      d_F22 = ptcls.dprops["F_22"].data();
      d_vpx_dx = ptcls.dprops["vpx_dx"].data();
      d_vpx_dy = ptcls.dprops["vpx_dy"].data();
      d_vpy_dx = ptcls.dprops["vpy_dx"].data();
      d_vpy_dy = ptcls.dprops["vpy_dy"].data();
      d_Fb_x = ptcls.dprops["Fb_x"].data();
      d_Fb_y = ptcls.dprops["Fb_y"].data();
      d_Fric_px = ptcls.dprops["Fric_px"].data();
      d_Fric_py = ptcls.dprops["Fric_py"].data();
      d_dZxp = ptcls.dprops["dZxp"].data();
      d_dZyp = ptcls.dprops["dZyp"].data();
      d_Zp = ptcls.dprops["Zp"].data();
      d_hpZ = ptcls.dprops["hpZ"].data();
      d_Fpx  = ptcls.dprops["Fpx"].data();
      d_Fpy  = ptcls.dprops["Fpy"].data();
      d_vpxL = ptcls.dprops["vpxL"].data();
      d_vpyL = ptcls.dprops["vpyL"].data();
      d_H = ptcls.dprops["H"].data();

      color_indices.resize(np);
      cell_start.resize(ncells + 1);
      cell_ptcls.resize(np);

      // Clamp particelle dopo merge/split
      double eps = 1e-10;
      double Lx = ncols * hx;
      double Ly = nrows * hy;
      for (int ip = 0; ip < np; ip++) {
        if (d_x[ip] < eps) d_x[ip] = eps;
        if (d_x[ip] > Lx - eps) d_x[ip] = Lx - eps;
        if (d_y[ip] < eps) d_y[ip] = eps;
        if (d_y[ip] > Ly - eps) d_y[ip] = Ly - eps;
      }

      ptcls.init_particle_mesh();
      d_p2g = ptcls.ptcl_to_grd.data();
      my_timer.toc("merge_split");

    }*/

    my_timer.tic("build_colors");
    {
        // CSR: celle → particelle
        std::vector<int> cell_count(ncells, 0);
        for (int ip = 0; ip < np; ip++)
            cell_count[d_p2g[ip]]++;

        cell_start[0] = 0;
        for (int i = 0; i < ncells; i++)
            cell_start[i + 1] = cell_start[i] + cell_count[i];

        cell_ptcls.resize(np);
        std::fill(cell_count.begin(), cell_count.end(), 0);
        for (int ip = 0; ip < np; ip++) {
            int cell = d_p2g[ip];
            cell_ptcls[cell_start[cell] + cell_count[cell]++] = ip;
        }

        // Colori per celle (non per particelle)
        int ccounts[4] = {0};
        for (int cell = 0; cell < ncells; cell++) {
            int r = cell % nrows;
            int c = cell / nrows;
            ccounts[(r % 2) * 2 + (c % 2)]++;
        }
        color_cell_offsets[0] = 0;
        for (int c = 0; c < 4; c++)
            color_cell_offsets[c + 1] = color_cell_offsets[c] + ccounts[c];

        int cpos[4] = {color_cell_offsets[0], color_cell_offsets[1],
                      color_cell_offsets[2], color_cell_offsets[3]};
        for (int cell = 0; cell < ncells; cell++) {
            int r = cell % nrows;
            int c = cell / nrows;
            int col = (r % 2) * 2 + (c % 2);
            color_cell_indices[cpos[col]++] = cell;
        }
    }
    my_timer.toc("build_colors");

    // step 0 — reset nodal arrays on host
    my_timer.tic("step 0");
    for (auto &v : vars)
      v.second.assign(v.second.size(), 0.0);
    for (auto &v : Plotvars)
      v.second.assign(v.second.size(), 0.0);
    vars["Z"] = data.Z;
    vars["dZdx"] = data.dZdx;
    vars["dZdy"] = data.dZdy;
    my_timer.toc("step 0");

    std::cout << "  Entering gpu_block..." << std::flush;
    my_timer.tic("gpu_block");
    {


      int* d_cstart = cell_start.data();
      int* d_cptcls = cell_ptcls.data();
      int* d_ccidx  = color_cell_indices.data();

#pragma omp target data map(to : d_p2g[0 : np], d_Mp[0 : np], d_cstart[0:ncells+1], d_cptcls[0:np], d_ccidx[0:ncells],  \
                                d_Z[0 : nn], d_dZdx[0 : nn], d_dZdy[0 : nn])   \
    map(tofrom : d_x[0 : np], d_y[0 : np], d_vpx[0 : np], d_vpy[0 : np],       \
            d_apx[0 : np], d_apy[0 : np], d_hp[0 : np], d_Ap[0 : np],          \
            d_Vp[0 : np], d_mom_px[0 : np], d_mom_py[0 : np],                  \
            d_vpx_dx[0 : np], d_vpx_dy[0 : np], d_vpy_dx[0 : np],              \
            d_vpy_dy[0 : np], d_Fb_x[0 : np], d_Fb_y[0 : np],                  \
            d_Fric_px[0 : np], d_Fric_py[0 : np], d_F11[0 : np],               \
            d_F12[0 : np], d_F21[0 : np], d_F22[0 : np], d_dZxp[0 : np],       \
            d_dZyp[0 : np], d_Zp[0 : np], d_hpZ[0 : np], d_Mv[0 : nn],         \
            d_mom_vx[0 : nn], d_mom_vy[0 : nn], d_F_ext_vx[0 : nn],            \
            d_F_ext_vy[0 : nn], d_F_int_vx[0 : nn], d_F_int_vy[0 : nn],        \
            d_Ftot_vx[0 : nn], d_Ftot_vy[0 : nn], d_avx[0 : nn],               \
            d_avy[0 : nn], d_vvx[0 : nn], d_vvy[0 : nn], d_Fric_x[0 : nn],     \
            d_Fric_y[0 : nn], d_Fpx[0 : np], d_Fpy[0 : np], d_vpxL[0 : np],    \
            d_vpyL[0 : np],  d_FPxv[0 : nn], d_FPyv[0 : nn], d_vvxL[0 : nn],  d_vvyL[0 : nn], d_H[0:np], \
          d_pv_rho[0:nn], d_pv_vvx[0:nn], d_pv_vvy[0:nn], d_pv_avx[0:nn], d_pv_avy[0:nn],d_HV[0:nn] )
      {


// STEP 1a: P2G mass + momentum — per-cell coloring, NO atomic
      for (int color = 0; color < 4; color++) {
        int cstart = color_cell_offsets[color];
        int cend   = color_cell_offsets[color + 1];
        if (cend == cstart) continue;

        #pragma omp target teams distribute parallel for
        for (int ic = cstart; ic < cend; ic++) {
          int cell = d_ccidx[ic];
          int r = cell % nrows;
          int c = cell / nrows;

          for (int jp = d_cstart[cell]; jp < d_cstart[cell + 1]; jp++) {
            int ip = d_cptcls[jp];
            double xx = d_x[ip], yy = d_y[ip];
            for (int in = 0; in < 4; in++) {
              double N  = shp(xx, yy, in, c, r, hx, hy);
              int  nidx = gt(in, c, r, nrows);
              d_Mv[nidx]     += N * d_Mp[ip];
              d_mom_vx[nidx] += N * d_mom_px[ip];
              d_mom_vy[nidx] += N * d_mom_py[ip];
            }
          }
        }
      }        

// STEP 1b: G2PD for Z gradients
#pragma omp target teams distribute parallel for
        for (int ip = 0; ip < np; ip++) {
          double xx = d_x[ip], yy = d_y[ip];
          int ci = d_p2g[ip];
          int r = gind2row(ci, nrows);
          int c = gind2col(ci, nrows);
          double dzx = 0.0, dzy = 0.0;
          for (int in = 0; in < 4; in++) {
            double Nx = shg(xx, yy, 0, in, c, r, hx, hy);
            double Ny = shg(xx, yy, 1, in, c, r, hx, hy);
            int nidx = gt(in, c, r, nrows);
            dzx += Nx * d_Z[nidx];
            dzy += Ny * d_Z[nidx];
          }
          d_dZxp[ip] = dzx;
          d_dZyp[ip] = dzy;
        }

// Gravity force along slope
#pragma omp target teams distribute parallel for
for (int ip = 0; ip < np; ip++) {
    double ratio = d_hp[ip] > 1e-10 ? d_H[ip] / d_hp[ip] : 0.0;
    d_Fpx[ip] = -g_c * d_Mp[ip] * (1.0 - ratio) * d_dZxp[ip];
    d_Fpy[ip] = -g_c * d_Mp[ip] * (1.0 - ratio) * d_dZyp[ip];
}

// STEP 2a: Fric_px/py per-particle
#pragma omp target teams distribute parallel for
        for (int ip = 0; ip < np; ip++) {
          // TODO: why no friction?
          d_Fric_px[ip] = d_Ap[ip] * d_Fb_x[ip];
          d_Fric_py[ip] = d_Ap[ip] * d_Fb_y[ip];
        }

// STEP 2b: P2G friction — per-cell coloring, NO atomic
      for (int color = 0; color < 4; color++) {
        int cstart = color_cell_offsets[color];
        int cend   = color_cell_offsets[color + 1];
        if (cend == cstart) continue;

        #pragma omp target teams distribute parallel for
        for (int ic = cstart; ic < cend; ic++) {
          int cell = d_ccidx[ic];
          int r = cell % nrows;
          int c = cell / nrows;

          for (int jp = d_cstart[cell]; jp < d_cstart[cell + 1]; jp++) {
            int ip = d_cptcls[jp];
            double xx = d_x[ip], yy = d_y[ip];
            for (int in = 0; in < 4; in++) {
              double N  = shp(xx, yy, in, c, r, hx, hy);
              int  nidx = gt(in, c, r, nrows);
              d_FPxv[nidx]  += N * d_Fpx[ip];
              d_FPyv[nidx]  += N * d_Fpy[ip];
              d_Fric_x[nidx] += N * d_Fric_px[ip];
              d_Fric_y[nidx] += N * d_Fric_py[ip];
            }
          }
        }
      }        

// STEP 2c: F_ext per-node
#pragma omp target teams distribute parallel for
        for (int iv = 0; iv < nn; iv++) {
          d_F_ext_vx[iv] = d_FPxv[iv] + d_Fric_x[iv];
          d_F_ext_vy[iv] = d_FPyv[iv] + d_Fric_y[iv];
        }

        // STEP 3a: P2GD internal forces — per-cell coloring, NO atomic
      for (int color = 0; color < 4; color++) {
        int cstart = color_cell_offsets[color];
        int cend   = color_cell_offsets[color + 1];
        if (cend == cstart) continue;

        #pragma omp target teams distribute parallel for
        for (int ic = cstart; ic < cend; ic++) {
          int cell = d_ccidx[ic];
          int r = cell % nrows;
          int c = cell / nrows;

          for (int jp = d_cstart[cell]; jp < d_cstart[cell + 1]; jp++) {
            int ip = d_cptcls[jp];
            double xx = d_x[ip], yy = d_y[ip];
            double vp = d_Vp[ip];
            for (int in = 0; in < 4; in++) {
              double Nx = shg(xx, yy, 0, in, c, r, hx, hy);
              double Ny = shg(xx, yy, 1, in, c, r, hx, hy);
              int  nidx = gt(in, c, r, nrows);
              d_F_int_vx[nidx] += (Nx * d_F11[ip] + Ny * d_F12[ip]) * vp;
              d_F_int_vy[nidx] += (Nx * d_F21[ip] + Ny * d_F22[ip]) * vp;
            }
          }
        }
      }

// === conservazione (Eq.44): DENTRO il target data,
//     dopo il P2G, PRIMA dello step 3b (mom_vx += dt*Ftot) ===
double sum_Mp = 0.0, sum_mpx = 0.0, sum_mpy = 0.0;
#pragma omp target teams distribute parallel for \
        reduction(+:sum_Mp,sum_mpx,sum_mpy)
for (int ip = 0; ip < np; ++ip) {
    sum_Mp  += d_Mp[ip];
    sum_mpx += d_mom_px[ip];
    sum_mpy += d_mom_py[ip];
}
double sum_Mv = 0.0, sum_mvx = 0.0, sum_mvy = 0.0;
#pragma omp target teams distribute parallel for \
        reduction(+:sum_Mv,sum_mvx,sum_mvy)
for (int in = 0; in < nn; ++in) {     // nn = numero nodi (usa il tuo nome)
    sum_Mv  += d_Mv[in];
    sum_mvx += d_mom_vx[in];
    sum_mvy += d_mom_vy[in];
}
// codice host (gira normalmente dentro la regione target-data)
double err_mass = std::abs(sum_Mp - sum_Mv) / sum_Mp;
double norm_mom = std::max(1.0, std::max(std::abs(sum_mpx), std::abs(sum_mpy)));
double err_mom  = std::max(std::abs(sum_mpx - sum_mvx),
                           std::abs(sum_mpy - sum_mvy)) / norm_mom;
err_file << t << "," << err_mass << "," << err_mom << "\n";      

// STEP 3b: Ftot + momentum per-node
#pragma omp target teams distribute parallel for
        for (int iv = 0; iv < nn; iv++) {
          d_Ftot_vx[iv] = d_F_ext_vx[iv] + d_F_int_vx[iv];
          d_Ftot_vy[iv] = d_F_ext_vy[iv] + d_F_int_vy[iv];
          d_mom_vx[iv] += dt * d_Ftot_vx[iv];
          d_mom_vy[iv] += dt * d_Ftot_vy[iv];
        }

// STEP 4: nodal vel/acc
#pragma omp target teams distribute parallel for
        for (int iv = 0; iv < nn; iv++) {
          bool active = d_Mv[iv] > 1e-2;
          d_avx[iv] = active ? d_Ftot_vx[iv] / d_Mv[iv] : 0.0;
          d_avy[iv] = active ? d_Ftot_vy[iv] / d_Mv[iv] : 0.0;
          d_vvx[iv] = active ? d_mom_vx[iv] / d_Mv[iv] : 0.0;
          d_vvy[iv] = active ? d_mom_vy[iv] / d_Mv[iv] : 0.0;
          d_vvxL[iv] = active ? dt * d_avx[iv] + d_vvx[iv] : 0.0;
          d_vvyL[iv] = active ? dt * d_avy[iv] + d_vvy[iv] : 0.0;
        }

// STEP 5: boundary conditions
if (bc_flag){
  #pragma omp target teams distribute parallel for
            for (int iv = 0; iv < nn; iv++) {
              int r = iv % (nrows + 1);
              int c = iv / (nrows + 1);
              // TODO: why not to use the method defined in quadgrid_cpp_imp.h??
              if (c == 0 || c == 1 || c == ncols - 1 || c == ncols) {
                d_mom_vx[iv] = 0.0;
                d_vvx[iv] = 0.0;
                d_vvxL[iv] = 0.0;
                d_avx[iv] = 0.0;
              }
              if (r == 0 || r == 1 || r == nrows - 1 || r == nrows) {
                d_mom_vy[iv] = 0.0;
                d_vvy[iv] = 0.0;
                d_vvyL[iv] = 0.0;
                d_avy[iv] = 0.0;
              }
            }
}

int    n_vcap = 0;
        double max_vmag = 0.0;

// STEP 6: G2P + velocity/position update
#pragma omp target teams distribute parallel for \
      reduction(+:n_vcap) reduction(max:max_vmag)
        for (int ip = 0; ip < np; ip++) {
          double xx = d_x[ip], yy = d_y[ip];
          int ci = d_p2g[ip];
          int r = gind2row(ci, nrows);
          int c = gind2col(ci, nrows);
          double ax = 0, ay = 0, vxL = 0, vyL = 0;
          for (int in = 0; in < 4; in++) {
            double N = shp(xx, yy, in, c, r, hx, hy);
            int nidx = gt(in, c, r, nrows);
            ax += N * d_avx[nidx];
            ay += N * d_avy[nidx];
            vxL += N * d_vvxL[nidx];
            vyL += N * d_vvyL[nidx];
          }
          
          // Metodo PIC Puro per stabilità (risolve le fratture a strisce)
          d_vpx[ip] = vxL;
          d_vpy[ip] = vyL;
          d_apx[ip] = ax;
          d_apy[ip] = ay;
          
          d_x[ip] += dt * vxL;
          d_y[ip] += dt * vyL;
          
          d_vpxL[ip] = vxL;
          d_vpyL[ip] = vyL;

          double vmag = sqrt(vxL*vxL + vyL*vyL);
          if (vmag > max_vmag) max_vmag = vmag;
          double vcap = 50.0;                 // m/s (una colata reale sta sotto ~20 m/s)
          if (vmag > vcap) {
              d_vpx[ip] = vxL * vcap / vmag;
              d_vpy[ip] = vyL * vcap / vmag;
              n_vcap++;
          }
          // Clamp positions to grid bounds
          /*double eps = 1e-10;
          double Lx = ncols * hx;
          double Ly = nrows * hy;
          if (d_x[ip] < eps) d_x[ip] = eps;
          if (d_x[ip] > Lx - eps) d_x[ip] = Lx - eps;
          if (d_y[ip] < eps) d_y[ip] = eps;
          if (d_y[ip] > Ly - eps) d_y[ip] = Ly - eps;*/

        }

// STEP 7: G2PD + height update
int    n_scfloor = 0, n_hpfloor = 0;
        double min_sc = 1e30, min_hp = 1e30;
#pragma omp target teams distribute parallel for \
        reduction(+:n_scfloor,n_hpfloor) reduction(min:min_sc,min_hp)
        for (int ip = 0; ip < np; ip++) {
          double xx = d_x[ip], yy = d_y[ip];
          int ci = d_p2g[ip];
          int r = gind2row(ci, nrows);
          int c = gind2col(ci, nrows);
          double vxdx = 0, vxdy = 0, vydx = 0, vydy = 0;
          for (int in = 0; in < 4; in++) {
            double Nx = shg(xx, yy, 0, in, c, r, hx, hy);
            double Ny = shg(xx, yy, 1, in, c, r, hx, hy);
            int nidx = gt(in, c, r, nrows);
            vxdx += Nx * d_vvxL[nidx];
            vxdy += Ny * d_vvxL[nidx];
            vydx += Nx * d_vvyL[nidx];
            vydy += Ny * d_vvyL[nidx];
          }
          d_vpx_dx[ip] = vxdx;
          d_vpx_dy[ip] = vxdy;
          d_vpy_dx[ip] = vydx;
          d_vpy_dy[ip] = vydy;

          // sc -> 1.0 + trace of axial deformations
          double sc = 1.0 + dt * (vxdx + vydy);
          if (sc < min_sc) min_sc = sc;
          if (sc < 0.1) { sc = 0.1; n_scfloor++; }
          d_hp[ip] /= sc;
          if (d_hp[ip] < min_hp) min_hp = d_hp[ip];   // hp "naturale" dopo /sc, prima del floor
          if (d_hp[ip] < 1e-2) { d_hp[ip] = 1e-2; n_hpfloor++; }

          if (d_Vp[ip] < 1e-10) d_Vp[ip] = 1e-10;
          d_Ap[ip] = d_Vp[ip] / d_hp[ip];
          
          d_mom_px[ip] = d_vpx[ip] * d_Mp[ip];
          d_mom_py[ip] = d_vpy[ip] * d_Mp[ip];
        }

        clamp_file << t << "," << n_vcap << "," << max_vmag << ","
                   << n_scfloor << "," << min_sc << ","
                   << n_hpfloor << "," << min_hp << "\n";

// FRICTION (Voellmy)
#pragma omp target teams distribute parallel for
        for (int ip = 0; ip < np; ip++) {
          double vx = d_vpx[ip], vy = d_vpy[ip], h = d_hp[ip];
          double nv = sqrt(vx * vx + vy * vy); // ||v||
          if (fric_on > 0 && nv > 1e-10 && xi_coeff > 0) {
            double v2 = vx*vx + vy*vy;
            d_Fb_x[ip] = -fric_on * (rho * g_c * h * tan_fa + rho * g_c * v2 / xi_coeff) * vx / nv;
            d_Fb_y[ip] = -fric_on * (rho * g_c * h * tan_fa + rho * g_c * v2 / xi_coeff) * vy / nv;
          } else {
              d_Fb_x[ip] = 0.0;
              d_Fb_y[ip] = 0.0;
          }
        }

// STEP 8: stress update (Bingham USL)
double max_coeff = 0.0, max_nv = 0.0, max_F11 = 0.0;
double max_dev = 0.0, max_pres = 0.0, max_Dxx = 0.0;
#pragma omp target teams distribute parallel for \
        reduction(max:max_coeff,max_nv,max_F11,max_dev,max_pres,max_Dxx) reduction(min:min_hp)
        for (int ip = 0; ip < np; ip++) {
          double vx = d_vpx[ip], vy = d_vpy[ip], h = d_hp[ip];
          double nv = sqrt(vx * vx + vy * vy); // ||v||
          // 'a' term on paper (2.2)
          //double alf = h > 1e-3 ? (6.0 * mu * nv) / ((h + 0.001) * tau_Y) : 0.0;
          double alf = (h > 1e-3 && tau_Y > 1e-12) ? (6.0 * mu * nv) / ((h + 0.001) * tau_Y) : 0.0;
          double bv = -114.0 / 32.0 - alf;
          double sq = sqrt(bv * bv - 4.0 * A_coeff * C_coeff);

          double z1 = (-bv + sq) / (2.0 * A_coeff);
          double z2 = (-bv - sq) / (2.0 * A_coeff);
          double zz = (fabs(z1 - 0.5) <= 0.5) ? z1 : z2;

          double sxx = d_vpx_dx[ip], sxy = 0.5 * (d_vpx_dy[ip] + d_vpy_dx[ip]),
                 syy = d_vpy_dy[ip];
          double Dxx = sxx, Dxy = sxy, Dyy = syy, Dzz = -(sxx + d_vpy_dy[ip]);
          double Dzx =
              h > 1e-3 ? 0.5 * (3.0 / (2.0 + zz)) * (vx / (h + 0.001)) : 0.0;
          double Dzy =
              h > 1e-3 ? 0.5 * (3.0 / (2.0 + zz)) * (vy / (h + 0.001)) : 0.0;

          double inv2 = 0.5 * (Dxx * Dxx + Dyy * Dyy + Dzz * Dzz) + Dzx * Dzx + Dzy * Dzy + Dxy * Dxy;

          double coeff_max = 5.0e6;                       // viscosità apparente massima (fisica)
          double inv2_floor = (tau_Y / coeff_max) * (tau_Y / coeff_max);  // cap su coeff per tau_y>0
          if (inv2_floor < 1e-9) inv2_floor = 1e-9;                       // cuscinetto minimo per tau_y=0
          double coeff = tau_Y / sqrt(inv2 + inv2_floor) + 2.0 * mu;
          //double coeff = inv2 != 0.0 ? (tau_Y / sqrt(inv2) + 2.0 * mu) : 0.0;
          double h_corr = h > 1e-10 ? h - d_H[ip]*d_H[ip]/h : 0.0;
          d_F11[ip] = -cc * coeff * Dxx + 0.5 * rho * g_c * h_corr;
          d_F22[ip] = -cc * coeff * Dyy + 0.5 * rho * g_c * h_corr;
          d_F12[ip] = -cc * coeff * Dxy;
          d_F21[ip] = -cc * coeff * Dxy;

          if (coeff > max_coeff) max_coeff = coeff;
    if (h < min_hp)        min_hp = h;
    if (nv > max_nv)       max_nv = nv;
    double aF = fabs(d_F11[ip]);
    if (aF > max_F11)      max_F11 = aF;

    double dev  = fabs(cc * coeff * Dxx);              // pezzo deviatorico (Bingham)
double pres = fabs(0.5 * rho * g_c * h_corr);      // pezzo pressione
if (dev  > max_dev)  max_dev  = dev;
if (pres > max_pres) max_pres = pres;
if (fabs(Dxx) > max_Dxx) max_Dxx = fabs(Dxx);
}
printf("max_coeff=%.3e  min_hp=%.3e  max_nv=%.3e  max_F11=%.3e\n",
       max_coeff, min_hp, max_nv, max_F11);

printf("max_dev=%.3e  max_pres=%.3e  max_Dxx=%.3e  max_coeff=%.3e\n",
       max_dev, max_pres, max_Dxx, max_coeff);

// hpZ: G2P for Z→Zp + hpZ = hp + Zp
#pragma omp target teams distribute parallel for
        for (int ip = 0; ip < np; ip++) {
          double xx = d_x[ip], yy = d_y[ip];
          int ci = d_p2g[ip];
          int r = gind2row(ci, nrows);
          int c = gind2col(ci, nrows);
          double zp = 0.0;
          for (int in = 0; in < 4; in++) {
            double N = shp(xx, yy, in, c, r, hx, hy);
            int nidx = gt(in, c, r, nrows);
            zp += N * d_Z[nidx];
          }
          d_Zp[ip] = zp;
          d_hpZ[ip] = d_hp[ip] + zp;
        }

// Plotvars P2G: accumulo — per-cell coloring
for (int color = 0; color < 4; color++) {
    int cstart = color_cell_offsets[color];
    int cend   = color_cell_offsets[color + 1];
    if (cend == cstart) continue;

    #pragma omp target teams distribute parallel for
    for (int ic = cstart; ic < cend; ic++) {
        int cell = d_ccidx[ic];
        int r = cell % nrows;
        int c = cell / nrows;

        for (int jp = d_cstart[cell]; jp < d_cstart[cell + 1]; jp++) {
            int ip = d_cptcls[jp];
            double xx = d_x[ip], yy = d_y[ip];
            for (int in = 0; in < 4; in++) {
                double N = shp(xx, yy, in, c, r, hx, hy);
                int nidx = gt(in, c, r, nrows);
                d_pv_rho[nidx] += N * d_Mp[ip];
                d_pv_vvx[nidx] += N * d_vpx[ip];
                d_pv_vvy[nidx] += N * d_vpy[ip];
                d_pv_avx[nidx] += N * d_apx[ip];
                d_pv_avy[nidx] += N * d_apy[ip];
            }
        }
    }
}

// Plotvars: normalizzazione per massa
#pragma omp target teams distribute parallel for
for (int iv = 0; iv < nn; iv++) {
    double mv = d_pv_rho[iv];
    if (mv > 1e-8) {
        d_pv_vvx[iv] /= mv;
        d_pv_vvy[iv] /= mv;
        d_pv_avx[iv] /= mv;
        d_pv_avy[iv] /= mv;
    }
}

// Step 8b: P2G hp → HV — per-cell coloring
for (int color = 0; color < 4; color++) {
    int cstart = color_cell_offsets[color];
    int cend   = color_cell_offsets[color + 1];
    if (cend == cstart) continue;

    #pragma omp target teams distribute parallel for
    for (int ic = cstart; ic < cend; ic++) {
        int cell = d_ccidx[ic];
        int r = cell % nrows;
        int c = cell / nrows;

        for (int jp = d_cstart[cell]; jp < d_cstart[cell + 1]; jp++) {
            int ip = d_cptcls[jp];
            double xx = d_x[ip], yy = d_y[ip];
            for (int in = 0; in < 4; in++) {
                double N = shp(xx, yy, in, c, r, hx, hy);
                int nidx = gt(in, c, r, nrows);
                d_HV[nidx] += N * d_hp[ip];
            }
        }
    }
}

      } // end target data
    }
    my_timer.toc("gpu_block");

    double KE = 0.0, vmax = 0.0;
    idx_t ip_bad = 0;
    for (idx_t ip = 0; ip < num_particles; ++ip) {
    double v2 = ptcls.dprops["vpx"][ip]*ptcls.dprops["vpx"][ip] +
                ptcls.dprops["vpy"][ip]*ptcls.dprops["vpy"][ip];
    KE += 0.5 * ptcls.dprops["Mp"][ip] * v2;
    if (v2 > vmax) { vmax = v2; ip_bad = ip; }
    }
    printf("KE = %.10e\n", KE);


    std::cout << "  Done." << std::endl;

     // Clamp posizioni prima del Plotvars P2G (CPU)
    {
      double eps = 1e-10;
      double Lx = ncols * hx;
      double Ly = nrows * hy;
      for (int ip = 0; ip < np; ip++) {
        if (d_x[ip] < eps) d_x[ip] = eps;
        if (d_x[ip] > Lx - eps) d_x[ip] = Lx - eps;
        if (d_y[ip] < eps) d_y[ip] = eps;
        if (d_y[ip] > Ly - eps) d_y[ip] = Ly - eps;
      }
    }
    ptcls.init_particle_mesh();

    d_p2g = ptcls.ptcl_to_grd.data();

    // Diagnostica: verifica
    {
      int bad = 0;
      for (int ip = 0; ip < np; ip++)
        if (d_p2g[ip] < 0 || d_p2g[ip] >= ncells) bad++;
      if (bad > 0) std::cout << "  BAD after clamp: " << bad << std::endl;
    }


     std::cout << "  max_vpx=" << *std::max_element(d_vpx, d_vpx+np)
              << " max_hp=" << *std::max_element(d_hp, d_hp+np)
              << " min_hp=" << *std::min_element(d_hp, d_hp+np) << std::endl;
    std::cout << "  Plotvars P2G..." << std::flush;

    std::cout << "done. step8b..." << std::flush;
    
     if (it % 10 == 0 && WRITE_OUTPUT==true) {
      my_timer.tic("save vts");
      std::string filename = "nc_grid_" + std::to_string(it) + ".vts";
      grid.vtk_export(filename.c_str(), vars);
      my_timer.toc("save vts");
    }

    t += dt;
    std::cout << "  vts done." << std::endl;
  } // end time loop

  my_timer.print_report();
  return 0;
}
