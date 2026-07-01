#include "merge_split_ops_cmes.h"
#include "mpm_data.h"
#include "gpu_kernels.h"
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <particles.h>
#include <quadgrid_cpp.h>
#include <timer.h>

using idx_t = quadgrid_t<std::vector<double>>::idx_t;
cdf::timer::timer_t my_timer{};

int main () {

  DATA data ("DATA.json");
  bool WRITE_OUTPUT = true;

  std::ofstream err_file ("conservation_errors_gpu.csv");
  err_file << "time,err_mass,err_mom\n";

  // -----------------------------------------------------------------------
  // Grid + particles setup
  // -----------------------------------------------------------------------
  quadgrid_t<std::vector<double>> grid;
  grid.set_sizes (data.Ney, data.Nex, data.hx, data.hy);

  idx_t num_particles = data.x.size ();
  particles_t ptcls (
    num_particles,
    {"label", "level"},
    {"Mp",     "Ap",     "vpx",    "vpy",    "mom_px",  "mom_py",
     "hp",     "Vp",     "F_ext_px","F_ext_py","apx",   "apy",
     "F_11",   "F_12",   "F_21",   "F_22",   "vpx_dx",  "vpx_dy",
     "vpy_dx", "vpy_dy", "Fb_x",   "Fb_y",   "hpZ",     "dZxp",
     "dZyp",   "Zp",     "xp",     "yp",     "Fric_px", "Fric_py",
     "Fpx",    "Fpy",    "vpxL",   "vpyL",   "H"},
    grid, data.x, data.y);

  ptcls.dprops["Mp"]     = data.Mp;
  ptcls.dprops["Ap"]     = data.Ap;
  ptcls.dprops["vpx"]    = data.vpx;
  ptcls.dprops["vpy"]    = data.vpy;
  ptcls.dprops["mom_px"] = data.mom_px;
  ptcls.dprops["mom_py"] = data.mom_py;
  ptcls.dprops["hp"]     = data.hp;
  ptcls.dprops["Vp"]     = data.Vp;
  ptcls.dprops["xp"]     = data.x;
  ptcls.dprops["yp"]     = data.y;

  #pragma omp parallel for schedule(static)
  for (idx_t ip = 0; ip < num_particles; ++ip) {
    ptcls.dprops["vpx_dx"][ip] = 0.0;
    ptcls.dprops["vpx_dy"][ip] = 0.0;
    ptcls.dprops["vpy_dx"][ip] = 0.0;
    ptcls.dprops["vpy_dy"][ip] = 0.0;
    ptcls.dprops["apx"][ip]    = 0.0;
    ptcls.dprops["apy"][ip]    = 0.0;
    ptcls.dprops["Fb_x"][ip]   = 0.0;
    ptcls.dprops["Fb_y"][ip]   = 0.0;
    ptcls.dprops["dZxp"][ip]   = 0.0;
    ptcls.dprops["dZyp"][ip]   = 0.0;
    ptcls.dprops["Zp"][ip]     = 0.0;
    ptcls.dprops["hpZ"][ip]    = 0.0;
    ptcls.dprops["Fpx"][ip]    = 0.0;
    ptcls.dprops["Fpy"][ip]    = 0.0;
    ptcls.dprops["vpxL"][ip]   = 0.0;
    ptcls.dprops["vpyL"][ip]   = 0.0;
  }

  std::iota (ptcls.iprops["label"].begin(), ptcls.iprops["label"].end(), 0);
  std::fill  (ptcls.iprops["level"].begin(), ptcls.iprops["level"].end(), 0);

  // -----------------------------------------------------------------------
  // Grid nodal arrays
  // -----------------------------------------------------------------------
  std::map<std::string, std::vector<double>> vars {
    {"Mv",       std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"mom_vx",   std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"mom_vy",   std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"F_ext_vx", std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"F_ext_vy", std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"F_int_vx", std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"F_int_vy", std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"Fric_x",   std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"Fric_y",   std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"avx",      std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"avy",      std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"vvx",      std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"vvy",      std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"Z",        std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"dZdx",     std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"dZdy",     std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"Ftot_vx",  std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"Ftot_vy",  std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"HV",       std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"FPxv",     std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"FPyv",     std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"vvxL",     std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"vvyL",     std::vector<double>(grid.num_global_nodes(), 0.0)}},

    Plotvars {
    {"rho_v", std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"avx",   std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"avy",   std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"vvx",   std::vector<double>(grid.num_global_nodes(), 0.0)},
    {"vvy",   std::vector<double>(grid.num_global_nodes(), 0.0)}};

  vars["Z"]    = data.Z;
  vars["dZdx"] = data.dZdx;
  vars["dZdy"] = data.dZdy;


  ptcls.g2p (vars, std::vector<std::string>{"Z"},
                   std::vector<std::string>{"Zp"});
  ptcls.g2p (vars, std::vector<std::string>{"dZdx","dZdy"},
                   std::vector<std::string>{"dZxp","dZyp"});

  double eq_level = data.eq_level;
  for (idx_t ip = 0; ip < num_particles; ++ip) {
    double diff = eq_level - ptcls.dprops["Zp"][ip];
    ptcls.dprops["H"][ip] = (eq_level > 0 && diff > 0) ? diff : 0.0;
  }

  #pragma omp parallel for schedule(static)
  for (idx_t ip = 0; ip < num_particles; ++ip) {
    ptcls.dprops["hpZ"][ip] = ptcls.dprops["hp"][ip] + ptcls.dprops["Zp"][ip];
    double hp_val  = ptcls.dprops["hp"][ip];
    double H_val   = ptcls.dprops["H"][ip];
    double h_corr  = hp_val > 1e-10 ? hp_val - H_val*H_val/hp_val : 0.0;
    ptcls.dprops["F_11"][ip] = 0.5 * data.rho * data.g * h_corr;
    ptcls.dprops["F_12"][ip] = 0.0;
    ptcls.dprops["F_21"][ip] = 0.0;
    ptcls.dprops["F_22"][ip] = 0.5 * data.rho * data.g * h_corr;
  }

  int it = 0;
  ptcls.build_mass ();
  grid.vtk_export ("GRID_forZ.vts", vars);


  int    np    = ptcls.num_particles;
  const int    nn    = grid.num_global_nodes ();
  const int    nrows = grid.num_rows ();
  const int    ncols = grid.num_cols ();
  const int    ncells = nrows * ncols;

  const double hx    = grid.hx ();
  const double hy    = grid.hy ();

  const double rho   = data.rho;
  const double g_c   = data.g;
  const double mu    = data.mu;
  const double tau_Y = data.tauy;
  const double xi    = data.xi;

  const double cc    = data.BINGHAM_ON;
  const double fric_on = data.FRICTION_ON;
  const bool   bc_flag = (data.BC_FLAG != 0);

  const bool   ms_on    = (data.MERGE_SPLIT_ON != 0);

  const double tan_fa  = std::tan (data.phi * M_PI / 180.0);
  const double A_coeff = 3.0 / 2.0;
  const double C_coeff = 65.0 / 32.0;
  
  const bool   fixed   = (data.DT_FIXED > 0.0);

  // -----------------------------------------------------------------------
  // Raw pointers — particle arrays
  // -----------------------------------------------------------------------
  double *d_x       = ptcls.x.data ();
  double *d_y       = ptcls.y.data ();
  double *d_vpx     = ptcls.dprops["vpx"].data ();
  double *d_vpy     = ptcls.dprops["vpy"].data ();
  double *d_apx     = ptcls.dprops["apx"].data ();
  double *d_apy     = ptcls.dprops["apy"].data ();
  double *d_hp      = ptcls.dprops["hp"].data ();
  double *d_Ap      = ptcls.dprops["Ap"].data ();
  double *d_Vp      = ptcls.dprops["Vp"].data ();
  double *d_Mp      = ptcls.dprops["Mp"].data ();
  double *d_mom_px  = ptcls.dprops["mom_px"].data ();
  double *d_mom_py  = ptcls.dprops["mom_py"].data ();
  double *d_F11     = ptcls.dprops["F_11"].data ();
  double *d_F12     = ptcls.dprops["F_12"].data ();
  double *d_F21     = ptcls.dprops["F_21"].data ();
  double *d_F22     = ptcls.dprops["F_22"].data ();
  double *d_vpx_dx  = ptcls.dprops["vpx_dx"].data ();
  double *d_vpx_dy  = ptcls.dprops["vpx_dy"].data ();
  double *d_vpy_dx  = ptcls.dprops["vpy_dx"].data ();
  double *d_vpy_dy  = ptcls.dprops["vpy_dy"].data ();
  double *d_Fb_x    = ptcls.dprops["Fb_x"].data ();
  double *d_Fb_y    = ptcls.dprops["Fb_y"].data ();
  double *d_Fric_px = ptcls.dprops["Fric_px"].data ();
  double *d_Fric_py = ptcls.dprops["Fric_py"].data ();
  double *d_dZxp    = ptcls.dprops["dZxp"].data ();
  double *d_dZyp    = ptcls.dprops["dZyp"].data ();
  double *d_Zp      = ptcls.dprops["Zp"].data ();
  double *d_hpZ     = ptcls.dprops["hpZ"].data ();
  double *d_Fpx     = ptcls.dprops["Fpx"].data ();
  double *d_Fpy     = ptcls.dprops["Fpy"].data ();
  double *d_vpxL    = ptcls.dprops["vpxL"].data ();
  double *d_vpyL    = ptcls.dprops["vpyL"].data ();
  double *d_H       = ptcls.dprops["H"].data ();

  // -----------------------------------------------------------------------
  // Raw pointers — nodal arrays
  // -----------------------------------------------------------------------
  double *d_Mv       = vars["Mv"].data ();
  double *d_mom_vx   = vars["mom_vx"].data ();
  double *d_mom_vy   = vars["mom_vy"].data ();
  double *d_F_ext_vx = vars["F_ext_vx"].data ();
  double *d_F_ext_vy = vars["F_ext_vy"].data ();
  double *d_F_int_vx = vars["F_int_vx"].data ();
  double *d_F_int_vy = vars["F_int_vy"].data ();
  double *d_Ftot_vx  = vars["Ftot_vx"].data ();
  double *d_Ftot_vy  = vars["Ftot_vy"].data ();
  double *d_avx      = vars["avx"].data ();
  double *d_avy      = vars["avy"].data ();
  double *d_vvx      = vars["vvx"].data ();
  double *d_vvy      = vars["vvy"].data ();
  double *d_vvxL     = vars["vvxL"].data ();
  double *d_vvyL     = vars["vvyL"].data ();
  double *d_Fric_x   = vars["Fric_x"].data ();
  double *d_Fric_y   = vars["Fric_y"].data ();
  double *d_FPxv     = vars["FPxv"].data ();
  double *d_FPyv     = vars["FPyv"].data ();
  double *d_Z        = vars["Z"].data ();
  double *d_HV       = vars["HV"].data ();
  double *d_pv_rho   = Plotvars["rho_v"].data ();
  double *d_pv_vvx   = Plotvars["vvx"].data ();
  double *d_pv_vvy   = Plotvars["vvy"].data ();
  double *d_pv_avx   = Plotvars["avx"].data ();
  double *d_pv_avy   = Plotvars["avy"].data ();

  // -----------------------------------------------------------------------
  // CSR + coloring pointers — built once by init_particle_mesh,
  // re-uploaded after each rebuild via #pragma omp target update to.
  // -----------------------------------------------------------------------
  ptcls.init_particle_mesh ();
  int *d_p2g    = ptcls.ptcl_to_grd.data ();

  int *d_cstart = ptcls.cell_start.data ();
  int *d_cptcls = ptcls.cell_ptcls.data ();
  int *d_ccidx  = ptcls.color_cell_idx.data ();
  int *d_coloff = ptcls.color_offsets;

  // -----------------------------------------------------------------------
  const int WARMUP = 5;
  std::chrono::high_resolution_clock::time_point t_start;
  double t  = 0.0;
  double dt = 1.0e-3;
  double cel;

#pragma omp target data map(to: d_Z[0:nn], d_cstart[0:ncells+1], d_ccidx[0:ncells], d_coloff[0:5]) \
                          map(alloc: d_Mv[0:nn], d_mom_vx[0:nn], d_mom_vy[0:nn], d_F_ext_vx[0:nn], d_F_ext_vy[0:nn], \
                                     d_F_int_vx[0:nn], d_F_int_vy[0:nn], d_Ftot_vx[0:nn], d_Ftot_vy[0:nn], d_avx[0:nn], \
                                     d_avy[0:nn], d_vvx[0:nn], d_vvy[0:nn], d_vvxL[0:nn], d_vvyL[0:nn], d_Fric_x[0:nn], \
                                     d_Fric_y[0:nn], d_FPxv[0:nn], d_FPyv[0:nn], d_HV[0:nn], d_pv_rho[0:nn], \
                                     d_pv_vvx[0:nn], d_pv_vvy[0:nn], d_pv_avx[0:nn], d_pv_avy[0:nn])
  {
    #pragma omp target enter data map(to: d_x[0:np], d_y[0:np], d_vpx[0:np], d_vpy[0:np], d_apx[0:np], d_apy[0:np], \
                                          d_hp[0:np], d_Ap[0:np], d_Vp[0:np], d_Mp[0:np], d_mom_px[0:np], d_mom_py[0:np], \
                                          d_vpx_dx[0:np], d_vpx_dy[0:np], d_vpy_dx[0:np], d_vpy_dy[0:np], d_Fb_x[0:np], \
                                          d_Fb_y[0:np], d_Fric_px[0:np], d_Fric_py[0:np], d_F11[0:np], d_F12[0:np], \
                                          d_F21[0:np], d_F22[0:np], d_dZxp[0:np], d_dZyp[0:np], d_Zp[0:np], d_hpZ[0:np], \
                                          d_Fpx[0:np], d_Fpy[0:np], d_vpxL[0:np], d_vpyL[0:np], d_H[0:np], \
                                          d_p2g[0:np], d_cptcls[0:np])

    // =======================================================================
    // TIME LOOP
    // =======================================================================
    while (fixed ? (it < data.NSTEPS) : (t < data.T)) {

      if (it == WARMUP)
        t_start = std::chrono::high_resolution_clock::now ();

      // ----------------------------------------------------------------
      if (fixed) {
        dt = data.DT_FIXED;
      } else {
        #pragma omp target update from(d_vpx[0:np], d_vpy[0:np], d_hp[0:np])
        my_timer.tic ("update dt");
        double max_vx = *std::max_element (d_vpx, d_vpx+np);
        double min_vx = *std::min_element (d_vpx, d_vpx+np);
        max_vx = std::max (std::abs(max_vx), std::abs(min_vx));
        double max_vy = *std::max_element (d_vpy, d_vpy+np);
        double min_vy = *std::min_element (d_vpy, d_vpy+np);
        max_vy = std::max (std::abs(max_vy), std::abs(min_vy));
        double hmax   = *std::max_element (d_hp, d_hp+np);
        cel = std::abs (std::max (std::sqrt(g_c*hmax)+max_vx,
                                  std::sqrt(g_c*hmax)+max_vy));
        if (it > 0)
          dt = data.CFL * std::min(hx, hy) / (1e-2 + cel);
        std::cout << "time=" << t << "  dt=" << dt
                  << "  cel=" << cel << std::endl;
        my_timer.toc ("update dt");
      }

      it++;

      // ================================================================
      // Merge-split (adaptive particle refinement/coarsening)
      // ================================================================
      ms_config ms_cfg;
      ms_cfg.alpha = data.ms_alpha;   ms_cfg.beta = data.ms_beta;
      ms_cfg.split_hp_min = data.ms_split_hp_min;
      ms_cfg.hp_min = data.ms_hp_min; ms_cfg.max_dv = data.ms_max_dv;
      ms_cfg.min_level = data.ms_min_level; ms_cfg.max_level = data.ms_max_level;
      ms_cfg.call_interval = data.ms_call_interval; ms_cfg.max_ops = (int)data.ms_max_ops;
      ms_cfg.shear_split = data.ms_shear_split;
      ms_cfg.min_particles_per_cell = data.ms_min_particles_per_cell;

      if (ms_on && it % ms_cfg.call_interval == 0 && it > 0) {
        my_timer.tic ("merge_split");
 
        const int np_old = np;
 
        // synchronize particle arrays from GPU to CPU before merge/split
        #pragma omp target update from(d_x[0:np_old], d_y[0:np_old], d_vpx[0:np_old], d_vpy[0:np_old], d_hp[0:np_old], \
                                       d_Mp[0:np_old], d_Vp[0:np_old], d_Ap[0:np_old], d_mom_px[0:np_old], d_mom_py[0:np_old])
 
        // delete old particle arrays from GPU memory
        #pragma omp target exit data map(delete: d_x[0:np_old], d_y[0:np_old], d_vpx[0:np_old], d_vpy[0:np_old], d_apx[0:np_old], \
                                                 d_apy[0:np_old], d_hp[0:np_old], d_Ap[0:np_old], d_Vp[0:np_old], d_Mp[0:np_old], \
                                                 d_mom_px[0:np_old], d_mom_py[0:np_old], d_F11[0:np_old], d_F12[0:np_old], \
                                                 d_F21[0:np_old], d_F22[0:np_old], d_vpx_dx[0:np_old], d_vpx_dy[0:np_old], \
                                                 d_vpy_dx[0:np_old], d_vpy_dy[0:np_old], d_Fb_x[0:np_old], d_Fb_y[0:np_old], \
                                                 d_Fric_px[0:np_old], d_Fric_py[0:np_old], d_dZxp[0:np_old], d_dZyp[0:np_old], \
                                                 d_Zp[0:np_old], d_hpZ[0:np_old], d_Fpx[0:np_old], d_Fpy[0:np_old], \
                                                 d_vpxL[0:np_old], d_vpyL[0:np_old], d_H[0:np_old], d_p2g[0:np_old], d_cptcls[0:np_old])
 
        // host side merge/split operation
        adaptive_merge_split<idx_t> (ptcls, ms_cfg, dt);
 
        
        np = ptcls.num_particles;
 
        d_x       = ptcls.x.data ();
        d_y       = ptcls.y.data ();
        d_vpx     = ptcls.dprops["vpx"].data ();
        d_vpy     = ptcls.dprops["vpy"].data ();
        d_apx     = ptcls.dprops["apx"].data ();
        d_apy     = ptcls.dprops["apy"].data ();
        d_hp      = ptcls.dprops["hp"].data ();
        d_Ap      = ptcls.dprops["Ap"].data ();
        d_Vp      = ptcls.dprops["Vp"].data ();
        d_Mp      = ptcls.dprops["Mp"].data ();
        d_mom_px  = ptcls.dprops["mom_px"].data ();
        d_mom_py  = ptcls.dprops["mom_py"].data ();
        d_F11     = ptcls.dprops["F_11"].data ();
        d_F12     = ptcls.dprops["F_12"].data ();
        d_F21     = ptcls.dprops["F_21"].data ();
        d_F22     = ptcls.dprops["F_22"].data ();
        d_vpx_dx  = ptcls.dprops["vpx_dx"].data ();
        d_vpx_dy  = ptcls.dprops["vpx_dy"].data ();
        d_vpy_dx  = ptcls.dprops["vpy_dx"].data ();
        d_vpy_dy  = ptcls.dprops["vpy_dy"].data ();
        d_Fb_x    = ptcls.dprops["Fb_x"].data ();
        d_Fb_y    = ptcls.dprops["Fb_y"].data ();
        d_Fric_px = ptcls.dprops["Fric_px"].data ();
        d_Fric_py = ptcls.dprops["Fric_py"].data ();
        d_dZxp    = ptcls.dprops["dZxp"].data ();
        d_dZyp    = ptcls.dprops["dZyp"].data ();
        d_Zp      = ptcls.dprops["Zp"].data ();
        d_hpZ     = ptcls.dprops["hpZ"].data ();
        d_Fpx     = ptcls.dprops["Fpx"].data ();
        d_Fpy     = ptcls.dprops["Fpy"].data ();
        d_vpxL    = ptcls.dprops["vpxL"].data ();
        d_vpyL    = ptcls.dprops["vpyL"].data ();
        d_H       = ptcls.dprops["H"].data ();
 
        ptcls.init_particle_mesh ();
        d_p2g    = ptcls.ptcl_to_grd.data ();

        d_cstart = ptcls.cell_start.data ();
        d_cptcls = ptcls.cell_ptcls.data ();
        d_ccidx  = ptcls.color_cell_idx.data ();
        d_coloff = ptcls.color_offsets;
 
        // update GPU with new particle arrays (new size, new addresses)
        #pragma omp target enter data map(to: d_x[0:np], d_y[0:np], d_vpx[0:np], d_vpy[0:np], d_apx[0:np], d_apy[0:np], \
                                              d_hp[0:np], d_Ap[0:np], d_Vp[0:np], d_Mp[0:np], d_mom_px[0:np], d_mom_py[0:np], \
                                              d_F11[0:np], d_F12[0:np], d_F21[0:np], d_F22[0:np], d_vpx_dx[0:np], d_vpx_dy[0:np], \
                                              d_vpy_dx[0:np], d_vpy_dy[0:np], d_Fb_x[0:np], d_Fb_y[0:np], d_Fric_px[0:np], \
                                              d_Fric_py[0:np], d_dZxp[0:np], d_dZyp[0:np], d_Zp[0:np], d_hpZ[0:np], \
                                              d_Fpx[0:np], d_Fpy[0:np], d_vpxL[0:np], d_vpyL[0:np], d_H[0:np], \
                                              d_p2g[0:np], d_cptcls[0:np])
 
        // update grid arrays that maintain size 'ncells' but have new values from init_particle_mesh
        #pragma omp target update to(d_cstart[0:ncells+1], d_ccidx[0:ncells], d_coloff[0:5])
 
        my_timer.toc ("merge_split");
      }


      std::cout << "  step " << it << "..." << std::flush;
      my_timer.tic ("gpu_block");

      // ================================================================
      // K1 — reset nodal accumulators + P2G mass/momentum
      // ================================================================
      gpu_kernels::reset_nodal_arrays (
          nn, d_Mv, d_mom_vx, d_mom_vy,
          d_F_ext_vx, d_F_ext_vy, d_F_int_vx, d_F_int_vy,
          d_Ftot_vx, d_Ftot_vy, d_avx, d_avy,
          d_vvx, d_vvy, d_vvxL, d_vvyL,
          d_Fric_x, d_Fric_y, d_FPxv, d_FPyv, d_HV,
          d_pv_rho, d_pv_vvx, d_pv_vvy, d_pv_avx, d_pv_avy);

      gpu_kernels::scatter_mass_momentum (
          np, nrows, hx, hy,
          d_cstart, d_cptcls, d_ccidx, d_coloff,
          d_x, d_y, d_Mp, d_mom_px, d_mom_py,
          d_Mv, d_mom_vx, d_mom_vy);

      // ================================================================
      // K2 — G2PD: grad(Z) -> particles
      // ================================================================
      gpu_kernels::gradZ_to_particles (
          np, nrows, hx, hy, d_x, d_y, d_p2g, d_Z, d_dZxp, d_dZyp);

      // ================================================================
      // K3 — gravity force + friction-to-particle-force
      // ================================================================
      gpu_kernels::gravity_force (
          np, g_c, d_hp, d_H, d_Mp, d_dZxp, d_dZyp, d_Fpx, d_Fpy);

      gpu_kernels::friction_particle_force (
          np, d_Ap, d_Fb_x, d_Fb_y, d_Fric_px, d_Fric_py);

      // ================================================================
      // K4 — P2G ext forces (split) + F_ext per node
      // ================================================================
      gpu_kernels::scatter_gravity_force (
          np, nrows, hx, hy,
          d_cstart, d_cptcls, d_ccidx, d_coloff,
          d_x, d_y, d_Fpx, d_Fpy, d_FPxv, d_FPyv);

      gpu_kernels::scatter_friction_force (
          np, nrows, hx, hy,
          d_cstart, d_cptcls, d_ccidx, d_coloff,
          d_x, d_y, d_Fric_px, d_Fric_py, d_Fric_x, d_Fric_y);

      gpu_kernels::compute_F_ext (
          nn, d_FPxv, d_FPyv, d_Fric_x, d_Fric_y, d_F_ext_vx, d_F_ext_vy);

      // ================================================================
      // K5 — P2GD internal forces
      // ================================================================
      gpu_kernels::scatter_internal_forces (
          np, nrows, hx, hy,
          d_cstart, d_cptcls, d_ccidx, d_coloff,
          d_x, d_y, d_Vp, d_F11, d_F12, d_F21, d_F22,
          d_F_int_vx, d_F_int_vy);

      double sum_Mv = 0.0, sum_mvx = 0.0, sum_mvy = 0.0;
#pragma omp target teams distribute parallel for \
      reduction(+:sum_Mv,sum_mvx,sum_mvy)
      for (int in = 0; in < nn; ++in) {
        sum_Mv  += d_Mv[in];
        sum_mvx += d_mom_vx[in];
        sum_mvy += d_mom_vy[in];
      }

      // ================================================================
      // K6 — Ftot+momentum, then nodal vel/acc+BC (split)
      // ================================================================
      gpu_kernels::nodal_force_and_momentum (
          nn, dt, d_F_ext_vx, d_F_ext_vy, d_F_int_vx, d_F_int_vy,
          d_Ftot_vx, d_Ftot_vy, d_mom_vx, d_mom_vy);

      gpu_kernels::nodal_velocity_and_bc (
          nn, nrows, ncols, dt, bc_flag,
          d_Mv, d_Ftot_vx, d_Ftot_vy, d_mom_vx, d_mom_vy,
          d_avx, d_avy, d_vvx, d_vvy, d_vvxL, d_vvyL);

      // ================================================================
      // K7 — G2P velocity + advect + momentum
      // ================================================================
      gpu_kernels::g2p_velocity_and_advect (
          np, nrows, ncols, hx, hy, dt,
          d_p2g, d_vvxL, d_vvyL, d_avx, d_avy,
          d_x, d_y, d_vpx, d_vpy, d_apx, d_apy,
          d_mom_px, d_mom_py,
          d_vpxL, d_vpyL, d_Mp);

      // ================================================================
      // K8 — G2PD velocity gradients + height update (fused)
      // ================================================================
      double min_sc = 1e30, min_hp_g8 = 1e30;
      gpu_kernels::g2pd_gradients_and_height_update (
          np, nrows, hx, hy, dt,
          d_x, d_y, d_p2g, d_vvxL, d_vvyL,
          d_vpx_dx, d_vpx_dy, d_vpy_dx, d_vpy_dy,
          d_hp, d_Vp, d_Ap, &min_sc, &min_hp_g8);

      // ================================================================
      // K9 — Voellmy friction + Bingham stress update (split)
      // ================================================================
      gpu_kernels::voellmy_friction (
          np, rho, g_c, tan_fa, xi, fric_on,
          d_vpx, d_vpy, d_hp, d_Fb_x, d_Fb_y);

      double max_coeff=0, min_hp=0, max_nv=0, max_F11=0;
      double max_dev=0, max_pres=0, max_Dxx=0;
      gpu_kernels::bingham_stress_update (
          np, rho, g_c, mu, tau_Y, cc, A_coeff, C_coeff,
          d_vpx, d_vpy, d_hp, d_H,
          d_vpx_dx, d_vpx_dy, d_vpy_dx, d_vpy_dy,
          d_F11, d_F12, d_F21, d_F22,
          &max_coeff, &min_hp, &max_nv, &max_F11,
          &max_dev, &max_pres, &max_Dxx);

      printf ("max_coeff=%.3e  min_hp=%.3e  max_nv=%.3e  max_F11=%.3e\n",
              max_coeff, min_hp, max_nv, max_F11);
      printf ("max_dev=%.3e  max_pres=%.3e  max_Dxx=%.3e\n",
              max_dev, max_pres, max_Dxx);

      // ================================================================
      // K10 — G2P (Z, hpZ) + Plotvars P2G (split) + hp->HV + normalize
      // ================================================================
      gpu_kernels::g2p_height (
          np, nrows, hx, hy, d_x, d_y, d_p2g, d_Z, d_hp, d_Zp, d_hpZ);

      gpu_kernels::scatter_plotvars (
          np, nrows, hx, hy,
          d_cstart, d_cptcls, d_ccidx, d_coloff,
          d_x, d_y, d_Mp, d_vpx, d_vpy, d_apx, d_apy, d_hp,
          d_pv_rho, d_pv_vvx, d_pv_vvy, d_pv_avx, d_pv_avy, d_HV);

      gpu_kernels::normalize_plotvars (
          nn, d_pv_rho, d_pv_vvx, d_pv_vvy, d_pv_avx, d_pv_avy);

      my_timer.toc ("gpu_block");
      std::cout << "  done." << std::endl;

      // ================================================================
      #pragma omp target update from(d_x[0:np], d_y[0:np])

      my_timer.tic ("reorder");
      ptcls.init_particle_mesh ();
      d_p2g    = ptcls.ptcl_to_grd.data ();

      d_cstart = ptcls.cell_start.data ();
      d_cptcls = ptcls.cell_ptcls.data ();
      d_ccidx  = ptcls.color_cell_idx.data ();
      d_coloff = ptcls.color_offsets;
      my_timer.toc ("reorder");

      #pragma omp target update to(d_x[0:np], d_y[0:np], d_p2g[0:np], d_cstart[0:ncells+1], d_cptcls[0:np], d_ccidx[0:ncells], d_coloff[0:5])

      // ================================================================
      if (WRITE_OUTPUT) {
        my_timer.tic ("save csv");
        #pragma omp target update from(d_vpx[0:np], d_vpy[0:np], d_hp[0:np], d_hpZ[0:np], d_Zp[0:np], d_apx[0:np], d_apy[0:np])
        // if (it % 10 == 0) {
          std::string fn = "nc_particles_" + std::to_string(it) + ".csv";
          std::ofstream OF (fn.c_str());
          ptcls.print<particles_t::output_format::csv>(OF);
          OF.close ();
        // }
        my_timer.toc ("save csv");
      }

      t += dt;

    } // end time loop

  } // end target data

  // performance
  auto t_end = std::chrono::high_resolution_clock::now ();
  double sec     = std::chrono::duration<double>(t_end - t_start).count ();
  int    counted = it - WARMUP;
  std::cout << "PERF np=" << np
            << " counted_steps=" << counted
            << " tot_s=" << sec
            << " per_step_ms=" << 1e3*sec/counted << std::endl;

  my_timer.print_report ();
  return 0;
}
