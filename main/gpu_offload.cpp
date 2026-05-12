#include <fstream>
#include <map>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <particles.h>
#include <quadgrid_cpp.h>
#include <timer.h>
#include "mpm_data.h"
#include "gpu_kernels.h"

using idx_t = quadgrid_t<std::vector<double>>::idx_t;
cdf::timer::timer_t my_timer{};

int main ()
{
  DATA data ("DATA.json");

  quadgrid_t<std::vector<double>> grid;
  grid.set_sizes (data.Ney, data.Nex, data.hx, data.hy);

  idx_t num_particles = data.x.size();
  particles_t ptcls (num_particles, {"label"}, {"Mp", "Ap","vpx","vpy","mom_px","mom_py","hp",
	      "Vp","F_ext_px","F_ext_py","apx","apy",
	      "F_11","F_12","F_21","F_22","vpx_dx","vpx_dy",
	      "vpy_dx","vpy_dy","Fb_x","Fb_y","hpZ","dZxp","dZyp","Zp","xp","yp","Fric_px","Fric_py"}, grid, data.x, data.y);
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
  }

  std::iota(ptcls.iprops["label"].begin(), ptcls.iprops["label"].end(), 0);

  double t = 0.0;
  double dt;
  double cel;
  double fric_ang = 34. * M_PI / 180.;

  std::map<std::string, std::vector<double>>
    vars{
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
      {"HV", std::vector<double>(grid.num_global_nodes(), 0.0)}
    },
    Plotvars{
      {"rho_v", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"avx", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"avy", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"vvx", std::vector<double>(grid.num_global_nodes(), 0.0)},
      {"vvy", std::vector<double>(grid.num_global_nodes(), 0.0)}
    };

  vars["Z"] = data.Z;
  vars["dZdx"] = data.dZdx;
  vars["dZdy"] = data.dZdy;

  ptcls.g2p(vars, std::vector<std::string>{"Z"}, std::vector<std::string>{"Zp"});
  ptcls.dprops.at("dZxp").assign(ptcls.num_particles, 0.0);
  ptcls.dprops.at("dZyp").assign(ptcls.num_particles, 0.0);
  ptcls.g2p(vars, std::vector<std::string>{"dZdx","dZdy"},
                   std::vector<std::string>{"dZxp","dZyp"});

  for (idx_t ip = 0; ip < num_particles; ++ip)
    ptcls.dprops["hpZ"][ip] = ptcls.dprops["hp"][ip] + ptcls.dprops["Zp"][ip];

  for (idx_t ip = 0; ip < num_particles; ++ip) {
    ptcls.dprops["F_11"][ip] = -0.5 * data.rho * data.g * ptcls.dprops["hp"][ip];
    ptcls.dprops["F_12"][ip] = 0.0;
    ptcls.dprops["F_21"][ip] = 0.0;
    ptcls.dprops["F_22"][ip] = -0.5 * data.rho * data.g * ptcls.dprops["hp"][ip];
  }

  int it = 0;
  ptcls.build_mass();
  grid.vtk_export("GRID_forZ.vts", vars);
  dt = 1.0e-5;

  // GPU INFO
  std::cout << "OpenMP target devices: " << omp_get_num_devices() << std::endl;
  if (omp_get_num_devices() > 0)
    std::cout << "GPU offload ENABLED (colored P2G)" << std::endl;
  else
    std::cout << "WARNING: No GPU, running on CPU" << std::endl;

  // EXTRACT RAW POINTERS (stable: vectors never resize in time loop)
  const int np     = ptcls.num_particles;
  const int nn     = grid.num_global_nodes();
  const int nrows  = grid.num_rows();
  const int ncols  = grid.num_cols();
  const double hx  = grid.hx();
  const double hy  = grid.hy();
  const double rho = data.rho;
  const double g_c = data.g;

  double* d_x       = ptcls.x.data();
  double* d_y       = ptcls.y.data();
  double* d_vpx     = ptcls.dprops["vpx"].data();
  double* d_vpy     = ptcls.dprops["vpy"].data();
  double* d_apx     = ptcls.dprops["apx"].data();
  double* d_apy     = ptcls.dprops["apy"].data();
  double* d_hp      = ptcls.dprops["hp"].data();
  double* d_Ap      = ptcls.dprops["Ap"].data();
  double* d_Vp      = ptcls.dprops["Vp"].data();
  double* d_Mp      = ptcls.dprops["Mp"].data();
  double* d_mom_px  = ptcls.dprops["mom_px"].data();
  double* d_mom_py  = ptcls.dprops["mom_py"].data();
  double* d_F11     = ptcls.dprops["F_11"].data();
  double* d_F12     = ptcls.dprops["F_12"].data();
  double* d_F21     = ptcls.dprops["F_21"].data();
  double* d_F22     = ptcls.dprops["F_22"].data();
  double* d_vpx_dx  = ptcls.dprops["vpx_dx"].data();
  double* d_vpx_dy  = ptcls.dprops["vpx_dy"].data();
  double* d_vpy_dx  = ptcls.dprops["vpy_dx"].data();
  double* d_vpy_dy  = ptcls.dprops["vpy_dy"].data();
  double* d_Fb_x    = ptcls.dprops["Fb_x"].data();
  double* d_Fb_y    = ptcls.dprops["Fb_y"].data();
  double* d_Fric_px = ptcls.dprops["Fric_px"].data();
  double* d_Fric_py = ptcls.dprops["Fric_py"].data();
  double* d_dZxp    = ptcls.dprops["dZxp"].data();
  double* d_dZyp    = ptcls.dprops["dZyp"].data();
  double* d_Zp      = ptcls.dprops["Zp"].data();
  double* d_hpZ     = ptcls.dprops["hpZ"].data();
  int*    d_p2g     = ptcls.ptcl_to_grd.data();

  double* d_Mv       = vars["Mv"].data();
  double* d_mom_vx   = vars["mom_vx"].data();
  double* d_mom_vy   = vars["mom_vy"].data();
  double* d_F_ext_vx = vars["F_ext_vx"].data();
  double* d_F_ext_vy = vars["F_ext_vy"].data();
  double* d_F_int_vx = vars["F_int_vx"].data();
  double* d_F_int_vy = vars["F_int_vy"].data();
  double* d_Ftot_vx  = vars["Ftot_vx"].data();
  double* d_Ftot_vy  = vars["Ftot_vy"].data();
  double* d_avx      = vars["avx"].data();
  double* d_avy      = vars["avy"].data();
  double* d_vvx      = vars["vvx"].data();
  double* d_vvy      = vars["vvy"].data();
  double* d_Fric_x   = vars["Fric_x"].data();
  double* d_Fric_y   = vars["Fric_y"].data();
  double* d_dZdx     = vars["dZdx"].data();
  double* d_dZdy     = vars["dZdy"].data();
  double* d_Z        = vars["Z"].data();

  const double A_coeff = 3.0 / 2.0;
  const double C_coeff = 65.0 / 32.0;
  const double mu      = 50.0;
  const double tau_Y   = 2000.0;
  const double cc      = 0.0;
  const double tan_fa  = std::tan(fric_ang);

  // Color index arrays (rebuilt each iteration)
  std::vector<int> color_indices(np);
  int color_offsets[5] = {0};

  // TIME LOOP
  while (t < data.T)
  {
    // CPU: dt update, I/O
    my_timer.tic("update dt");
      double max_vx = *std::max_element(d_vpx, d_vpx + np);
      double min_vx = *std::min_element(d_vpx, d_vpx + np);
      max_vx = std::max(std::abs(max_vx), std::abs(min_vx));
      double max_vy = *std::max_element(d_vpy, d_vpy + np);
      double min_vy = *std::min_element(d_vpy, d_vpy + np);
      max_vy = std::max(std::abs(max_vy), std::abs(min_vy));
      double hmax = *std::max_element(d_hp, d_hp + np);
      double max_vel = std::max(std::sqrt(g_c*hmax)+max_vx, std::sqrt(g_c*hmax)+max_vy);
      cel = std::abs(max_vel);
      if (it > 0) dt = 0.2 * data.hx / (1e-2 + cel);
      std::cout << "time = " << t << "  dt = " << dt << std::endl;
    my_timer.toc("update dt");

    my_timer.tic("save csv");
      std::string filename = "nc_particles_" + std::to_string(it++) + ".csv";
      if (t >= 0.0) {
        std::ofstream OF(filename.c_str());
        ptcls.print<particles_t::output_format::csv>(OF);
        OF.close();
      }
    my_timer.toc("save csv");

    // CPU: connectivity + color index building
    my_timer.tic("reorder");
      ptcls.init_particle_mesh();
    my_timer.toc("reorder");  

    my_timer.tic("build_colors");
    {
      int counts[4] = {0, 0, 0, 0};
      for (int ip = 0; ip < np; ip++) {
        int cell = d_p2g[ip];
        int r = cell % nrows;
        int c = cell / nrows;
        counts[(r % 2) * 2 + (c % 2)]++;
      }
      color_offsets[0] = 0;
      for (int c = 0; c < 4; c++) color_offsets[c+1] = color_offsets[c] + counts[c];

      int pos[4] = {color_offsets[0], color_offsets[1], color_offsets[2], color_offsets[3]};
      for (int ip = 0; ip < np; ip++) {
        int cell = d_p2g[ip];
        int r = cell % nrows;
        int c = cell / nrows;
        int col = (r % 2) * 2 + (c % 2);
        color_indices[pos[col]++] = ip;
      }
    }
    my_timer.toc("build_colors");

    if (it <= 1) {
      std::cout << "[COLORS] offsets: " << color_offsets[0] << " " << color_offsets[1]
                << " " << color_offsets[2] << " " << color_offsets[3]
                << " " << color_offsets[4] << " np=" << np << std::endl;
    }

    // CPU: Step 0 — reset nodal arrays on host
    my_timer.tic("step 0");
      for (auto &v : vars)     v.second.assign(v.second.size(), 0.0);
      for (auto &v : Plotvars) v.second.assign(v.second.size(), 0.0);
      vars["Z"] = data.Z;
      vars["dZdx"] = data.dZdx;
      vars["dZdy"] = data.dZdy;
    my_timer.toc("step 0");

    // GPU PHASE: ALL computation in one target data block
    my_timer.tic("gpu_block");
    {
      const double dt_local = dt;
      int* d_cidx = color_indices.data();

      #pragma omp target data                                                \
          map(to:     d_p2g[0:np], d_Mp[0:np], d_cidx[0:np],  \
                      d_Z[0:nn], d_dZdx[0:nn], d_dZdy[0:nn])               \
          map(tofrom: d_x[0:np], d_y[0:np],                                 \
                      d_vpx[0:np], d_vpy[0:np],                             \
                      d_apx[0:np], d_apy[0:np],                             \
                      d_hp[0:np], d_Ap[0:np], d_Vp[0:np],                  \
                      d_mom_px[0:np], d_mom_py[0:np],                       \
                      d_vpx_dx[0:np], d_vpx_dy[0:np],                      \
                      d_vpy_dx[0:np], d_vpy_dy[0:np],                      \
                      d_Fb_x[0:np], d_Fb_y[0:np],                          \
                      d_Fric_px[0:np], d_Fric_py[0:np],                    \
                      d_F11[0:np], d_F12[0:np],                             \
                      d_F21[0:np], d_F22[0:np],                             \
                      d_dZxp[0:np], d_dZyp[0:np],                          \
                      d_Zp[0:np], d_hpZ[0:np],                             \
                      d_Mv[0:nn], d_mom_vx[0:nn], d_mom_vy[0:nn],         \
                      d_F_ext_vx[0:nn], d_F_ext_vy[0:nn],                  \
                      d_F_int_vx[0:nn], d_F_int_vy[0:nn],                  \
                      d_Ftot_vx[0:nn], d_Ftot_vy[0:nn],                    \
                      d_avx[0:nn], d_avy[0:nn],                             \
                      d_vvx[0:nn], d_vvy[0:nn],                             \
                      d_Fric_x[0:nn], d_Fric_y[0:nn])
      {

        #pragma omp target update to(d_p2g[0:np], d_cidx[0:np])

      // STEP 1a: P2G mass + momentum — COLORED (4 kernels)
      for (int color = 0; color < 4; color++) {
        int start = color_offsets[color];
        int count = color_offsets[color+1] - start;
        if (count == 0) continue;

        #pragma omp target teams distribute parallel for
        for (int ii = start; ii < start + count; ii++) {
          int ip = d_cidx[ii];
          double xx = d_x[ip], yy = d_y[ip];
          int ci = d_p2g[ip];
          int r = gpu_gind2row(ci, nrows);
          int c = gpu_gind2col(ci, nrows);
          for (int in = 0; in < 4; in++) {
            double N  = gpu_shp(xx, yy, in, c, r, hx, hy);
            int  nidx = gpu_gt(in, c, r, nrows);
            #pragma omp atomic
            d_Mv[nidx]     += N * d_Mp[ip];
            #pragma omp atomic
            d_mom_vx[nidx] += N * d_mom_px[ip];
            #pragma omp atomic
            d_mom_vy[nidx] += N * d_mom_py[ip];
          }
        }
      }

      // STEP 1a: P2G mass + momentum — ATOMIC
      /*#pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double xx = d_x[ip], yy = d_y[ip];
        int ci = d_p2g[ip];
        int r = gpu_gind2row(ci, nrows);
        int c = gpu_gind2col(ci, nrows);
        for (int in = 0; in < 4; in++) {
          double N  = gpu_shp(xx, yy, in, c, r, hx, hy);
          int  nidx = gpu_gt(in, c, r, nrows);
          #pragma omp atomic
          d_Mv[nidx]     += N * d_Mp[ip];
          #pragma omp atomic
          d_mom_vx[nidx] += N * d_mom_px[ip];
          #pragma omp atomic
          d_mom_vy[nidx] += N * d_mom_py[ip];
        }
      }*/

      // STEP 1b: G2PD for Z gradients — flat (no race)
      #pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double xx = d_x[ip], yy = d_y[ip];
        int ci = d_p2g[ip];
        int r = gpu_gind2row(ci, nrows);
        int c = gpu_gind2col(ci, nrows);
        double dzx = 0.0, dzy = 0.0;
        for (int in = 0; in < 4; in++) {
          double Nx = gpu_shg(xx, yy, 0, in, c, r, hx, hy);
          double Ny = gpu_shg(xx, yy, 1, in, c, r, hx, hy);
          int  nidx = gpu_gt(in, c, r, nrows);
          dzx += Nx * d_Z[nidx];
          dzy += Ny * d_Z[nidx];
        }
        d_dZxp[ip] = dzx;
        d_dZyp[ip] = dzy;
      }

      // STEP 2a: Fric_px/py per-particle — flat (no race)
      #pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        d_Fric_px[ip] = 0.0 * d_Ap[ip] * d_Fb_x[ip];
        d_Fric_py[ip] = 0.0 * d_Ap[ip] * d_Fb_y[ip];
      }

      // STEP 2b: P2G friction — COLORED (4 kernels)
      for (int color = 0; color < 4; color++) {
        int start = color_offsets[color];
        int count = color_offsets[color+1] - start;
        if (count == 0) continue;

        #pragma omp target teams distribute parallel for
        for (int ii = start; ii < start + count; ii++) {
          int ip = d_cidx[ii];
          double xx = d_x[ip], yy = d_y[ip];
          int ci = d_p2g[ip];
          int r = gpu_gind2row(ci, nrows);
          int c = gpu_gind2col(ci, nrows);
          for (int in = 0; in < 4; in++) {
            double N  = gpu_shp(xx, yy, in, c, r, hx, hy);
            int  nidx = gpu_gt(in, c, r, nrows);
            #pragma omp atomic
            d_Fric_x[nidx] += N * d_Fric_px[ip];
            #pragma omp atomic
            d_Fric_y[nidx] += N * d_Fric_py[ip];
          }
        }
      }

      // STEP 2b: P2G friction — ATOMIC
      /*#pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double xx = d_x[ip], yy = d_y[ip];
        int ci = d_p2g[ip];
        int r = gpu_gind2row(ci, nrows);
        int c = gpu_gind2col(ci, nrows);
        for (int in = 0; in < 4; in++) {
          double N  = gpu_shp(xx, yy, in, c, r, hx, hy);
          int  nidx = gpu_gt(in, c, r, nrows);
          #pragma omp atomic
          d_Fric_x[nidx] += N * d_Fric_px[ip];
          #pragma omp atomic
          d_Fric_y[nidx] += N * d_Fric_py[ip];
        }
      }*/

      // STEP 2c: F_ext per-node — flat
      #pragma omp target teams distribute parallel for
      for (int iv = 0; iv < nn; iv++) {
        d_F_ext_vx[iv] = -d_Mv[iv]*9.81*0.0*d_dZdx[iv] + d_Fric_x[iv];
        d_F_ext_vy[iv] = -d_Mv[iv]*9.81*0.0*d_dZdy[iv] + d_Fric_y[iv];
      }

      // STEP 3a: P2GD internal forces — COLORED (4 kernels)
      for (int color = 0; color < 4; color++) {
        int start = color_offsets[color];
        int count = color_offsets[color+1] - start;
        if (count == 0) continue;

        #pragma omp target teams distribute parallel for
        for (int ii = start; ii < start + count; ii++) {
          int ip = d_cidx[ii];
          double xx = d_x[ip], yy = d_y[ip];
          int ci = d_p2g[ip];
          int r = gpu_gind2row(ci, nrows);
          int c = gpu_gind2col(ci, nrows);
          double vp = d_Vp[ip];
          for (int in = 0; in < 4; in++) {
            double Nx = gpu_shg(xx, yy, 0, in, c, r, hx, hy);
            double Ny = gpu_shg(xx, yy, 1, in, c, r, hx, hy);
            int  nidx = gpu_gt(in, c, r, nrows);
            #pragma omp atomic
            d_F_int_vx[nidx] += (Nx * d_F11[ip] + Ny * d_F12[ip]) * vp;
            #pragma omp atomic
            d_F_int_vy[nidx] += (Nx * d_F21[ip] + Ny * d_F22[ip]) * vp;
          }
        }
      }

      // STEP 3a: P2GD internal forces — ATOMIC
      /*#pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double xx = d_x[ip], yy = d_y[ip];
        int ci = d_p2g[ip];
        int r = gpu_gind2row(ci, nrows);
        int c = gpu_gind2col(ci, nrows);
        double vp = d_Vp[ip];
        for (int in = 0; in < 4; in++) {
          double Nx = gpu_shg(xx, yy, 0, in, c, r, hx, hy);
          double Ny = gpu_shg(xx, yy, 1, in, c, r, hx, hy);
          int  nidx = gpu_gt(in, c, r, nrows);
          #pragma omp atomic
          d_F_int_vx[nidx] += (Nx * d_F11[ip] + Ny * d_F12[ip]) * vp;
          #pragma omp atomic
          d_F_int_vy[nidx] += (Nx * d_F21[ip] + Ny * d_F22[ip]) * vp;
        }
      }*/

      // STEP 3b: Ftot + momentum per-node — flat
      #pragma omp target teams distribute parallel for
      for (int iv = 0; iv < nn; iv++) {
        double ftx = d_F_ext_vx[iv] - d_F_int_vx[iv];
        double fty = d_F_ext_vy[iv] - d_F_int_vy[iv];
        d_Ftot_vx[iv] = ftx;
        d_Ftot_vy[iv] = fty;
        d_mom_vx[iv] += dt_local * ftx;
        d_mom_vy[iv] += dt_local * fty;
      }

      // STEP 4: nodal vel/acc — flat
      #pragma omp target teams distribute parallel for
      for (int iv = 0; iv < nn; iv++) {
        double mv = d_Mv[iv];
        bool active = mv > 1e-8;
        d_avx[iv] = active ? d_Ftot_vx[iv] / mv : 0.0;
        d_avy[iv] = active ? d_Ftot_vy[iv] / mv : 0.0;
        d_vvx[iv] = active ? d_mom_vx[iv]  / mv : 0.0;
        d_vvy[iv] = active ? d_mom_vy[iv]  / mv : 0.0;
      }

      // STEP 5: boundary conditions — flat
      #pragma omp target teams distribute parallel for
      for (int iv = 0; iv < nn; iv++) {
        int r = iv % (nrows + 1);
        int c = iv / (nrows + 1);
        if (c == 0 || c == 1 || c == ncols-1 || c == ncols) {
          d_avx[iv] = 0.0;  d_vvx[iv] = 0.0;
        }
        if (r == 0 || r == 1 || r == nrows-1 || r == nrows) {
          d_avy[iv] = 0.0;  d_vvy[iv] = 0.0;
        }
      }

      // STEP 6: G2P + velocity/position update — flat
      #pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double xx = d_x[ip], yy = d_y[ip];
        int ci = d_p2g[ip];
        int r = gpu_gind2row(ci, nrows);
        int c = gpu_gind2col(ci, nrows);
        double vx=0, vy=0, ax=0, ay=0;
        for (int in = 0; in < 4; in++) {
          double N  = gpu_shp(xx, yy, in, c, r, hx, hy);
          int  nidx = gpu_gt(in, c, r, nrows);
          vx += N * d_vvx[nidx];  vy += N * d_vvy[nidx];
          ax += N * d_avx[nidx];  ay += N * d_avy[nidx];
        }
        vx += dt_local * ax;  vy += dt_local * ay;
        d_vpx[ip] = vx;  d_vpy[ip] = vy;
        d_apx[ip] = ax;  d_apy[ip] = ay;
        d_x[ip] += dt_local * vx;
        d_y[ip] += dt_local * vy;
      }

      // STEP 7: G2PD + height update — flat
      #pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double xx = d_x[ip], yy = d_y[ip];
        int ci = d_p2g[ip];
        int r = gpu_gind2row(ci, nrows);
        int c = gpu_gind2col(ci, nrows);
        double vxdx=0, vxdy=0, vydx=0, vydy=0;
        for (int in = 0; in < 4; in++) {
          double Nx = gpu_shg(xx, yy, 0, in, c, r, hx, hy);
          double Ny = gpu_shg(xx, yy, 1, in, c, r, hx, hy);
          int  nidx = gpu_gt(in, c, r, nrows);
          vxdx += Nx * d_vvx[nidx];  vxdy += Ny * d_vvx[nidx];
          vydx += Nx * d_vvy[nidx];  vydy += Ny * d_vvy[nidx];
        }
        d_vpx_dx[ip]=vxdx; d_vpx_dy[ip]=vxdy;
        d_vpy_dx[ip]=vydx; d_vpy_dy[ip]=vydy;
        double sc = 1.0 + dt_local*(vxdx+vydy);
        d_hp[ip] /= sc;  d_Ap[ip] /= sc;
        d_Vp[ip] = d_hp[ip] * d_Ap[ip];
        d_mom_px[ip] = d_vpx[ip] * d_Mp[ip];
        d_mom_py[ip] = d_vpy[ip] * d_Mp[ip];
      }

      // FRICTION (Voellmy) — flat
      #pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double vx=d_vpx[ip], vy=d_vpy[ip], h=d_hp[ip];
        double nv = sqrt(vx*vx + vy*vy);
        d_Fb_x[ip] = h>1e-3 ? -0.0*2.5*(rho*h*g_c*tan_fa+rho*g_c*vx*vx/100.0)*vx/(nv+0.001) : 0.0;
        d_Fb_y[ip] = h>1e-3 ? -0.0*2.5*(rho*h*g_c*tan_fa+rho*g_c*vy*vy/100.0)*vy/(nv+0.001) : 0.0;
      }

      // STEP 8: stress update (Bingham USL) — flat
      #pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double vx=d_vpx[ip], vy=d_vpy[ip], h=d_hp[ip];
        double nv = sqrt(vx*vx + vy*vy);
        double alf = h>1e-3 ? (6.0*mu*nv)/((h+0.001)*tau_Y) : 0.0;
        double bv = -114.0/32.0 - alf;
        double sq = sqrt(bv*bv - 4.0*A_coeff*C_coeff);
        double zz = 0.0;
        double sxx=d_vpx_dx[ip], sxy=0.5*(d_vpx_dy[ip]+d_vpy_dx[ip]), syy=d_vpy_dy[ip];
        double Dxx=sxx, Dxy=sxy, Dyy=syy, Dzz=-(sxx+d_vpy_dy[ip]);
        double Dzx = h>1e-3 ? 0.5*(3.0/(2.0+zz))*(vx/(h+0.001)) : 0.0;
        double Dzy = h>1e-3 ? 0.5*(3.0/(2.0+zz))*(vy/(h+0.001)) : 0.0;
        double inv2 = 0.5*(Dxx*Dxx+Dyy*Dyy+Dzz*Dzz+Dzx*Dzx+Dzy*Dzy+Dxy*Dxy);
        double coeff = inv2!=0.0 ? (tau_Y/sqrt(inv2)+2.0*mu) : 0.0;
        d_F11[ip] = cc*coeff*Dxx - 0.5*rho*g_c*h;
        d_F12[ip] = cc*coeff*Dxy;
        d_F21[ip] = cc*coeff*Dxy;
        d_F22[ip] = cc*coeff*Dyy - 0.5*rho*g_c*h;
      }

      // hpZ: G2P for Z→Zp + hpZ = hp + Zp — flat (no race)
      #pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double xx = d_x[ip], yy = d_y[ip];
        int ci = d_p2g[ip];
        int r = gpu_gind2row(ci, nrows);
        int c = gpu_gind2col(ci, nrows);
        double zp = 0.0;
        for (int in = 0; in < 4; in++) {
          double N  = gpu_shp(xx, yy, in, c, r, hx, hy);
          int  nidx = gpu_gt(in, c, r, nrows);
          zp += N * d_Z[nidx];
        }
        d_Zp[ip] = zp;
        d_hpZ[ip] = d_hp[ip] + zp;
      }

      } // end target data
    }

    if (it <= 3) {
      double sum_Mv = 0, max_vpx = 0, max_Fint = 0, max_hp = 0;
      for (int i = 0; i < nn; i++) sum_Mv += d_Mv[i];
      for (int i = 0; i < np; i++) {
        if (std::abs(d_vpx[i]) > max_vpx) max_vpx = std::abs(d_vpx[i]);
        if (d_hp[i] > max_hp) max_hp = d_hp[i];
      }
      for (int i = 0; i < nn; i++)
        if (std::abs(d_F_int_vx[i]) > max_Fint) max_Fint = std::abs(d_F_int_vx[i]);
      std::cout << "[DIAG it=" << it << "] sum_Mv=" << sum_Mv
                << " max_vpx=" << max_vpx << " max_Fint=" << max_Fint
                << " max_hp=" << max_hp << std::endl;
    }

    my_timer.toc("gpu_block");

    // CPU: Plotvars P2G + VTS export (only for I/O, not critical path)
      ptcls.p2g(Plotvars, std::vector<std::string>{"Mp","vpx","vpy","apx","apy"},
                           std::vector<std::string>{"rho_v","vvx","vvy","avx","avy"}, true);

    my_timer.tic("step 8b");
      ptcls.p2g(vars, std::vector<std::string>{"hp"}, std::vector<std::string>{"HV"});
    my_timer.toc("step 8b");

    my_timer.tic("save vts");
      filename = "nc_grid_" + std::to_string(it) + ".vts";
      grid.vtk_export(filename.c_str(), vars);
      t += dt;
    my_timer.toc("save vts");

  } // end time loop

  my_timer.print_report();
  return 0;
}