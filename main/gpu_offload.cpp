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
  std::cout << "OpenMP target devices available: " << omp_get_num_devices() << std::endl;
  if (omp_get_num_devices() > 0)
    std::cout << "GPU offload ENABLED (target data optimization)" << std::endl;
  else
    std::cout << "WARNING: No GPU devices found, running on CPU" << std::endl;

  // EXTRACT RAW POINTERS ONCE (vectors never resize during time loop)
  const int np     = ptcls.num_particles;
  const int nn     = grid.num_global_nodes();
  const int nrows  = grid.num_rows();
  const int ncols  = grid.num_cols();
  const double hx  = grid.hx();
  const double hy  = grid.hy();
  const double rho = data.rho;
  const double g_c = data.g;

  // Particle arrays
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
  int*    d_p2g     = ptcls.ptcl_to_grd.data();

  // Nodal arrays
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

  // Constants
  const double A_coeff = 3.0 / 2.0;
  const double C_coeff = 65.0 / 32.0;
  const double mu      = 50.0;
  const double tau_Y   = 2000.0;
  const double cc      = 0.0;
  const double tan_fa  = std::tan(fric_ang);

  // TIME LOOP
  while (t < 1.0)
  {
    // CPU PHASE: dt, I/O, connectivity, P2G/P2GD (race condition ops)
    my_timer.tic("update dt");
      double max_vel_x = *std::max_element(d_vpx, d_vpx + np);
      double min_vel_x = *std::min_element(d_vpx, d_vpx + np);
      max_vel_x = std::max(std::abs(max_vel_x), std::abs(min_vel_x));
      double max_vel_y = *std::max_element(d_vpy, d_vpy + np);
      double min_vel_y = *std::min_element(d_vpy, d_vpy + np);
      max_vel_y = std::max(std::abs(max_vel_y), std::abs(min_vel_y));
      double hmean = *std::max_element(d_hp, d_hp + np);
      double max_vel = std::max(std::sqrt(g_c * hmean) + max_vel_x,
                                std::sqrt(g_c * hmean) + max_vel_y);
      cel = std::abs(max_vel);
      if (it > 0)
        dt = 0.2 * data.hx / (1e-2 + cel);
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

    // (0) Connectivity + reset
    my_timer.tic("reorder");
      ptcls.init_particle_mesh();

    my_timer.tic("step 0");
      for (auto &v : vars)     v.second.assign(v.second.size(), 0.0);
      for (auto &v : Plotvars) v.second.assign(v.second.size(), 0.0);
      ptcls.dprops.at("dZxp").assign(np, 0.0);
      ptcls.dprops.at("dZyp").assign(np, 0.0);
      vars["Z"] = data.Z;
      vars["dZdx"] = data.dZdx;
      vars["dZdy"] = data.dZdy;
    my_timer.toc("step 0");

    // (1) P2G + g2pd — CPU (race condition)
    my_timer.tic("step 1");
      ptcls.p2g(vars, std::vector<std::string>{"Mp","mom_px","mom_py"},
                       std::vector<std::string>{"Mv","mom_vx","mom_vy"});
      ptcls.g2pd(vars, std::vector<std::string>{"Z"},
                        std::vector<std::string>{"dZxp"},
                        std::vector<std::string>{"dZyp"});
    my_timer.toc("step 1");

    // (2a) Friction per-particle — GPU kernel with own map
    // (2b) P2G friction — CPU (race condition)
    my_timer.tic("step 2");
    {
      double* d_Fric_px = ptcls.dprops["Fric_px"].data();
      double* d_Fric_py = ptcls.dprops["Fric_py"].data();

      #pragma omp target teams distribute parallel for \
          map(to: d_Ap[0:np], d_Fb_x[0:np], d_Fb_y[0:np]) \
          map(from: d_Fric_px[0:np], d_Fric_py[0:np])
      for (int ip = 0; ip < np; ip++) {
        d_Fric_px[ip] = 0.0 * d_Ap[ip] * d_Fb_x[ip];
        d_Fric_py[ip] = 0.0 * d_Ap[ip] * d_Fb_y[ip];
      }
    }
      ptcls.p2g(vars, std::vector<std::string>{"Fric_px","Fric_py"},
                       std::vector<std::string>{"Fric_x","Fric_y"});
    my_timer.toc("step 2");

    // (3a) P2GD internal forces — CPU (race condition)
    my_timer.tic("step 3a");
      ptcls.p2gd(vars, std::vector<std::string>{"F_11","F_21"},
                        std::vector<std::string>{"F_12","F_22"},
                        "Vp", std::vector<std::string>{"F_int_vx","F_int_vy"});
    my_timer.toc("step 3a");

    // GPU PHASE: single target data region for ALL GPU kernels
    // Steps 2c, 3b, 4, 5, 6, 7, friction, 8 all execute on device.
    // Data is transferred once at the start and once at the end,
    // instead of per-kernel.
    my_timer.tic("gpu_block");
    {
      const double dt_local = dt;

      #pragma omp target data                                          \
          map(to:     d_p2g[0:np], d_Mp[0:np],                        \
                      d_Mv[0:nn], d_Fric_x[0:nn], d_Fric_y[0:nn],    \
                      d_dZdx[0:nn], d_dZdy[0:nn],                     \
                      d_F_int_vx[0:nn], d_F_int_vy[0:nn])             \
          map(tofrom: d_x[0:np], d_y[0:np],                           \
                      d_vpx[0:np], d_vpy[0:np],                       \
                      d_apx[0:np], d_apy[0:np],                       \
                      d_hp[0:np], d_Ap[0:np], d_Vp[0:np],            \
                      d_mom_px[0:np], d_mom_py[0:np],                 \
                      d_vpx_dx[0:np], d_vpx_dy[0:np],                \
                      d_vpy_dx[0:np], d_vpy_dy[0:np],                \
                      d_Fb_x[0:np], d_Fb_y[0:np],                    \
                      d_F11[0:np], d_F12[0:np],                       \
                      d_F21[0:np], d_F22[0:np],                       \
                      d_mom_vx[0:nn], d_mom_vy[0:nn],                \
                      d_F_ext_vx[0:nn], d_F_ext_vy[0:nn],            \
                      d_Ftot_vx[0:nn], d_Ftot_vy[0:nn],              \
                      d_avx[0:nn], d_avy[0:nn],                       \
                      d_vvx[0:nn], d_vvy[0:nn])
      {

      // (2c) F_ext per-node
      #pragma omp target teams distribute parallel for
      for (int iv = 0; iv < nn; iv++) {
        d_F_ext_vx[iv] = -d_Mv[iv] * 9.81 * 0.0 * d_dZdx[iv] + d_Fric_x[iv];
        d_F_ext_vy[iv] = -d_Mv[iv] * 9.81 * 0.0 * d_dZdy[iv] + d_Fric_y[iv];
      }

      // (3b) Ftot + momentum per-node
      #pragma omp target teams distribute parallel for
      for (int iv = 0; iv < nn; iv++) {
        double ftx = d_F_ext_vx[iv] - d_F_int_vx[iv];
        double fty = d_F_ext_vy[iv] - d_F_int_vy[iv];
        d_Ftot_vx[iv] = ftx;
        d_Ftot_vy[iv] = fty;
        d_mom_vx[iv] += dt_local * ftx;
        d_mom_vy[iv] += dt_local * fty;
      }

      // (4) Nodal accelerations and velocities
      #pragma omp target teams distribute parallel for
      for (int iv = 0; iv < nn; iv++) {
        double mv = d_Mv[iv];
        bool active = mv > 1e-8;
        d_avx[iv] = active ? d_Ftot_vx[iv] / mv : 0.0;
        d_avy[iv] = active ? d_Ftot_vy[iv] / mv : 0.0;
        d_vvx[iv] = active ? d_mom_vx[iv]  / mv : 0.0;
        d_vvy[iv] = active ? d_mom_vy[iv]  / mv : 0.0;
      }

      // (5) Boundary conditions — GPU offloaded
      #pragma omp target teams distribute parallel for
      for (int iv = 0; iv < nn; iv++) {
        int r = iv % (nrows + 1);
        int c = iv / (nrows + 1);
        if (c == 0 || c == 1 || c == ncols - 1 || c == ncols) {
          d_avx[iv] = 0.0;
          d_vvx[iv] = 0.0;
        }
        if (r == 0 || r == 1 || r == nrows - 1 || r == nrows) {
          d_avy[iv] = 0.0;
          d_vvy[iv] = 0.0;
        }
      }

      // (6) G2P + velocity/position update
      #pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double xx = d_x[ip];
        double yy = d_y[ip];
        int ci = d_p2g[ip];
        int r = gpu_gind2row(ci, nrows);
        int c = gpu_gind2col(ci, nrows);

        double vpx_v = 0.0, vpy_v = 0.0;
        double apx_v = 0.0, apy_v = 0.0;
        for (int in = 0; in < 4; in++) {
          double N  = gpu_shp(xx, yy, in, c, r, hx, hy);
          int  nidx = gpu_gt(in, c, r, nrows);
          vpx_v += N * d_vvx[nidx];
          vpy_v += N * d_vvy[nidx];
          apx_v += N * d_avx[nidx];
          apy_v += N * d_avy[nidx];
        }
        vpx_v += dt_local * apx_v;
        vpy_v += dt_local * apy_v;

        d_vpx[ip] = vpx_v;  d_vpy[ip] = vpy_v;
        d_apx[ip] = apx_v;  d_apy[ip] = apy_v;
        d_x[ip] += dt_local * vpx_v;
        d_y[ip] += dt_local * vpy_v;
      }

      // (7) G2PD + height update
      #pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double xx = d_x[ip];
        double yy = d_y[ip];
        int ci = d_p2g[ip];
        int r = gpu_gind2row(ci, nrows);
        int c = gpu_gind2col(ci, nrows);

        double vxdx = 0.0, vxdy = 0.0, vydx = 0.0, vydy = 0.0;
        for (int in = 0; in < 4; in++) {
          double Nx = gpu_shg(xx, yy, 0, in, c, r, hx, hy);
          double Ny = gpu_shg(xx, yy, 1, in, c, r, hx, hy);
          int  nidx = gpu_gt(in, c, r, nrows);
          vxdx += Nx * d_vvx[nidx];
          vxdy += Ny * d_vvx[nidx];
          vydx += Nx * d_vvy[nidx];
          vydy += Ny * d_vvy[nidx];
        }
        d_vpx_dx[ip] = vxdx;  d_vpx_dy[ip] = vxdy;
        d_vpy_dx[ip] = vydx;  d_vpy_dy[ip] = vydy;

        double sc = 1.0 + dt_local * (vxdx + vydy);
        d_hp[ip] /= sc;
        d_Ap[ip] /= sc;
        d_Vp[ip]  = d_hp[ip] * d_Ap[ip];
        d_mom_px[ip] = d_vpx[ip] * d_Mp[ip];
        d_mom_py[ip] = d_vpy[ip] * d_Mp[ip];
      }

      //Friction (Voellmy)
      #pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double vx = d_vpx[ip], vy = d_vpy[ip], h = d_hp[ip];
        double nv = sqrt(vx*vx + vy*vy);
        d_Fb_x[ip] = h > 1e-3 ? -0.0*2.5*(rho*h*g_c*tan_fa + rho*g_c*vx*vx/100.0)*vx/(nv+0.001) : 0.0;
        d_Fb_y[ip] = h > 1e-3 ? -0.0*2.5*(rho*h*g_c*tan_fa + rho*g_c*vy*vy/100.0)*vy/(nv+0.001) : 0.0;
      }

      //(8) Stress update (Bingham USL)
      #pragma omp target teams distribute parallel for
      for (int ip = 0; ip < np; ip++) {
        double vx = d_vpx[ip], vy = d_vpy[ip], h = d_hp[ip];
        double nv = sqrt(vx*vx + vy*vy);

        double alf = h > 1e-3 ? (6.0*mu*nv)/((h+0.001)*tau_Y) : 0.0;
        double bv = -114.0/32.0 - alf;
        double sq = sqrt(bv*bv - 4.0*A_coeff*C_coeff);
        double z1 = (-bv+sq)/(2.0*A_coeff);
        double z2 = (-bv-sq)/(2.0*A_coeff);
        double zz = 0.0; // ZZ=0 in original code

        double sxx = d_vpx_dx[ip];
        double sxy = 0.5*(d_vpx_dy[ip] + d_vpy_dx[ip]);
        double syy = d_vpy_dy[ip];
        double Dxx=sxx, Dxy=sxy, Dyy=syy;
        double Dzz = -(sxx + d_vpy_dy[ip]);
        double Dzx = h>1e-3 ? 0.5*(3.0/(2.0+zz))*(vx/(h+0.001)) : 0.0;
        double Dzy = h>1e-3 ? 0.5*(3.0/(2.0+zz))*(vy/(h+0.001)) : 0.0;
        double inv2 = 0.5*(Dxx*Dxx+Dyy*Dyy+Dzz*Dzz+Dzx*Dzx+Dzy*Dzy+Dxy*Dxy);
        double coeff = inv2!=0.0 ? (tau_Y/sqrt(inv2)+2.0*mu) : 0.0;

        d_F11[ip] = cc*coeff*Dxx - 0.5*rho*g_c*h;
        d_F12[ip] = cc*coeff*Dxy;
        d_F21[ip] = cc*coeff*Dxy;
        d_F22[ip] = cc*coeff*Dyy - 0.5*rho*g_c*h;
      }

      } // end target data
    }
    my_timer.toc("gpu_block");

    // CPU PHASE (post-GPU): hpZ, Plotvars P2G, VTS export
    
    // hpZ: g2p on CPU, then hpZ kernel on GPU with own map
    my_timer.tic("step 8_hpZ");
      ptcls.dprops.at("hpZ").assign(np, 0.0);
      ptcls.dprops.at("Zp").assign(np, 0.0);
      ptcls.g2p(vars, std::vector<std::string>{"Z"}, std::vector<std::string>{"Zp"});
    {
      double* d_hpZ = ptcls.dprops["hpZ"].data();
      double* d_Zp  = ptcls.dprops["Zp"].data();

      #pragma omp target teams distribute parallel for \
          map(to: d_hp[0:np], d_Zp[0:np]) \
          map(from: d_hpZ[0:np])
      for (int ip = 0; ip < np; ip++)
        d_hpZ[ip] = d_hp[ip] + d_Zp[ip];
    }
    my_timer.toc("step 8_hpZ");

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