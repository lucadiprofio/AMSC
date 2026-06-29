#include "counter.h"
#ifdef USE_DPL
#  include <oneapi/dpl/algorithm>
#  include <oneapi/dpl/execution>
#  include <oneapi/tbb/global_control.h>
#else
#  include <execution>
#endif
#include <fstream>
#include <functional>
#include <map>
#include <cmath>
#include <iostream>
#include <particles.h>
#include <quadgrid_cpp.h>
#include <timer.h>
#include "mpm_data.h"
#include <chrono>

#ifdef USE_DPL
using dpl::transform;
using dpl::for_each;
auto policy = dpl::execution::par;
#else
using std::transform;
using std::for_each;
auto policy = std::execution::par;
#endif

#ifdef USE_COMPRESSION
#  include <boost/iostreams/device/file.hpp>
#  include <boost/iostreams/filtering_stream.hpp>
#  include <boost/iostreams/filter/gzip.hpp>
#endif

struct stress_tensor_t {

  const std::vector<double>& vpx;
  const std::vector<double>& vpy;
  const std::vector<double>& hp;
  const std::vector<double>& H;
  const std::vector<double>& vpx_dx;
  const std::vector<double>& vpy_dy;
  const std::vector<double>& vpx_dy;
  const std::vector<double>& vpy_dx;
  std::vector<double>& F_11;
  std::vector<double>& F_12;
  std::vector<double>& F_21;
  std::vector<double>& F_22;
  const DATA& data;

  stress_tensor_t (particles_t &ptcls, DATA &data_) :
    vpx{ptcls.dprops.at ("vpx")},
    vpy{ptcls.dprops.at ("vpy")},
    hp{ptcls.dprops.at ("hp")},
    H{ptcls.dprops.at ("H")},
    vpx_dx{ptcls.dprops.at ("vpx_dx")},
    vpy_dy{ptcls.dprops.at ("vpy_dy")},
    vpx_dy{ptcls.dprops.at ("vpx_dy")},
    vpy_dx{ptcls.dprops.at ("vpy_dx")},
    F_11{ptcls.dprops.at ("F_11")},
    F_12{ptcls.dprops.at ("F_12")},
    F_21{ptcls.dprops.at ("F_21")},
    F_22{ptcls.dprops.at ("F_22")},
    data{data_} { }


  void operator()(int ip) {

    double A{3./2.}, B{-114./32.}, C{65./32.};
    double nrm{0.0}, ALF{0.0}, Z1{0.0}, Z2{0.0}, ZZ{0.0};
    double s_xx{0.0}, s_xy{0.0}, s_yy{0.0};
    double D_xx{0.0}, D_yx{0.0}, D_zx{0.0}, D_xy{0.0}, D_yy{0.0}, D_zy{0.0}, D_xz{0.0}, D_yz{0.0}, D_zz{0.0};
    double invII{0.0}, sig_yy{0.0}, sig_xy{0.0}, sig_xx{0.0};
    double BING = data.BINGHAM_ON;

    nrm = std::sqrt(vpx[ip] * vpx[ip] +vpy[ip] * vpy[ip] );

    double mu= data.mu;
    double tau_Y=data.tauy;

    ALF = hp[ip] > 1.e-3
      ? (6. * mu * nrm)/((hp[ip]+0.001) * tau_Y)
      : 0.0;
    B = -114./32. - ALF;
    Z1 = (-B + std::sqrt(B * B - 4. * A * C))/(2. * A);
    Z2 = (-B - std::sqrt(B * B - 4. * A * C))/(2. * A);
    ZZ = std::abs (Z1 - .5) <= .5 ? Z1 : Z2;

    s_xx = vpx_dx[ip];
    s_xy = 0.5 * (vpx_dy[ip] + vpy_dx[ip]);
    s_yy = vpy_dy[ip];

    D_xx = s_xx;
    D_yx = s_xy;
    D_zx = hp[ip] > 1.e-3 ? 0.5 * (3. / (2. + ZZ)) * (vpx[ip] / (hp[ip] + 0.001) ) : 0.0;

    D_xy = s_xy;
    D_yy = s_yy;
    D_zy = hp[ip] > 1.e-3 ? 0.5 * (3. / (2. + ZZ))  * (vpy[ip] / (hp[ip]+ 0.001) ) : 0.0;

    D_xz =  0.0;
    D_yz =  0.0;
    D_zz = - (vpx_dx[ip] + vpy_dy[ip]);

    invII = 0.5 * (D_xx * D_xx + D_yy * D_yy + D_zz * D_zz) +
                   D_zx * D_zx + D_zy * D_zy + D_xy * D_xy;

    sig_xx = invII != 0 ? (tau_Y/std::sqrt(invII) + 2. * mu) * D_xx : 0.0;
    sig_xy = invII != 0 ? (tau_Y/std::sqrt(invII) + 2. * mu) * D_xy : 0.0;
    sig_yy = invII != 0 ? (tau_Y/std::sqrt(invII) + 2. * mu) * D_yy : 0.0;

    double h_corr = hp[ip] > 1e-10 ? hp[ip] - (H[ip]*H[ip]) / hp[ip] : 0.0;

    F_11[ip] =  BING * sig_xx - .5 * data.rho * data.g * h_corr;
    F_12[ip] =  BING * sig_xy;
    F_21[ip] =  BING * sig_xy;
    F_22[ip] =  BING * sig_yy - .5 * data.rho * data.g * h_corr;

  }

};

using idx_t = quadgrid_t<std::vector<double>>::idx_t;
cdf::timer::timer_t my_timer{};


int main ()
{

  //tbb::global_control(tbb::global_control::max_allowed_parallelism, 1);

  DATA data ("DATA.json");

  quadgrid_t<std::vector<double>> grid;
  grid.set_sizes (data.Ney, data.Nex, data.hx, data.hy);

  idx_t num_particles = data.x.size();
  particles_t ptcls (num_particles, {"label"},
                     {"Mp", "Ap","vpx","vpy","mom_px","mom_py","hp",
                         "Vp","F_ext_px","F_ext_py","apx","apy",
                         "F_11","F_12","F_21","F_22","vpx_dx","vpx_dy",
                         "vpy_dx","vpy_dy","Fb_x","Fb_y","hpZ","dZxp","dZyp",
                         "Zp","xp","yp","Fric_px","Fric_py","Fpx","Fpy","vpxL","vpyL", "H"},
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


  for (idx_t ip = 0; ip < num_particles; ++ip)
    {
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


  std::iota(ptcls.iprops["label"].begin(),ptcls.iprops["label"].end(),0);

  double t = 0.0;
  double dt;
  double cel;

  bool WRITE_OUTPUT = true;
  double phi = data.phi;
  double atm = 100000.;

  //CAMBIARE QUI IN BASE AL TEST
  //TODO RIVECERLO DAL JSON?
  double fric_ang = phi * M_PI / 180.;
  double atan_grad_z;
  std::vector<double> norm_v (num_particles, 0.0);
  std::vector<double> div_vp (num_particles, 0.0);


  std::map<std::string, std::vector<double>>
    vars{
    {"Mv", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"mom_vx", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"mom_vy", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"F_ext_vx", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"F_ext_vy", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"F_int_vx", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"F_int_vy", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"Fric_x", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"Fric_y", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"div_v", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"avx", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"avy", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"vvx", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"vvy", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"Z", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"dZdx", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"dZdy", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"Ftot_vx", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"Ftot_vy", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"HV", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"FPxv", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"FPyv", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"vvxL", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"vvyL", std::vector<double>(grid.num_global_nodes (), 0.0)}
  },
    Plotvars{
    {"rho_v", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"avx", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"avy", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"vvx", std::vector<double>(grid.num_global_nodes (), 0.0)},
    {"vvy", std::vector<double>(grid.num_global_nodes (), 0.0)}
    };


    vars["Z"] = data.Z;
    vars["dZdx"] = data.dZdx;
    vars["dZdy"] = data.dZdy;

    my_timer.tic ("g2p");
    ptcls.g2p (vars, {"Z"}, {"Zp"});
    my_timer.toc ("g2p");

    ptcls.dprops.at("dZxp").assign(ptcls.num_particles, 0.0);
    ptcls.dprops.at("dZyp").assign(ptcls.num_particles, 0.0);

    my_timer.tic ("g2p");
    ptcls.g2p (vars, {"dZdx","dZdy"}, {"dZxp","dZyp"});
    my_timer.toc ("g2p");  

    for (idx_t ip = 0; ip < num_particles; ++ip) {
      ptcls.dprops["hpZ"][ip] = ptcls.dprops["hp"][ip] + ptcls.dprops["Zp"][ip];

      double diff = data.eq_level - ptcls.dprops["Zp"][ip];
      ptcls.dprops["H"][ip] = (data.eq_level > 0 && diff > 0) ? diff : 0.0;
    }

    for (idx_t ip = 0; ip<num_particles; ++ip)  {

        double h_corr = ptcls.dprops["hp"][ip] > 1e-10 ? ptcls.dprops["hp"][ip] - (ptcls.dprops["H"][ip]*ptcls.dprops["H"][ip]) / ptcls.dprops["hp"][ip] : 0.0;
        ptcls.dprops["F_11"][ip] = - .5 * data.rho * data.g * h_corr;
        ptcls.dprops["F_12"][ip] = 0.0;
        ptcls.dprops["F_21"][ip] = 0.0;
        ptcls.dprops["F_22"][ip] = - .5 * data.rho * data.g * h_corr;
    }

    int it = 0;

    ptcls.build_mass();
    grid.vtk_export("GRID_forZ.vts", vars);
    dt = 1.0e-3;
    std::vector<idx_t> ordering (ptcls.num_particles);

    // Inizializzazione file per gli errori di conservazione
    std::ofstream err_file("conservation_errors.csv");
    err_file << "time,err_mass,err_mom\n";

    const int WARMUP = 5;
  std::chrono::high_resolution_clock::time_point t_start;
  bool fixed = (data.DT_FIXED > 0.0);

  while (fixed ? (it < data.NSTEPS) : (t < data.T)) {
    if (it == WARMUP) t_start = std::chrono::high_resolution_clock::now();

       double dt;
    if (fixed) {
      dt = data.DT_FIXED;
    } else {
      my_timer.tic ("update dt");
      double max_vel_x = *std::max_element(ptcls.dprops["vpx"].begin(), ptcls.dprops["vpx"].end());
        double min_vel_x = *std::min_element(ptcls.dprops["vpx"].begin(), ptcls.dprops["vpx"].end());

        max_vel_x = std::max (std::abs(max_vel_x), std::abs(min_vel_x));

        double max_vel_y = *std::max_element(ptcls.dprops["vpy"].begin(), ptcls.dprops["vpy"].end());
        double min_vel_y = *std::min_element(ptcls.dprops["vpy"].begin(), ptcls.dprops["vpy"].end());

        max_vel_y = std::max (std::abs(max_vel_y), std::abs(min_vel_y));

        double hmean = *std::max_element (ptcls.dprops["hp"].begin(), ptcls.dprops["hp"].end());
        double max_vel = std::max(std::sqrt(data.g * hmean) + max_vel_x, std::sqrt(data.g * hmean) + max_vel_y);
        cel = std::abs(max_vel);

        if (it > 0)
        dt = data.CFL * std::min(data.hx, data.hy) / (1e-2 + cel); // to avoid cell-crossing
        std::cout << "time = " << t << "  " << " dt = " <<  dt << std::endl;
        std::cout << "cel = " << cel << std::endl;
        my_timer.toc ("update dt");
    }
    it++;

        if(WRITE_OUTPUT==true){
        my_timer.tic ("save csv");
        std::string filename = "nc_particles_";
        filename = filename + std::to_string (it);
        filename = filename + ".csv";
        #ifdef USE_COMPRESSION
          filename = filename + ".gz";
        #endif
                if (it % 10 == 0)
                  {
        #ifdef USE_COMPRESSION
              boost::iostreams::filtering_ostream OF;
              OF.push (boost::iostreams::gzip_compressor());
              OF.push (boost::iostreams::file_sink (filename));
        #else
                    std::ofstream OF (filename.c_str ());
        #endif
                    ptcls.print<particles_t::output_format::csv>(OF);
        #ifdef USE_COMPRESSION
              boost::iostreams::close (OF);
        #else
                    OF.close ();
        #endif
                  }
                my_timer.toc ("save csv");

        }

        //  (0)  CONNECTIVITY and BASIS FUNCTIONS
        my_timer.tic ("reorder");
        ptcls.init_particle_mesh ();

        // ordering.resize (ptcls.num_particles);
        // idx_t iordering = 0;
        // for (auto const & ii : ptcls.grd_to_ptcl) {
        //   for (auto const & jj : ii.second) {
        //     ordering[iordering++] = jj;
        //   }
        // }

        // ptcls.reorder (ordering);
        // iordering = 0;
        // for (auto & ii : ptcls.grd_to_ptcl) {
        //   for (auto & jj : ii.second) {
        //     jj = iordering++;
        //   }
        // }
        // ptcls.init_particle_mesh ();
        my_timer.toc ("reorder");

        my_timer.tic ("step 0");
        for (auto &v : vars) {
          v.second.assign (v.second.size (), 0.0);
        }

        for (auto &v : Plotvars) {
          v.second.assign (v.second.size (), 0.0);
        }
        vars["Z"] = data.Z;
        vars["dZdx"] = data.dZdx;
        vars["dZdy"] = data.dZdy;

        ptcls.dprops.at("dZxp").assign(ptcls.num_particles, 0.0);
        ptcls.dprops.at("dZyp").assign(ptcls.num_particles, 0.0);

        my_timer.toc ("step 0");

        // (1) PROJECTION FROM MP TO NODES (P2G)

        my_timer.tic("cpu_block");

        my_timer.tic ("p2g");
        ptcls.p2g (vars, {"Mp","mom_px","mom_py"}, {"Mv","mom_vx","mom_vy"});
        my_timer.toc ("p2g");

        my_timer.tic ("g2pd");
        ptcls.g2pd (vars, {"Z"}, {"dZxp"}, {"dZyp"});
        my_timer.toc ("g2pd");


        if(WRITE_OUTPUT){
        // ==========================================
        // TEST CONSERVAZIONE MASSA E MOMENTO (L-inf)
        // ==========================================
        double total_mass_particles = 0.0;
        double total_mom_px_particles = 0.0;
        double total_mom_py_particles = 0.0;

        for (idx_t ip = 0; ip < ptcls.num_particles; ++ip) {
            total_mass_particles += ptcls.dprops["Mp"][ip];
            total_mom_px_particles += ptcls.dprops["mom_px"][ip];
            total_mom_py_particles += ptcls.dprops["mom_py"][ip];
        }

        double total_mass_nodes = 0.0;
        double total_mom_px_nodes = 0.0;
        double total_mom_py_nodes = 0.0;

        for (idx_t in = 0; in < grid.num_global_nodes(); ++in) {
            total_mass_nodes += vars["Mv"][in];
            total_mom_px_nodes += vars["mom_vx"][in];
            total_mom_py_nodes += vars["mom_vy"][in];
        }

// Calcolo dell'errore RELATIVO (Normalizzato)
        double err_mass = std::abs(total_mass_particles - total_mass_nodes) / total_mass_particles;
        
        // Per il momento, usiamo un denominatore sicuro per evitare divisioni per zero a t=0 (quando la frana è ferma)
        double norm_mom = std::max(1.0, std::max(std::abs(total_mom_px_particles), std::abs(total_mom_py_particles)));
        
        double err_mom_x = std::abs(total_mom_px_particles - total_mom_px_nodes) / norm_mom;
        double err_mom_y = std::abs(total_mom_py_particles - total_mom_py_nodes) / norm_mom;
        
        // Norma L-infinito del momento 
        double err_mom = std::max(err_mom_x, err_mom_y);

        // Salvataggio su file CSV (a ogni iterazione)
        err_file << t << "," << err_mass << "," << err_mom << "\n";

        // Stampa a schermo (solo ogni 100 iterazioni per non spammare)
        if (it % 100 == 0) {
            std::cout << "[Test] t = " << t << " | Err Massa: " << err_mass 
                      << " | Err Momento: " << err_mom << std::endl;
        }
        // ==========================================
      }     
        // (2)  EXTERNAL FORCES ON VERTICES (P2G)
        my_timer.tic ("step 2a");
        auto& Vp = ptcls.dprops.at("Vp");
        auto& dZxp = ptcls.dprops.at("dZxp");
        auto& dZyp = ptcls.dprops.at("dZyp");
        auto& Fpx = ptcls.dprops.at("Fpx");
        auto& Fpy = ptcls.dprops.at("Fpy");
        auto& hp = ptcls.dprops.at("hp");
        auto& H = ptcls.dprops.at("H");

        for_each(policy, ptcls.iprops["label"].begin(), ptcls.iprops["label"].end(), [=, &Vp, &dZxp, &dZyp, &Fpx, &Fpy, &hp, &H, &data](int ip) {
            
            // Fattore di bilanciamento. Se H = 0 (frana normale), il moltiplicatore è 1.0.
            double wb_factor = hp[ip] > 1e-10 ? 1.0 - (H[ip] / hp[ip]) : 1.0;
            
            Fpx[ip] = -data.g * Vp[ip] * data.rho * dZxp[ip] * wb_factor;
            Fpy[ip] = -data.g * Vp[ip] * data.rho * dZyp[ip] * wb_factor;
        });

        transform (policy, ptcls.dprops["Ap"].begin (), ptcls.dprops["Ap"].end (), ptcls.dprops["Fb_x"].begin (),
                   ptcls.dprops["Fric_px"].begin (), std::multiplies<double> ());
        transform (policy, ptcls.dprops["Ap"].begin (), ptcls.dprops["Ap"].end (), ptcls.dprops["Fb_y"].begin (),
                   ptcls.dprops["Fric_py"].begin (), std::multiplies<double> ());
        my_timer.toc ("step 2a");

        my_timer.tic ("p2g");
        ptcls.p2g (vars, {"Fpx","Fpy","Fric_px","Fric_py"}, {"FPxv","FPyv","Fric_x","Fric_y"});
        my_timer.toc ("p2g");

        my_timer.tic ("step 2b");
        transform (policy, vars["FPxv"].begin (), vars["FPxv"].end (), vars["Fric_x"].begin (),
                   vars["F_ext_vx"].begin (), std::plus<double> ());
        transform (policy, vars["FPyv"].begin (), vars["FPyv"].end (), vars["Fric_y"].begin (),
                   vars["F_ext_vy"].begin (), std::plus<double> ());

        my_timer.toc ("step 2b");

        // (3) INTERNAL FORCES (p2gd) and MOMENTUM BALANCE
        my_timer.tic ("p2gd");
        ptcls.p2gd (vars, {"F_11","F_21"}, {"F_12","F_22"}, "Vp", {"F_int_vx","F_int_vy"});
        my_timer.toc ("p2gd");

        my_timer.tic ("step 3");

        transform (policy, vars["F_ext_vx"].begin (), vars["F_ext_vx"].end (), vars["F_int_vx"].begin (), vars["Ftot_vx"].begin (), std::minus<double> ());
        transform (policy, vars["F_ext_vy"].begin (), vars["F_ext_vy"].end (), vars["F_int_vy"].begin (), vars["Ftot_vy"].begin (), std::minus<double> ());

        transform (policy, vars["Ftot_vx"].begin (), vars["Ftot_vx"].end (), vars["mom_vx"].begin (), vars["mom_vx"].begin (), [=] (double x, double y) { return dt*x + y; });
        transform (policy, vars["Ftot_vy"].begin (), vars["Ftot_vy"].end (), vars["mom_vy"].begin (), vars["mom_vy"].begin (), [=] (double x, double y) { return dt*x + y; });

        my_timer.toc ("step 3");

        // (4)  COMPUTE NODAL ACCELERATIONS AND VELOCITIES
        my_timer.tic ("step 4");

        transform (policy, vars["Ftot_vx"].begin (), vars["Ftot_vx"].end (), vars["Mv"].begin (), vars["avx"].begin (), [] (double x, double y) { return y > 1.e-2 ? x/y : 0.0; });
        transform (policy, vars["Ftot_vy"].begin (), vars["Ftot_vy"].end (), vars["Mv"].begin (), vars["avy"].begin (), [] (double x, double y) { return y > 1.e-2 ? x/y : 0.0; });
        transform (policy, vars["mom_vx"].begin (), vars["mom_vx"].end (), vars["Mv"].begin (), vars["vvx"].begin (), [] (double x, double y) { return y > 1.e-2 ? x/y : 0.0; });
        transform (policy, vars["mom_vy"].begin (), vars["mom_vy"].end (), vars["Mv"].begin (), vars["vvy"].begin (), [] (double x, double y) { return y > 1.e-2 ? x/y : 0.0; });
        transform (policy, vars["avx"].begin (), vars["avx"].end (),vars["vvx"].begin (), vars["vvxL"].begin (), [=] (double x, double y) { return dt * x  + y; });
        transform (policy, vars["avy"].begin (), vars["avy"].end (),vars["vvy"].begin (), vars["vvyL"].begin (), [=] (double x, double y) { return dt * x  + y; });
        my_timer.toc ("step 4");

        // (5) BOUNDARY CONDITIONS - TO DO
        my_timer.tic ("step 5");
        // We assume the slide will never reach the boundary and do nothing here! 
        if (data.BC_FLAG)
        {
          for (auto icell = grid.begin_cell_sweep ();
              icell != grid.end_cell_sweep (); ++icell)
              {
                if ( (icell->e (2) == 2) || (icell->e(3)==3)  )
                {
                  for (idx_t inode = 0; inode < 4; ++inode)
                  {
                    vars["avx"][icell->gt(inode)] = 0.0;
                    vars["vvx"][icell->gt(inode)] = 0.0;
                    vars["vvxL"][icell->gt(inode)] = 0.0; // <- Il VERO blocco per il metodo PIC
                    vars["mom_vx"][icell->gt(inode)] = 0.0;
                  }
                }
                if ( (icell->e(0) == 0) || (icell->e(1)==1) )
                  {
                   for (idx_t inode = 0; inode < 4; ++inode)
                  {
                    vars["avy"][icell->gt(inode)] = 0.0;
                    vars["vvy"][icell->gt(inode)] = 0.0;
                    vars["vvyL"][icell->gt(inode)] = 0.0; // <- Il VERO blocco per il metodo PIC                    
                    vars["mom_vy"][icell->gt(inode)] = 0.0;
                  }
              }
          }
        }
        my_timer.toc ("step 5");
// (6) RETURN TO POINTS (G2P) and UPDATE POS AND VEL ON PARTICLES
        my_timer.tic ("step 6");
        // Non ci serve più la storia precedente per il FLIP, implementiamo il PIC puro.
        ptcls.dprops.at("vpxL").assign(ptcls.num_particles, 0.0);
        ptcls.dprops.at("vpyL").assign(ptcls.num_particles, 0.0);
        ptcls.dprops.at("apx").assign(ptcls.num_particles, 0.0);
        ptcls.dprops.at("apy").assign(ptcls.num_particles, 0.0);
        my_timer.toc ("step 6");

        my_timer.tic ("g2p");
        // Mappiamo le velocità aggiornate (vvxL, vvyL) dalla griglia alle particelle (vpxL, vpyL)
        ptcls.g2p (vars, {"vvxL","vvyL","avx","avy"}, {"vpxL","vpyL","apx","apy"});
        my_timer.toc ("g2p");

        my_timer.tic ("step 6b");  
        
        // PIC Puro come prescritto a pagina 6, equazione 33 del paper
        transform (policy, ptcls.dprops["vpxL"].begin (), ptcls.dprops["vpxL"].end (), ptcls.dprops["vpx"].begin (), [] (double x) { return x; } );
        transform (policy, ptcls.dprops["vpyL"].begin (), ptcls.dprops["vpyL"].end (), ptcls.dprops["vpy"].begin (), [] (double x) { return x; } );

        // Aggiornamento delle posizioni (Eq 35 del paper)
        transform (policy, ptcls.x.begin (), ptcls.x.end (),  ptcls.dprops["vpx"].begin (), ptcls.x.begin (), [=] (double x, double y) { return x + dt * y; } );
        transform (policy, ptcls.y.begin (), ptcls.y.end (),  ptcls.dprops["vpy"].begin (), ptcls.y.begin (), [=] (double x, double y) { return x + dt * y; } );

        // Gestione dei bordi (evita che le particelle escano dal dominio)
        double eps = 1e-10;
        double Lx = data.Nex * data.hx;
        double Ly = data.Ney * data.hy;
        for (idx_t ip = 0; ip < ptcls.num_particles; ++ip) {
          if (ptcls.x[ip] < eps) ptcls.x[ip] = eps;
          if (ptcls.x[ip] > Lx - eps) ptcls.x[ip] = Lx - eps;
          if (ptcls.y[ip] < eps) ptcls.y[ip] = eps;
          if (ptcls.y[ip] > Ly - eps) ptcls.y[ip] = Ly - eps;
        }
        my_timer.toc ("step 6b");

        // (7) COMPUTE HEIGHT WITH STRAIN (divergence of velocities)
        my_timer.tic ("step 7a");
        ptcls.dprops.at("vpx_dx").assign(ptcls.num_particles, 0.0);
        ptcls.dprops.at("vpy_dy").assign(ptcls.num_particles, 0.0);
        // FIX 1: Azzeriamo anche le derivate di taglio per evitare l'esplosione dello stress!
        ptcls.dprops.at("vpx_dy").assign(ptcls.num_particles, 0.0);
        ptcls.dprops.at("vpy_dx").assign(ptcls.num_particles, 0.0);
        my_timer.toc ("step 7a");

        my_timer.tic ("g2pd");
        // FIX 2: Usiamo vvxL e vvyL (velocità aggiornate) come richiede l'Equazione 36 del paper
        ptcls.g2pd (vars, {"vvxL","vvyL"}, {"vpx_dx","vpy_dx"}, {"vpx_dy","vpy_dy"});
        my_timer.toc ("g2pd");


        /*my_timer.tic ("step 7");
        transform (policy, ptcls.dprops["vpx_dx"].begin (), ptcls.dprops["vpx_dx"].end (), ptcls.dprops["vpy_dy"].begin (), div_vp.begin (), std::plus<double> ());
        transform (policy, ptcls.dprops["hp"].begin (), ptcls.dprops["hp"].end (), div_vp.begin (), ptcls.dprops["hp"].begin (), [dt] (double x, double y) { return x / (1 + dt * y); } );
        transform (policy, ptcls.dprops["vpx"].begin (), ptcls.dprops["vpx"].end (), ptcls.dprops["Mp"].begin (), ptcls.dprops["mom_px"].begin (), std::multiplies<double> () );
        transform (policy, ptcls.dprops["vpy"].begin (), ptcls.dprops["vpy"].end (), ptcls.dprops["Mp"].begin (), ptcls.dprops["mom_py"].begin (), std::multiplies<double> () );
        transform (policy, ptcls.dprops["Vp"].begin (), ptcls.dprops["Vp"].end (), div_vp.begin (), ptcls.dprops["Vp"].begin (), [dt] (double x, double y) { return x / (1 + dt * y); } );
       // transform (policy, ptcls.dprops["hp"].begin (), ptcls.dprops["hp"].end (), ptcls.dprops["Mp"].begin (),ptcls.dprops["Ap"].begin (), [=] (double x, double y) { return y / (1e-4 + data.rho * x); } );
      // transform (policy, ptcls.dprops["hp"].begin (), ptcls.dprops["hp"].end (), ptcls.dprops["Ap"].begin (),ptcls.dprops["Mp"].begin (), [] (double x, double y) { return x * y * 1291.0 ; } );
        transform (policy, ptcls.dprops["Vp"].begin (), ptcls.dprops["Vp"].end (), ptcls.dprops["hp"].begin (), ptcls.dprops["Ap"].begin (), [] (double x, double y) { return y > 1.e-4 ? x/y : 0.0; } );//std::divides<double>()

        my_timer.toc ("step 7");*/
        
        my_timer.tic ("step 7");
        transform (policy, ptcls.dprops["vpx_dx"].begin (), ptcls.dprops["vpx_dx"].end (), ptcls.dprops["vpy_dy"].begin (), div_vp.begin (), std::plus<double> ());
        transform (policy, ptcls.dprops["hp"].begin (), ptcls.dprops["hp"].end (), div_vp.begin (), ptcls.dprops["hp"].begin (), [=] (double x, double y) { return x / (1 + dt * y); } );
        transform (policy, ptcls.dprops["hp"].begin (), ptcls.dprops["hp"].end (),
           ptcls.dprops["hp"].begin (),
           [] (double h) { return h < 1e-2 ? 1e-2 : h; });
        transform (policy, ptcls.dprops["vpx"].begin (), ptcls.dprops["vpx"].end (), ptcls.dprops["Mp"].begin (), ptcls.dprops["mom_px"].begin (), std::multiplies<double> () );
        transform (policy, ptcls.dprops["vpy"].begin (), ptcls.dprops["vpy"].end (), ptcls.dprops["Mp"].begin (), ptcls.dprops["mom_py"].begin (), std::multiplies<double> () );
        //transform (policy, ptcls.dprops["Vp"].begin (), ptcls.dprops["Vp"].end (), div_vp.begin (), ptcls.dprops["Vp"].begin (), [=] (double x, double y) { return x / (1 + dt * y); } );
       // transform (policy, ptcls.dprops["hp"].begin (), ptcls.dprops["hp"].end (), ptcls.dprops["Mp"].begin (),ptcls.dprops["Ap"].begin (), [=] (double x, double y) { return y / (1e-4 + data.rho * x); } );
      // transform (policy, ptcls.dprops["hp"].begin (), ptcls.dprops["hp"].end (), ptcls.dprops["Ap"].begin (),ptcls.dprops["Mp"].begin (), [] (double x, double y) { return x * y * 1291.0 ; } );
        transform (policy, ptcls.dprops["Vp"].begin (), ptcls.dprops["Vp"].end (),
           ptcls.dprops["hp"].begin (), ptcls.dprops["Ap"].begin (),
           [] (double x, double y) { return y > 1.e-2 ? x/y : 0.0; } );

        my_timer.toc ("step 7");

        // (7b) UPDATE FRICTIONS
        my_timer.tic ("step 7b");
         for (idx_t ip = 0; ip < ptcls.num_particles; ++ip) {
          double vx = ptcls.dprops["vpx"][ip];
          double vy = ptcls.dprops["vpy"][ip];
          double h = ptcls.dprops["hp"][ip];
          double nv = std::sqrt(vx*vx + vy*vy);
          if (data.FRICTION_ON > 0 && nv > 1e-10 && data.xi > 0) {
            double v2 = vx*vx + vy*vy;
            ptcls.dprops["Fb_x"][ip] = -data.FRICTION_ON * (data.rho * data.g * h * std::tan(fric_ang) + data.rho * data.g * v2 / data.xi) * vx / nv;
            ptcls.dprops["Fb_y"][ip] = -data.FRICTION_ON * (data.rho * data.g * h * std::tan(fric_ang) + data.rho * data.g * v2 / data.xi) * vy / nv;
          } else {
            ptcls.dprops["Fb_x"][ip] = 0.0;
            ptcls.dprops["Fb_y"][ip] = 0.0;
          }
        }
        my_timer.toc ("step 7b");


        // (8) UPDATE PARTICLE STRESS (USL)
        my_timer.tic ("step 8");

        range<idx_t> rng(0, ptcls.num_particles);
        stress_tensor_t st(ptcls, data);
        for_each (policy, rng.begin (), rng.end (), st);

        ptcls.dprops.at("hpZ").assign(ptcls.num_particles, 0.0);
        ptcls.dprops.at("Zp").assign(ptcls.num_particles, 0.0);
        my_timer.toc ("step 8");

        my_timer.tic ("g2p");
        ptcls.g2p (vars, {"Z"}, {"Zp"});
        my_timer.toc ("g2p");

        my_timer.tic ("step 8b");
        transform (policy, ptcls.dprops["hp"].begin (), ptcls.dprops["hp"].end (), ptcls.dprops["Zp"].begin (), ptcls.dprops["hpZ"].begin (), std::plus<double> ());
        my_timer.toc ("step 8b");

        my_timer.tic ("p2g");
        ptcls.p2g (vars,std::vector<std::string>{"hp"}, std::vector<std::string>{"HV"});
        my_timer.toc ("p2g");

        my_timer.toc("cpu_block");

        if(WRITE_OUTPUT==true){
         my_timer.tic ("save vts");
         std::string filename = "nc_grid_";
         filename = filename + std::to_string (it);
         filename = filename + ".vts";
        if (it % 10 == 0) {
          grid.vtk_export(filename.c_str(), Plotvars);
          grid.vtk_export(filename.c_str(), vars);
         }
         my_timer.toc ("save vts");
        }
        t +=dt;

      }

      auto t_end = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t_end - t_start).count();
  int counted = it - WARMUP;
  std::cout << "PERF np=" << ptcls.num_particles
            << " counted_steps=" << counted
            << " tot_s=" << sec
            << " per_step_ms=" << 1e3*sec/counted << std::endl;

    my_timer.print_report ();
    //  ptcls.print<particles_t::output_format::csv>(std::cout);

    return 0;
}