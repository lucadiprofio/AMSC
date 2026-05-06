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

    }


  std::iota(ptcls.iprops["label"].begin(),ptcls.iprops["label"].end(),0);

  double t = 0.0;
  double dt  ;
  double cel;
  double phi = 0.0;
  double atm = 100000.;
  double fric_ang = 34. * M_PI / 180.;
  double atan_grad_z;
  std::vector<double> norm_v (num_particles, 0.0);
  double A = 3./2.;
  std::vector<double> ALF (num_particles, 0.0);
  std::vector<double> B (num_particles, -114./32.);
  double C = 65./32.;
  std::vector<double> s_xx (num_particles, 0.0);
  std::vector<double> s_xy (num_particles, 0.0);
  std::vector<double> s_yy (num_particles, 0.0);
  std::vector<double> D_xx (num_particles, 0.0);
  std::vector<double> D_xy (num_particles, 0.0);
  std::vector<double> D_xz (num_particles, 0.0);
  std::vector<double> D_yx (num_particles, 0.0);
  std::vector<double> D_yy (num_particles, 0.0);
  std::vector<double> D_yz (num_particles, 0.0);
  std::vector<double> D_zx (num_particles, 0.0);
  std::vector<double> D_zy (num_particles, 0.0);
  std::vector<double> D_zz (num_particles, 0.0);
  std::vector<double> invII (num_particles, 0.0);
  std::vector<double> Z1 (num_particles, 0.0);
  std::vector<double> Z2 (num_particles, 0.0);
  std::vector<double> ZZ (num_particles, 0.0);
  std::vector<double> sig_xx (num_particles, 0.0);
  std::vector<double> sig_xy (num_particles, 0.0);
  std::vector<double> sig_yy (num_particles, 0.0);




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
      {"HV", std::vector<double>(grid.num_global_nodes (), 0.0)}
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


    ptcls.g2p (vars,std::vector<std::string>{"Z"},
	       std::vector<std::string>{"Zp"});
    ptcls.dprops.at("dZxp").assign(ptcls.num_particles, 0.0);
    ptcls.dprops.at("dZyp").assign(ptcls.num_particles, 0.0);
    ptcls.g2p (vars,std::vector<std::string>{"dZdx","dZdy"},
	       std::vector<std::string>{"dZxp","dZyp"});

    for (idx_t ip = 0; ip < num_particles; ++ip)
      {
	      ptcls.dprops["hpZ"][ip] = ptcls.dprops["hp"][ip] + ptcls.dprops["Zp"][ip];
      }

      for (idx_t ip = 0; ip<num_particles; ++ip)
  {
    ptcls.dprops["F_11"][ip] = -  0.5 * data.rho * data.g *   (ptcls.dprops["hp"][ip]  ) ;
    ptcls.dprops["F_12"][ip] = 0.0;
    ptcls.dprops["F_21"][ip] = 0.0;
    ptcls.dprops["F_22"][ip] = - 0.5 * data.rho * data.g *   (ptcls.dprops["hp"][ip] );
  }

    int it = 0;

    ptcls.build_mass();


    grid.vtk_export("GRID_forZ.vts", vars);


    dt = 1.0e-5;
    std::vector<idx_t> ordering (ptcls.num_particles);
    while (t < 1.0) //data.T;
        {

  	my_timer.tic ("update dt");
    /*      double max_vel_x = *std::max_element(ptcls.dprops["vpx"].begin(), ptcls.dprops["vpx"].end());
            double max_vel_y = *std::max_element(ptcls.dprops["vpy"].begin(), ptcls.dprops["vpy"].end());
            double hmean = accumulate(ptcls.dprops["hp"].begin(), ptcls.dprops["hp"].end(), 0.0/ptcls.dprops["hp"].size());
            double max_vel = std::max(std::sqrt(data.g * hmean)+max_vel_x,-std::sqrt(data.g * hmean)+max_vel_y);
          //  double max_vel = std::max(1+max_vel_x,1+max_vel_y);
            cel = std::abs(max_vel); */

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
	      dt = 0.2* data.hx / (1e-2 + cel); // 0.7      0.2 *  data.hx / (1e-4 + cel);
	      std::cout << "time = " << t << "  " << " dt = " <<  dt << std::endl;
    my_timer.toc ("update dt");

    my_timer.tic ("save csv");

            std::string filename = "nc_particles_";
            filename = filename + std::to_string (it++);
            filename = filename + ".csv";
            if (t>=0.0)//(it % 50 == 0)
            {
              std::ofstream OF (filename.c_str ());
              ptcls.print<particles_t::output_format::csv>(OF);
              OF.close ();
            }
    my_timer.toc("save csv");

	//  (0)  CONNECTIVITY and BASIS FUNCTIONS
	my_timer.tic ("reorder");
        ptcls.init_particle_mesh ();

  /*      ordering.resize (ptcls.num_particles);
        idx_t iordering = 0;
        for (auto const & ii : ptcls.grd_to_ptcl) {
          for (auto const & jj : ii.second) {
            ordering[iordering++] = jj;
          }
        }

        ptcls.reorder (ordering);
        iordering = 0;
        for (auto & ii : ptcls.grd_to_ptcl) {
          for (auto & jj : ii.second) {
            jj = iordering++;
          }
        }
    my_timer.toc ("reorder"); */

    my_timer.tic("step 0");
        for (auto &v : vars)
	  {
            v.second.assign (v.second.size (),0.0);
	  }

        for (auto &v : Plotvars)
	  {
            v.second.assign (v.second.size (),0.0);
	  }




        ptcls.dprops.at("dZxp").assign(ptcls.num_particles, 0.0);
        ptcls.dprops.at("dZyp").assign(ptcls.num_particles, 0.0);
        vars["Z"] = data.Z;
        vars["dZdx"] = data.dZdx;
        vars["dZdy"] = data.dZdy;

	my_timer.toc ("step 0");

	// (1) PROJECTION FROM MP TO NODES (P2G)
	my_timer.tic ("step 1");
        ptcls.p2g (vars,std::vector<std::string>{"Mp","mom_px","mom_py"},
		   std::vector<std::string>{"Mv","mom_vx","mom_vy"});
	ptcls.g2pd (vars,std::vector<std::string>{"Z"},
		    std::vector<std::string>{"dZxp"},
		    std::vector<std::string>{"dZyp"});

	my_timer.toc ("step 1");

	// (2)  EXTERNAL FORCES ON VERTICES (P2G)
	my_timer.tic ("step 2");
	for (auto icell = grid.begin_cell_sweep (); icell != grid.end_cell_sweep (); ++icell)
	  {
	    for (auto inode = 0; inode < quadgrid_t<std::vector<double>>::cell_t::nodes_per_cell; ++inode)
	      {
		auto gv = icell -> gt (inode);
   	if (ptcls.grd_to_ptcl.count (icell->get_global_cell_idx ()) > 0)
		  for (auto ip = 0; ip < ptcls.grd_to_ptcl.at (icell->get_global_cell_idx ()).size (); ++ip)
		     {
		      auto gp = ptcls.grd_to_ptcl.at(icell->get_global_cell_idx ())[ip];
		      ptcls.dprops["Fric_px"][gp] = 0.0 * ptcls.dprops["Ap"][gp] *  ptcls.dprops["Fb_x"][gp]  ; // - ptcls.dprops["Mp"][gp] *    9.81 * ptcls.dprops["dZxp"][gp] + ptcls.dprops["Ap"][gp] * ptcls.dprops["Fb_x"][gp]  ;
		      ptcls.dprops["Fric_py"][gp] = 0.0 * ptcls.dprops["Ap"][gp] * ptcls.dprops["Fb_y"][gp]  ;//  - ptcls.dprops["Mp"][gp] *    9.81 * ptcls.dprops["dZyp"][gp]  + ptcls.dprops["Ap"][gp] *  ptcls.dprops["Fb_y"][gp]  ;
		     }
	     }

	  }

	ptcls.p2g (vars,std::vector<std::string>{"Fric_px","Fric_py"},
		  std::vector<std::string>{"Fric_x","Fric_y"});
//    ptcls.p2g (vars,std::vector<std::string>{"Fric_px","Fric_py"},
//  		   std::vector<std::string>{"F_ext_vx","F_ext_vy"});

	for (auto icell = grid.begin_cell_sweep ();
             icell != grid.end_cell_sweep (); ++icell)
	  {
	    for (auto inode = 0; inode < quadgrid_t<std::vector<double>>::cell_t::nodes_per_cell; ++inode)
	      {
		auto gv = icell -> gt (inode);

		vars["F_ext_vx"][gv] = -  vars["Mv"][gv] *  9.81 * 0.0 * vars["dZdx"][gv] + vars["Fric_x"][gv]; // vars["dZdx"][gv]; // vars["dZdx"][gv]; //  dZdx[gv]  ; //-  ptcls.dprops["Ap"][gp] * ptcls.dprops["Fb_x"][gp];

		vars["F_ext_vy"][gv] = -  vars["Mv"][gv] *  9.81 * 0.0 *  vars["dZdy"][gv] + vars["Fric_y"][gv]; // vars["dZdy"][gv]; //  -   ptcls.dprops["Ap"][gp] *  ptcls.dprops["Fb_y"][gp];

	      }

	  }

	my_timer.toc ("step 2");

	// (3) INTERNAL FORCES (p2gd) and MOMENTUM BALANCE
	my_timer.tic ("step 3");
        ptcls.p2gd (vars, std::vector<std::string>{"F_11","F_21"},
		    std::vector<std::string>{"F_12","F_22"},
		    "Vp",std::vector<std::string>{"F_int_vx","F_int_vy"});



	for (auto icell = grid.begin_cell_sweep ();
	     icell != grid.end_cell_sweep (); ++icell)
	  {
            for (auto inode = 0; inode < quadgrid_t<std::vector<double>>::cell_t::nodes_per_cell; ++inode)
	      {
		auto iv = icell -> gt (inode);
		vars["Ftot_vx"][iv] = vars["F_ext_vx"][iv] - vars["F_int_vx"][iv];
		vars["Ftot_vy"][iv] = vars["F_ext_vy"][iv] - vars["F_int_vy"][iv];
	      }

	  }





        for (auto icell = grid.begin_cell_sweep ();
             icell != grid.end_cell_sweep (); ++icell)
	  {
	    for (auto inode = 0; inode < quadgrid_t<std::vector<double>>::cell_t::nodes_per_cell; ++inode)
	      {
		auto iv = icell -> gt (inode);
		vars["mom_vx"][iv] += dt * vars["Ftot_vx"][iv];
		vars["mom_vy"][iv] += dt * vars["Ftot_vy"][iv];

	      }

	  }
	my_timer.toc ("step 3");



	// (4)  COMPUTE NODAL ACCELERATIONS AND VELOCITIES
	my_timer.tic ("step 4");
	for (auto icell = grid.begin_cell_sweep ();
	     icell != grid.end_cell_sweep (); ++icell)
	  {
	    for (auto inode = 0; inode < quadgrid_t<std::vector<double>>::cell_t::nodes_per_cell; ++inode)
	      {
		auto iv = icell -> gt (inode);
		vars["avx"][iv] = vars["Mv"][iv] > 1e-8 ?  vars["Ftot_vx"][iv] / vars["Mv"][iv] : 0.0;
		vars["avy"][iv] = vars["Mv"][iv] > 1e-8 ? vars[ "Ftot_vy"][iv] / vars["Mv"][iv] : 0.0;

		vars["vvx"][iv] = vars["Mv"][iv] > 1e-8 ?  vars["mom_vx"][iv] / vars["Mv"][iv] : 0.0;
		vars["vvy"][iv]= vars["Mv"][iv] > 1e-8 ?  vars["mom_vy"][iv] / vars["Mv"][iv] : 0.0;
	      }

	  }
	my_timer.toc ("step 4");

  // (5) BOUNDARY CONDITIONS - TO DO
 my_timer.tic ("step 5");
  for (auto icell = grid.begin_cell_sweep ();
      icell != grid.end_cell_sweep (); ++icell)
   {
        if ( (icell->e (2) == 2) || (icell->e(3)==3)  )
          {
          for (idx_t inode = 0; inode < 4; ++inode)
          {
                vars["avx"][icell->gt(inode)] = 0.0;
              vars["vvx"][icell->gt(inode)] = 0.0;
              //  vars["mom_vx"][icell->gt(inode)] = 0.0;
          }
          }
       if ( (icell->e(0) == 0) || (icell->e(1)==1) )
         {
          for (idx_t inode = 0; inode < 4; ++inode)
               {
                 vars["avy"][icell->gt(inode)] = 0.0;
                vars["vvy"][icell->gt(inode)] = 0.0;
                //   vars["mom_vy"][icell->gt(inode)] = 0.0;
               }
         }
      }


 my_timer.toc ("step 5");

	// (6) G2P + VELOCITY/POSITION UPDATE
	// Fuses 3 operations into 1 kernel:
	//   a) G2P: interpolate nodal vvx,vvy,avx,avy onto particles
	//   b) Velocity update: vpx += dt * apx
	//   c) Position update: x  += dt * vpx
	// Each particle reads from its 4 cell nodes (read-only on grid)
	// and writes only to its own propertiesw so there is no race condition.
	my_timer.tic ("step 6");
	{
	  // Extract raw pointers
	  // Particle data
	  double* d_x       = ptcls.x.data();
	  double* d_y       = ptcls.y.data();
	  double* d_vpx     = ptcls.dprops["vpx"].data();
	  double* d_vpy     = ptcls.dprops["vpy"].data();
	  double* d_apx     = ptcls.dprops["apx"].data();
	  double* d_apy     = ptcls.dprops["apy"].data();
	  const int* d_p2g  = ptcls.ptcl_to_grd.data();
 
	  // Grid data (read-only for this kernel)
	  const double* d_vvx = vars["vvx"].data();
	  const double* d_vvy = vars["vvy"].data();
	  const double* d_avx = vars["avx"].data();
	  const double* d_avy = vars["avy"].data();
 
	  const int np    = ptcls.num_particles;
	  const int nrows = grid.num_rows();
	  const double hx = grid.hx();
	  const double hy = grid.hy();
	  const int nn    = grid.num_global_nodes();
	  const double dt_local = dt;
 
	  #pragma omp target teams distribute parallel for \
	      map(to:     d_x[0:np], d_y[0:np], d_p2g[0:np],      \
	                  d_vvx[0:nn], d_vvy[0:nn],                 \
	                  d_avx[0:nn], d_avy[0:nn])                 \
	      map(from:   d_vpx[0:np], d_vpy[0:np],                \
	                  d_apx[0:np], d_apy[0:np])                 \
	      map(tofrom: d_x[0:np], d_y[0:np])
	  for (int ip = 0; ip < np; ip++) {
	    double xx = d_x[ip];
	    double yy = d_y[ip];
	    int cell_idx = d_p2g[ip];
	    int r = gpu_gind2row(cell_idx, nrows);
	    int c = gpu_gind2col(cell_idx, nrows);
 
	    // G2P: interpolate nodal values to particle
	    double vpx_val = 0.0, vpy_val = 0.0;
	    double apx_val = 0.0, apy_val = 0.0;
 
	    for (int inode = 0; inode < 4; inode++) {
	      double N    = gpu_shp(xx, yy, inode, c, r, hx, hy);
	      int    nidx = gpu_gt(inode, c, r, nrows);
	      vpx_val += N * d_vvx[nidx];
	      vpy_val += N * d_vvy[nidx];
	      apx_val += N * d_avx[nidx];
	      apy_val += N * d_avy[nidx];
	    }
 
	    // Velocity update: v += dt * a
	    vpx_val += dt_local * apx_val;
	    vpy_val += dt_local * apy_val;
 
	    // Store velocities and accelerations
	    d_vpx[ip] = vpx_val;
	    d_vpy[ip] = vpy_val;
	    d_apx[ip] = apx_val;
	    d_apy[ip] = apy_val;
 
	    // Position update: x += dt * v
	    d_x[ip] += dt_local * vpx_val;
	    d_y[ip] += dt_local * vpy_val;
	  }
	}
	my_timer.toc ("step 6");
 
	// G2PD + HEIGHT UPDATE — GPU OFFLOADED
	// Fuses 2 operations into 1 kernel:
	//   a) G2PD: interpolate velocity gradients onto particles
	//      (d(vvx)/dx, d(vvx)/dy, d(vvy)/dx, d(vvy)/dy)
	//   b) Height/area/volume/momentum update using divergence
	// Each particle reads from its 4 cell nodes (read-only on grid)
	// and writes only to its own properties so there is no race condition.
	my_timer.tic ("step 7");
	{
	  // --- Extract raw pointers ---
	  const double* d_x   = ptcls.x.data();
	  const double* d_y   = ptcls.y.data();
	  const int* d_p2g    = ptcls.ptcl_to_grd.data();
 
	  // Grid data (read-only)
	  const double* d_vvx = vars["vvx"].data();
	  const double* d_vvy = vars["vvy"].data();
 
	  // Particle data to update
	  double* d_vpx_dx  = ptcls.dprops["vpx_dx"].data();
	  double* d_vpx_dy  = ptcls.dprops["vpx_dy"].data();
	  double* d_vpy_dx  = ptcls.dprops["vpy_dx"].data();
	  double* d_vpy_dy  = ptcls.dprops["vpy_dy"].data();
	  double* d_hp      = ptcls.dprops["hp"].data();
	  double* d_Ap      = ptcls.dprops["Ap"].data();
	  double* d_Vp      = ptcls.dprops["Vp"].data();
	  double* d_mom_px  = ptcls.dprops["mom_px"].data();
	  double* d_mom_py  = ptcls.dprops["mom_py"].data();
	  const double* d_vpx = ptcls.dprops["vpx"].data();
	  const double* d_vpy = ptcls.dprops["vpy"].data();
	  const double* d_Mp  = ptcls.dprops["Mp"].data();
 
	  const int np    = ptcls.num_particles;
	  const int nrows = grid.num_rows();
	  const double hx = grid.hx();
	  const double hy = grid.hy();
	  const int nn    = grid.num_global_nodes();
	  const double dt_local = dt;
 
	  #pragma omp target teams distribute parallel for \
	      map(to:     d_x[0:np], d_y[0:np], d_p2g[0:np],          \
	                  d_vvx[0:nn], d_vvy[0:nn],                     \
	                  d_vpx[0:np], d_vpy[0:np], d_Mp[0:np])         \
	      map(from:   d_vpx_dx[0:np], d_vpx_dy[0:np],              \
	                  d_vpy_dx[0:np], d_vpy_dy[0:np],               \
	                  d_mom_px[0:np], d_mom_py[0:np])                \
	      map(tofrom: d_hp[0:np], d_Ap[0:np], d_Vp[0:np])
	  for (int ip = 0; ip < np; ip++) {
	    double xx = d_x[ip];
	    double yy = d_y[ip];
	    int cell_idx = d_p2g[ip];
	    int r = gpu_gind2row(cell_idx, nrows);
	    int c = gpu_gind2col(cell_idx, nrows);
 
	    // G2PD: interpolate velocity gradients
	    double vpx_dx_val = 0.0;
	    double vpx_dy_val = 0.0;
	    double vpy_dx_val = 0.0;
	    double vpy_dy_val = 0.0;
 
	    for (int inode = 0; inode < 4; inode++) {
	      double Nx   = gpu_shg(xx, yy, 0, inode, c, r, hx, hy);
	      double Ny   = gpu_shg(xx, yy, 1, inode, c, r, hx, hy);
	      int    nidx = gpu_gt(inode, c, r, nrows);
	      vpx_dx_val += Nx * d_vvx[nidx];   // d(vx)/dx
	      vpx_dy_val += Ny * d_vvx[nidx];   // d(vx)/dy
	      vpy_dx_val += Nx * d_vvy[nidx];   // d(vy)/dx
	      vpy_dy_val += Ny * d_vvy[nidx];   // d(vy)/dy
	    }
 
	    // Store gradients (needed by step 8 for stress computation)
	    d_vpx_dx[ip] = vpx_dx_val;
	    d_vpx_dy[ip] = vpx_dy_val;
	    d_vpy_dx[ip] = vpy_dx_val;
	    d_vpy_dy[ip] = vpy_dy_val;
 
	    // Height update: h /= (1 + dt * div(v))
	    double div_v = vpx_dx_val + vpy_dy_val;
	    double scale = 1.0 + dt_local * div_v;
	    d_hp[ip] /= scale;
	    d_Ap[ip] /= scale;
	    d_Vp[ip]  = d_hp[ip] * d_Ap[ip];
 
	    // Momentum update
	    d_mom_px[ip] = d_vpx[ip] * d_Mp[ip];
	    d_mom_py[ip] = d_vpy[ip] * d_Mp[ip];
	  }
	}
	my_timer.toc ("step 7");

  // FRICTION COMPUTATION — GPU OFFLOADED (flat particle loop)
   my_timer.tic ("friction");
  {
    double* d_vpx    = ptcls.dprops["vpx"].data();
    double* d_vpy    = ptcls.dprops["vpy"].data();
    double* d_hp     = ptcls.dprops["hp"].data();
    double* d_Fb_x   = ptcls.dprops["Fb_x"].data();
    double* d_Fb_y   = ptcls.dprops["Fb_y"].data();
    const int np     = ptcls.num_particles;
    const double rho = data.rho;
    const double g   = data.g;
    const double tan_fa = std::tan(fric_ang);
 
    #pragma omp target teams distribute parallel for \
        map(to: d_vpx[0:np], d_vpy[0:np], d_hp[0:np]) \
        map(from: d_Fb_x[0:np], d_Fb_y[0:np])
    for (int ip = 0; ip < np; ip++) {
      double vx = d_vpx[ip];
      double vy = d_vpy[ip];
      double hp_val = d_hp[ip];
      double nv = sqrt(vx * vx + vy * vy);
 
      // Voellmy friction (currently scaled by 0.0, structure preserved)
      d_Fb_x[ip] = hp_val > 1.e-3 ?
        -0.0 * 2.5 * (rho * hp_val * g * tan_fa +
                       rho * g * vx * vx / 100.0) * vx / (nv + 0.001) : 0.0;
 
      d_Fb_y[ip] = hp_val > 1.e-3 ?
        -0.0 * 2.5 * (rho * hp_val * g * tan_fa +
                       rho * g * vy * vy / 100.0) * vy / (nv + 0.001) : 0.0;
    }
  }
  my_timer.toc ("friction");

	// (8) UPDATE PARTICLE STRESS (USL)
	my_timer.tic ("step 8");
  {
    // --- Extract raw pointers for GPU mapping ---
    double* d_vpx     = ptcls.dprops["vpx"].data();
    double* d_vpy     = ptcls.dprops["vpy"].data();
    double* d_vpx_dx  = ptcls.dprops["vpx_dx"].data();
    double* d_vpx_dy  = ptcls.dprops["vpx_dy"].data();
    double* d_vpy_dx  = ptcls.dprops["vpy_dx"].data();
    double* d_vpy_dy  = ptcls.dprops["vpy_dy"].data();
    double* d_hp      = ptcls.dprops["hp"].data();
    double* d_F11     = ptcls.dprops["F_11"].data();
    double* d_F12     = ptcls.dprops["F_12"].data();
    double* d_F21     = ptcls.dprops["F_21"].data();
    double* d_F22     = ptcls.dprops["F_22"].data();
    const int np      = ptcls.num_particles;
    const double rho  = data.rho;
    const double g    = data.g;
 
    // Bingham model constants (from original code)
    const double A_coeff  = 3.0 / 2.0;
    const double C_coeff  = 65.0 / 32.0;
    const double mu       = 50.0;    // viscosity
    const double tau_Y    = 2000.0;  // yield stress
    const double cc       = 0.0;     // stress scaling (0 = off)
 
    #pragma omp target teams distribute parallel for \
        map(to:     d_vpx[0:np], d_vpy[0:np],              \
                    d_vpx_dx[0:np], d_vpx_dy[0:np],        \
                    d_vpy_dx[0:np], d_vpy_dy[0:np],        \
                    d_hp[0:np])                             \
        map(tofrom: d_F11[0:np], d_F12[0:np],              \
                    d_F21[0:np], d_F22[0:np])
    for (int ip = 0; ip < np; ip++) {
 
      double vx  = d_vpx[ip];
      double vy  = d_vpy[ip];
      double hp_val = d_hp[ip];
 
      // Velocity norm
      double nv = sqrt(vx * vx + vy * vy);
 
      // Depth profile parameter (ZZ=0 in original code)
      double alf = hp_val > 1.e-3 ?
        (6.0 * mu * nv) / ((hp_val + 0.001) * tau_Y) : 0.0;
      double b_val  = -114.0 / 32.0 - alf;
      double discr  = b_val * b_val - 4.0 * A_coeff * C_coeff;
      double sq     = sqrt(discr);
      double z1     = (-b_val + sq) / (2.0 * A_coeff);
      double z2     = (-b_val - sq) / (2.0 * A_coeff);
      double zz_val = (fabs(z1 - 0.5) <= 0.5) ? z1 : z2;
      // NOTE: zz_val is computed but not used below (ZZ=0 in original).
      // Preserving original behavior: using 0.0 for D_zx, D_zy denom.
      double zz_use = 0.0;
 
      // 2D strain rate tensor
      double sxx = d_vpx_dx[ip];
      double sxy = 0.5 * (d_vpx_dy[ip] + d_vpy_dx[ip]);
      double syy = d_vpy_dy[ip];
 
      // Full 3D strain rate tensor D (depth-integrated)
      double Dxx = sxx;
      double Dxy = sxy;
      double Dyx = sxy;
      double Dyy = syy;
      double Dzz = -(sxx + d_vpy_dy[ip]);
      double Dzx = hp_val > 1.e-3 ?
        0.5 * (3.0 / (2.0 + zz_use)) * (vx / (hp_val + 0.001)) : 0.0;
      double Dzy = hp_val > 1.e-3 ?
        0.5 * (3.0 / (2.0 + zz_use)) * (vy / (hp_val + 0.001)) : 0.0;
      double Dxz = 0.0;
      double Dyz = 0.0;
 
      // Second invariant I_2 = 0.5 * D:D
      double inv2 = 0.5 * (Dxx*Dxx + Dyy*Dyy + Dzz*Dzz +
                            Dxz*Dxz + Dyz*Dyz + Dxy*Dxy);
 
      // Bingham stress: sigma = (tau_Y / sqrt(I_2) + 2*mu) * D
      double coeff = (inv2 != 0.0) ?
        (tau_Y / sqrt(inv2) + 2.0 * mu) : 0.0;
      double sigxx = coeff * Dxx;
      double sigxy = coeff * Dxy;
      double sigyy = coeff * Dyy;
 
      // Update stress tensor (cc scales Bingham, rest is hydrostatic)
      d_F11[ip] = cc * sigxx - 0.5 * rho * g * hp_val;
      d_F12[ip] = cc * sigxy;
      d_F21[ip] = cc * sigxy;
      d_F22[ip] = cc * sigyy - 0.5 * rho * g * hp_val;
    }
  }
 
  // hpZ update (g2p call stays on CPU, per-particle update offloaded)
  ptcls.dprops.at("hpZ").assign(ptcls.num_particles, 0.0);
  ptcls.dprops.at("Zp").assign(ptcls.num_particles, 0.0);
  ptcls.g2p (vars,std::vector<std::string>{"Z"},
       std::vector<std::string>{"Zp"});
 
  {
    double* d_hpZ = ptcls.dprops["hpZ"].data();
    double* d_hp  = ptcls.dprops["hp"].data();
    double* d_Zp  = ptcls.dprops["Zp"].data();
    const int np  = ptcls.num_particles;
 
    #pragma omp target teams distribute parallel for \
        map(to: d_hp[0:np], d_Zp[0:np]) \
        map(from: d_hpZ[0:np])
    for (int ip = 0; ip < np; ip++) {
      d_hpZ[ip] = d_hp[ip] + d_Zp[ip];
    }
  }
 
  my_timer.toc("step 8");

	        ptcls.p2g (Plotvars,std::vector<std::string>{"Mp","vpx","vpy","apx","apy"},
             std::vector<std::string>{"rho_v","vvx","vvy","avx","avy"},true);

 my_timer.tic("step 8b");
        ptcls.p2g (vars,std::vector<std::string>{"hp"}, std::vector<std::string>{"HV"});

	my_timer.toc ("step 8b");

	my_timer.tic ("save vts");
        filename = "nc_grid_";
        filename = filename + std::to_string (it);
      filename = filename + ".vts";
	 // grid.vtk_export(filename.c_str(), Plotvars);
	grid.vtk_export(filename.c_str(), vars);
        t +=dt;
	my_timer.toc ("save vts");

      }

    my_timer.print_report ();
    //  ptcls.print<particles_t::output_format::csv>(std::cout);

    return 0;
}
