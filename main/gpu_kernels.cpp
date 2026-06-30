#include <cmath>
#include <cstdio>
#include "gpu_kernels.h"

#pragma omp declare target

inline int gt (int inode, int c, int r, int nrows) {
  int bl = r + c * (nrows + 1);
  switch (inode) {
    case 0: return bl;
    case 1: return bl + 1;
    case 2: return bl + (nrows + 1);
    case 3: return bl + (nrows + 2);
    default: return -1;
  }
}

inline int grow (int idx, int nrows) { return idx % nrows; }
inline int gcol (int idx, int nrows) { return idx / nrows; }

inline double pnode (int idir, int inode, int c, int r, double hx, double hy) {
  double bl;
  if (idir == 0) {
    bl = c * hx;
    if (inode > 1) bl += hx;
  } else {
    bl = r * hy;
    if (inode == 1 || inode == 3) bl += hy;
  }
  return bl;
}

inline double shp (double x, double y, int inode, int c, int r, double hx, double hy) {
  switch (inode) {
    case 3: return ((x - pnode(0,0,c,r,hx,hy))/hx) * ((y - pnode(1,0,c,r,hx,hy))/hy);
    case 2: return ((x - pnode(0,0,c,r,hx,hy))/hx) * (1.0 - (y - pnode(1,0,c,r,hx,hy))/hy);
    case 1: return (1.0 - (x - pnode(0,0,c,r,hx,hy))/hx) * ((y - pnode(1,0,c,r,hx,hy))/hy);
    case 0: return (1.0 - (x - pnode(0,0,c,r,hx,hy))/hx) * (1.0 - (y - pnode(1,0,c,r,hx,hy))/hy);
    default: return 0.0;
  }
}

inline double shg (double x, double y, int idir, int inode, int c, int r, double hx, double hy) {
  switch (inode) {
    case 3:
      if (idir == 0) return (1.0/hx) * ((y - pnode(1,0,c,r,hx,hy))/hy);
      else            return ((x - pnode(0,0,c,r,hx,hy))/hx) * (1.0/hy);
    case 2:
      if (idir == 0) return (1.0/hx) * (1.0 - (y - pnode(1,0,c,r,hx,hy))/hy);
      else            return ((x - pnode(0,0,c,r,hx,hy))/hx) * (-1.0/hy);
    case 1:
      if (idir == 0) return (-1.0/hx) * ((y - pnode(1,0,c,r,hx,hy))/hy);
      else            return (1.0 - (x - pnode(0,0,c,r,hx,hy))/hx) * (1.0/hy);
    case 0:
      if (idir == 0) return (-1.0/hx) * (1.0 - (y - pnode(1,0,c,r,hx,hy))/hy);
      else            return (1.0 - (x - pnode(0,0,c,r,hx,hy))/hx) * (-1.0/hy);
    default: return 0.0;
  }
}

#pragma omp end declare target

namespace gpu_kernels {


void reset_nodal_arrays (
    int nn,
    double* d_Mv, double* d_mom_vx, double* d_mom_vy,
    double* d_F_ext_vx, double* d_F_ext_vy,
    double* d_F_int_vx, double* d_F_int_vy,
    double* d_Ftot_vx,  double* d_Ftot_vy,
    double* d_avx, double* d_avy,
    double* d_vvx, double* d_vvy, double* d_vvxL, double* d_vvyL,
    double* d_Fric_x, double* d_Fric_y,
    double* d_FPxv, double* d_FPyv, double* d_HV,
    double* d_pv_rho, double* d_pv_vvx, double* d_pv_vvy,
    double* d_pv_avx, double* d_pv_avy) {

  #pragma omp target teams loop firstprivate(nn)
  for (int iv = 0; iv < nn; iv++) {
    d_Mv[iv]=0;       d_mom_vx[iv]=0;   d_mom_vy[iv]=0;
    d_F_ext_vx[iv]=0; d_F_ext_vy[iv]=0;
    d_F_int_vx[iv]=0; d_F_int_vy[iv]=0;
    d_Ftot_vx[iv]=0;  d_Ftot_vy[iv]=0;
    d_avx[iv]=0;      d_avy[iv]=0;
    d_vvx[iv]=0;      d_vvy[iv]=0;
    d_vvxL[iv]=0;     d_vvyL[iv]=0;
    d_Fric_x[iv]=0;   d_Fric_y[iv]=0;
    d_FPxv[iv]=0;     d_FPyv[iv]=0;
    d_HV[iv]=0;
    d_pv_rho[iv]=0;   d_pv_vvx[iv]=0;   d_pv_vvy[iv]=0;
    d_pv_avx[iv]=0;   d_pv_avy[iv]=0;
  }
}


void scatter_mass_momentum (
    int np, int nrows, double hx, double hy,
    const int* cstart, const int* cptcls, const int* ccidx, const int* coloff,
    const double* d_x, const double* d_y,
    const double* d_Mp, const double* d_mom_px, const double* d_mom_py,
    double* d_Mv, double* d_mom_vx, double* d_mom_vy) {

  for (int color = 0; color < 4; ++color) {
    const int cbegin = coloff[color];
    const int cend   = coloff[color + 1];
    if (cbegin == cend) continue;

    #pragma omp target teams loop firstprivate(cbegin, cend, nrows, hx, hy)
    for (int ic = cbegin; ic < cend; ++ic) {
      const int cell = ccidx[ic];
      const int r    = cell % nrows;
      const int c    = cell / nrows;
      for (int jp = cstart[cell]; jp < cstart[cell+1]; ++jp) {
        const int    ip = cptcls[jp];
        const double xx = d_x[ip], yy = d_y[ip];
        const double mp = d_Mp[ip], mpx = d_mom_px[ip], mpy = d_mom_py[ip];
        for (int inode = 0; inode < 4; ++inode) {
          const double N    = shp (xx, yy, inode, c, r, hx, hy);
          const int    nidx = gt  (inode, c, r, nrows);
          d_Mv[nidx]     += N * mp;
          d_mom_vx[nidx] += N * mpx;
          d_mom_vy[nidx] += N * mpy;
        }
      }
    }
  }
}


void gradZ_to_particles (
    int np, int nrows, double hx, double hy,
    const double* d_x, const double* d_y, const int* d_p2g,
    const double* d_Z,
    double* d_dZxp, double* d_dZyp) {

  #pragma omp target teams loop firstprivate(np, nrows, hx, hy)
  for (int ip = 0; ip < np; ++ip) {
    const double xx = d_x[ip], yy = d_y[ip];
    const int    ci = d_p2g[ip];
    const int    r  = grow (ci, nrows);
    const int    c  = gcol (ci, nrows);
    double dzx = 0.0, dzy = 0.0;
    for (int inode = 0; inode < 4; ++inode) {
      const int nidx = gt (inode, c, r, nrows);
      dzx += shg (xx, yy, 0, inode, c, r, hx, hy) * d_Z[nidx];
      dzy += shg (xx, yy, 1, inode, c, r, hx, hy) * d_Z[nidx];
    }
    d_dZxp[ip] = dzx;
    d_dZyp[ip] = dzy;
  }
}


void gravity_force (
    int np, double g_c,
    const double* d_hp, const double* d_H, const double* d_Mp,
    const double* d_dZxp, const double* d_dZyp,
    double* d_Fpx, double* d_Fpy) {

  #pragma omp target teams loop firstprivate(np, g_c)
  for (int ip = 0; ip < np; ip++) {
    const double ratio = d_hp[ip] > 1e-10 ? d_H[ip] / d_hp[ip] : 0.0;
    d_Fpx[ip] = -g_c * d_Mp[ip] * (1.0 - ratio) * d_dZxp[ip];
    d_Fpy[ip] = -g_c * d_Mp[ip] * (1.0 - ratio) * d_dZyp[ip];
  }
}


void friction_particle_force (
    int np,
    const double* d_Ap, const double* d_Fb_x, const double* d_Fb_y,
    double* d_Fric_px, double* d_Fric_py) {

  #pragma omp target teams loop firstprivate(np)
  for (int ip = 0; ip < np; ip++) {
    d_Fric_px[ip] = d_Ap[ip] * d_Fb_x[ip];
    d_Fric_py[ip] = d_Ap[ip] * d_Fb_y[ip];
  }
}


void scatter_gravity_force (
    int np, int nrows, double hx, double hy,
    const int* cstart, const int* cptcls, const int* ccidx, const int* coloff,
    const double* d_x, const double* d_y,
    const double* d_Fpx, const double* d_Fpy,
    double* d_FPxv, double* d_FPyv) {

  for (int color = 0; color < 4; ++color) {
    const int cbegin = coloff[color];
    const int cend   = coloff[color + 1];
    if (cbegin == cend) continue;

    #pragma omp target teams loop firstprivate(cbegin, cend, nrows, hx, hy)
    for (int ic = cbegin; ic < cend; ++ic) {
      const int cell = ccidx[ic];
      const int r    = cell % nrows;
      const int c    = cell / nrows;
      for (int jp = cstart[cell]; jp < cstart[cell+1]; ++jp) {
        const int    ip = cptcls[jp];
        const double xx = d_x[ip], yy = d_y[ip];
        const double fpx = d_Fpx[ip], fpy = d_Fpy[ip];
        for (int inode = 0; inode < 4; ++inode) {
          const double N    = shp (xx, yy, inode, c, r, hx, hy);
          const int    nidx = gt  (inode, c, r, nrows);
          d_FPxv[nidx] += N * fpx;
          d_FPyv[nidx] += N * fpy;
        }
      }
    }
  }
}


void scatter_friction_force (
    int np, int nrows, double hx, double hy,
    const int* cstart, const int* cptcls, const int* ccidx, const int* coloff,
    const double* d_x, const double* d_y,
    const double* d_Fric_px, const double* d_Fric_py,
    double* d_Fric_x, double* d_Fric_y) {

  for (int color = 0; color < 4; ++color) {
    const int cbegin = coloff[color];
    const int cend   = coloff[color + 1];
    if (cbegin == cend) continue;

    #pragma omp target teams loop firstprivate(cbegin, cend, nrows, hx, hy)
    for (int ic = cbegin; ic < cend; ++ic) {
      const int cell = ccidx[ic];
      const int r    = cell % nrows;
      const int c    = cell / nrows;
      for (int jp = cstart[cell]; jp < cstart[cell+1]; ++jp) {
        const int    ip = cptcls[jp];
        const double xx = d_x[ip], yy = d_y[ip];
        const double frx = d_Fric_px[ip], fry = d_Fric_py[ip];
        for (int inode = 0; inode < 4; ++inode) {
          const double N    = shp (xx, yy, inode, c, r, hx, hy);
          const int    nidx = gt  (inode, c, r, nrows);
          d_Fric_x[nidx] += N * frx;
          d_Fric_y[nidx] += N * fry;
        }
      }
    }
  }
}


void compute_F_ext (
    int nn,
    const double* d_FPxv, const double* d_FPyv,
    const double* d_Fric_x, const double* d_Fric_y,
    double* d_F_ext_vx, double* d_F_ext_vy) {

  #pragma omp target teams loop firstprivate(nn)
  for (int iv = 0; iv < nn; iv++) {
    d_F_ext_vx[iv] = d_FPxv[iv] + d_Fric_x[iv];
    d_F_ext_vy[iv] = d_FPyv[iv] + d_Fric_y[iv];
  }
}


void scatter_internal_forces (
    int np, int nrows, double hx, double hy,
    const int* cstart, const int* cptcls, const int* ccidx, const int* coloff,
    const double* d_x, const double* d_y, const double* d_Vp,
    const double* d_F11, const double* d_F12,
    const double* d_F21, const double* d_F22,
    double* d_F_int_vx, double* d_F_int_vy) {

  for (int color = 0; color < 4; ++color) {
    const int cbegin = coloff[color];
    const int cend   = coloff[color + 1];
    if (cbegin == cend) continue;

    #pragma omp target teams loop firstprivate(cbegin, cend, nrows, hx, hy)
    for (int ic = cbegin; ic < cend; ++ic) {
      const int cell = ccidx[ic];
      const int r    = cell % nrows;
      const int c    = cell / nrows;
      for (int jp = cstart[cell]; jp < cstart[cell+1]; ++jp) {
        const int    ip = cptcls[jp];
        const double xx = d_x[ip], yy = d_y[ip];
        const double vp = d_Vp[ip];
        const double f11=d_F11[ip], f12=d_F12[ip], f21=d_F21[ip], f22=d_F22[ip];
        for (int inode = 0; inode < 4; ++inode) {
          const double Nx   = shg (xx, yy, 0, inode, c, r, hx, hy);
          const double Ny   = shg (xx, yy, 1, inode, c, r, hx, hy);
          const int    nidx = gt  (inode, c, r, nrows);
          d_F_int_vx[nidx] += (Nx*f11 + Ny*f21) * vp;
          d_F_int_vy[nidx] += (Nx*f12 + Ny*f22) * vp;
        }
      }
    }
  }
}


void nodal_force_and_momentum (
    int nn, double dt,
    const double* d_F_ext_vx, const double* d_F_ext_vy,
    const double* d_F_int_vx, const double* d_F_int_vy,
    double* d_Ftot_vx, double* d_Ftot_vy,
    double* d_mom_vx,  double* d_mom_vy) {

  #pragma omp target teams loop firstprivate(nn, dt)
  for (int iv = 0; iv < nn; iv++) {
    const double ftx = d_F_ext_vx[iv] + d_F_int_vx[iv];
    const double fty = d_F_ext_vy[iv] + d_F_int_vy[iv];
    d_Ftot_vx[iv] = ftx;
    d_Ftot_vy[iv] = fty;
    d_mom_vx[iv] += dt * ftx;
    d_mom_vy[iv] += dt * fty;
  }
}


void nodal_velocity_and_bc (
    int nn, int nrows, int ncols, double dt, bool bc_flag,
    const double* d_Mv, const double* d_Ftot_vx, const double* d_Ftot_vy,
    double* d_mom_vx, double* d_mom_vy,
    double* d_avx, double* d_avy,
    double* d_vvx, double* d_vvy,
    double* d_vvxL, double* d_vvyL) {

  const int bc_flag_i = bc_flag ? 1 : 0;

  #pragma omp target teams loop firstprivate(nn, nrows, ncols, dt, bc_flag_i)
  for (int iv = 0; iv < nn; iv++) {
    const bool active = d_Mv[iv] > 1e-2;
    double avx  = active ? d_Ftot_vx[iv] / d_Mv[iv] : 0.0;
    double avy  = active ? d_Ftot_vy[iv] / d_Mv[iv] : 0.0;
    double vvx  = active ? d_mom_vx[iv]  / d_Mv[iv] : 0.0;
    double vvy  = active ? d_mom_vy[iv]  / d_Mv[iv] : 0.0;
    double vvxL = active ? dt * avx + vvx : 0.0;
    double vvyL = active ? dt * avy + vvy : 0.0;

    if (bc_flag_i) {
      const int r = iv % (nrows + 1);
      const int c = iv / (nrows + 1);
      if (c==0 || c==1 || c==ncols-1 || c==ncols) {
        d_mom_vx[iv] = 0.0; vvx = 0.0; vvxL = 0.0; avx = 0.0;
      }
      if (r==0 || r==1 || r==nrows-1 || r==nrows) {
        d_mom_vy[iv] = 0.0; vvy = 0.0; vvyL = 0.0; avy = 0.0;
      }
    }

    d_avx[iv]  = avx;  d_avy[iv]  = avy;
    d_vvx[iv]  = vvx;  d_vvy[iv]  = vvy;
    d_vvxL[iv] = vvxL; d_vvyL[iv] = vvyL;
  }
}


void g2p_velocity_and_advect (
    int np, int nrows, int ncols, double hx, double hy, double dt,
    const int* d_p2g,
    const double* d_vvxL, const double* d_vvyL,
    const double* d_avx,  const double* d_avy,
    double* d_x, double* d_y,
    double* d_vpx, double* d_vpy,
    double* d_apx, double* d_apy,
    double* d_vpxL, double* d_vpyL,
    double* d_mom_px, double* d_mom_py,
    const double* d_Mp,
    double* max_vmag_out) {

  double max_vmag = 0.0;

  #pragma omp target teams loop firstprivate(np, nrows, hx, hy, dt) reduction(max:max_vmag)
  for (int ip = 0; ip < np; ip++) {
    const double xx = d_x[ip], yy = d_y[ip];
    const int    ci = d_p2g[ip];
    const int    r  = grow (ci, nrows);
    const int    c  = gcol (ci, nrows);

    double ax=0, ay=0, vxL=0, vyL=0;
    for (int inode = 0; inode < 4; ++inode) {
      const double N    = shp (xx, yy, inode, c, r, hx, hy);
      const int    nidx = gt  (inode, c, r, nrows);
      ax  += N * d_avx[nidx];
      ay  += N * d_avy[nidx];
      vxL += N * d_vvxL[nidx];
      vyL += N * d_vvyL[nidx];
    }

    d_apx[ip] = ax;
    d_apy[ip] = ay;
    d_vpxL[ip] = vxL;
    d_vpyL[ip] = vyL;

    const double vmag = sqrt (vxL*vxL + vyL*vyL);
    if (vmag > max_vmag) max_vmag = vmag;

    d_x[ip] = xx + dt * vxL;
    d_y[ip] = yy + dt * vyL;
    d_vpx[ip] = vxL;
    d_vpy[ip] = vyL;
    d_mom_px[ip] = vxL * d_Mp[ip];
    d_mom_py[ip] = vyL * d_Mp[ip];
  }

  *max_vmag_out = max_vmag;
}


void g2pd_gradients_and_height_update (
    int np, int nrows, double hx, double hy, double dt,
    const double* d_x, const double* d_y, const int* d_p2g,
    const double* d_vvxL, const double* d_vvyL,
    double* d_vpx_dx, double* d_vpx_dy,
    double* d_vpy_dx, double* d_vpy_dy,
    double* d_hp, double* d_Vp, double* d_Ap,
    int* n_scfloor_out, int* n_hpfloor_out,
    double* min_sc_out, double* min_hp_out) {

  int n_scfloor=0, n_hpfloor=0;
  double min_sc=1e30, min_hp=1e30;

  #pragma omp target teams loop firstprivate(np, nrows, hx, hy, dt) reduction(+:n_scfloor,n_hpfloor) reduction(min:min_sc,min_hp)
  for (int ip = 0; ip < np; ip++) {
    const double xx = d_x[ip], yy = d_y[ip];
    const int    ci = d_p2g[ip];
    const int    r  = grow (ci, nrows);
    const int    c  = gcol (ci, nrows);

    double vxdx=0, vxdy=0, vydx=0, vydy=0;
    for (int inode = 0; inode < 4; ++inode) {
      const int nidx = gt (inode, c, r, nrows);
      const double Nx = shg (xx, yy, 0, inode, c, r, hx, hy);
      const double Ny = shg (xx, yy, 1, inode, c, r, hx, hy);
      vxdx += Nx * d_vvxL[nidx];
      vxdy += Ny * d_vvxL[nidx];
      vydx += Nx * d_vvyL[nidx];
      vydy += Ny * d_vvyL[nidx];
    }
    d_vpx_dx[ip] = vxdx;
    d_vpx_dy[ip] = vxdy;
    d_vpy_dx[ip] = vydx;
    d_vpy_dy[ip] = vydy;

    double sc = 1.0 + dt * (vxdx + vydy);
    if (sc < min_sc) min_sc = sc;
    if (sc < 0.1) { sc = 0.1; n_scfloor++; }
    d_hp[ip] /= sc;
    if (d_hp[ip] < min_hp) min_hp = d_hp[ip];
    if (d_hp[ip] < 1e-2)  { d_hp[ip] = 1e-2; n_hpfloor++; }
    if (d_Vp[ip] < 1e-10)   d_Vp[ip] = 1e-10;
    d_Ap[ip] = d_Vp[ip] / d_hp[ip];
  }

  *n_scfloor_out = n_scfloor;
  *n_hpfloor_out = n_hpfloor;
  *min_sc_out    = min_sc;
  *min_hp_out    = min_hp;
}


void voellmy_friction (
    int np, double rho, double g_c, double tan_fa, double xi, double fric_on,
    const double* d_vpx, const double* d_vpy, const double* d_hp,
    double* d_Fb_x, double* d_Fb_y) {

  #pragma omp target teams loop firstprivate(np, rho, g_c, tan_fa, xi, fric_on)
  for (int ip = 0; ip < np; ip++) {
    const double vx = d_vpx[ip], vy = d_vpy[ip], h = d_hp[ip];
    const double nv = sqrt (vx*vx + vy*vy);
    if (fric_on > 0 && nv > 1e-10 && xi > 0) {
      const double v2 = vx*vx + vy*vy;
      d_Fb_x[ip] = -fric_on*(rho*g_c*h*tan_fa + rho*g_c*v2/xi)*vx/nv;
      d_Fb_y[ip] = -fric_on*(rho*g_c*h*tan_fa + rho*g_c*v2/xi)*vy/nv;
    } else {
      d_Fb_x[ip] = 0.0;
      d_Fb_y[ip] = 0.0;
    }
  }
}


void bingham_stress_update (
    int np, double rho, double g_c,
    double mu, double tau_Y, double cc, double A_coeff, double C_coeff,
    const double* d_vpx, const double* d_vpy, const double* d_hp,
    const double* d_H,
    const double* d_vpx_dx, const double* d_vpx_dy,
    const double* d_vpy_dx, const double* d_vpy_dy,
    double* d_F11, double* d_F12, double* d_F21, double* d_F22,
    double* max_coeff_out, double* min_hp_out, double* max_nv_out,
    double* max_F11_out, double* max_dev_out, double* max_pres_out,
    double* max_Dxx_out) {

  double max_coeff=0, max_nv=0, max_F11=0, max_dev=0, max_pres=0, max_Dxx=0;
  double min_hp = 1e30;

  #pragma omp target teams loop firstprivate(np, rho, g_c, mu, tau_Y, cc, A_coeff, C_coeff) reduction(max:max_coeff,max_nv,max_F11,max_dev,max_pres,max_Dxx) reduction(min:min_hp)
  for (int ip = 0; ip < np; ip++) {
    const double vx = d_vpx[ip], vy = d_vpy[ip], h = d_hp[ip];
    const double nv = sqrt (vx*vx + vy*vy);

    const double alf = (h > 1e-3 && tau_Y > 1e-12)
                       ? (6.0*mu*nv) / ((h+0.001)*tau_Y) : 0.0;
    const double bv  = -114.0/32.0 - alf;
    const double sq  = sqrt (bv*bv - 4.0*A_coeff*C_coeff);
    const double z1  = (-bv + sq) / (2.0*A_coeff);
    const double z2  = (-bv - sq) / (2.0*A_coeff);
    const double zz  = (fabs(z1-0.5) <= 0.5) ? z1 : z2;

    const double Dxx = d_vpx_dx[ip];
    const double Dxy = 0.5*(d_vpx_dy[ip] + d_vpy_dx[ip]);
    const double Dyy = d_vpy_dy[ip];
    const double Dzz = -(Dxx + Dyy);
    const double Dzx = h > 1e-3 ? 0.5*(3.0/(2.0+zz))*(vx/(h+0.001)) : 0.0;
    const double Dzy = h > 1e-3 ? 0.5*(3.0/(2.0+zz))*(vy/(h+0.001)) : 0.0;

    const double inv2 = 0.5*(Dxx*Dxx + Dyy*Dyy + Dzz*Dzz)
                        + Dzx*Dzx + Dzy*Dzy + Dxy*Dxy;

    const double coeff_max  = 5.0e6;
    double inv2_floor = (tau_Y/coeff_max)*(tau_Y/coeff_max);
    if (inv2_floor < 1e-9) inv2_floor = 1e-9;
    const double coeff = tau_Y / sqrt(inv2 + inv2_floor) + 2.0*mu;

    const double h_corr = h > 1e-10 ? h - d_H[ip]*d_H[ip]/h : 0.0;
    d_F11[ip] = -cc*coeff*Dxx + 0.5*rho*g_c*h_corr;
    d_F22[ip] = -cc*coeff*Dyy + 0.5*rho*g_c*h_corr;
    d_F12[ip] = -cc*coeff*Dxy;
    d_F21[ip] = -cc*coeff*Dxy;

    if (coeff > max_coeff) max_coeff = coeff;
    if (h     < min_hp)    min_hp    = h;
    if (nv    > max_nv)    max_nv    = nv;
    if (fabs(d_F11[ip]) > max_F11) max_F11 = fabs(d_F11[ip]);
    const double dev_  = fabs(cc*coeff*Dxx);
    const double pres_ = fabs(0.5*rho*g_c*h_corr);
    if (dev_  > max_dev)  max_dev  = dev_;
    if (pres_ > max_pres) max_pres = pres_;
    if (fabs(Dxx) > max_Dxx) max_Dxx = fabs(Dxx);
  }

  *max_coeff_out = max_coeff;
  *min_hp_out    = min_hp;
  *max_nv_out    = max_nv;
  *max_F11_out   = max_F11;
  *max_dev_out   = max_dev;
  *max_pres_out  = max_pres;
  *max_Dxx_out   = max_Dxx;
}


void g2p_height (
    int np, int nrows, double hx, double hy,
    const double* d_x, const double* d_y, const int* d_p2g,
    const double* d_Z, const double* d_hp,
    double* d_Zp, double* d_hpZ) {

  #pragma omp target teams loop firstprivate(np, nrows, hx, hy)
  for (int ip = 0; ip < np; ip++) {
    const double xx = d_x[ip], yy = d_y[ip];
    const int    ci = d_p2g[ip];
    const int    r  = grow (ci, nrows);
    const int    c  = gcol (ci, nrows);
    double zp = 0.0;
    for (int inode = 0; inode < 4; ++inode) {
      const int nidx = gt (inode, c, r, nrows);
      zp += shp (xx, yy, inode, c, r, hx, hy) * d_Z[nidx];
    }
    d_Zp[ip]  = zp;
    d_hpZ[ip] = d_hp[ip] + zp;
  }
}


void scatter_plotvars (
    int np, int nrows, double hx, double hy,
    const int* cstart, const int* cptcls, const int* ccidx, const int* coloff,
    const double* d_x, const double* d_y,
    const double* d_Mp, const double* d_vpx, const double* d_vpy,
    const double* d_apx, const double* d_apy, const double* d_hp,
    double* d_pv_rho, double* d_pv_vvx, double* d_pv_vvy,
    double* d_pv_avx, double* d_pv_avy, double* d_HV) {

  for (int color = 0; color < 4; ++color) {
    const int cbegin = coloff[color];
    const int cend   = coloff[color + 1];
    if (cbegin == cend) continue;

    #pragma omp target teams loop firstprivate(cbegin, cend, nrows, hx, hy)
    for (int ic = cbegin; ic < cend; ++ic) {
      const int cell = ccidx[ic];
      const int r    = cell % nrows;
      const int c    = cell / nrows;
      for (int jp = cstart[cell]; jp < cstart[cell+1]; ++jp) {
        const int    ip = cptcls[jp];
        const double xx = d_x[ip], yy = d_y[ip];
        const double mp = d_Mp[ip], vx = d_vpx[ip], vy = d_vpy[ip];
        const double ax = d_apx[ip], ay = d_apy[ip], hh = d_hp[ip];
        for (int inode = 0; inode < 4; ++inode) {
          const double N    = shp (xx, yy, inode, c, r, hx, hy);
          const int    nidx = gt  (inode, c, r, nrows);
          d_pv_rho[nidx] += N * mp;
          d_pv_vvx[nidx] += N * vx;
          d_pv_vvy[nidx] += N * vy;
          d_pv_avx[nidx] += N * ax;
          d_pv_avy[nidx] += N * ay;
          d_HV[nidx]     += N * hh;
        }
      }
    }
  }
}


void normalize_plotvars (
    int nn,
    double* d_pv_rho, double* d_pv_vvx, double* d_pv_vvy,
    double* d_pv_avx, double* d_pv_avy) {

  #pragma omp target teams loop firstprivate(nn)
  for (int iv = 0; iv < nn; iv++) {
    const double mv = d_pv_rho[iv];
    if (mv > 1e-8) {
      d_pv_vvx[iv] /= mv;
      d_pv_vvy[iv] /= mv;
      d_pv_avx[iv] /= mv;
      d_pv_avy[iv] /= mv;
    }
  }
}

} // namespace gpu_kernels
