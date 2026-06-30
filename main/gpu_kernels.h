#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

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
    double* d_pv_avx, double* d_pv_avy);

// -------------------------------------------------------------------
// K1 — P2G: scatter mass + momentum into Mv,mom_vx,mom_vy
// -------------------------------------------------------------------
void scatter_mass_momentum (
    int np, int nrows, double hx, double hy,
    const int* cstart, const int* cptcls, const int* ccidx, const int* coloff,
    const double* d_x, const double* d_y,
    const double* d_Mp, const double* d_mom_px, const double* d_mom_py,
    double* d_Mv, double* d_mom_vx, double* d_mom_vy);

// -------------------------------------------------------------------
// K2 — G2PD: interpolate grad(Z) to particles (dZxp, dZyp)
// -------------------------------------------------------------------
void gradZ_to_particles (
    int np, int nrows, double hx, double hy,
    const double* d_x, const double* d_y, const int* d_p2g,
    const double* d_Z,
    double* d_dZxp, double* d_dZyp);

// -------------------------------------------------------------------
// K3 — Gravity-along-slope force (Fpx, Fpy)
// -------------------------------------------------------------------
void gravity_force (
    int np, double g_c,
    const double* d_hp, const double* d_H, const double* d_Mp,
    const double* d_dZxp, const double* d_dZyp,
    double* d_Fpx, double* d_Fpy);

// -------------------------------------------------------------------
// K3b — Friction-to-nodal-force conversion (Fric_px, Fric_py)
// -------------------------------------------------------------------
void friction_particle_force (
    int np,
    const double* d_Ap, const double* d_Fb_x, const double* d_Fb_y,
    double* d_Fric_px, double* d_Fric_py);

// -------------------------------------------------------------------
// K4 — P2G: scatter Fpx,Fpy (gravity) into FPxv,FPyv
// -------------------------------------------------------------------
void scatter_gravity_force (
    int np, int nrows, double hx, double hy,
    const int* cstart, const int* cptcls, const int* ccidx, const int* coloff,
    const double* d_x, const double* d_y,
    const double* d_Fpx, const double* d_Fpy,
    double* d_FPxv, double* d_FPyv);

// -------------------------------------------------------------------
// K4b — P2G: scatter Fric_px,Fric_py into Fric_x,Fric_y
// -------------------------------------------------------------------
void scatter_friction_force (
    int np, int nrows, double hx, double hy,
    const int* cstart, const int* cptcls, const int* ccidx, const int* coloff,
    const double* d_x, const double* d_y,
    const double* d_Fric_px, const double* d_Fric_py,
    double* d_Fric_x, double* d_Fric_y);

// -------------------------------------------------------------------
// K4c — F_ext per node = FPxv+Fric_x, FPyv+Fric_y
// -------------------------------------------------------------------
void compute_F_ext (
    int nn,
    const double* d_FPxv, const double* d_FPyv,
    const double* d_Fric_x, const double* d_Fric_y,
    double* d_F_ext_vx, double* d_F_ext_vy);

// -------------------------------------------------------------------
// K5 — P2GD: scatter internal forces into F_int_vx, F_int_vy
// -------------------------------------------------------------------
void scatter_internal_forces (
    int np, int nrows, double hx, double hy,
    const int* cstart, const int* cptcls, const int* ccidx, const int* coloff,
    const double* d_x, const double* d_y, const double* d_Vp,
    const double* d_F11, const double* d_F12,
    const double* d_F21, const double* d_F22,
    double* d_F_int_vx, double* d_F_int_vy);

// -------------------------------------------------------------------
// K6 — Ftot + momentum integration (nodal)
// -------------------------------------------------------------------
void nodal_force_and_momentum (
    int nn, double dt,
    const double* d_F_ext_vx, const double* d_F_ext_vy,
    const double* d_F_int_vx, const double* d_F_int_vy,
    double* d_Ftot_vx, double* d_Ftot_vy,
    double* d_mom_vx,  double* d_mom_vy);

// -------------------------------------------------------------------
// K6b — Nodal velocity/acceleration + boundary conditions
// -------------------------------------------------------------------
void nodal_velocity_and_bc (
    int nn, int nrows, int ncols, double dt, bool bc_flag,
    const double* d_Mv, const double* d_Ftot_vx, const double* d_Ftot_vy,
    double* d_mom_vx, double* d_mom_vy,
    double* d_avx, double* d_avy,
    double* d_vvx, double* d_vvy,
    double* d_vvxL, double* d_vvyL);

// -------------------------------------------------------------------
// K7 — G2P: interpolate vvxL,vvyL,avx,avy to particles, then advect
//      and update momentum
// -------------------------------------------------------------------
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
    double* max_vmag_out);

// -------------------------------------------------------------------
// K8 — G2PD: interpolate vvxL,vvyL gradients to particles, then
//      height update (sc, hp, Ap floor logic)
// -------------------------------------------------------------------
void g2pd_gradients_and_height_update (
    int np, int nrows, double hx, double hy, double dt,
    const double* d_x, const double* d_y, const int* d_p2g,
    const double* d_vvxL, const double* d_vvyL,
    double* d_vpx_dx, double* d_vpx_dy,
    double* d_vpy_dx, double* d_vpy_dy,
    double* d_hp, double* d_Vp, double* d_Ap,
    int* n_scfloor_out, int* n_hpfloor_out,
    double* min_sc_out, double* min_hp_out);

// -------------------------------------------------------------------
// K9a — Voellmy friction (Fb_x, Fb_y)
// -------------------------------------------------------------------
void voellmy_friction (
    int np, double rho, double g_c, double tan_fa, double xi, double fric_on,
    const double* d_vpx, const double* d_vpy, const double* d_hp,
    double* d_Fb_x, double* d_Fb_y);

// -------------------------------------------------------------------
// K9b — Bingham, stress update (F11,F12,F21,F22)
// -------------------------------------------------------------------
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
    double* max_Dxx_out);

// -------------------------------------------------------------------
// K10a — G2P: Z -> Zp, hpZ = hp + Zp
// -------------------------------------------------------------------
void g2p_height (
    int np, int nrows, double hx, double hy,
    const double* d_x, const double* d_y, const int* d_p2g,
    const double* d_Z, const double* d_hp,
    double* d_Zp, double* d_hpZ);

// -------------------------------------------------------------------
// K10b — P2G: scatter Mp,vpx,vpy,apx,apy,hp -> rho_v,vvx,vvy,avx,avy,HV
// -------------------------------------------------------------------
void scatter_plotvars (
    int np, int nrows, double hx, double hy,
    const int* cstart, const int* cptcls, const int* ccidx, const int* coloff,
    const double* d_x, const double* d_y,
    const double* d_Mp, const double* d_vpx, const double* d_vpy,
    const double* d_apx, const double* d_apy, const double* d_hp,
    double* d_pv_rho, double* d_pv_vvx, double* d_pv_vvy,
    double* d_pv_avx, double* d_pv_avy, double* d_HV);

// -------------------------------------------------------------------
// K10d — normalise Plotvars by nodal mass
// -------------------------------------------------------------------
void normalize_plotvars (
    int nn,
    double* d_pv_rho, double* d_pv_vvx, double* d_pv_vvy,
    double* d_pv_avx, double* d_pv_avy);

    
} // namespace gpu_kernels

#endif /* GPU_KERNELS_H */
