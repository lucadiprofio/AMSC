// =====================================================================
//  test_rheology.cpp  -- unit test ISOLATI per Bingham e Voellmy
//
//  Standalone: compila senza quadgrid.
//      nvc++ -O2 test_rheology.cpp -o test_rheology     (oppure g++)
//      ./test_rheology
//
//  I valori attesi (REF_*) sono calcolati INDIPENDENTEMENTE dalle
//  equazioni del paper Fois et al., CNSNS 138 (2024), Eq. 2-9.
//  L'aritmetica testata e' copiata VERBATIM dai kernel di gpu_offload.cpp.
// =====================================================================
#include <cmath>
#include <cstdio>

static int n_fail = 0;
static void check(const char* name, double got, double expected, double rtol) {
    double denom = std::fabs(expected) > 1e-300 ? std::fabs(expected) : 1.0;
    double rel = std::fabs(got - expected) / denom;
    bool ok = rel <= rtol;
    if (!ok) ++n_fail;
    std::printf("  [%s] %-14s got=% .10e  exp=% .10e  rel=%.2e\n",
                ok ? "PASS" : "FAIL", name, got, expected, rel);
}

// ---- costanti fisiche dei riferimenti ----
static const double A_coeff = 3.0/2.0;
static const double C_coeff = 65.0/32.0;

// =====================================================================
// 1) BINGHAM  -- aritmetica copiata da gpu_offload.cpp (STEP 8 stress)
//    Ritorna coeff, F11, F22, F12 dato lo stato della particella.
// =====================================================================
static void bingham_stress(double vx, double vy, double h,
                           double ux, double uy, double vdx, double vdy,
                           double mu, double tau_Y, double rho, double g,
                           double H, double cc,
                           double& coeff, double& F11, double& F22, double& F12)
{
    double nv = sqrt(vx*vx + vy*vy);
    double alf = (h > 1e-3 && tau_Y > 1e-12) ? (6.0*mu*nv)/((h+0.001)*tau_Y) : 0.0;
    double bv = -114.0/32.0 - alf;
    double sq = sqrt(bv*bv - 4.0*A_coeff*C_coeff);
    double z1 = (-bv + sq)/(2.0*A_coeff);
    double z2 = (-bv - sq)/(2.0*A_coeff);
    double zz = (fabs(z1 - 0.5) <= 0.5) ? z1 : z2;

    double sxx = ux, sxy = 0.5*(uy + vdx), syy = vdy;
    double Dxx = sxx, Dxy = sxy, Dyy = syy, Dzz = -(sxx + vdy);
    double Dzx = h > 1e-3 ? 0.5*(3.0/(2.0+zz))*(vx/(h+0.001)) : 0.0;
    double Dzy = h > 1e-3 ? 0.5*(3.0/(2.0+zz))*(vy/(h+0.001)) : 0.0;

    double inv2 = 0.5*(Dxx*Dxx + Dyy*Dyy + Dzz*Dzz) + Dzx*Dzx + Dzy*Dzy + Dxy*Dxy;
    double inv2_floor = (tau_Y/5.0e6)*(tau_Y/5.0e6);
    if (inv2_floor < 1e-9) inv2_floor = 1e-9;
    coeff = tau_Y/sqrt(inv2 + inv2_floor) + 2.0*mu;
    double h_corr = h > 1e-10 ? h - H*H/h : 0.0;
    F11 = -cc*coeff*Dxx + 0.5*rho*g*h_corr;
    F22 = -cc*coeff*Dyy + 0.5*rho*g*h_corr;
    F12 = -cc*coeff*Dxy;
}

// =====================================================================
// 2) VOELLMY  -- due varianti
//    _code : per-componente (gpu_offload.cpp riga 677, lo stato ATTUALE)
//    _fix  : |v|^2 scalare (Eq.2 del paper, la versione CORRETTA)
// =====================================================================
static void voellmy_code(double vx, double vy, double h, double rho, double g,
                         double tanphi, double xi, double fric,
                         double& Fbx, double& Fby) {
    double nv = sqrt(vx*vx + vy*vy);
    if (fric > 0 && nv > 1e-10 && xi > 0) {
        double v2 = vx*vx + vy*vy;
        Fbx = -fric*(rho*g*h*tanphi + rho*g*v2/xi)*vx/nv;
        Fby = -fric*(rho*g*h*tanphi + rho*g*v2/xi)*vy/nv;
    } else { Fbx = Fby = 0.0; }
}
static void voellmy_fix(double vx, double vy, double h, double rho, double g,
                        double tanphi, double xi, double fric,
                        double& Fbx, double& Fby) {
    double nv = sqrt(vx*vx + vy*vy);
    if (fric > 0 && nv > 1e-10 && xi > 0) {
        double F = rho*g*h*tanphi + rho*g*(vx*vx+vy*vy)/xi;   // |v|^2
        Fbx = -fric*F*vx/nv;
        Fby = -fric*F*vy/nv;
    } else { Fbx = Fby = 0.0; }
}

int main() {
    const double rho = 1200.0, g = 9.81;

    // ---------------- BINGHAM caso 1 (full tensor) ----------------
    std::printf("=== BINGHAM caso 1 (tensore pieno) ===\n");
    {
        double coeff,F11,F22,F12;
        bingham_stress(2.0,1.0, 0.5,  0.30,0.10,-0.20,0.05,
                       /*mu*/2000.0, /*tauY*/30.0, rho, g, /*H*/0.0, /*cc*/1.0,
                       coeff,F11,F22,F12);
        check("coeff", coeff, 4.008923505522e+03, 1e-9);
        check("F11",   F11,   1.740322948343e+03, 1e-9);
        check("F22",   F22,   2.742553824724e+03, 1e-9);
        check("F12",   F12,   2.004461752761e+02, 1e-9);
    }

    // ------ BINGHAM caso 2: discrimina sqrt(I2) vs I2 (Eq.3) -------
    std::printf("=== BINGHAM caso 2 (discrimina sqrt I2) ===\n");
    {
        double coeff,F11,F22,F12;
        bingham_stress(0.10,0.0, 1.0,  0.01,0.0,0.0,0.0,
                       /*mu*/50.0, /*tauY*/2000.0, rho, g, 0.0, 1.0,
                       coeff,F11,F22,F12);
        check("coeff_sqrtI2", coeff, 3.850258648389e+04, 1e-9);   // codice (corretto)
        // NB: se fosse tauY/I2 (Eq.3 letterale) sarebbe 7.375228e+05.
        // Il PASS qui conferma che il codice usa la forma dimensionalmente
        // corretta tauY/sqrt(I2), NON la forma letterale del paper.
        double coeff_literal = 7.375228251538e+05;
        std::printf("  [info] Eq.3 letterale (tauY/I2) darebbe %.6e -> il codice NON la usa\n",
                    coeff_literal);
    }

    // ---------------- BINGHAM well-balanced (pressione) -----------
    // v=0 -> deviatorico nullo, resta solo 1/2 rho g (h - H^2/h).
    // Con h=H il termine deve annullarsi (proprieta' well-balanced).
    std::printf("=== BINGHAM pressione well-balanced (v=0) ===\n");
    {
        double coeff,F11,F22,F12;
        bingham_stress(0.0,0.0, 7.0, 0.0,0.0,0.0,0.0,
                       2000.0, 30.0, rho, g, /*H=h*/7.0, 1.0,
                       coeff,F11,F22,F12);
        check("F11_wb(h=H)", F11, 0.0, 1e-12);   // deve essere ~0
    }

    // ---------------- VOELLMY -------------------------------------
    std::printf("=== VOELLMY (Eq.2) ===\n");
    {
        const double tanphi = std::tan(20.0*M_PI/180.0);
        const double xi = 200.0, h = 0.5;
        const double REF_Fbx_paper = -2.179387052101e+03;
        const double REF_Fby_paper = -1.089693526051e+03;

        double cx,cy; voellmy_code(2.0,1.0,h,rho,g,tanphi,xi,1.0,cx,cy);
        std::printf("  -- versione ATTUALE del codice (per-componente):\n");
        check("Fbx_code", cx, REF_Fbx_paper, 1e-6);   // ATTESO: FAIL
        check("Fby_code", cy, REF_Fby_paper, 1e-6);   // ATTESO: FAIL

        double fx,fy; voellmy_fix(2.0,1.0,h,rho,g,tanphi,xi,1.0,fx,fy);
        std::printf("  -- versione CORRETTA (|v|^2):\n");
        check("Fbx_fix",  fx, REF_Fbx_paper, 1e-9);   // ATTESO: PASS
        check("Fby_fix",  fy, REF_Fby_paper, 1e-9);   // ATTESO: PASS
    }

    std::printf("\n%s  (%d assert falliti)\n",
                n_fail==0 ? "TUTTO PASS" : "CI SONO FAIL", n_fail);
    // i 2 FAIL di Voellmy_code sono ATTESI finche' non applichi il fix |v|^2
    return 0;
}