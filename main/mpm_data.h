#ifndef HAVE_MPM_DATA
#define HAVE_MPM_DATA

#include <json.hpp>

struct DATA
{
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> Mp;
  std::vector<double> Ap;
  std::vector<double> vpx;
  std::vector<double> vpy;
  int Nex;
  int Ney;
  double hx;
  double hy;
  std::vector<double> hp;
  std::vector<double> mom_px;
  std::vector<double> mom_py;
  double g;
  double T;
  double rho;
  std::vector<double> Vp;
  double xi;
  std::vector<double> Z;
  std::vector<double> dZdx;
  std::vector<double> dZdy;

  // rheology
  double mu;
  double phi;
  double tauy;

  double BINGHAM_ON;
  double FRICTION_ON;
  double CFL;
  int BC_FLAG;

  int MERGE_SPLIT_ON;

  double eq_level;

  double DT_FIXED;   // dal JSON: >0 = dt fisso (scaling), 0 = adattivo
  int    NSTEPS;     // dal JSON: passi in modalità dt fisso

    // merge/split (opzionali: default = valori di ms_config) -> si tarano dal .m
  double ms_alpha;
  double ms_beta;
  double ms_split_hp_min;
  double ms_shear_split;
  double ms_hp_min;
  double ms_max_dv;
  int    ms_min_level;
  int    ms_max_level;
  int    ms_min_particles_per_cell;
  int    ms_call_interval;
  double ms_max_ops; // double per evitare overflow int da JSON, poi cast

  DATA (const char* filename);
};

void
from_json (const nlohmann::json &j, DATA &d)
{
  j.at("x").get_to(d.x);
  j.at("y").get_to(d.y);
  j.at("Mp").get_to(d.Mp);
  j.at("Ap").get_to(d.Ap);
  j.at("vpx").get_to(d.vpx);
  j.at("vpy").get_to(d.vpy);
  j.at("Nex").get_to(d.Nex);
  j.at("Ney").get_to(d.Ney);
  j.at("hx").get_to(d.hx);
  j.at("hy").get_to(d.hy);
  j.at("hp").get_to(d.hp);
  j.at("mom_px").get_to(d.mom_px);
  j.at("mom_py").get_to(d.mom_py);
  j.at("rho").get_to(d.rho);
  j.at("g").get_to(d.g);
  j.at("T").get_to(d.T);
  j.at("Vp").get_to(d.Vp);
  j.at("xi").get_to(d.xi);
  j.at("Z").get_to(d.Z);
  j.at("dZdx").get_to(d.dZdx);
  j.at("dZdy").get_to(d.dZdy);

  j.at("mu").get_to(d.mu);
  j.at("phi").get_to(d.phi);
  j.at("tauy").get_to(d.tauy);

  j.at("BINGHAM_ON").get_to(d.BINGHAM_ON);
  j.at("FRICTION_ON").get_to(d.FRICTION_ON);
  j.at("CFL").get_to(d.CFL);
  j.at("BC_FLAG").get_to(d.BC_FLAG);

  j.at("MERGE_SPLIT_ON").get_to(d.MERGE_SPLIT_ON);

  j.at("eq_level").get_to(d.eq_level);

  d.DT_FIXED = j.value("DT_FIXED", 0.0);
  d.NSTEPS   = j.value("NSTEPS", 0);

    // merge/split: tutti opzionali (default = ms_config). DATA.json vecchi restano validi.
  d.ms_alpha                  = j.value("ms_alpha", 0.9);
  d.ms_beta                   = j.value("ms_beta", 0.30);
  d.ms_split_hp_min           = j.value("ms_split_hp_min", 0.0);
  d.ms_shear_split            = j.value("ms_shear_split", 1e30);
  d.ms_hp_min                 = j.value("ms_hp_min", 0.05);
  d.ms_max_dv                 = j.value("ms_max_dv", 0.01);
  d.ms_min_level              = j.value("ms_min_level", -2);
  d.ms_max_level              = j.value("ms_max_level", 2);
  d.ms_min_particles_per_cell = j.value("ms_min_particles_per_cell", 2);
  d.ms_call_interval          = j.value("ms_call_interval", 10);
  d.ms_max_ops                = j.value("ms_max_ops", 1e9);

}

DATA::DATA (const char* filename) {
  nlohmann::json json;
  std::ifstream json_file (filename);
  json_file>>json;
  json_file.close ();
  json.get_to (*this);
};

#endif
