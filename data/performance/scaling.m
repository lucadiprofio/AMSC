% =====================================================================
%  make_bench.m  --  genera DATA_bench_<N>.json per lo studio di scaling
%
%  Scala griglia E particelle insieme mantenendo ~4 particelle/cella,
%  cosi il problema cresce in modo uniforme (no sbilanciamento di carico).
%  Geometria: pendio dolce uniforme + strato di materiale costante.
%  Stabile e con costo per-step ~proporzionale a N a ogni taglia.
%
%  Due modalita (controllate dai campi nel JSON, gestite dal solver):
%    - scaling:    DT_FIXED > 0, NSTEPS passi a dt costante  -> curva di scaling
%    - applicativo: DT_FIXED = 0, dt adattivo fino a T        -> speedup end-to-end
% =====================================================================
clear all; clc;

% =====================================================================
function write_json(fname, xp, yp, Mp, Ap, vp, momp, Nex, Ney, hx, hy, hp, ...
                    g, T, xi, rho, Vp, Z, dZdx, dZdy, mu, phi, tauy, ...
                    BINGHAM, FRICTION, CFL, BC_FLAG, eq_level, DT_FIXED, NSTEPS)
  FID = fopen(fname, "w");
  fprintf(FID, "{\n");
  v2j(FID,"x",xp,false);            v2j(FID,"y",yp,false);
  v2j(FID,"Mp",Mp,false);          v2j(FID,"Ap",Ap,false);
  v2j(FID,"vpx",vp(:,1),false);    v2j(FID,"vpy",vp(:,2),false);
  v2j(FID,"Nex",Nex,false);        v2j(FID,"Ney",Ney,false);
  v2j(FID,"hx",hx,false);          v2j(FID,"hy",hy,false);
  v2j(FID,"hp",hp,false);
  v2j(FID,"mom_px",momp(:,1),false); v2j(FID,"mom_py",momp(:,2),false);
  v2j(FID,"g",g,false);            v2j(FID,"T",T,false);
  v2j(FID,"xi",xi,false);          v2j(FID,"rho",rho,false);
  v2j(FID,"Vp",Vp,false);          v2j(FID,"Z",Z,false);
  v2j(FID,"dZdx",dZdx,false);      v2j(FID,"dZdy",dZdy,false);
  v2j(FID,"mu",mu,false);          v2j(FID,"phi",phi,false);
  v2j(FID,"tauy",tauy,false);
  v2j(FID,"BINGHAM_ON",BINGHAM,false);  v2j(FID,"FRICTION_ON",FRICTION,false);
  v2j(FID,"CFL",CFL,false);        v2j(FID,"BC_FLAG",BC_FLAG,false);
  v2j(FID,"eq_level",eq_level,false);
  v2j(FID,"DT_FIXED",DT_FIXED,false); v2j(FID,"MERGE_SPLIT_ON",FRICTION,false);
  v2j(FID,"NSTEPS",NSTEPS,true);
  fprintf(FID, "}\n");
  fclose(FID);
endfunction

function v2j(fid, name, x, last)
  if isscalar(x) && ~ischar(x)
    fprintf(fid, '"%s": %.17g', name, x);
  elseif ischar(x)
    fprintf(fid, '"%s": "%s"', name, x);
  else
    x = x(:);
    fprintf(fid, '"%s": [', name);
    fprintf(fid, '%.17g', x(1));
    for k = 2:numel(x)
      fprintf(fid, ',%.17g', x(k));
    endfor
    fprintf(fid, ']');
  endif
  if (~last) fprintf(fid, ',\n'); else fprintf(fid, '\n'); endif
endfunction

% --- taglie da generare (geometriche) ---
N_list = [1000 3000 10000 30000 100000 300000 500000];

% --- modalita benchmark ---
DT_FIXED = 1e-3;     % >0 = dt fisso (scaling). Metti 0 per dt adattivo (applicativo).
NSTEPS   = 100;      % passi a dt fisso (usato solo se DT_FIXED>0)
T        = 20.0;     % tempo finale (usato solo se DT_FIXED==0)

% --- parametri fisici (reologia ATTESA: viscosita grande, yield piccolo) ---
g    = 9.81;
rho  = 1200.0;
xi   = 200.0;
mu   = 2000.0;       % Pa s
phi  = 20.0;         % gradi
tauy = 30.0;         % Pa
CFL  = 0.2;          % usato solo in modalita adattiva
eq_level = 0.0;      % non well-balanced
BINGHAM  = 1.0;      % reologia ACCESA (e' il caso d'uso realistico)
FRICTION = 1.0;
BC_FLAG  = 1.0;

hx = 1.0; hy = 1.0;  % lato cella fisso; il dominio cresce con N
s  = 0.5;            % passo particelle (2 per cella per lato -> 4 ppc)
slope = tan(2*pi/180);  % pendio dolce ~2 gradi lungo x
h0 = 5.0;            % spessore strato di materiale [m]

for N = N_list
  ncell = round(sqrt(N)/2);           % celle per lato
  Nex = ncell; Ney = ncell;
  Lx = Nex*hx;  Ly = Ney*hy;

  % --- nodi e topografia (pendio dolce uniforme) ---
  x = linspace(0, Lx, Nex+1);
  y = linspace(0, Ly, Ney+1);
  [X, Y] = meshgrid(x, y);
  Zz = 100.0 - slope.*X;              % quota: scende lungo x
  [gx, gy] = gradient(Zz, hx, hy);
  Z = Zz(:); dZdx = gx(:); dZdy = gy(:);

  % --- particelle: riempiono l interno con un piccolo margine ---
  m = 1.0;                             % margine [m] per non toccare il bordo a t=0
  xp = (m + s/2):s:(Lx - m);
  yp = (m + s/2):s:(Ly - m);
  [XP, YP] = meshgrid(xp, yp);
  xp = XP(:); yp = YP(:);
  nmp = numel(xp);
  MERGE_SPLIT_ON=1.0;

  hp = h0 * ones(nmp, 1);

  % --- massa FISICA (Eq.18: V_p = A_p h_p, m_p = rho V_p) ---
  Ap = (s*s) * ones(nmp, 1);          % footprint particella = passo^2
  Vp = Ap .* hp;
  Mp = rho * Vp;
  vp   = zeros(nmp, 2);
  momp = zeros(nmp, 2);

  fname = sprintf("DATA_bench_%d.json", N);
  write_json(fname, xp, yp, Mp, Ap, vp, momp, Nex, Ney, hx, hy, hp, ...
             g, T, xi, rho, Vp, Z, dZdx, dZdy, mu, phi, tauy, ...
             BINGHAM, FRICTION, CFL, BC_FLAG, eq_level, DT_FIXED, NSTEPS);
  printf("%s : %d particelle, griglia %dx%d (%d celle), ppc~%.1f\n", ...
         fname, nmp, Nex, Ney, Nex*Ney, nmp/(Nex*Ney));
end
disp("fatto.");
