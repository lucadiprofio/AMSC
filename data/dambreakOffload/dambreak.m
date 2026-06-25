% bench_dambreak.m
% Dam break su dominio piatto per benchmark GPU.
% ~40.000 particelle al centro, si espandono per pressione idrostatica.
% Dominio grande → le particelle non escono.

clear all;
clc;

% Griglia: 200x200, dominio [0,100]x[0,100]
nelex = 200;
neley = 200;
Lx = 100;
Ly = 100;

hx = Lx / nelex;   % 0.5
hy = Ly / neley;    % 0.5

x = linspace(0, Lx, nelex+1);
y = linspace(0, Ly, neley+1);
[X, Y] = meshgrid(x, y);

% Dominio piatto: Z = 0 ovunque
Z = zeros(size(X));
[gx, gy] = gradient(Z, hx, hy);

% Particelle: blocco centrale [25,75] x [25,75]
DX = 0.3;  DY = 0.3;
[xp, yp] = meshgrid(25.15:DX:74.85, 25.15:DY:74.85);
hp = 3.0 * ones(size(xp));

hp = hp(:);
xp = xp(:);
yp = yp(:);

nmp = numel(xp);
fprintf('Numero particelle: %d\n', nmp);
fprintf('Celle griglia: %d x %d = %d\n', nelex, neley, nelex*neley);
fprintf('Nodi griglia: %d\n', (nelex+1)*(neley+1));

% Costanti
g     = 9.81;
xi    = 200;
rhosy = 1000.0;
T     = 5.0;

% Quantita delle particelle

Msys  = sum(hp * DX * DY * rhosy);
Mp    = Msys / nmp * ones(nmp, 1);
Vp    = Mp ./ rhosy;
Ap    = Vp ./ hp;
vp    = zeros(nmp, 2);

mu = 50.;
phi = 0.;
tauy = 2000.;

MERGE_SPLIT_ON = 1;

BINGHAM = 1.0;
FRICTION = 1.0;
CFL = 0.2;
BC_FLAG = 1;
eq_level = 0.0;

momp  = zeros(nmp, 2);

Zz = Z(:);
dZdx = gx(:);
dZdy = gy(:);

% Scrittura DATA.json
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
  if (~last)
    fprintf(fid, ',\n');
  else
    fprintf(fid, '\n');
  endif
endfunction

FID = fopen("DATA.json", "w");
fprintf(FID, "{\n");
v2j(FID, "x",           xp,        false);
v2j(FID, "y",           yp,        false);
v2j(FID, "Mp",          Mp,        false);
v2j(FID, "Ap",          Ap,        false);
v2j(FID, "vpx",         vp(:,1),   false);
v2j(FID, "vpy",         vp(:,2),   false);
v2j(FID, "Nex",         nelex,     false);
v2j(FID, "Ney",         neley,     false);
v2j(FID, "hx",          hx,        false);
v2j(FID, "hy",          hy,        false);
v2j(FID, "hp",          hp,        false);
v2j(FID, "mom_px",      momp(:,1), false);
v2j(FID, "mom_py",      momp(:,2), false);
v2j(FID, "g",           g,         false);
v2j(FID, "T",           T,         false);
v2j(FID, "xi",          xi,        false);
v2j(FID, "rho",         rhosy,     false);
v2j(FID, "Vp",          Vp,        false);
v2j(FID, "Z",           Zz,        false);
v2j(FID, "dZdx",        dZdx,      false);
v2j(FID, "dZdy",        dZdy,      false);
v2j(FID, "mu",         mu,         false);
v2j(FID, "phi",        phi,        false);
v2j(FID, "tauy",       tauy,       false);
v2j(FID, "MERGE_SPLIT_ON", MERGE_SPLIT_ON, false);
v2j(FID, "BINGHAM_ON",  BINGHAM,         false);
v2j(FID, "FRICTION_ON", FRICTION,         false);
v2j(FID, "eq_level",eq_level,           false); 
v2j(FID, "CFL",         CFL,       false);
v2j(FID, "BC_FLAG",     BC_FLAG,         true);
fprintf(FID, "}\n");
fclose(FID);

fprintf('DATA.json scritto correttamente: %d particelle, griglia %dx%d\n', nmp, nelex, neley);