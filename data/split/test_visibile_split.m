% test_visible_split.m
% Caso test per rendere visibile il merge/splitting.
% Canale stretto, rampa ripida, poche particelle a sinistra.
% Le particelle scivolano a destra, le celle a sinistra si svuotano → split.

clear all;
clc;

% Griglia piccola: 40x10, dominio [0,20]x[0,5]
nelex = 40;
neley = 10;
Lx = 20;
Ly = 5;

hx = Lx / nelex;   % 0.5
hy = Ly / neley;    % 0.5

x = linspace (0, Lx, nelex+1);
y = linspace (0, Ly, neley+1);
[X, Y] = meshgrid (x, y);

% Rampa: Z scende da 4 (sinistra) a 0 (destra)
Z = 4 * (1 - X/Lx);

[gx, gy] = gradient (Z, hx, hy);

% Particelle: SOLO nella meta sinistra
DX = 0.3;  DY = 0.3;
[xp, yp] = meshgrid (0.2:DX:9.8, 0.2:DY:4.8);
hp = 5.0 * ones(size(xp));   % acqua alta 3m

hp = hp(:);
xp = xp(:);
yp = yp(:);

% Rimuovi particelle con hp troppo basso
hzero = find(hp <= 0.01);
hp(hzero) = [];
xp(hzero) = [];
yp(hzero) = [];

nmp = numel(xp);
fprintf('Numero particelle: %d\n', nmp);
fprintf('Celle griglia: %d x %d = %d\n', nelex, neley, nelex*neley);
fprintf('Particelle per cella (media zona occupata): %.1f\n', nmp / (20*6));

% Costanti
g     = 9.81;
xi    = 200;
rhosy = 1000.0;
T     = 3.0;    % 3 secondi bastano per vedere il movimento

% Quantita' delle particelle
Msys  = sum(hp * DX * DY * rhosy);
Mp    = Msys / nmp * ones(nmp, 1);
Vp    = Mp ./ rhosy;
Ap    = Vp ./ hp;
vp    = zeros(nmp, 2);
momp  = zeros(nmp, 2);

Zz = Z(:);
dZdx = gx(:);
dZdy = gy(:);

% Scrittura manuale del DATA.json
function v2j (fid, name, x, last)
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

FID = fopen("DATA.json","w");
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
v2j(FID, "BINGHAM_ON",  0,         false);
v2j(FID, "FRICTION_ON", 0,         false);
v2j(FID, "CFL",         0.1,       false);
v2j(FID, "BC_FLAG",     1,         true);
fprintf(FID, "}\n");
fclose(FID);
disp("DATA.json scritto correttamente");