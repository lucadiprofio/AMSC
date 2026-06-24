% mass and momentum conservation test for dam break on inclined plane with gaussian obstacle

clear all;
close all;
clc;

% Domain: [0, 150] x [0, 20]
nelex = 75;
neley = 30;
Lx = 150;
Ly = 20;
hx = Lx / nelex;
hy = Ly / neley;

x = linspace(0, Lx, nelex+1);
y = linspace(0, Ly, neley+1);
[X, Y] = meshgrid(x, y);

% Topograhpy: inclned plane + gaussian bump
Z = 60 - (1/3).*X + 15.*exp(-((X-50).^2)./7 - ((Y-10).^2)./7);

% Gradient
[gx, gy] = gradient(Z, hx, hy);


figure(1);
surf(X, Y, Z);
title('Topography Z(x,y)');
xlabel('x [m]'); ylabel('y [m]'); zlabel('Z [m]');

% Particles: only in region [0, 20] x [0, 20]
Xmin = 0; Xmax = 20; Ymin = 0; Ymax = 20;
DX = 0.45; DY = 0.45;
[xp, yp] = meshgrid(Xmin+DX/2:DX:Xmax-DX/2, Ymin+DY/2:DY:Ymax-DY/2);
xp = xp(:);
yp = yp(:);

% Initial height: h = 60 - Z interpolated at particle positions
hp = 60 - interp2(X, Y, Z, xp, yp, 'linear');

% Remove particles with zero or negative height
hzero = find(hp <= 0.001);
hp(hzero) = [];
xp(hzero) = [];
yp(hzero) = [];

nmp = numel(xp);
fprintf('Particles: %d\n', nmp);


figure(2);
surf(X, Y, Z);
hold on;
scatter3(xp, yp, hp + interp2(X, Y, Z, xp, yp, 'linear'), 7, 'b', 'filled');
title('Initial condition: h_p + Z');
xlabel('x [m]'); ylabel('y [m]'); zlabel('h + Z [m]');

%% Physical parameters (from paper Section 4.2)
g     = 9.81;
xi    = 200;      % turbulence coefficient [m/s^2]
rhosy = 1200.0;   % mudflow density [kg/m^3]
T     = 7;        % final time [s]

%% Initial conditions for particles
Msys  = sum(hp * DX * DY * rhosy);
Mp    = Msys / nmp * ones(nmp, 1);
Vp    = Mp ./ rhosy;
Ap    = Vp ./ hp;
vp    = zeros(nmp, 2);
momp  = zeros(nmp, 2);

% Flag (from paper: friction on, Bingham on)
BINGHAM  = 1.0;
FRICTION = 1.0;
CFL      = 0.2;
BC_FLAG  = 1;

MERGE_SPLIT_ON = 1;
physical_boundary = [1, 0, 0, 0]  % top, right, bottom, left

eq_level = 0.0;   % non well-balanced

% Rheological parameters (not specified in the paper
% we choose them to be consistent with the well-balanced test):
mu = 10; % Pa·s (viscosity)
phi = 12; % degrees (friction angle)
tauy = 30; % Pa·s (as in well-balanced test)

Zz   = Z(:);
dZdx = gx(:);
dZdy = gy(:);

%% writing DATA.json
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
v2j(FID, "x",           xp,         false);
v2j(FID, "y",           yp,         false);
v2j(FID, "Mp",          Mp,         false);
v2j(FID, "Ap",          Ap,         false);
v2j(FID, "vpx",         vp(:,1),    false);
v2j(FID, "vpy",         vp(:,2),    false);
v2j(FID, "Nex",         nelex,      false);
v2j(FID, "Ney",         neley,      false);
v2j(FID, "hx",          hx,         false);
v2j(FID, "hy",          hy,         false);
v2j(FID, "hp",          hp,         false);
v2j(FID, "mom_px",      momp(:,1),  false);
v2j(FID, "mom_py",      momp(:,2),  false);
v2j(FID, "g",           g,          false);
v2j(FID, "T",           T,          false);
v2j(FID, "xi",          xi,         false);
v2j(FID, "rho",         rhosy,      false);
v2j(FID, "Vp",          Vp,         false);
v2j(FID, "Z",           Zz,         false);
v2j(FID, "dZdx",        dZdx,       false);
v2j(FID, "dZdy",        dZdy,       false);

v2j(FID, "mu",        mu,       false);
v2j(FID, "phi",       phi,      false);
v2j(FID, "tauy",      tauy,      false);

v2j(FID, "MERGE_SPLIT_ON", MERGE_SPLIT_ON, false);
v2j(FID, "physical_boundary", physical_boundary, false);

v2j(FID, "BINGHAM_ON",  BINGHAM,    false);
v2j(FID, "CFL",         CFL,        false);
v2j(FID, "BC_FLAG",     BC_FLAG,    false);
v2j(FID, "eq_level",    eq_level,   false);
v2j(FID, "FRICTION_ON", FRICTION,   true);
fprintf(FID, "}\n");
fclose(FID);
fprintf('DATA.json written: %d particles, grid %dx%d\n', nmp, nelex, neley);
fprintf('Domain: [0,%.0f] x [0,%.0f], hx=%.3f, hy=%.3f\n', Lx, Ly, hx, hy);
fprintf('Parameters: rho=%.0f, xi=%.0f, T=%.0f, CFL=%.2f\n', rhosy, xi, T, CFL);
