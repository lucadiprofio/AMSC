% WELL-BALANCED

clear all;
clc;

Lx = 10; Ly = 10;
nEx = 50; %180
nEy = 50;
hl = [Lx/nEx Ly/nEy];
nl=2;
[xv,yv] = meshgrid(0:hl(1):Lx,0:hl(2):Ly);
NVx = size(xv,2);
NVy = size(yv,1);
NV  = NVx*NVy;
NP = (NVx-1)*(NVy-1);
Nex = NVx-1;
Ney = NVy-1;
x = xv(:); y = yv(:);

DT_FIXED = 0;   
NSTEPS   = 100;

Zplot =5.*exp(-2/5.*((xv-5).^2 + (yv-5).^2));  % choose this one for Z_1
%Zplot(xv>=4 & xv<=8 & yv>=4 & yv<=8) = 4;     % choose this one for Z_2

h = 0*Zplot;
Ymin = 0; Ymax = 10; Xmin = 0; Xmax = 10;
select = (yv>Ymin & yv<Ymax & xv >Xmin & xv < Xmax);

[gx,gy] = gradient(Zplot, hl(1), hl(2));

X = xv; Y = yv;
h= 10 - Zplot;
h(h < 0) = 0;

surf (xv, yv, Zplot)
hold all
surf (X, Y, Zplot+h)
axis equal
xlabel('x [m]')
ylabel('y [m]')
zlabel('z [m]')
title('Initial condition')

DX = .17; DY = .17;
[xp, yp] = meshgrid (Xmin:DX:Xmax, Ymin:DY:Ymax);

hp = interp2 (X, Y, h, xp, yp, 'linear');
hp = hp(:);
xp = xp(:);
yp = yp(:);

%{
xL      = 0+(0.5*hl(1)/nl):hl(1)/nl:10;
yL      = 0+(0.5*hl(1)/nl):hl(1)/nl: Ly-(0.5*hl(1)/nl);

[xpl,ypl] = meshgrid(xL,yL);
L = Lx;

xpr = [];
ypr = xpr;

xp = [xpl(:); xpr(:)];
yp = [ypl(:); ypr(:)];

hp =  10.*ones(size(xp(:),1),1);
%}

Xp = [xp(:),yp(:)];



figure()
surf(xv,yv,Zplot);
title('normal')




Z = Zplot(:); %reshape(Zplot,NVy*NVx,1);
dZdx = gx(:); %reshape(gx,NV,1);
dZdy = gy(:); %reshape(gy,NV,1);
nmp = numel(xp(:));







figure()
surf(xv,yv,Zplot)
hold on
scatter3(xp,yp,hp,7,'b','filled');
axis([0 10 0 10 0 10])
xlabel('x');
ylabel('y');



%% Constants
g     = 9.81;
xi    = 200.;
vis   = 0;
ty    = 0;
T     = 1.;

nmp   = numel(xp);
rhosy = 1200.0;
Ap    = DX*DY * ones(nmp, 1);   % footprint = spaziatura particelle
Vp    = Ap .* hp;               % volume colonna = area × profondità
Mp    = rhosy * Vp;             % massa = ρ × volume  → Mp ∝ hp

vp = zeros(nmp, 2);
momp  = zeros (nmp,2);

Fb(:,1) = zeros (nmp,1);
Fb(:,2) = zeros (nmp,1);

mu = 2000.; % Pa
phi = 20.; % degrees
tauy = 30.; % Pa * s

BINGHAM = 0.0;
FRICTION = 0.0;
CFL = 0.1;
BC_FLAG = 0.0;

MERGE_SPLIT_ON = 0;

eq_level = 10.0;

%%
DATA = struct ( ...
     "x", xp, ...
     "y", yp, ...
     "Mp", Mp, ...
     "Ap", Ap, ...
     "vpx", vp(:,1), ...
     "vpy", vp(:,2), ...
     "Nex", nEx, ...
     "Ney", nEy, ...
     "hx", hl(1), ...
     "hy", hl(2), ...
     "hp", hp, ...
     "mom_px", momp(:,1), ...
     "mom_py", momp(:,2), ...
     "g", g, ...
     "T", T, ...
     "xi", xi, ...
     "rho", rhosy, ...
     "Vp", Vp, ...
     "Z", Z, ...
     "dZdx", dZdx, ...
     "dZdy", dZdy, ...

     "mu", mu, ...
     "phi", phi, ...
     "tauy", tauy, ...
    "DT_FIXED", DT_FIXED, ...
    "NSTEPS", NSTEPS, ...
     "BINGHAM_ON", BINGHAM, ...
     "FRICTION_ON", FRICTION, ...
     "CFL", CFL, ...
     "BC_FLAG", BC_FLAG, ...
     "MERGE_SPLIT_ON", MERGE_SPLIT_ON, ...
     "eq_level", eq_level ...
   );

% --- writing DATA.json ---
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
v2j(FID, "x",           xp,         false);
v2j(FID, "y",           yp,         false);
v2j(FID, "Mp",          Mp,         false);
v2j(FID, "Ap",          Ap,         false);
v2j(FID, "vpx",         vp(:,1),    false);
v2j(FID, "vpy",         vp(:,2),    false);
v2j(FID, "Nex",         Nex,        false);
v2j(FID, "Ney",         Ney,        false);
v2j(FID, "hx",          hl(1),      false);
v2j(FID, "hy",          hl(2),      false);
v2j(FID, "hp",          hp,         false);
v2j(FID, "mom_px",      momp(:,1),  false);
v2j(FID, "mom_py",      momp(:,2),  false);
v2j(FID, "g",           g,          false);
v2j(FID, "T",           T,          false);
v2j(FID, "xi",          xi,         false);
v2j(FID, "rho",         rhosy,      false);
v2j(FID, "Vp",          Vp,         false);
v2j(FID, "Z",           Z,          false);
v2j(FID, "dZdx",        dZdx,       false);
v2j(FID, "dZdy",        dZdy,       false);

v2j(FID, "mu",          mu,         false);
v2j(FID, "phi",         phi,        false);
v2j(FID, "tauy",        tauy,       false);

v2j(FID, "BC_FLAG",     BC_FLAG,    false);
v2j(FID, "CFL",         CFL,        false);
v2j(FID, "BINGHAM_ON",  BINGHAM,    false);
v2j(FID, "FRICTION_ON", FRICTION,   false);
v2j(FID, "DT_FIXED",  DT_FIXED,    false);
v2j(FID, "NSTEPS", NSTEPS,   false);
v2j(FID, "MERGE_SPLIT_ON", MERGE_SPLIT_ON, false);

v2j(FID, "eq_level", eq_level, true);
fprintf(FID, "}\n");
fclose(FID);
disp("DATA.json scritto correttamente");
