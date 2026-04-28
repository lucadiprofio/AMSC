
clear all;
close all;
clc;

neley = 60;
nelex = 200;

##hx = 0.5;
##hy = 0.5;

y = linspace (0, 30,neley+1);
x = linspace (0, 100, nelex+1);


hx = 100/nelex;
hy = 30/neley;

[X, Y] = meshgrid (x, y);

Z = 5 - 0.05 * X;
Z(Z < 0) = 0;


##h = 0*Z;
Ymin = 0; Ymax = 30; Xmin = 0; Xmax = 100;
##select = (Y>=Ymin & Y<=Ymax & X >=Xmin & X <= Xmax);
####
##h(select)= 5 - Z(select);
##h(h < 0) = 0;
####
##surf (X, Y, Z)
##hold all
##surf (X, Y, Z+h)
##axis equal


DX = .5; DY = .5;
[xp, yp] = meshgrid (Xmin:DX:Xmax, Ymin:DY:Ymax);

%hp = interp2 (X, Y, h, xp, yp, 'spline');
hp = 5.*ones(size(xp));
hp(xp > 50) = 0;
hp = hp(:);
xp = xp(:);
yp = yp(:);

hzero = find (hp <= 0.001);
hp(hzero) = [];
xp(hzero) = [];
yp(hzero) = [];

immagine_out = uint16(Z);
%imwrite (immagine_out, "sfondo.tif")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



Zz        = double (Z);
[gx, gy]  = gradient (Zz);

Z    = Zz(:);
dZdx = gx(:);
dZdy = gy(:);


figure()
mesh(X,Y,Zz)
axis equal
hold on
scatter3(xp(:),yp(:),hp(:))


%% Constants
g     = 9.81;
xi    = 200;
vis   = 50;
ty    = 2000;
T     = 10;

%% Material point quantities initialization
nmp   = numel(xp);
rhosy = 1000.0;
Msys  = sum (hp*DX*DY*rhosy);
Mp    = Msys/nmp * ones(nmp, 1);
Vp    = Mp./rhosy;
Ap    = Vp./hp;
vp    = zeros (nmp,2);
BINGHAM = 0.0;
FRICTION = 0.0;
CFL = 0.001;
BC_FLAG = 1.0;
momp  = zeros (nmp,2);

Fb(:,1) = zeros (nmp,1);
Fb(:,2) = zeros (nmp,1);

%%
DATA = struct (
	   "x", xp, ...
	   "y", yp, ...
	   "Mp", Mp, ...
	   "Ap", Ap, ...
	   "vpx", vp(:,1), ...
	   "vpy", vp(:,2), ...
	   "Nex", nelex, ...
	   "Ney", neley, ...
	   "hx", hx, ...
	   "hy", hy, ...
	   "hp", hp, ...
	   "mom_px", momp(:,1), ...
	   "mom_py", momp(:,2), ...
	   "g", g, ...
	   "T", T, ...
	   "xi", xi, ...
	   "vis", vis, ...
	   "ty", ty, ...
	   "rho", rhosy, ...
	   "Vp", Vp, ...
	   "Z", Z, ...
	   "dZdx", dZdx,
	   "dZdy", dZdy,
     "BINGHAM_ON", BINGHAM,
     "FRICTION_ON", FRICTION,
     "CFL", CFL,
     "BC_FLAG",BC_FLAG
	 );
% --- Scrittura manuale del DATA.json ---
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
v2j(FID, "x",          xp,         false);
v2j(FID, "y",          yp,         false);
v2j(FID, "Mp",         Mp,         false);
v2j(FID, "Ap",         Ap,         false);
v2j(FID, "vpx",        vp(:,1),    false);
v2j(FID, "vpy",        vp(:,2),    false);
v2j(FID, "Nex",        nelex,      false);
v2j(FID, "Ney",        neley,      false);
v2j(FID, "hx",         hx,         false);
v2j(FID, "hy",         hy,         false);
v2j(FID, "hp",         hp,         false);
v2j(FID, "mom_px",     momp(:,1),  false);
v2j(FID, "mom_py",     momp(:,2),  false);
v2j(FID, "g",          g,          false);
v2j(FID, "T",          T,          false);
v2j(FID, "xi",         xi,         false);
v2j(FID, "rho",        rhosy,      false);
v2j(FID, "Vp",         Vp,         false);
v2j(FID, "Z",          Z,          false);
v2j(FID, "dZdx",       dZdx,       false);
v2j(FID, "dZdy",       dZdy,       false);
v2j(FID, "BINGHAM_ON", BINGHAM,    false);
v2j(FID, "FRICTION_ON",FRICTION,   false);
v2j(FID, "CFL",        CFL,        false);
v2j(FID, "BC_FLAG",    BC_FLAG,    true);
fprintf(FID, "}\n");
fclose(FID);
disp("DATA.json scritto correttamente");









