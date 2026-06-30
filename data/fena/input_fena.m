
clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rowstart = 550;
rowend   = 900;

colstart = 171;
colend   = 430;

ndivrows = rowend - rowstart;
ndivcols = colend - colstart;

hx = hy = 5;
immagine_int = load("dtm.txt");
immagine_int = immagine_int(rowstart:rowend, colstart:colend);
immagine_int = immagine_int(end:-1:1, :);
immagine_int(immagine_int < 0) = -1;
disp(size(immagine_int));                          % deve essere 351 x 260
disp([min(immagine_int(:)) max(immagine_int(:))]); % range quote plausibile

y = linspace (0, hy*ndivrows, ndivrows+1);
x = linspace (0, hx*ndivcols, ndivcols+1);

[X, Y] = meshgrid (x, y);
Z = double (immagine_int);

h = 0*Z; %ymin 1400 ymax=1700   xmin = 700 xmax = 900
Ymin = 1350; Ymax = 1450; Xmin = 700; Xmax = 900;
select = (Y>Ymin & Y<Ymax & X >Xmin & X < Xmax);

h(select)= 101 - Z(select);
h(h < 0) = 0;

DX = 0.851; DY = 0.851;
[xp, yp] = meshgrid (Xmin:DX:Xmax, Ymin:DY:Ymax);
hp = interp2 (X, Y, h, xp, yp, 'spline');
hp = hp(:);
xp = xp(:);
yp = yp(:);

hzero = find (hp <= 1);
hp(hzero) = [];
xp(hzero) = [];
yp(hzero) = [];

%data = jsonencode (struct ('X', X(:), 'Y', Y(:), 'Z', Z(:), 'h', h(:)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



Zz        = double (immagine_int);
Zz(Zz<=0) = -1.0;
[gx, gy]  = gradient (Zz);

Z    = Zz(:);
dZdx = gx(:);
dZdy = gy(:);


figure()
mesh(X,Y,Zz)

figure()
scatter3(xp(:),yp(:),hp(:))


%% Constants
g     = 9.81;
xi    = 200;
vis   = 50;
ty    = 2000;
T     = 10;

%% Material point quantities initialization
nmp   = numel(xp);
rhosy = 1291.0;
Msys  = sum (hp*DX*DY*rhosy);
Mp    = Msys/nmp * ones(nmp, 1);
Vp    = Mp./rhosy;
Ap    = Vp./hp;
vp    = zeros (nmp,2);
BINGHAM = 0.0;
FRICTION = 0.0;
CFL = 0.01;
MERGE_SPLIT_ON = 1;
eq_level = 0.0;   % non well-balanced
momp  = zeros (nmp,2);
DT_FIXED = 0;   
NSTEPS   = 100;
mu = 50.; phi = 34.; tauy = 2000.; 
Fb(:,1) = zeros (nmp,1);
Fb(:,2) = zeros (nmp,1);
BC_FLAG = 1.0;

%%
DATA = struct (
	   "x", xp, ...
	   "y", yp, ...
	   "Mp", Mp, ...
	   "Ap", Ap, ...
	   "vpx", vp(:,1), ...
	   "vpy", vp(:,2), ...
	   "Nex", ndivcols, ...
	   "Ney", ndivrows, ...
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
	   "mu", mu, ...
		 "phi", phi, ...
		 "tauy", tauy, ...
      "DT_FIXED", DT_FIXED, ...
    "NSTEPS", NSTEPS, ...
		 "MERGE_SPLIT_ON", MERGE_SPLIT_ON, ...
     "BINGHAM_ON", BINGHAM, ...
     "FRICTION_ON", FRICTION, ...
     "CFL", CFL, ...
	 "BC_FLAG",BC_FLAG, ...
	 "eq_level", eq_level
	 );

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
v2j(FID, "Nex",         ndivcols,   false);
v2j(FID, "Ney",         ndivrows,   false);
v2j(FID, "hx",          hx,         false);
v2j(FID, "hy",          hy,         false);
v2j(FID, "hp",          hp,         false);
v2j(FID, "mom_px",      momp(:,1),  false);
v2j(FID, "mom_py",      momp(:,2),  false);
v2j(FID, "g",           g,          false);
v2j(FID, "T",           T,          false);
v2j(FID, "xi",          xi,         false);
v2j(FID, "vis",         vis,        false);
v2j(FID, "ty",          ty,         false);
v2j(FID, "rho",         rhosy,      false);
v2j(FID, "Vp",          Vp,         false);
v2j(FID, "Z",           Z,          false);
v2j(FID, "dZdx",        dZdx,       false);
v2j(FID, "dZdy",        dZdy,       false);
v2j(FID, "mu",         mu,         false);
v2j(FID, "phi",        phi,        false);
v2j(FID, "tauy",       tauy,       false);
v2j(FID, "eq_level",eq_level,           false); 
v2j(FID, "MERGE_SPLIT_ON", MERGE_SPLIT_ON, false);
v2j(FID, "DT_FIXED",  DT_FIXED,    false);
v2j(FID, "NSTEPS", NSTEPS,   false);
v2j(FID, "BINGHAM_ON",  BINGHAM,    false);
v2j(FID, "FRICTION_ON", FRICTION,   false);
v2j(FID, "CFL",         CFL,        false);
v2j(FID, "BC_FLAG",     BC_FLAG,    true);
fprintf(FID, "}\n");
fclose(FID);
disp("DATA.json scritto correttamente");









