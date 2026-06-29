
clear all;
close all;
clc;

load('mask_in_vladi.mat')
load('dem_real.mat')
% load('mask_out.tif');

PP = zeros(676,676);
pm = zeros(175,175);

for i = 1:175
    pm(176-i,i)=1;
end

mask_in = pm*mask_in;

for i = 1:676


    PP(677-i,i) = 1;

end

dem = PP*dem;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rowstart = 1;
rowend   = 676;

colstart =1;
colend   = 756;

ndivrows = rowend - rowstart;
ndivcols = colend - colstart;

Xmin = 180;
Xmax = 520;
Ymin = 180;
Ymax = 520;

hx =5;
hy = 5;

y = linspace (0, hy*ndivrows, ndivrows+1);
x = linspace (0, hx*ndivcols, ndivcols+1);

%y = linspace (0, hy*(Ymax-Ymin), Ymax-Ymin );
%x = linspace (0, hx*(Xmax-Xmin), Xmax - Xmin );

[X, Y] = meshgrid (x, y);
%Z = double (immagine_int);
Z = dem; %double(dem(Xmin:Xmax,Ymin:Ymax));

zplot = dem; % double( dem(Xmin:Xmax,Ymin:Ymax));



%h = 0*Z; %ymin 1400 ymax=1700   xmin = 700 xmax = 900
%{
%Ymin = 1350; Ymax = 1450; Xmin = 700; Xmax = 900;
select = (Y>Ymin & Y<Ymax & X >Xmin & X < Xmax);

h(select)= 38.*double(mask_in(select)) ; %(700.*double(mask_in(select))) - Z(select);
h(h < 0) = 0;
%}


%h = 38.*double(mask_in);

%h(h < 0) = 0;
% DX = 1; DY = 1;
% %[xp, yp] = meshgrid (Xmin:DX:Xmax, Ymin:DY:Ymax);
% %hp = interp2 (X, Y, h, xp, yp, 'spline');
% %select = (Y>Ymin & Y<Ymax & X >Xmin & X < Xmax);
%
%
%
%
%
% hp = 38.*double(mask_in);
%
% %yy = linspace (0, hy*ndivrows, ndivrows+1);
% %xx = linspace (0, hx*ndivcols, ndivcols+1);
%
% yy = linspace (0, hy*size(hp,1), size(hp,1));
% xx = linspace (0, hx*size(hp,2), size(hp,2));
%
% [Xx, Yy] = meshgrid (xx, yy);
% xp = Xx;
% yp = Yy;
%
% hp = hp(:);
% xp = xp(:);
% yp = yp(:);
%
% hzero = find (hp < 38);
% hp(hzero) = [];
% xp(hzero) = [];
% yp(hzero) = [];

%data = jsonencode (struct ('X', X(:), 'Y', Y(:), 'Z', Z(:), 'h', h(:)));
%immagine_out = uint16(immagine_int);
%imwrite (immagine_out, "sfondo.tif")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
AA = mask_in; %double(mask_in);
AA = AA(:,:,1);

CC = AA > 0.5;
[boundary_y, boundary_x] = find(CC);

% Tracing del contorno manuale: trova i pixel di bordo
% (pixel accesi con almeno un vicino spento)
[nr, nc_img] = size(CC);
is_border = false(nr, nc_img);
for r = 1:nr
  for c = 1:nc_img
    if CC(r,c)
      if r==1 || r==nr || c==1 || c==nc_img || ...
         ~CC(r-1,c) || ~CC(r+1,c) || ~CC(r,c-1) || ~CC(r,c+1)
        is_border(r,c) = true;
      end
    end
  end
end
[br, bc] = find(is_border);
boundary = [br, bc];

%figure()
%plot(boundary(:,2), boundary(:,1), 'rx-', 'LineWidth', 1.0)

% xL      = 0+(0.5*hl/nl):hl/nl:Lx;
% yL      = 0+(0.5*hl/nl):hl/nl: Ly-(0.5*hl/nl);
NPX = 3650;
NPY = NPX;
yy = linspace (0, 5*size(mask_in,1), NPX);
xx = linspace (0, 5*size(mask_in,2), NPY);
[xp,yp] = meshgrid(xx,yy);
xp = xp(:);
yp = yp(:);
[in,on] = inpolygon(xp(:),yp(:),boundary(:,2),boundary(:,1));
nbound = numel(boundary(:,2));
xp = 5.*[  xp(in)];
yp = 5.*[ yp(in)];
hp = 38.*ones(numel(xp),1);
DX = 5*size(mask_in,1)/NPX;
DY = 5*size(mask_in,2)/NPY;
xp = xp + 1530;
yp = yp + 1141;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Zz        = double (Z);
Zz(Zz<=0) = -1.0;
[gx, gy]  = gradient (Zz);

Z    = Zz(:);
dZdx = gx(:);
dZdy = gy(:);


%% Constants
g     = 9.81;
xi    = 200;
vis   = 2000;
ty    = 50;
T     = 20;

%% Material point quantities initialization
nmp   = numel(xp);
rhosy = 1291.0;
Msys  = sum (hp*DX*DY*rhosy);
Ap = DX*DY*ones(nmp,1); 
Vp = Ap.*hp;  
Mp = rhosy*Vp;
mu = 2000.; phi = 34.; tauy = 50.; 
eq_level = 0.0;
MERGE_SPLIT_ON = 1;
momp  = zeros (nmp,2);
vp    = zeros (nmp,2);

Fb(:,1) = zeros (nmp,1);
Fb(:,2) = zeros (nmp,1);

BINGHAM = 1.0;
FRICTION = 1.0;
CFL = 0.01;
BC_FLAG = 1.0;

DT_FIXED = 0;   
NSTEPS   = 100;

%%
DATA = struct( "x", xp, ...
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
	   "dZdx", dZdx,...
	   "dZdy", dZdy,...
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









