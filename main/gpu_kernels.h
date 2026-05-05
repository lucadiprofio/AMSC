#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H
 
//
// Device-compatible grid functions for OpenMP target offload.
// These replicate the static methods of quadgrid_t so they can
// be called inside #pragma omp target regions.
//
// Usage: include this header in gpu_offload.cpp and use these
// functions inside target kernels instead of the class methods.
//
 
#pragma omp declare target
 
inline int gpu_gind2col(int idx, int numrows) {
    return idx / numrows;
}
 
inline int gpu_gind2row(int idx, int numrows) {
    return idx % numrows;
}
 
inline int gpu_gt(int inode, int cidx, int ridx, int numrows) {
    int bottom_left = ridx + cidx * (numrows + 1);
    switch (inode) {
    case 0: return bottom_left;
    case 1: return bottom_left + 1;
    case 2: return bottom_left + (numrows + 1);
    case 3: return bottom_left + (numrows + 2);
    default: return -1;
    }
}
 
// Returns the coordinate of the bottom-left corner of cell (colidx, rowidx)
// idir=0 -> x coordinate, idir=1 -> y coordinate
// inode selects the corner (only matters for deciding +hx or +hy offset)
inline double gpu_p(int idir, int inode, int colidx, int rowidx,
                    double hx, double hy) {
    double val = 0.0;
    if (idir == 0) {
        val = colidx * hx;
        if (inode > 1) val += hx;
    } else {
        val = rowidx * hy;
        if (inode == 1 || inode == 3) val += hy;
    }
    return val;
}
 
// Bilinear shape function N_inode evaluated at (x,y)
// for cell at column c, row r, with cell size hx x hy.
//
//   Node numbering:
//   1 --- 3
//   |     |
//   0 --- 2
//
inline double gpu_shp(double x, double y, int inode,
                      int c, int r, double hx, double hy) {
    double x0 = c * hx;       // = gpu_p(0, 0, c, r, hx, hy)
    double y0 = r * hy;       // = gpu_p(1, 0, c, r, hx, hy)
    double xi  = (x - x0) / hx;
    double eta = (y - y0) / hy;
 
    switch (inode) {
    case 0: return (1.0 - xi) * (1.0 - eta);
    case 1: return (1.0 - xi) * eta;
    case 2: return xi * (1.0 - eta);
    case 3: return xi * eta;
    default: return 0.0;
    }
}
 
// Gradient of bilinear shape function.
// idir=0 -> dN/dx, idir=1 -> dN/dy
inline double gpu_shg(double x, double y, int idir, int inode,
                      int c, int r, double hx, double hy) {
    double x0 = c * hx;
    double y0 = r * hy;
    double xi  = (x - x0) / hx;
    double eta = (y - y0) / hy;
 
    if (idir == 0) {
        switch (inode) {
        case 0: return -(1.0 - eta) / hx;
        case 1: return -eta / hx;
        case 2: return  (1.0 - eta) / hx;
        case 3: return  eta / hx;
        default: return 0.0;
        }
    } else {
        switch (inode) {
        case 0: return -(1.0 - xi) / hy;
        case 1: return  (1.0 - xi) / hy;
        case 2: return -xi / hy;
        case 3: return  xi / hy;
        default: return 0.0;
        }
    }
}
 
#pragma omp end declare target
 
#endif // GPU_KERNELS_H