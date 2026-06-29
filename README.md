# Particles Merge-Split for & GPU Parallelization

GPU parallelization (via OpenMP target offload) of a depth-averaged Material
Point Method (MPM) solver for landslide and mudflow simulation. The solver
implements the semi-conservative, well-balanced depth-averaged MPM of the
reference model below, with Bingham + Voellmy rheology, exact mass/momentum
conservation, and adaptive particle refinement (merge/split). The project ports
the CPU solver to the GPU, validates the physics against the reference
benchmarks, and analyses the CPU-vs-GPU performance scaling.

The work is split into two main parts:
the **GPU offloading** of the solver together with the **validation and
performance** study, and the **adaptive merge/split**
(particle refinement), which keeps the particle distribution regular as the
material deforms. Both parts are complete and functional, and are toggled and
configured through the input file (see §5 and §9).

---

## 1. Repository structure

```
AMSC/
├── main/                 # solver sources + Makefile; executables are built here
│   ├── optim_par.cpp     # CPU solver (parallel STL)
│   ├── gpu_offload.cpp   # GPU solver (OpenMP target offload)
│   ├── mpm_data.h        # DATA struct: reads DATA.json
│   └── merge_split_ops.h # merge/split operators
├── quadgrid/             # background grid (git submodule)
└── data/                 # one folder per test, each containing its DATA.json
    ├── well_balance/     # well-balancing test 
    ├── conservation_test/     # full dynamic conservation test
    ├── ALTRO TEST FORSE    
```

The grid library `quadgrid` is included as a **git submodule** and must be
initialized after cloning (see below). Executables are produced in `main/` and
launched from each test folder via a relative path (see §6).

---

## 2. Requirements

- **NVIDIA HPC SDK (NVHPC) 26.1** — provides the `nvc++` compiler with OpenMP
  target offload.
- **NVIDIA GPU** — tested on an **NVIDIA L4** (Ada Lovelace, compute capability
  `cc89` / `sm_89`), 24 GB VRAM.
- **Apptainer** — the toolchain runs inside an Apptainer container
  (NVHPC 26.1 image).
- **GNU Octave** — to generate the `DATA.json` input files from the `.m` scripts.
- **Python 3** with `numpy` for the
  validation and performance analysis scripts.

---

## 3. Getting the code

```bash
git clone https://github.com/lucadiprofio/AMSC.git
cd AMSC
git submodule update --init --recursive   # fetch the quadgrid submodule
```

---

## 4. Building

Builds run **inside the Apptainer container**, on a GPU node of the cluster.
From a GPU node, open the container (the `--nv` flag exposes the GPU; `--bind`
makes the work and home directories visible inside):

```bash
apptainer shell --nv --bind /work,/home \
    /software/containers/nvhpc/nvhpc_26.1-devel-cuda_multi-ubuntu24.04.sif
```

Then, from the repository root, build with `make`:

```bash
make
```

This produces both executables (CPU and GPU) in the `build/` folder. The build
uses the `nvc++` compiler from NVHPC 26.1; the GPU target is compiled with OpenMP
offload for compute capability `cc89` (NVIDIA L4). The exact flags are defined in
the `Makefile`.

---

## 5. Generating the input (`DATA.json`)

Each test case has an Octave generator script that writes a `DATA.json` file
(particle positions, masses, topography, physical parameters, run options).
The solver reads `DATA.json` from its working directory.

```bash
octave «script name».m      # writes DATA.json
```

### Key fields in `DATA.json`

| Field | Meaning |
|---|---|
| `Nex`, `Ney` | number of grid **cells** in x, y |
| `hx`, `hy` | cell size |
| `x`, `y`, `hp`, `Mp`, `Ap`, `Vp` | per-particle position, depth, mass, area, volume |
| `Z`, `dZdx`, `dZdy` | nodal bathymetry and its gradients |
| `mu`, `tauy`, `phi`, `xi`, `rho` | rheology (Bingham + Voellmy) parameters |
| `BINGHAM_ON`, `FRICTION_ON` | toggle rheology / friction (1 = on) |
| `eq_level` | still-water level (`> 0` only for well-balancing; `0` otherwise) |
| `BC_FLAG` | boundary conditions (1 = enabled) |
| `CFL` | CFL number for the adaptive time step |
| `DT_FIXED` | if `> 0`, use a fixed time step (benchmark mode); if `0`, adaptive |
| `NSTEPS` | number of steps in fixed-`dt` mode |
| `MERGE_SPLIT_ON` | adaptive merge/split toggle |

---

## 6. Running a simulation

Each test lives in its own folder containing a `DATA.json`. From inside that
folder, launch the executable in `main/` via its relative path and redirect the
output to a log file:

```bash
../../main/optim_par   > result_cpu.txt 2>&1    # CPU
../../main/gpu_offload > result_gpu.txt 2>&1    # GPU
```

(Adjust the relative path `../../main/` to the depth of your test folder, and the
executable names to those produced by `make`.)

### Output files

- `nc_particles_<step>.csv` — per-particle fields (written every 10 steps when
  output is enabled).
- `GRID_forZ.vts` — nodal fields, for ParaView.

> ParaView tip: plot the particles at elevation `hpZ` ( = `hp + Zp`) so they sit
> on the terrain surface; plotting at a constant height makes them appear to
> "float".
