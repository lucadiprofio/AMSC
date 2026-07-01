# dampm (with merge-split and gpu offloading)

Porting on GPU (via OpenMP target offload) of a depth-averaged Material
Point Method (MPM) solver for landslide and mudflow simulation. The solver
implements the semi-conservative, well-balanced depth-averaged MPM of the
reference model below, with Bingham + Voellmy rheology, exact mass/momentum
conservation, and adaptive particle refinement (merge/split).

Also, the **adaptive merge/split**
(particle refinement) is implemented, which keeps the particle distribution regular as the
material deforms.

---

## 1. Repository structure

```
AMSC/
├── main/
│   ├── optim_par.cpp        # CPU solver (parallel STL)
│   ├── gpu_offload.cpp      # GPU solver (OpenMP target offload)
│   ├── gpu_kernels.h/.cpp   # GPU kernels used by gpu_offload
│   ├── mpm_data.h           # DATA struct: reads DATA.json
│   └── merge_split_ops_cmes.h # merge/split implementation
├── quadgrid/                # mpm library (submodule)
├── timer/                   # timing utility (submodule)
├── json/                    # nlohmann/json (submodule)
└── data/                    # one folder per test
    ├── conservation_test/   # mass and linear momentum conservation
    ├── cortenova/
    ├── dambreak/
    ├── fena/
    ├── hongkong/
    ├── performance/
    ├── tubo/
    ├── well_balance/        # well-balancing test
    └── well_balanced_small_hp/
```

`quadgrid`, `timer`, and `json` are included as **git submodules** and must be
initialized after cloning (see below). Executables are produced in `build/` and
launched from each test folder via a relative path (see §6).

---

## 2. Requirements

- **NVIDIA HPC SDK (NVHPC) 26.1** — provides the `nvc++` compiler with OpenMP
  target offload. Verified via the container: `nvc++ 26.1-0`.
  > ⚠️ On the cluster, use the **fully versioned** image
  > `nvhpc_26.1-devel-cuda_multi-ubuntu24.04.sif`. A generic `nvhpc.sif` may
  > point to a different, older image (e.g. `nvc++ 25.9-0` was found under
  > that name) — always check `nvc++ --version` inside the container before
  > building.
- **NVIDIA GPU** — tested on an **NVIDIA L4** (Ada Lovelace, compute capability
  `cc89` / `sm_89`), 24 GB VRAM (verified via `nvidia-smi`, driver 580.159.03,
  CUDA 13.0).
- **Apptainer** (tested with **1.4.5**) — the toolchain runs inside an
  Apptainer container (NVHPC 26.1 image, Ubuntu 24.04).
- **GNU Octave** (tested with **8.4.0**) — to generate the `DATA.json` input
  files from the `.m` scripts.

## 3. Getting the code

```bash
git clone git@github.com:lucadiprofio/AMSC.git
cd AMSC
git submodule update --init --recursive   # fetch quadgrid, timer, json submodules
```
or alternatively
```bash
git clone --recurse-submodules git@github.com:lucadiprofio/AMSC.git
```

---

## 4. Building

Builds run **inside the Apptainer container**, on a node of the cluster via

```bash
apptainer shell --nv /software/containers/nvhpc/nvhpc.sif
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
octave <script name.m>      # generate DATA.json
```

### Key fields in `DATA.json`

| Field | Meaning |
|---|---|
| `Nex`, `Ney` | number of grid **cells** in x, y |
| `hx`, `hy` | cell size |
| `x`, `y`, `hp`, `Mp`, `Ap`, `Vp` | per-particle position, depth, mass, area, volume |
| `Z`, `dZdx`, `dZdy` | nodal bathymetry and its gradients |
| `mu`, `tauy`, `phi`, `xi`, `rho` | rheology (Bingham + Voellmy) parameters |
| `BINGHAM_ON`, `FRICTION_ON` | toggle rheology / friction |
| `eq_level` | still-water level (`> 0` only for well-balancing; `0` otherwise) |
| `BC_FLAG` | boundary conditions (1 = enabled) |
| `CFL` | CFL number for the adaptive time step |
| `DT_FIXED` | if `> 0`, use a fixed time step (benchmark mode); if `0`, adaptive |
| `NSTEPS` | number of steps in fixed-`dt` mode |
| `MERGE_SPLIT_ON` | adaptive merge/split toggle |
| `ms_*` | tuning parameters for the merge/split algorithm (thresholds, refinement levels, call interval, etc.) |

---

## 6. Running a simulation

Each test lives in its own folder containing a `DATA.json`. From inside that
folder, launch the executable in `build/` via its relative path and redirect the
output to a log file:

```bash
../../build/optim_par   > result_cpu.txt 2>&1    # CPU
../../build/gpu_offload > result_gpu.txt 2>&1    # GPU
```

(Adjust the relative path `../../build/` to the depth of your test folder, and the
executable names to those produced by `make`)

### Output files

- `nc_particles_<step>.csv` — per-particle fields (written every 10 steps when
  output is enabled).
- `GRID_forZ.vts` — nodal fields on the background grid (bathymetry and
  derived grid quantities), for visualization in ParaView.

> ParaView tip: plot the particles at elevation `hpZ` ( = `hp + Zp`) so they sit
> on the terrain surface; plotting at a constant height makes them appear to
> "float".