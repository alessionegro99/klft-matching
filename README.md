# klft
A library for lattice field theory simulation accelerated using Kokkos

## Installation
```bash
cmake -DKokkos_ROOT=/path/to/kokkos/installation [-DCMAKE_CXX_COMPILER=/path/to/kokkos/nvcc/wrapper] /path/to/klft
make
```

## Usage
```bash
Usage: ./metropolis [options]
Options:
--gauge-group U1
--ndim 2, 3, or 4
--LX lattice size in x direction
--LY lattice size in y direction
--LZ lattice size in z direction
--LT lattice size in t direction
--n-hit number of hits per sweep
--beta inverse coupling constant
--delta step size
--seed random number generator seed
--n-sweep number of sweeps
--n-meas measure every ... sweeps
--cold-start true or false
--outfilename output filename
--open-bc-x true or false
--open-bc-y true or false
--open-bc-z true or false
--x0 X starting point for OBC observables
--y0 Y ...
--z0 Z ...
--Wt Wilson loop W(Wt,Ws)
--Ws Wilson loop W(Wt,Ws)
--verbose verbose output true or false
```