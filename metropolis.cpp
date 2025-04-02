#include "include/klft.hpp"
#include <iostream>

using real_t = double;

int main(int argc, char **argv) {
  std::string gauge_group = "SU2";
  int ndim = 4;
  size_t LX = 8;
  size_t LY = 8;
  size_t LZ = 8;
  size_t LT = 16;
  size_t n_hit = 100;
  real_t beta = 2.0;
  real_t delta = 0.05;
  size_t seed = 1234;
  size_t n_sweep = 1000;
  size_t n_meas = 100;
  bool cold_start = true;
  std::string outfilename = "";
  bool open_bc_x = false;
  bool open_bc_y = false;
  bool open_bc_z = false;
  bool open_bc_t = false;
  int x0 = 0;
  int y0 = 0;
  int z0 = 0;
  int max_R_Wilson_loop = 0;
  bool verbose = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--gauge-group") {
      gauge_group = argv[i + 1];
    }
    if (std::string(argv[i]) == "--ndim") {
      ndim = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--LX") {
      LX = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--LY") {
      LY = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--LZ") {
      LZ = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--LT") {
      LT = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--n-hit") {
      n_hit = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--beta") {
      beta = std::stod(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--delta") {
      delta = std::stod(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--seed") {
      seed = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--n-sweep") {
      n_sweep = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--n-meas") {
      n_meas = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--cold-start") {
      cold_start = std::string(argv[i + 1]) == "true";
    }
    if (std::string(argv[i]) == "--outfilename") {
      outfilename = argv[i + 1];
    }
    if (std::string(argv[i]) == "--open-bc-x") {
      open_bc_x = std::string(argv[i + 1]) == "true";
    }
    if (std::string(argv[i]) == "--open-bc-y") {
      open_bc_y = std::string(argv[i + 1]) == "true";
    }
    if (std::string(argv[i]) == "--open-bc-z") {
      open_bc_z = std::string(argv[i + 1]) == "true";
    }
    if (std::string(argv[i]) == "--open-bc-t") {
      open_bc_t = std::string(argv[i + 1]) == "true";
    }
    if (std::string(argv[i]) == "--x0") {
      x0 = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--y0") {
      y0 = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--z0") {
      z0 = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--max_R_Wilson_loop") {
      max_R_Wilson_loop = std::stoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--verbose") {
      verbose = std::string(argv[i + 1]) == "true";
    }
    if (std::string(argv[i]) == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "--gauge-group SU3, SU2 or U1" << std::endl;
      std::cout << "--ndim 2, 3, or 4" << std::endl;
      std::cout << "--LX lattice size in x direction" << std::endl;
      std::cout << "--LY lattice size in y direction" << std::endl;
      std::cout << "--LZ lattice size in z direction" << std::endl;
      std::cout << "--LT lattice size in t direction" << std::endl;
      std::cout << "--n-hit number of hits per sweep" << std::endl;
      std::cout << "--beta inverse coupling constant" << std::endl;
      std::cout << "--delta step size" << std::endl;
      std::cout << "--seed random number generator seed" << std::endl;
      std::cout << "--n-sweep number of sweeps" << std::endl;
      std::cout << "--n-meas number of measurements" << std::endl;
      std::cout << "--cold-start true or false" << std::endl;
      std::cout << "--outfilename output filename" << std::endl;
      std::cout << "--open-bc-x true or false" << std::endl;
      std::cout << "--open-bc-y true or false" << std::endl;
      std::cout << "--open-bc-z true or false" << std::endl;
      std::cout << "--(x0, y0, z0) starting point for OBC Wloop" << std::endl;
      std::cout << "--max_R_Wilson_loop maximum R for Wilson loop" << std::endl;
      std::cout << "--verbose true or false" << std::endl;

      return 0;
    }
  }
  bool open_bc[4] = {open_bc_x, open_bc_y, open_bc_z, open_bc_t};
  int v0[4] = {x0, y0, z0, t0};
  // if(gauge_group == "SU2" && ndim == 4)
  // klft::Metropolis_SU2_4D<real_t>(LX,LY,LZ,LT,n_hit,beta,delta,seed,n_sweep,cold_start,outfilename,open_bc);
  // if(gauge_group == "SU2" && ndim == 3)
  // klft::Metropolis_SU2_3D<real_t>(LX,LY,LT,n_hit,beta,delta,seed,n_sweep,cold_start,outfilename,open_bc);
  // if(gauge_group == "SU2" && ndim == 2)
  // klft::Metropolis_SU2_2D<real_t>(LX,LT,n_hit,beta,delta,seed,n_sweep,cold_start,outfilename,open_bc);
  if (gauge_group == "U1" && ndim == 4)
    klft::Metropolis_U1_4D<real_t>(LX, LY, LZ, LT, n_hit, beta, delta, seed,
                                   n_sweep, cold_start, outfilename, open_bc);
  if (gauge_group == "U1" && ndim == 3)
    klft::Metropolis_U1_3D<real_t>(LX, LY, LT, n_hit, beta, delta, seed,
                                   n_sweep, n_meas, cold_start, outfilename,
                                   open_bc, v0, max_R_Wilson_loop, verbose);
  if (gauge_group == "U1" && ndim == 2)
    klft::Metropolis_U1_2D<real_t>(LX, LT, n_hit, beta, delta, seed, n_sweep,
                                   cold_start, outfilename, open_bc);
  // if(gauge_group == "SU3" && ndim == 4)
  // klft::Metropolis_SU3_4D<real_t>(LX,LY,LZ,LT,n_hit,beta,delta,seed,n_sweep,cold_start,outfilename,open_bc);
  // if(gauge_group == "SU3" && ndim == 3)
  // klft::Metropolis_SU3_3D<real_t>(LX,LY,LT,n_hit,beta,delta,seed,n_sweep,cold_start,outfilename,open_bc);
  // if(gauge_group == "SU3" && ndim == 2)
  // klft::Metropolis_SU3_2D<real_t>(LX,LT,n_hit,beta,delta,seed,n_sweep,cold_start,outfilename,open_bc);
  return 0;
}