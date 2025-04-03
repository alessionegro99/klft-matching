#include "../include/Metropolis.hpp"
#include "../include/klft.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>

namespace klft {

template <typename T>
void Metropolis_U1_4D(const size_t &LX, const size_t &LY, const size_t &LZ,
                      const size_t &LT, const size_t &n_hit, const T &beta,
                      const T &delta, const size_t &seed, const size_t &n_sweep,
                      const bool cold_start, const std::string &outfilename,
                      const bool open_bc[3]) {
  std::cout << "Running Metropolis_U1_4D" << std::endl;
  std::cout << "Gauge Field Dimensions:" << std::endl;
  std::cout << "LX = " << LX << std::endl;
  std::cout << "LY = " << LY << std::endl;
  std::cout << "LZ = " << LZ << std::endl;
  std::cout << "LT = " << LT << std::endl;
  std::cout << "Metropolis Parameters:" << std::endl;
  std::cout << "beta = " << beta << std::endl;
  std::cout << "delta = " << delta << std::endl;
  std::cout << "n_hit = " << n_hit << std::endl;
  std::cout << "n_sweep = " << n_sweep << std::endl;
  std::cout << "seed = " << seed << std::endl;
  std::cout << "start condition = " << (cold_start ? "cold" : "hot")
            << std::endl;
  std::cout << "output file = " << outfilename << std::endl;
  std::ofstream outfile;
  if (outfilename != "") {
    outfile.open(outfilename);
    outfile << "step, plaquette, acceptance_rate, time" << std::endl;
  }
  Kokkos::initialize();
  {
    using Group = U1<T>;
    using GaugeFieldType = GaugeField<T, Group, 4, 1>;
    using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
    RNG rng = RNG(seed);
    GaugeFieldType gauge_field = GaugeFieldType(LX, LY, LZ, LT);
    Metropolis<T, Group, GaugeFieldType, RNG> metropolis =
        Metropolis<T, Group, GaugeFieldType, RNG>(gauge_field, rng, n_hit, beta,
                                                  delta);
    metropolis.initGauge(cold_start);
    if (open_bc[0])
      gauge_field.set_open_bc_x();
    if (open_bc[1])
      gauge_field.set_open_bc_y();
    if (open_bc[2])
      gauge_field.set_open_bc_z();
    std::cout << "Starting Plaquette: " << gauge_field.get_plaquette()
              << std::endl;
    std::cout << "Starting Metropolis: " << std::endl;
    auto metropolis_start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n_sweep; i++) {
      auto start_time = std::chrono::high_resolution_clock::now();
      T acceptance_rate = metropolis.sweep();
      T plaquette = gauge_field.get_plaquette();
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> sweep_time = end_time - start_time;
      std::cout << "Step: " << i << " Plaquette: " << plaquette
                << " Acceptance Rate: " << acceptance_rate
                << " Time: " << sweep_time.count() << std::endl;
      if (outfilename != "") {
        outfile << i << ", " << plaquette << ", " << acceptance_rate << ", "
                << sweep_time.count() << std::endl;
      }
    }
    auto metropolis_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> metropolis_time =
        metropolis_end_time - metropolis_start_time;
    std::cout << "Metropolis Time: " << metropolis_time.count() << std::endl;
  }
  Kokkos::finalize();
  outfile.close();
}

template <typename T>
void Metropolis_U1_3D(const size_t &LX, const size_t &LY, const size_t &LT,
                      const size_t &n_hit, const T &beta, const T &delta,
                      const size_t &seed, const size_t &n_sweep,
                      const size_t &n_meas, const bool cold_start,
                      const std::string &outfilename, const bool open_bc[3],
                      const bool open_bc_t, const int v0[3], const int t0,
                      const bool non_planar, const int &max_T_Wilson_loop,
                      const int &max_R_Wilson_loop, const bool &verbose) {
  std::cout << "Running Metropolis_U1_3D" << std::endl;
  std::cout << "Gauge Field Dimensions:" << std::endl;
  std::cout << "LX = " << LX << std::endl;
  std::cout << "LY = " << LY << std::endl;
  std::cout << "LT = " << LT << std::endl;
  std::cout << "Metropolis Parameters:" << std::endl;
  std::cout << "beta = " << beta << std::endl;
  std::cout << "delta = " << delta << std::endl;
  std::cout << "n_hit = " << n_hit << std::endl;
  std::cout << "n_sweep = " << n_sweep << std::endl;
  std::cout << "n_meas = " << n_meas << std::endl;
  std::cout << "seed = " << seed << std::endl;
  std::cout << "start condition = " << (cold_start ? "cold" : "hot")
            << std::endl;
  std::cout << "output file = " << outfilename << std::endl;
  std::cout << "x0 = " << v0[0] << std::endl;
  std::cout << "y0 = " << v0[1] << std::endl;
  std::cout << "z0 = " << v0[2] << std::endl;
  std::cout << "t0 = " << t0 << std::endl;
  std::cout << "open_bc_x = " << open_bc[0] << std::endl;
  std::cout << "open_bc_y = " << open_bc[1] << std::endl;
  std::cout << "open_bc_z = " << open_bc[2] << std::endl;
  std::cout << "open_bc_t = " << open_bc_t << std::endl;
  std::cout << "non_planar = " << non_planar << std::endl;
  std::cout << "max T Wilson loop = " << max_T_Wilson_loop << std::endl;
  std::cout << "max R Wilson loop = " << max_R_Wilson_loop << std::endl;
  std::cout << "verbose output = " << verbose << std::endl;
  std::ofstream outfile;
  if (outfilename != "") {
    outfile.open(outfilename); // print all the parameters to the output file
    outfile << "Running Metropolis_U1_3D" << "\n";
    outfile << "Gauge Field Dimensions:" << "\n";
    outfile << "LX = " << LX << "\n";
    outfile << "LY = " << LY << "\n";
    outfile << "LT = " << LT << "\n";
    outfile << "Metropolis Parameters:" << "\n";
    outfile << "beta = " << beta << "\n";
    outfile << "delta = " << delta << "\n";
    outfile << "n_hit = " << n_hit << "\n";
    outfile << "n_sweep = " << n_sweep << "\n";
    outfile << "n_meas = " << n_meas << "\n";
    outfile << "seed = " << seed << "\n";
    outfile << "start condition = " << (cold_start ? "cold" : "hot") << "\n";
    outfile << "output file = " << outfilename << "\n";
    outfile << "x0 = " << v0[0] << "\n";
    outfile << "y0 = " << v0[1] << "\n";
    outfile << "z0 = " << v0[2] << "\n";
    outfile << "t0 = " << t0 << "\n";
    outfile << "open_bc_x = " << open_bc[0] << "\n";
    outfile << "open_bc_y = " << open_bc[1] << "\n";
    outfile << "open_bc_z = " << open_bc[2] << "\n";
    outfile << "open_bc_t = " << open_bc_t << "\n";
    outfile << "non_planar = " << non_planar << "\n";
    outfile << "max T Wilson loop = " << max_T_Wilson_loop << "\n";
    outfile << "max R Wilson loop = " << max_R_Wilson_loop << "\n";
    outfile << "verbose output = " << verbose << "\n";

    outfile << "step plaquette acceptance_rate time "; // printing measurement
                                                       // header
    if (open_bc[0] && open_bc[1]) {
      for (int j = 1; j < std::min(LT, static_cast<size_t>(max_T_Wilson_loop));
           j++) {
        if (non_planar) {
          for (int k = 1; k < std::min(LX, LY); k++) {
            for (int l = 0; l <= k; l++) {
              if (sqrt(k * k + l * l) < max_R_Wilson_loop)
                outfile << "Wt(R = " << sqrt(k * k + l * l) << ", T = " << j
                        << ") ";
            }
          }
        } else if (!non_planar) {
          for (int k = 1;
               k < std::min({LX, LY, static_cast<size_t>(max_R_Wilson_loop)});
               k++) {
            outfile << "Wt(R = " << k << ", T = " << j << ") ";
          }
        }
      }
      outfile << std::endl;
    } else if (!open_bc[0] && !open_bc[1]) {
      if (!non_planar) {
        for (int j = 1;
             j < std::min(LT, static_cast<size_t>(max_T_Wilson_loop)); j++) {
          for (int k = 1;
               k < std::min({LX, LY, static_cast<size_t>(max_R_Wilson_loop)});
               k++) {
            outfile << "Wt(R = " << k << ", T = " << j << ") ";
          }
        }
        outfile << std::endl;
      }
    }
  }
  Kokkos::initialize();
  {
    using Group = U1<T>;
    using GaugeFieldType = GaugeField<T, Group, 3, 1>;
    using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
    RNG rng = RNG(seed);
    GaugeFieldType gauge_field = GaugeFieldType(LX, LY, LT);
    Metropolis<T, Group, GaugeFieldType, RNG> metropolis =
        Metropolis<T, Group, GaugeFieldType, RNG>(gauge_field, rng, n_hit, beta,
                                                  delta);
    metropolis.initGauge(cold_start);
    if (open_bc[0])
      gauge_field.set_open_bc_x();
    if (open_bc[1])
      gauge_field.set_open_bc_y();
    if (open_bc[0] && open_bc[1])
      std::cout << "Starting Plaquette: " << gauge_field.get_plaquette_obc()
                << std::endl;
    else
      std::cout << "Starting Plaquette: " << gauge_field.get_plaquette()
                << std::endl;
    std::cout << "Starting Metropolis: " << std::endl;
    auto metropolis_start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_sweep; i++) {
      auto start_time = std::chrono::high_resolution_clock::now();
      T acceptance_rate = metropolis.sweep();
      T plaquette = 0.0;
      T wloop_temporal = 0.0;
      if (open_bc[0] && open_bc[1]) {
        plaquette = gauge_field.get_plaquette_obc();
      } else {
        plaquette = gauge_field.get_plaquette();
      }
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> sweep_time = end_time - start_time;
      if (verbose) {
        std::cout << "Step: " << i << " Plaquette: " << plaquette
                  << " Acceptance Rate: " << acceptance_rate
                  << " Time: " << sweep_time.count() << std::endl;
      }
      if (outfilename != "") {
        if (open_bc[0] && open_bc[1]) {
          if (!((i + 1) % n_meas) || (i == (n_sweep - 1))) {
            outfile << i + 1 << " " << plaquette << " " << acceptance_rate
                    << " " << sweep_time.count();
            for (int j = 1;
                 j < std::min(LT, static_cast<size_t>(max_T_Wilson_loop));
                 j++) {
              if (non_planar) {
                for (int k = 1; k < std::min(LX, LY); k++) {
                  for (int l = 0; l <= k; l++) {
                    if (sqrt(k * k + l * l) < max_R_Wilson_loop)
                      outfile << " "
                              << gauge_field.wloop_np_temporal_obc(
                                     v0[0], v0[1], v0[2], j, k, l);
                  }
                }
              } else if (non_planar == false) {
                for (int k = 1;
                     k <
                     std::min({LX, LY, static_cast<size_t>(max_R_Wilson_loop)});
                     k++) {
                  outfile << " "
                          << gauge_field.wloop_temporal_obc(v0[0], v0[1], v0[2],
                                                            j, k);
                }
              }
            }
            outfile << std::endl;
          }
        }
      }
    }
    auto metropolis_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> metropolis_time =
        metropolis_end_time - metropolis_start_time;
    std::cout << "Metropolis Time: " << metropolis_time.count() << std::endl;
  }
  Kokkos::finalize();
  outfile.close();
}

template <typename T>
void Metropolis_U1_2D(const size_t &LX, const size_t &LT, const size_t &n_hit,
                      const T &beta, const T &delta, const size_t &seed,
                      const size_t &n_sweep, const bool cold_start,
                      const std::string &outfilename, const bool open_bc[3]) {
  std::cout << "Running Metropolis_U1_2D" << std::endl;
  std::cout << "Gauge Field Dimensions:" << std::endl;
  std::cout << "LX = " << LX << std::endl;
  std::cout << "LT = " << LT << std::endl;
  std::cout << "Metropolis Parameters:" << std::endl;
  std::cout << "beta = " << beta << std::endl;
  std::cout << "delta = " << delta << std::endl;
  std::cout << "n_hit = " << n_hit << std::endl;
  std::cout << "n_sweep = " << n_sweep << std::endl;
  std::cout << "seed = " << seed << std::endl;
  std::cout << "start condition = " << (cold_start ? "cold" : "hot")
            << std::endl;
  std::cout << "output file = " << outfilename << std::endl;
  std::ofstream outfile;
  if (outfilename != "") {
    outfile.open(outfilename);
    outfile << "step, plaquette, acceptance_rate, time" << std::endl;
  }
  Kokkos::initialize();
  {
    using Group = U1<T>;
    using GaugeFieldType = GaugeField<T, Group, 2, 1>;
    using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
    RNG rng = RNG(seed);
    GaugeFieldType gauge_field = GaugeFieldType(LX, LT);
    Metropolis<T, Group, GaugeFieldType, RNG> metropolis =
        Metropolis<T, Group, GaugeFieldType, RNG>(gauge_field, rng, n_hit, beta,
                                                  delta);
    metropolis.initGauge(cold_start);
    if (open_bc[0])
      gauge_field.set_open_bc_x();
    std::cout << "Starting Plaquette: " << gauge_field.get_plaquette()
              << std::endl;
    std::cout << "Starting Metropolis: " << std::endl;
    auto metropolis_start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n_sweep; i++) {
      auto start_time = std::chrono::high_resolution_clock::now();
      T acceptance_rate = metropolis.sweep();
      T plaquette = gauge_field.get_plaquette();
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> sweep_time = end_time - start_time;
      std::cout << "Step: " << i << " Plaquette: " << plaquette
                << " Acceptance Rate: " << acceptance_rate
                << " Time: " << sweep_time.count() << std::endl;
      if (outfilename != "") {
        outfile << i << ", " << plaquette << ", " << acceptance_rate << ", "
                << sweep_time.count() << std::endl;
      }
    }
    auto metropolis_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> metropolis_time =
        metropolis_end_time - metropolis_start_time;
    std::cout << "Metropolis Time: " << metropolis_time.count() << std::endl;
  }
  Kokkos::finalize();
  outfile.close();
}

template void Metropolis_U1_4D<float>(
    const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
    const size_t &n_hit, const float &beta, const float &delta,
    const size_t &seed, const size_t &n_sweep, const bool cold_start,
    const std::string &outfilename, const bool open_bc[3]);

template void Metropolis_U1_4D<double>(
    const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
    const size_t &n_hit, const double &beta, const double &delta,
    const size_t &seed, const size_t &n_sweep, const bool cold_start,
    const std::string &outfilename, const bool open_bc[3]);
;

template void Metropolis_U1_3D<float>(
    const size_t &LX, const size_t &LY, const size_t &LT, const size_t &n_hit,
    const float &beta, const float &delta, const size_t &seed,
    const size_t &n_sweep, const size_t &n_meas, const bool cold_start,
    const std::string &outfilename, const bool open_bc[3], const bool open_bc_t,
    const int v0[3], const int t0, const bool non_planar,
    const int &max_T_Wilson_loop, const int &max_R_Wilson_loop,
    const bool &verbose);

template void Metropolis_U1_3D<double>(
    const size_t &LX, const size_t &LY, const size_t &LT, const size_t &n_hit,
    const double &beta, const double &delta, const size_t &seed,
    const size_t &n_sweep, const size_t &n_meas, const bool cold_start,
    const std::string &outfilename, const bool open_bc[3], const bool open_bc_t,
    const int v0[3], const int t0, const bool non_planar,
    const int &max_T_wilson_loop, const int &max_R_Wilson_loop,
    const bool &verbose);
;

template void Metropolis_U1_2D<float>(const size_t &LX, const size_t &LT,
                                      const size_t &n_hit, const float &beta,
                                      const float &delta, const size_t &seed,
                                      const size_t &n_sweep,
                                      const bool cold_start,
                                      const std::string &outfilename,
                                      const bool open_bc[3]);

template void Metropolis_U1_2D<double>(const size_t &LX, const size_t &LT,
                                       const size_t &n_hit, const double &beta,
                                       const double &delta, const size_t &seed,
                                       const size_t &n_sweep,
                                       const bool cold_start,
                                       const std::string &outfilename,
                                       const bool open_bc[3]);
;

} // namespace klft