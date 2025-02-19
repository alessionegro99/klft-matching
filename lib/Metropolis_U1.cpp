#include "../include/klft.hpp"
#include "../include/Metropolis.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>

namespace klft {

  template <typename T>
  void Metropolis_U1_4D(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename, const bool open_bc[3]) {
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
    std::cout << "start condition = " << (cold_start ? "cold" : "hot") << std::endl;
    std::cout << "output file = " << outfilename << std::endl;
    std::ofstream outfile;
    if(outfilename != "") {
      outfile.open(outfilename);
      outfile << "step, plaquette, acceptance_rate, time" << std::endl;
    }
    Kokkos::initialize();
    {
      using Group = U1<T>;
      using GaugeFieldType = GaugeField<T,Group,4,1>;
      using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
      RNG rng = RNG(seed);
      GaugeFieldType gauge_field = GaugeFieldType(LX,LY,LZ,LT);
      Metropolis<T,Group,GaugeFieldType,RNG> metropolis = Metropolis<T,Group,GaugeFieldType,RNG>(gauge_field,rng,n_hit,beta,delta);
      metropolis.initGauge(cold_start);
      if(open_bc[0]) gauge_field.set_open_bc_x();
      if(open_bc[1]) gauge_field.set_open_bc_y();
      if(open_bc[2]) gauge_field.set_open_bc_z();
      std::cout << "Starting Plaquette: " << gauge_field.get_plaquette() << std::endl;
      std::cout << "Starting Metropolis: " << std::endl;
      auto metropolis_start_time = std::chrono::high_resolution_clock::now();
      for(size_t i = 0; i < n_sweep; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        T acceptance_rate = metropolis.sweep();
        T plaquette = gauge_field.get_plaquette();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sweep_time = end_time - start_time;
        std::cout << "Step: " << i << " Plaquette: " << plaquette << " Acceptance Rate: " << acceptance_rate << " Time: " << sweep_time.count() << std::endl;
        if(outfilename != "") {
          outfile << i << ", " << plaquette << ", " << acceptance_rate << ", " << sweep_time.count() << std::endl;
        }
      }
    auto metropolis_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> metropolis_time = metropolis_end_time - metropolis_start_time;
    std::cout << "Metropolis Time: " << metropolis_time.count() << std::endl;
    }
    Kokkos::finalize();
    outfile.close();
  }

  template <typename T>
  void Metropolis_U1_3D(const size_t &LX, const size_t &LY, const size_t &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const size_t &n_meas, const bool cold_start,
                         const std::string &outfilename, const bool open_bc[3], const int v0[3]) {
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
    std::cout << "start condition = " << (cold_start ? "cold" : "hot") << std::endl;
    std::cout << "output file = " << outfilename << std::endl;
    std::cout << "x0 = " << v0[0] << std::endl;
    std::cout << "y0 = " << v0[1] << std::endl;
    std::cout << "z0 = " << v0[2] << std::endl;
    std::ofstream outfile;
    if(outfilename != "") {
      outfile.open(outfilename);
      outfile << "step plaquette acceptance_rate time ";
      for(int j = 1; j < LT; j++){
        outfile << "Wt(R = " << 1 << ", T = " << j << ") ";
        outfile << "Wt(R = " << sqrt(2) << ", T = " << j << ") "; 
        if(LX >= 3 && LY >= 3)
        {
          outfile << "Wt(R = " << 2 << ", T = " << j << ") ";
          outfile << "Wt(R = " << sqrt(5) << ", T = " << j << ") ";
          outfile << "Wt(R = " << sqrt(8) << ", T = " << j << ") ";
        }
        if(LX >= 4 && LY >= 4)
        {
          outfile << "Wt(R = " << sqrt(10) << ", T = " << j << ") ";
          outfile << "Wt(R = " << sqrt(18) << ", T = " << j << ") ";
        }
        if(LX >= 5 && LY >= 5)
        {
          outfile << "Wt(R = " << 5 << ", T = " << j << ") ";
          outfile << "Wt(R = " << sqrt(32) << ", T = " << j << ") ";
        }
      }
      outfile << std::endl;
    }
    Kokkos::initialize();
    {
      using Group = U1<T>;
      using GaugeFieldType = GaugeField<T,Group,3,1>;
      using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
      RNG rng = RNG(seed);
      GaugeFieldType gauge_field = GaugeFieldType(LX,LY,LT);
      Metropolis<T,Group,GaugeFieldType,RNG> metropolis = Metropolis<T,Group,GaugeFieldType,RNG>(gauge_field,rng,n_hit,beta,delta);
      metropolis.initGauge(cold_start);
      if(open_bc[0]) gauge_field.set_open_bc_x();
      if(open_bc[1]) gauge_field.set_open_bc_y();
      if(open_bc[0] && open_bc[1])
        std::cout << "Starting Plaquette: " << gauge_field.get_plaquette_obc() << std::endl;
      else
        std::cout << "Starting Plaquette: " << gauge_field.get_plaquette() << std::endl;
      std::cout << "Starting Metropolis: " << std::endl;
      auto metropolis_start_time = std::chrono::high_resolution_clock::now();
      for(int i = 0; i < n_sweep; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        T acceptance_rate = metropolis.sweep();
        T plaquette = 0.0;
        T wloop_temporal = 0.0;
        if(open_bc[0] && open_bc[1]){
          plaquette = gauge_field.get_plaquette_obc();
        }
        else{
          plaquette = gauge_field.get_plaquette();
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sweep_time = end_time - start_time;
        std::cout << "Step: " << i << " Plaquette: " << plaquette << " Acceptance Rate: " << acceptance_rate << " Time: " << sweep_time.count() << std::endl;
        if(outfilename != "") {
          if(!((i+1)%n_meas) || (i==(n_sweep-1))){
            outfile << i + 1 << " " << plaquette << " " << acceptance_rate << " " << sweep_time.count();
            for(int j = 1; j < LT; j++){
              outfile << " " << gauge_field.wloop_temporal_obc(v0[0], v0[1], v0[2], j, 1); // r = 1
              outfile << " " << gauge_field.wloop_np_temporal_obc(v0[0], v0[1], v0[2], j, 1, 1); // r = sqrt(2)
              if(LX >= 3 && LY >= 3)
              {
              outfile << " " << gauge_field.wloop_temporal_obc(v0[0], v0[1], v0[2], j, 2); // r = 2
              outfile << " " << gauge_field.wloop_np_temporal_obc(v0[0], v0[1], v0[2], j, 1, 2); // r = sqrt(5)
              outfile << " " << gauge_field.wloop_np_temporal_obc(v0[0], v0[1], v0[2], j, 2, 2); // r = sqrt(8)
              }
              if(LX >= 4 && LY >= 4)
              {
              outfile << " " << gauge_field.wloop_np_temporal_obc(v0[0], v0[1], v0[2], j, 3, 1); // r = sqrt(10)
              outfile << " " << gauge_field.wloop_np_temporal_obc(v0[0], v0[1], v0[2], j, 3, 3); // r = sqrt(18)
              }
              if(LX >= 5 && LY >= 5)
              {
              outfile << " " << gauge_field.wloop_np_temporal_obc(v0[0], v0[1], v0[2], j, 3, 4); // r = 5
              outfile << " " << gauge_field.wloop_np_temporal_obc(v0[0], v0[1], v0[2], j, 4, 4); // r = sqrt(32)
              }
            }            
            outfile << std::endl;
          }
        }
      }
    auto metropolis_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> metropolis_time = metropolis_end_time - metropolis_start_time;
    std::cout << "Metropolis Time: " << metropolis_time.count() << std::endl;
    }
    Kokkos::finalize();
    outfile.close();
  }

  template <typename T>
  void Metropolis_U1_2D(const size_t &LX, const size_t &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
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
    std::cout << "start condition = " << (cold_start ? "cold" : "hot") << std::endl;
    std::cout << "output file = " << outfilename << std::endl;
    std::ofstream outfile;
    if(outfilename != "") {
      outfile.open(outfilename);
      outfile << "step, plaquette, acceptance_rate, time" << std::endl;
    }
    Kokkos::initialize();
    {
      using Group = U1<T>;
      using GaugeFieldType = GaugeField<T,Group,2,1>;
      using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
      RNG rng = RNG(seed);
      GaugeFieldType gauge_field = GaugeFieldType(LX,LT);
      Metropolis<T,Group,GaugeFieldType,RNG> metropolis = Metropolis<T,Group,GaugeFieldType,RNG>(gauge_field,rng,n_hit,beta,delta);
      metropolis.initGauge(cold_start);
      if(open_bc[0]) gauge_field.set_open_bc_x();
      std::cout << "Starting Plaquette: " << gauge_field.get_plaquette() << std::endl;
      std::cout << "Starting Metropolis: " << std::endl;
      auto metropolis_start_time = std::chrono::high_resolution_clock::now();
      for(size_t i = 0; i < n_sweep; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        T acceptance_rate = metropolis.sweep();
        T plaquette = gauge_field.get_plaquette();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sweep_time = end_time - start_time;
        std::cout << "Step: " << i << " Plaquette: " << plaquette << " Acceptance Rate: " << acceptance_rate << " Time: " << sweep_time.count() << std::endl;
        if(outfilename != "") {
          outfile << i << ", " << plaquette << ", " << acceptance_rate << ", " << sweep_time.count() << std::endl;
        }
      }
    auto metropolis_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> metropolis_time = metropolis_end_time - metropolis_start_time;
    std::cout << "Metropolis Time: " << metropolis_time.count() << std::endl;
    }
    Kokkos::finalize();
    outfile.close();
  }


  template void Metropolis_U1_4D<float>(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT, 
                                        const size_t &n_hit, const float &beta, const float &delta,
                                        const size_t &seed, const size_t &n_sweep, const bool cold_start,
                                        const std::string &outfilename, const bool open_bc[3]);

  template void Metropolis_U1_4D<double>(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT, 
                                         const size_t &n_hit, const double &beta, const double &delta,
                                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                                         const std::string &outfilename, const bool open_bc[3]);;

  template void Metropolis_U1_3D<float>(const size_t &LX, const size_t &LY, const size_t &LT, 
                                        const size_t &n_hit, const float &beta, const float &delta,
                                        const size_t &seed, const size_t &n_sweep, const size_t &n_meas, const bool cold_start,
                                        const std::string &outfilename, const bool open_bc[3], const int v0[3]);

  template void Metropolis_U1_3D<double>(const size_t &LX, const size_t &LY, const size_t &LT,
                                         const size_t &n_hit, const double &beta, const double &delta,
                                         const size_t &seed, const size_t &n_sweep, const size_t &n_meas, const bool cold_start,
                                         const std::string &outfilename, const bool open_bc[3], const int v0[3]);;

  template void Metropolis_U1_2D<float>(const size_t &LX, const size_t &LT,
                                        const size_t &n_hit, const float &beta, const float &delta,
                                        const size_t &seed, const size_t &n_sweep, const bool cold_start,
                                        const std::string &outfilename, const bool open_bc[3]);  

  template void Metropolis_U1_2D<double>(const size_t &LX, const size_t &LT,
                                         const size_t &n_hit, const double &beta, const double &delta,
                                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                                         const std::string &outfilename, const bool open_bc[3]);;                                                                                                                                                                                                                                          

}