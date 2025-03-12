#include "GLOBAL.hpp"

namespace klft {

  template <typename T>
  void Metropolis_SU2_4D(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);
  
  template <typename T>
  void Metropolis_SU2_3D(const size_t &LX, const size_t &LY, const size_t &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);

  template <typename T>
  void Metropolis_SU2_2D(const size_t &LX, const size_t &LT,
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);

  template <typename T>
  void Metropolis_U1_4D(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT, 
                        const size_t &n_hit, const T &beta, const T &delta,
                        const size_t &seed, const size_t &n_sweep, const bool cold_start,
                        const std::string &outfilename, const bool open_bc[3]);

  template <typename T>
  void Metropolis_U1_3D(const size_t &LX, const size_t &LY, const size_t &LT, 
                        const size_t &n_hit, const T &beta, const T &delta,
                        const size_t &seed, const size_t &n_sweep, const size_t &n_meas, const bool cold_start,
                        const std::string &outfilename, const bool open_bc[3], const int v0[3], const int &max_R_Wilson_loop, const bool &verbose);

  template <typename T>
  void Metropolis_U1_2D(const size_t &LX, const size_t &LT,
                        const size_t &n_hit, const T &beta, const T &delta,
                        const size_t &seed, const size_t &n_sweep, const bool cold_start,
                        const std::string &outfilename, const bool open_bc[3]);

  template <typename T>
  void Metropolis_SU3_4D(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);

  template <typename T>
  void Metropolis_SU3_3D(const size_t &LX, const size_t &LY, const size_t &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);

  template <typename T>
  void Metropolis_SU3_2D(const size_t &LX, const size_t &LT,
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);
                        
}