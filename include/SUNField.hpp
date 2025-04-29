// define structs for initializing SUN field

#pragma once
#include "GLOBAL.hpp"
#include "Tuner.hpp"

namespace klft
{

  template <size_t Nc>
  struct deviceSUNField {

    deviceSUNField() = delete;

    // initialize all sites to a given value
    deviceSUNField(const index_t L0, const index_t L1, 
                const index_t L2, const index_t L3, 
                const complex_t init) : dimensions({L0, L1, L2, L3}) {
      do_init(L0, L1, L2, L3, field, init);
    }

    deviceSUNField(const IndexArray<4> &dimensions,
                const complex_t init) : dimensions(dimensions) {
      do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field, init);
    }

    // initialize all sites to a given SUN matrix
    deviceSUNField(const index_t L0, const index_t L1, 
                const index_t L2, const index_t L3, 
                const SUN<Nc> &init) : dimensions({L0, L1, L2, L3}) {
      do_init(L0, L1, L2, L3, field, init);
    }

    deviceSUNField(const IndexArray<4> &dimensions,
                const SUN<Nc> &init) : dimensions(dimensions) {
      do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field, init);
    }

    void do_init(const index_t L0, const index_t L1, 
                const index_t L2, const index_t L3, 
                SUNField<Nc> &V, const complex_t init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<4>("init_deviceSUNField",
        IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1,
                      const index_t i2, const index_t i3) {
          #pragma unroll
          for(index_t c1 = 0; c1 < Nc; ++c1) {
            #pragma unroll
            for(index_t c2 = 0; c2 < Nc; ++c2) {
              V(i0,i1,i2,i3)[c1][c2] = init;
            }
          }
        });
      Kokkos::fence();
    }

    void do_init(const index_t L0, const index_t L1, 
                const index_t L2, const index_t L3, 
                SUNField<Nc> &V, const SUN<Nc> &init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<4>("init_deviceSUNField",
        IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1,
                      const index_t i2, const index_t i3) {
          V(i0,i1,i2,i3) = init;
        });
      Kokkos::fence();
    }

    SUNField<Nc> field;
    const IndexArray<4> dimensions;

    // define accessors
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const indexType i0, const indexType i1, const indexType i2, const indexType i3) const {
      return field(i0,i1,i2,i3);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const indexType i0, const indexType i1, const indexType i2, const indexType i3) {
      return field(i0,i1,i2,i3);
    }

    // define accessors with 4D Kokkos array
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const Kokkos::Array<indexType,4> site) const {
      return field(site[0],site[1],site[2],site[3]);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const Kokkos::Array<indexType,4> site) {
      return field(site[0],site[1],site[2],site[3]);
    }

  };

  template <size_t Nc>
  struct deviceSUNField3D {

    deviceSUNField3D() = delete;

    // initialize all sites to a given value
    deviceSUNField3D(const index_t L0, const index_t L1, 
                  const index_t L2, 
                  const complex_t init) : dimensions({L0, L1, L2}) {
      do_init(L0, L1, L2, field, init);
    }

    deviceSUNField3D(const IndexArray<3> &dimensions,
                  const complex_t init) : dimensions(dimensions) {
      do_init(dimensions[0], dimensions[1], dimensions[2], field, init);
    }

    // initialize all sites to a given SUN matrix
    deviceSUNField3D(const index_t L0, const index_t L1, 
                  const index_t L2, 
                  const SUN<Nc> &init) : dimensions({L0, L1, L2}) {
      do_init(L0, L1, L2, field, init);
    }

    deviceSUNField3D(const IndexArray<3> &dimensions,
                  const SUN<Nc> &init) : dimensions(dimensions) {
      do_init(dimensions[0], dimensions[1], dimensions[2], field, init);
    }

    void do_init(const index_t L0, const index_t L1, 
                const index_t L2, 
                SUNField3D<Nc> &V, const complex_t init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
      tune_and_launch_for<3>("init_deviceSUNField3D",
        IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1,
                      const index_t i2) {
          #pragma unroll
          for(index_t c1 = 0; c1 < Nc; ++c1) {
            #pragma unroll
            for(index_t c2 = 0; c2 < Nc; ++c2) {
              V(i0,i1,i2)[c1][c2] = init;
            }
          }
        });
      Kokkos::fence();
    }

    void do_init(const index_t L0, const index_t L1, 
                const index_t L2, 
                SUNField3D<Nc> &V, const SUN<Nc> &init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
      tune_and_launch_for<3>("init_deviceSUNField3D",
        IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1,
                      const index_t i2) {
          V(i0,i1,i2) = init;
        });
      Kokkos::fence();
    }

    SUNField3D<Nc> field;
    const IndexArray<3> dimensions;

    // define accessors
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const indexType i0, const indexType i1, const indexType i2) const {
      return field(i0,i1,i2);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const indexType i0, const indexType i1, const indexType i2) {
      return field(i0,i1,i2);
    }

    // define accessors with 3D Kokkos array
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const Kokkos::Array<indexType,3> site) const {
      return field(site[0],site[1],site[2]);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const Kokkos::Array<indexType,3> site) {
      return field(site[0],site[1],site[2]);
    }

  };

  template <size_t Nc>
  struct deviceSUNField2D {

    deviceSUNField2D() = delete;

    // initialize all sites to a given value
    deviceSUNField2D(const index_t L0, const index_t L1, 
                  const complex_t init) : dimensions({L0, L1}) {
      do_init(L0, L1, field, init);
    }

    deviceSUNField2D(const IndexArray<2> &dimensions,
                  const complex_t init) : dimensions(dimensions) {
      do_init(dimensions[0], dimensions[1], field, init);
    }

    // initialize all sites to a given SUN matrix
    deviceSUNField2D(const index_t L0, const index_t L1, 
                  const SUN<Nc> &init) : dimensions({L0, L1}) {
      do_init(L0, L1, field, init);
    }

    deviceSUNField2D(const IndexArray<2> &dimensions,
                  const SUN<Nc> &init) : dimensions(dimensions) {
      do_init(dimensions[0], dimensions[1], field, init);
    }

    void do_init(const index_t L0, const index_t L1, 
                SUNField2D<Nc> &V, const complex_t init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
      tune_and_launch_for<2>("init_deviceSUNField2D",
        IndexArray<2>{0, 0}, IndexArray<2>{L0, L1},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          #pragma unroll
          for(index_t c1 = 0; c1 < Nc; ++c1) {
            #pragma unroll
            for(index_t c2 = 0; c2 < Nc; ++c2) {
              V(i0,i1)[c1][c2] = init;
            }
          }
        });
      Kokkos::fence();
    }

    void do_init(const index_t L0, const index_t L1, 
                SUNField2D<Nc> &V, const SUN<Nc> &init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
      tune_and_launch_for<2>("init_deviceSUNField2D",
        IndexArray<2>{0, 0}, IndexArray<2>{L0, L1},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          V(i0,i1) = init;
        });
      Kokkos::fence();
    }

    SUNField2D<Nc> field;
    const IndexArray<2> dimensions;

    // define accessors
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const indexType i0, const indexType i1) const {
      return field(i0,i1);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const indexType i0, const indexType i1) {
      return field(i0,i1);
    }

    // define accessors with 2D Kokkos array
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const Kokkos::Array<indexType,2> site) const {
      return field(site[0],site[1]);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION
    SUN<Nc> & operator()(const Kokkos::Array<indexType,2> site) {
      return field(site[0],site[1]);
    }

  };

}