#pragma once
#include "GLOBAL.hpp"

namespace klft {
  template <typename T>
  struct U1 {

    Kokkos::complex<T> v;

    U1() = default;

    KOKKOS_INLINE_FUNCTION U1(const Kokkos::complex<T> &a) {
      v = a;
    }

    KOKKOS_INLINE_FUNCTION U1(const U1<T> &in) {
      v = in.v;
    }

    KOKKOS_INLINE_FUNCTION U1(const Kokkos::complex<T> v_in[1]) {
      v = v_in[0];
    }

    KOKKOS_INLINE_FUNCTION U1(const Kokkos::Array<Kokkos::complex<T>,1> &v_in) {
      v = v_in[0];
    }

    KOKKOS_INLINE_FUNCTION void set_identity() {
      v = Kokkos::complex<T>(1.0,0.0);
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> operator()(const int &i) {
      return v;
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> operator()(const int &i) const {
      return v;
    }

    KOKKOS_INLINE_FUNCTION void dagger() {
      v.imag(-v.imag());
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator+=(const U1<Tin> &in) {
      v += in.v;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator-=(const U1<Tin> &in) {
      v -= in.v;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION U1<T> operator+(const U1<Tin> &in) const {
      return U1<T>(v + in.v);
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION U1<T> operator-(const U1<Tin> &in) const {
      return U1<T>(v - in.v);
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator*=(const U1<Tin> &in) {
      T a = v.real()*in.v.real() - v.imag()*in.v.imag();
      T b = v.real()*in.v.imag() + v.imag()*in.v.real();
      v = Kokkos::complex<T>(a,b);
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION U1<T> operator*(const U1<Tin> &in) const {
      return U1<T>(Kokkos::complex<T>(v.real()*in.v.real() - v.imag()*in.v.imag(),v.real()*in.v.imag() + v.imag()*in.v.real()));
    }

    KOKKOS_INLINE_FUNCTION T retrace() const {
      return v.real();
    }

    KOKKOS_INLINE_FUNCTION void restoreGauge() {
      T norm = Kokkos::sqrt(v.real()*v.real() + v.imag()*v.imag());
      if(Kokkos::abs(norm-0.0) > 1e-12) v /= Kokkos::sqrt(v.real()*v.real() + v.imag()*v.imag());
      else v = Kokkos::complex<T>(0.0,0.0);
    }

    template <class RNG>
    KOKKOS_INLINE_FUNCTION void get_random(RNG &generator, T delta) {
      v = Kokkos::exp(Kokkos::complex(0.0,generator.drand(-delta*Kokkos::numbers::pi_v<T>,delta*Kokkos::numbers::pi_v<T>)));
    }
  };

  template <typename T>
  KOKKOS_INLINE_FUNCTION U1<T> dagger(const U1<T> &in) {
    return U1<T>(Kokkos::complex(in.v.real(),-in.v.imag()));
  }

  // template <typename T, class G, class RNG, std::enable_if_t<std::is_same<G,U1<T>>::value,int> = 0>
  // KOKKOS_INLINE_FUNCTION U1<T> get_random(RNG &generator, T delta) {
  //   return U1<T>(generator.drand(-delta*Kokkos::numbers::pi_v<T>,delta*Kokkos::numbers::pi_v<T>));
  // }

} // namespace klqcd
