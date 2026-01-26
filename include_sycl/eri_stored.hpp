#pragma once
#include <sycl/sycl.hpp>
//#include <dpct/dpct.hpp>
#include "eri.hpp"

namespace gansu {
/*
static double block_reduce_sum(double x, uint8_t *dpct_local){
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto sdata = (double *)dpct_local;
  int tid = item_ct1.get_local_id(2);
  sdata[tid] = x;
  item_ct1.barrier();

  for (int s = item_ct1.get_local_range(2) / 2; s > 0; s >>= 1) {
    if(tid < s) sdata[tid] += sdata[tid + s];
    item_ct1.barrier();
  }
  return sdata[0];
}
*/

static size_t idx4_to_1(int num_basis, int mu, int nu, int la, int si){
  return ( ( (size_t(mu)*num_basis + nu)*num_basis + la)*num_basis + si );
}

static inline real_t antisym_eri(const real_t *__restrict__ eri_mo,
                                          const int num_basis, const int p,
                                          const int q, const int r, const int s)
{
    assert(p >= 0 && p < num_basis*2);
    assert(q >= 0 && q < num_basis*2);
    assert(r >= 0 && r < num_basis*2);
    assert(s >= 0 && s < num_basis*2);

    // <pq||rs> = (pr|qs) - (ps|qr)
    real_t prqs = ((p%2)==(r%2) && ((q%2)==(s%2))) ? eri_mo[idx4_to_1(num_basis, p/2, r/2, q/2, s/2)] : 0.0;
    real_t psqr = ((p%2)==(s%2) && ((q%2)==(r%2))) ? eri_mo[idx4_to_1(num_basis, p/2, s/2, q/2, r/2)] : 0.0;
    return prqs - psqr;
}

inline void require_fp64(const sycl::queue& q)
{
    if (!q.get_device().has(sycl::aspect::fp64)) {
        throw std::runtime_error("FP64 not supported on selected SYCL device");
    }
}

template <typename T>
//[[nodiscard]] inline T atomic_add(T* addr, T value) noexcept
inline T atomic_add(T* addr, T value) noexcept
{
    sycl::atomic_ref<
        T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        atom(*addr);

    return atom.fetch_add(value);
}

template <typename T>
inline T atomic_max(T* addr, T value) noexcept
{
    sycl::atomic_ref<
        T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        atom(*addr);

    return atom.fetch_max(value);
}

template <typename T>
inline T atomic_add_local(T* addr, T value) noexcept {
    sycl::atomic_ref<T,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space> atom(*addr);
    return atom.fetch_add(value);
}

template <typename T>
inline void atomic_max_fp_local(T* addr, T value) {
    sycl::atomic_ref<
        T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::work_group,
        sycl::access::address_space::local_space
    > atom(*addr);

    T old = atom.load();
    while (old < value &&
           !atom.compare_exchange_strong(old, value)) {}
}

template <typename T>
inline void atomic_max_fp_global(T* addr, T value) {
    sycl::atomic_ref<
        T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space
    > atom(*addr);

    T old = atom.load();
    while (old < value &&
           !atom.compare_exchange_strong(old, value)) {}
}



} // namespace gansu
