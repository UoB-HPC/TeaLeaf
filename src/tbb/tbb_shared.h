#pragma once

#include <cassert>
#include <ostream>
#include <tbb/tbb.h>

#if defined(PARTITIONER_AUTO)
using tbb_partitioner = tbb::auto_partitioner;
  #define PARTITIONER_NAME "auto_partitioner"
#elif defined(PARTITIONER_AFFINITY)
using tbb_partitioner = tbb::affinity_partitioner;
  #define PARTITIONER_NAME "affinity_partitioner"
#elif defined(PARTITIONER_STATIC)
using tbb_partitioner = tbb::static_partitioner;
  #define PARTITIONER_NAME "static_partitioner"
#elif defined(PARTITIONER_SIMPLE)
using tbb_partitioner = tbb     ::simple_partitioner;
  #define PARTITIONER_NAME "simple_partitioner"
#else
// default to auto
using tbb_partitioner = tbb::auto_partitioner;
  #define PARTITIONER_NAME "auto_partitioner"
#endif

static tbb_partitioner partitioner{};

template <typename N = int> struct Range2d {
  const N fromX, toX;
  const N fromY, toY;

  constexpr inline Range2d(N fromX, N fromY, N toX, N toY) : fromX(fromX), toX(toX), fromY(fromY), toY(toY) {
    assert(fromX < toX);
    assert(fromY < toY);
    assert(sizeX() >= 0);
    assert(sizeY() >= 0);
  }
  [[nodiscard]] constexpr inline N sizeX() const { return toX - fromX; }
  [[nodiscard]] constexpr inline N sizeY() const { return toY - fromY; }
  [[nodiscard]] constexpr inline N sizeXY() const { return sizeX() * sizeY(); }

  constexpr inline N restore(N i, N xLimit) const {
    const int jj = (i / sizeX()) + fromX;
    const int kk = (i % sizeX()) + fromY;
    return kk + jj * xLimit;
  }

  friend std::ostream &operator<<(std::ostream &os, const Range2d &d) {
    os << "Range2d{"
       << " X[" << d.fromX << "->" << d.toX << " (" << d.sizeX() << ")]"
       << " Y[" << d.fromY << "->" << d.toY << " (" << d.sizeY() << ")]"
       << "}";
    return os;
  }
};

template <typename T> T *alloc_raw(size_t size) { return static_cast<T *>(std::malloc(size * sizeof(T))); }
template <typename T> void dealloc_raw(T *ptr) { std::free(ptr); }