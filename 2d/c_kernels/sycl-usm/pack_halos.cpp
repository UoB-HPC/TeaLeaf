#include "../../shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Packs the top halo buffer(s)
void pack_top(const int x,          //
              const int y,          //
              const int halo_depth, //
              SyclBuffer &buffer,   //
              SyclBuffer &field,    //
              const int depth,      //
              queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class pack_top>(range<1>(x * depth), [=](id<1> idx) {
          const int offset = x * (y - halo_depth - depth);
          buffer[idx[0]] = field[offset + idx[0]];
        });
      })
      .wait_and_throw();
}

// Packs the bottom halo buffer(s)
void pack_bottom(const int x,          //
                 const int y,          //
                 const int halo_depth, //
                 SyclBuffer &buffer,   //
                 SyclBuffer &field,    //
                 const int depth,      //
                 queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class pack_bottom>(range<1>(x * depth), [=](id<1> idx) {
          const int offset = x * halo_depth;
          buffer[idx[0]] = field[offset + idx[0]];
        });
      })
      .wait_and_throw();
}

// Packs the left halo buffer(s)
void pack_left(const int x,          //
               const int y,          //
               const int halo_depth, //
               SyclBuffer &buffer,   //
               SyclBuffer &field,    //
               const int depth,      //
               queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class pack_left>(range<1>(y * depth), [=](id<1> idx) {
          const auto lines = idx[0] / depth;
          const auto offset = halo_depth + lines * (x - depth);
          buffer[idx[0]] = field[offset + idx[0]];
        });
      })
      .wait_and_throw();
}

// Packs the right halo buffer(s)
void pack_right(const int x,          //
                const int y,          //
                const int halo_depth, //
                SyclBuffer &buffer,   //
                SyclBuffer &field,    //
                const int depth,      //
                queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class pack_right>(range<1>(y * depth), [=](id<1> idx) {
          const auto lines = idx[0] / depth;
          const auto offset = x - halo_depth - depth + lines * (x - depth);
          buffer[idx[0]] = field[offset + idx[0]];
        });
      })
      .wait_and_throw();
}

// Unpacks the top halo buffer(s)
void unpack_top(const int x,          //
                const int y,          //
                const int halo_depth, //
                SyclBuffer &buffer,   //
                SyclBuffer &field,    //
                const int depth,      //
                queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class unpack_top>(range<1>(x * depth), [=](id<1> idx) {
          const int offset = x * (y - halo_depth);
          field[offset + idx[0]] = buffer[idx[0]];
        });
      })
      .wait_and_throw();
}

// Unpacks the bottom halo buffer(s)
void unpack_bottom(const int x,          //
                   const int y,          //
                   const int halo_depth, //
                   SyclBuffer &buffer,   //
                   SyclBuffer &field,    //
                   const int depth,      //
                   queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class unpack_bottom>(range<1>(x * depth), [=](id<1> idx) {
          const int offset = x * (halo_depth - depth);
          field[offset + idx[0]] = buffer[idx[0]];
        });
      })
      .wait_and_throw();
}

// Unpacks the left halo buffer(s)
void unpack_left(const int x,          //
                 const int y,          //
                 const int halo_depth, //
                 SyclBuffer &buffer,   //
                 SyclBuffer &field,    //
                 const int depth,      //
                 queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class unpack_left>(range<1>(y * depth), [=](id<1> idx) {
          const auto lines = idx[0] / depth;
          const auto offset = halo_depth - depth + lines * (x - depth);
          field[offset + idx[0]] = buffer[idx[0]];
        });
      })
      .wait_and_throw();
}

// Unpacks the right halo buffer(s)
void unpack_right(const int x,          //
                  const int y,          //
                  const int halo_depth, //
                  SyclBuffer &buffer,   //
                  SyclBuffer &field,    //
                  const int depth,      //
                  queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class unpack_right>(range<1>(y * depth), [=](id<1> idx) {
          const auto lines = idx[0] / depth;
          const auto offset = x - halo_depth + lines * (x - depth);
          field[offset + idx[0]] = buffer[idx[0]];
        });
      })
      .wait_and_throw();
}
