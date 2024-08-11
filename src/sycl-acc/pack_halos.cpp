#include <sstream>

#include "chunk.h"
#include "comms.h"
#include "shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Packs the top halo buffer(s)
void pack_top(const int x,            //
              const int y,            //
              const int depth,        //
              const int halo_depth,   //
              SyclBuffer &fieldBuff,  //
              SyclBuffer &bufferBuff, //
              const int buffer_offset, queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::write>(h);
    auto field = fieldBuff.get_access<access::mode::read>(h);
    h.parallel_for<class pack_top>(range<1>(x * depth), [=](id<1> idx) {
      const int offset = x * (y - halo_depth - depth);
      buffer[idx[0] + buffer_offset] = field[offset + idx[0]];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Packs the bottom halo buffer(s)
void pack_bottom(const int x,            //
                 const int y,            //
                 const int depth,        //
                 const int halo_depth,   //
                 SyclBuffer &fieldBuff,  //
                 SyclBuffer &bufferBuff, //
                 const int buffer_offset, queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::write>(h);
    auto field = fieldBuff.get_access<access::mode::read>(h);
    h.parallel_for<class pack_bottom>(range<1>(x * depth), [=](id<1> idx) {
      const int offset = x * halo_depth;
      buffer[idx[0] + buffer_offset] = field[offset + idx[0]];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Packs the left halo buffer(s)
void pack_left(const int x,            //
               const int y,            //
               const int depth,        //
               const int halo_depth,   //
               SyclBuffer &fieldBuff,  //
               SyclBuffer &bufferBuff, //
               const int buffer_offset, queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::write>(h);
    auto field = fieldBuff.get_access<access::mode::read>(h);
    h.parallel_for<class pack_left>(range<1>(y * depth), [=](id<1> idx) {
      const auto lines = idx[0] / depth;
      const auto offset = halo_depth + lines * (x - depth);
      buffer[idx[0] + buffer_offset] = field[offset + idx[0]];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Packs the right halo buffer(s)
void pack_right(const int x,            //
                const int y,            //
                const int depth,        //
                const int halo_depth,   //
                SyclBuffer &fieldBuff,  //
                SyclBuffer &bufferBuff, //
                const int buffer_offset, queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::write>(h);
    auto field = fieldBuff.get_access<access::mode::read>(h);
    h.parallel_for<class pack_right>(range<1>(y * depth), [=](id<1> idx) {
      const auto lines = idx[0] / depth;
      const auto offset = x - halo_depth - depth + lines * (x - depth);
      buffer[idx[0] + buffer_offset] = field[offset + idx[0]];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Unpacks the top halo buffer(s)
void unpack_top(const int x,            //
                const int y,            //
                const int depth,        //
                const int halo_depth,   //
                SyclBuffer &fieldBuff,  //
                SyclBuffer &bufferBuff, //
                const int buffer_offset, queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read>(h);
    auto field = fieldBuff.get_access<access::mode::write>(h);
    h.parallel_for<class unpack_top>(range<1>(x * depth), [=](id<1> idx) {
      const int offset = x * (y - halo_depth);
      field[offset + idx[0]] = buffer[idx[0] + buffer_offset];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Unpacks the bottom halo buffer(s)
void unpack_bottom(const int x,            //
                   const int y,            //
                   const int depth,        //
                   const int halo_depth,   //
                   SyclBuffer &fieldBuff,  //
                   SyclBuffer &bufferBuff, //
                   const int buffer_offset, queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read>(h);
    auto field = fieldBuff.get_access<access::mode::write>(h);
    h.parallel_for<class unpack_bottom>(range<1>(x * depth), [=](id<1> idx) {
      const int offset = x * (halo_depth - depth);
      field[offset + idx[0]] = buffer[idx[0] + buffer_offset];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Unpacks the left halo buffer(s)
void unpack_left(const int x,            //
                 const int y,            //
                 const int depth,        //
                 const int halo_depth,   //
                 SyclBuffer &fieldBuff,  //
                 SyclBuffer &bufferBuff, //
                 const int buffer_offset, queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read>(h);
    auto field = fieldBuff.get_access<access::mode::write>(h);
    h.parallel_for<class unpack_left>(range<1>(y * depth), [=](id<1> idx) {
      const auto lines = idx[0] / depth;
      const auto offset = halo_depth - depth + lines * (x - depth);
      field[offset + idx[0]] = buffer[idx[0] + buffer_offset];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Unpacks the right halo buffer(s)
void unpack_right(const int x,            //
                  const int y,            //
                  const int depth,        //
                  const int halo_depth,   //
                  SyclBuffer &fieldBuff,  //
                  SyclBuffer &bufferBuff, //
                  const int buffer_offset, queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read>(h);
    auto field = fieldBuff.get_access<access::mode::write>(h);
    h.parallel_for<class unpack_right>(range<1>(y * depth), [=](id<1> idx) {
      const auto lines = idx[0] / depth;
      const auto offset = x - halo_depth + lines * (x - depth);
      field[offset + idx[0]] = buffer[idx[0] + buffer_offset];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

void run_pack_or_unpack(Chunk *chunk, Settings &settings, int depth, int face, bool pack, FieldBufferType field, FieldBufferType buffer,
                        int offset) {
  START_PROFILING(settings.kernel_profile);
  switch (face) {
    case CHUNK_LEFT:
      if (pack) pack_left(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset, *chunk->ext->device_queue);
      else
        unpack_left(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset, *chunk->ext->device_queue);
      break;
    case CHUNK_RIGHT:
      if (pack) pack_right(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset, *chunk->ext->device_queue);
      else
        unpack_right(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset, *chunk->ext->device_queue);
      break;
    case CHUNK_TOP:
      if (pack) pack_top(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset, *chunk->ext->device_queue);
      else
        unpack_top(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset, *chunk->ext->device_queue);
      break;
    case CHUNK_BOTTOM:
      if (pack) pack_bottom(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset, *chunk->ext->device_queue);
      else
        unpack_bottom(chunk->x, chunk->y, depth, settings.halo_depth, *field, *buffer, offset, *chunk->ext->device_queue);
      break;
    default: die(__LINE__, __FILE__, "Incorrect face provided: %d.\n", face);
  }
  STOP_PROFILING(settings.kernel_profile, __func__);
}

#if !(defined(__HIPSYCL__) || defined(__OPENSYCL__))

template <typename A> decltype(auto) get_native_ptr_or_throw(sycl::interop_handle &ih, A accessor) {
  using sycl::backend;
  using T = std::remove_cv_t<typename decltype(accessor)::value_type>;
  switch (ih.get_backend()) {
    case backend::ext_oneapi_level_zero: return reinterpret_cast<T *>(ih.get_native_mem<backend::ext_oneapi_level_zero>(accessor));
  #ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
    case backend::ext_oneapi_cuda: return reinterpret_cast<T *>(ih.get_native_mem<backend::ext_oneapi_cuda>(accessor));
  #endif
  #ifdef SYCL_EXT_ONEAPI_BACKEND_HIP
    case backend::ext_oneapi_hip: return reinterpret_cast<T *>(ih.get_native_mem<backend::ext_oneapi_hip>(accessor));
  #endif
    default:
      std::stringstream ss;
      ss << "backend " << ih.get_backend() << " does not support a pointer-based sycl::interop_handle::get_native_mem";
      throw std::logic_error(ss.str());
  }
}
#endif

void run_send_recv_halo(Chunk *chunk, Settings &settings,                                 //
                        FieldBufferType src_send_buffer, FieldBufferType src_recv_buffer, //
                        StagingBufferType, StagingBufferType,                             //
                        int buffer_len, int neighbour,                                    //
                        int send_tag, int recv_tag,                                       //
                        MPI_Request *send_request, MPI_Request *recv_request) {

#ifdef USE_HOSTTASK
  if (settings.staging_buffer) {
    chunk->ext->device_queue->submit([&](sycl::handler &h) {
      auto snd_buffer_acc = src_send_buffer->get_host_access(h, sycl::read_only);
      auto rcv_buffer_acc = src_recv_buffer->get_host_access(h, sycl::write_only);
      h.host_task([=, &settings]() {
        send_recv_message(settings,                     //
                          snd_buffer_acc.get_pointer(), //
                          rcv_buffer_acc.get_pointer(), //
                          buffer_len, neighbour, send_tag, recv_tag, send_request, recv_request);
      });
    });
  } else {
    chunk->ext->device_queue->submit([&](sycl::handler &h) {
      auto snd_buffer_acc = src_send_buffer->get_access<access_mode::read>(h);
      auto rcv_buffer_acc = src_recv_buffer->get_access<access_mode::write>(h);
      h.host_task([=, &settings](sycl::interop_handle ih) {            // XXX pass handle arg here as copy, not ref!
        send_recv_message(settings,                                    //
                          get_native_ptr_or_throw(ih, snd_buffer_acc), //
                          get_native_ptr_or_throw(ih, rcv_buffer_acc), //
                          buffer_len, neighbour, send_tag, recv_tag, send_request, recv_request);
      });
    });
  }
#else
  if (settings.staging_buffer) {
    chunk->ext->device_queue->wait_and_throw();
    send_recv_message(settings, //
                      host_accessor<double, 1, access_mode::read_write>{*src_send_buffer, buffer_len}.get_pointer(),
                      host_accessor<double, 1, access_mode::read_write>{*src_recv_buffer, buffer_len}.get_pointer(), buffer_len, neighbour,
                      send_tag, recv_tag, send_request, recv_request);
  } else {
  #if defined(__HIPSYCL__) || defined(__OPENSYCL__)
    //    chunk->ext->device_queue->wait_and_throw();
    auto d = chunk->ext->device_queue->get_device();
    // Construct the buffers so that get_pointer is not nullptr, only happens once per rank for the lifetime of the program
    if (!src_recv_buffer->get_pointer(d))
      chunk->ext->device_queue->submit([&](sycl::handler &h) { h.update(sycl::accessor{*src_recv_buffer, h}); }).wait_and_throw();
    if (!src_send_buffer->get_pointer(d))
      chunk->ext->device_queue->submit([&](sycl::handler &h) { h.update(sycl::accessor{*src_send_buffer, h}); }).wait_and_throw();
    // We can't use host_task here, but since we can pull out the pointers directly, if we synchronise before MPI_Waitall
    // the desired concurrency should still be there
    chunk->ext->device_queue->submit([&](sycl::handler &h) {
                              h.update(sycl::accessor{*src_send_buffer, h, sycl::read_only});
                            })
        .wait_and_throw();
    chunk->ext->device_queue->submit([&](sycl::handler &h) {
                              h.update(sycl::accessor{*src_recv_buffer, h, sycl::write_only});
                            })
        .wait_and_throw();
    send_recv_message(settings,                        //
                      src_send_buffer->get_pointer(d), //
                      src_recv_buffer->get_pointer(d), //
                      buffer_len, neighbour, send_tag, recv_tag, send_request, recv_request);
  #else
    throw std::logic_error("host_task is disabled and staging is also disabled, this won't work");
  #endif
  }
#endif
}

void run_before_waitall_halo(Chunk *chunk, Settings &settings) {
#ifdef USE_HOSTTASK
  chunk->ext->device_queue->wait_and_throw();
#else
  if (settings.staging_buffer) {
    // drop-through to waitall directly
  } else {
  #if defined(__HIPSYCL__) || defined(__OPENSYCL__)
    chunk->ext->device_queue->wait_and_throw();
  #else
    throw std::logic_error("host_task is disabled and staging is also disabled, this won't work");
  #endif
  }
#endif
}
void run_restore_recv_halo(Chunk *, Settings &, FieldBufferType, StagingBufferType, int) {}
