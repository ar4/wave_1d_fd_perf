#include <stdio.h>
#include <omp.h>

void step(
    float * restrict f,
    float * restrict fp,
    const int nx_padded,
    const float * restrict const model_padded,
    const float dt,
    const float dx,
    const float * restrict const sources,
    const int * restrict const sources_x,
    const int num_sources,
    const int source_len,
    const int num_steps) {

  int step;
  int i;
  int sx;
  int threadIdx;
  int thread_start;
  int thread_end;
  int thread_source_start;
  int thread_source_end;
  int per_thread;
  float f_xx;
  float * tmp;

#pragma omp parallel private(thread_start, thread_end, thread_source_start,\
    thread_source_end, step, i, f_xx, tmp, threadIdx, per_thread) \
    firstprivate(f, fp)
  {
    per_thread = (int) ((nx_padded - 16.0) / omp_get_num_threads() + 0.5);
    threadIdx = omp_get_thread_num();

    thread_start = 8 + per_thread * threadIdx;
    thread_end = thread_start + per_thread;
    thread_end = thread_end < nx_padded - 8 ? thread_end : nx_padded - 8;

    thread_source_start = -1;
    thread_source_end = -1;
    // Find the first source index that is within this thread's range
    for (i = 0; i < num_sources; i++) {
      if (sources_x[i] < thread_start) continue;
      if (sources_x[i] > thread_end) break;
      thread_source_start = i;
      thread_source_end = i + 1;
      break;
    }

    // Find the last source index that is within this thread's range
    if (thread_source_end >= 0) {
      for (i = thread_source_end; i < num_sources; i++) {
        if (sources_x[i] > thread_end) break;
        thread_source_end = i + 1;
      }
    }

    for (step = 0; step < num_steps; step++) {
      for (i = thread_start; i < thread_end; i++) {
        f_xx = (
            -735*f[i-8]+15360*f[i-7]
            -156800*f[i-6]+1053696*f[i-5]
            -5350800*f[i-4]+22830080*f[i-3]
            -94174080*f[i-2]+538137600*f[i-1]
            -924708642*f[i+0]
            +538137600*f[i+1]-94174080*f[i+2]
            +22830080*f[i+3]-5350800*f[i+4]
            +1053696*f[i+5]-156800*f[i+6]
            +15360*f[i+7]-735*f[i+8])/(302702400*dx*dx);
        fp[i] = (model_padded[i] * model_padded[i] * dt * dt * f_xx
            + 2 * f[i] - fp[i]);
      }

      for (i = thread_source_start; i < thread_source_end; i++) {
        sx = sources_x[i] + 8;
        fp[sx] += (model_padded[sx] * model_padded[sx] * dt * dt
            * sources[i * source_len + step]);
      }

      tmp = f;
      f = fp;
      fp = tmp;
#pragma omp barrier
    }
  }

}
