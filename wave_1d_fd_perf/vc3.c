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
  float f_xx;
  float * tmp;

  for (step = 0; step < num_steps; step++) {
#pragma omp parallel for private(f_xx)
    for (i = 8; i < nx_padded - 8; i++) {
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

    for (i = 0; i < num_sources; i++) {
      sx = sources_x[i] + 8;
      fp[sx] += (model_padded[sx] * model_padded[sx] * dt * dt
          * sources[i * source_len + step]);
    }

    tmp = f;
    f = fp;
    fp = tmp;
  }
}
