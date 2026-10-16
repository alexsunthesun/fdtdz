"""Throughput benchmark for ``fdtdz`` across launch parameters.

Reports cell-updates per second for the full simulated domain (including the
z-PML), i.e. ``xx * yy * (zz + pml) * tt / seconds``.

Example:
  python bench/bench_fdtdz.py --sizes 400 800 --tt 2000 --sweep
"""

import argparse
import itertools
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

import fdtdz_jax


def _ramped_sin(wavelength, ramp, dt, tt, delay=4):
  t = (2 * np.pi * dt / wavelength) * np.arange(tt, dtype=np.float32)
  return ((1 + np.tanh(t / ramp - delay)) / 2) * np.sin(t)


def _absorption_mask(xx, yy, width, smoothness):
  def profile(n):
    center = (n - 1) / 2
    pos = np.arange(n, dtype=np.float32) + np.array([[0], [0.5]], np.float32)
    pos = np.abs(pos - center) - center + width
    return smoothness * np.maximum(pos, 0)**2

  x, y = profile(xx), profile(yy)
  return jnp.array(np.stack([x[1][:, None] + y[0][None, :],
                             x[0][:, None] + y[1][None, :],
                             x[0][:, None] + y[0][None, :]]))


def _pml_sigma(pml_widths, zz, ln_R=16.0, m=4.0):
  z = np.arange(zz, dtype=np.float32)[:, None] + np.array([[0, 0.5]])
  z = np.maximum(np.maximum(pml_widths[0] - z, z - (zz - pml_widths[1])), 0)
  w = np.array(pml_widths)[np.argmax([pml_widths[0] - z, z], axis=0)]
  sigma = (m + 1) * ln_R / (2 * np.maximum(w, 1)) * (z / np.maximum(w, 1))**m
  return jnp.array(sigma.astype(np.float32))


def make_inputs(xx, yy, tt, use_reduced_precision, pml_widths=(8, 8), dt=0.5,
                abs_width=50):
  zz = (128 if use_reduced_precision else 64) - sum(pml_widths)
  source_field = jnp.zeros((2, xx, 1, zz)).at[0, xx // 2, 0, zz // 2].set(1.0)
  waveform = jnp.broadcast_to(_ramped_sin(10.0, 4, dt, tt)[:, None], (tt, 2))
  return dict(
      # Small output subvolume, so timing is dominated by the update kernel.
      epsilon=jnp.ones((3, 8, 8, 8)),
      dt=dt,
      source_field=source_field,
      source_waveform=waveform,
      source_position=yy // 2,
      absorption_mask=_absorption_mask(xx, yy, abs_width, 1e-2),
      pml_kappa=jnp.ones((zz, 2)),
      pml_sigma=_pml_sigma(pml_widths, zz),
      pml_alpha=0.05 * jnp.ones((zz, 2)),
      pml_widths=pml_widths,
      output_steps=(tt - 1, tt, 1),
      use_reduced_precision=use_reduced_precision,
      offset=(xx // 2, yy // 2, zz // 2 - 4),
  ), zz + sum(pml_widths)


def run(inputs, launch_params):
  return fdtdz_jax.fdtdz(**inputs, launch_params=launch_params)


def bench(xx, yy, tt, use_reduced_precision, launch_params, reps=3):
  inputs, zfull = make_inputs(xx, yy, tt, use_reduced_precision)
  run(inputs, launch_params).block_until_ready()  # Compile + warm-up.
  times = []
  for _ in range(reps):
    t0 = time.perf_counter()
    run(inputs, launch_params).block_until_ready()
    times.append(time.perf_counter() - t0)
  t = statistics.median(times)
  return t, xx * yy * zfull * tt / t / 1e9


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--sizes", type=int, nargs="+", default=[400, 800, 1200])
  p.add_argument("--tt", type=int, default=2000)
  p.add_argument("--reps", type=int, default=3)
  p.add_argument("--sweep", action="store_true",
                 help="Sweep grid/spacing/cc instead of using the preset.")
  p.add_argument("--precision", choices=["f16", "f32", "both"], default="both")
  args = p.parse_args()

  print(f"device: {jax.devices()[0].device_kind}, jax {jax.__version__}")
  precisions = {"f16": [True], "f32": [False], "both": [True, False]}
  if args.sweep:
    grids = [(10, 7), (7, 10), (5, 14)]
    configs = [((2, 4), g, s, cc) for g, s, cc in itertools.product(
        grids, (2, 4), ((8, 0), (12, 0)))]
  else:
    configs = [None]

  print(f"{'prec':4} {'size':>9} {'grid':>8} {'sp':>2} {'cc':>6} "
        f"{'time (s)':>9} {'Gcell/s':>8}")
  for rp, size, lp in itertools.product(precisions[args.precision],
                                        args.sizes, configs):
    t, rate = bench(size, size, args.tt, rp, lp, args.reps)
    lp_ = lp or fdtdz_jax.fdtdz_jax._preset_launch_params(
        jax.devices()[0].device_kind)
    print(f"{'f16' if rp else 'f32':4} {size:>4}x{size:<4} "
          f"{str(lp_[1]):>8} {lp_[2]:>2} {str(lp_[3]):>6} "
          f"{t:9.3f} {rate:8.2f}", flush=True)


if __name__ == "__main__":
  main()
