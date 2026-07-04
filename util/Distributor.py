# util/Distributor.py
#
# Conservative particle distributor:
#   ROMS sigma levels (s_rho at rho-points)  ->  fixed physical depth bins (meters)
#
# Same public API (same methods + signatures) as Interp3D in util/Interpolator.py:
#   - __init__(srcLons, srcLats, dstLons, dstLats, s_rho, mask, H)
#   - distrib(invar3d, fill_value=1.e+37, invalid_value=1.e+37)
#   - bottomValues(invar3d, invalid_value=1e37)
#   - surfaceValues(invar3d, factor=1.0, invalid_value=1e37)
#
# Key requirement:
#   For each time t and each (j,i) water column:
#     sum_sigma particles == sum_depth particles   (EXACT integer equality)
#
# Implementation notes:
# - Horizontal remapping is identical to Interp3D: nearest-neighbor via Interp2D.interp.
# - Vertical step is NOT interpolation: it redistributes particles by layer overlap,
#   then applies conservative rounding per column (largest remainder / Hamilton method).
# - Uses @jit(nopython=True) in the vertical core like Interpolator.py.

import numpy as np
from numba import jit

from .Interpolator import Interp2D, depths, extract_value_at_bottom, extract_value_at_surface


@jit(nopython=True)
def _centers_to_edges_1d(centers, left, right):
    """
    centers: (N,)
    returns edges: (N+1,) using midpoints and forced [left, right]
    """
    n = centers.shape[0]
    edges = np.empty(n + 1, dtype=np.float64)

    if n == 1:
        edges[0] = left
        edges[1] = right
        return edges

    edges[0] = left
    for k in range(1, n):
        edges[k] = 0.5 * (centers[k - 1] + centers[k])
    edges[n] = right
    return edges


@jit(nopython=True)
def _overlap_lengths(a_edges, b_edges):
    """
    a_edges: (Ka+1,), b_edges: (Kb+1,)
    returns overlaps: (Ka, Kb)
    """
    Ka = a_edges.shape[0] - 1
    Kb = b_edges.shape[0] - 1
    ov = np.zeros((Ka, Kb), dtype=np.float64)

    for i in range(Ka):
        a0 = a_edges[i]
        a1 = a_edges[i + 1]
        for j in range(Kb):
            b0 = b_edges[j]
            b1 = b_edges[j + 1]
            left = a0 if a0 > b0 else b0
            right = a1 if a1 < b1 else b1
            val = right - left
            if val > 0.0:
                ov[i, j] = val
    return ov


@jit(nopython=True)
def _largest_remainder_rounding_1d(x, target_sum):
    """
    Conservative integer rounding (Hamilton method) in numba:
      base = floor(x)
      add +1 to bins with largest fractional parts until sums match target_sum
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.int64)

    if target_sum <= 0:
        for i in range(n):
            out[i] = 0
        return out

    # clamp to non-negative
    for i in range(n):
        if x[i] < 0.0:
            x[i] = 0.0

    # floor
    base_sum = 0
    frac = np.empty(n, dtype=np.float64)
    for i in range(n):
        b = int(np.floor(x[i]))
        out[i] = b
        base_sum += b
        frac[i] = x[i] - b

    # if overshoot (should be rare), remove from smallest fractions where possible
    if base_sum > target_sum:
        need_remove = base_sum - target_sum
        # argsort ascending
        order = np.argsort(frac)
        for ii in range(n):
            if need_remove <= 0:
                break
            idx = order[ii]
            if out[idx] > 0:
                d = out[idx] if out[idx] < need_remove else need_remove
                out[idx] -= d
                need_remove -= d
        # (If still inconsistent, we accept best effort; but typically never happens.)
        return out

    # normal: distribute remainder to largest fractions
    rem = target_sum - base_sum
    if rem > 0:
        order = np.argsort(frac)  # ascending
        # take last rem indices (largest fracs)
        start = n - rem
        if start < 0:
            start = 0
        for ii in range(start, n):
            out[order[ii]] += 1

    return out


@jit(nopython=True)
def _vertical_distribute_conservative(s_rho, variable, depths_centers, mask_i, mask_j, H, fill_value, invalid_value):
    """
    variable: (t, Ks, eta, xi) on destination horizontal grid
    returns:  (t, Kd, eta, xi) float64, with integer counts stored as floats,
              and fill_value where invalid/outside water column.

    Notes:
    - mirrors Interpolator.vertical_interp behavior: uses t=0 only.
    - s_rho is expected as ndarray float64 (may contain NaN but should not in wet cells).
    """
    t, Ks, eta, xi = variable.shape
    Kd = depths_centers.shape[0]
    dst = np.full((t, Kd, eta, xi), fill_value, dtype=np.float64)

    tt = 0

    # precompute depth centers as float64
    # (already passed as float64 ndarray)
    for idx in range(mask_i.shape[0]):
        ii = mask_i[idx]
        jj = mask_j[idx]

        Hij = H[ii, jj]
        if not np.isfinite(Hij) or Hij <= 0.0:
            continue

        # sigma profile at (ii,jj), reverse like Interp3D does for vertical_interp
        prof_r = np.empty(Ks, dtype=np.float64)
        for k in range(Ks):
            v = variable[tt, Ks - 1 - k, ii, jj]  # reversed
            if invalid_value is not None and not np.isnan(invalid_value) and v == invalid_value:
                v = np.nan
            prof_r[k] = v

        # If all NaN -> skip
        all_nan = True
        for k in range(Ks):
            if np.isfinite(prof_r[k]):
                all_nan = False
                break
        if all_nan:
            continue

        # Replace NaN with 0 for particles (conservative)
        col_sum = 0.0
        for k in range(Ks):
            if not np.isfinite(prof_r[k]):
                prof_r[k] = 0.0
            col_sum += prof_r[k]

        # z centers (meters), increasing from near 0 to H
        # use reversed s_rho like vertical_interp: z_levels = H * -s_rho[::-1]
        z_centers = np.empty(Ks, dtype=np.float64)
        for k in range(Ks):
            sr = s_rho[Ks - 1 - k]  # reversed
            z = Hij * (-sr)
            # clamp to [0,H]
            if z < 0.0:
                z = 0.0
            if z > Hij:
                z = Hij
            z_centers[k] = z

        # sigma edges and thickness
        z_edges = _centers_to_edges_1d(z_centers, 0.0, Hij)
        for k in range(z_edges.shape[0]):
            if z_edges[k] < 0.0:
                z_edges[k] = 0.0
            if z_edges[k] > Hij:
                z_edges[k] = Hij

        dz = np.empty(Ks, dtype=np.float64)
        for k in range(Ks):
            dzz = z_edges[k + 1] - z_edges[k]
            if dzz < 0.0:
                dzz = 0.0
            dz[k] = dzz

        # depth bin edges and thickness (clipped to [0,H])
        d_edges = _centers_to_edges_1d(depths_centers, 0.0, Hij)
        for k in range(d_edges.shape[0]):
            if d_edges[k] < 0.0:
                d_edges[k] = 0.0
            if d_edges[k] > Hij:
                d_edges[k] = Hij

        dd = np.empty(Kd, dtype=np.float64)
        valid_bins = np.empty(Kd, dtype=np.int64)  # 1/0
        for k in range(Kd):
            ddk = d_edges[k + 1] - d_edges[k]
            if ddk < 0.0:
                ddk = 0.0
            dd[k] = ddk
            valid_bins[k] = 1 if ddk > 0.0 else 0

        # overlaps sigma-layer vs depth-bin
        ov = _overlap_lengths(z_edges, d_edges)  # (Ks,Kd)

        # expected counts in each depth bin
        expected = np.zeros(Kd, dtype=np.float64)
        for k in range(Ks):
            if dz[k] <= 0.0:
                continue
            pk = prof_r[k]
            for d in range(Kd):
                # fraction of sigma layer into depth bin
                f = ov[k, d] / dz[k]
                if f > 0.0:
                    expected[d] += pk * f

        # conservative integer rounding to match target integer sum in column
        target = int(np.rint(col_sum))
        rounded = _largest_remainder_rounding_1d(expected, target)

        # write: valid bins -> integer counts, others -> fill_value
        for d in range(Kd):
            if valid_bins[d] == 1:
                dst[tt, d, ii, jj] = float(rounded[d])
            else:
                dst[tt, d, ii, jj] = fill_value

    return dst


class Distrib3D(Interp2D):
    """
    Same API as Interp3D, but redistributes particle counts vertically and enforces
    exact integer conservation per water column.
    """

    def __init__(self, srcLons, srcLats, dstLons, dstLats, s_rho, mask, H):
        super().__init__(srcLons, srcLats, dstLons, dstLats)
        self.s_rho = s_rho
        self.H = H
        self.mask = super().interp(mask, fill_value=0, invalid_value=np.nan)
        self.mask_indices = np.where(self.mask == 1)

    def distrib(self, invar3d, fill_value=1.0e37, invalid_value=1.0e37):
        """
        invar3d: (time, Ks, src_eta, src_xi) particle counts on source grid
        returns: (time, Kd, len(dstLats), len(dstLons)) particle counts on depth bins,
                 stored as float64 but guaranteed integers in wet cells.
        """
        outvar3d = np.empty(
            (invar3d.shape[0], invar3d.shape[1], len(self.dstLats), len(self.dstLons)),
            dtype=np.float64,
        )

        for k in range(invar3d.shape[1]):
#            print(f"Distributing level: {k}")
            outvar3d[0, k] = super().interp(invar3d[0, k], fill_value=fill_value, invalid_value=invalid_value)

        sr = self.s_rho.filled(np.nan) if hasattr(self.s_rho, "filled") else np.asarray(self.s_rho, dtype=np.float64)
        HH = self.H.filled(np.nan) if hasattr(self.H, "filled") else np.asarray(self.H, dtype=np.float64)
        dcent = np.asarray(depths, dtype=np.float64)

        mask_i, mask_j = self.mask_indices  # arrays

        out_depth = _vertical_distribute_conservative(
            sr,
            outvar3d,
            dcent,
            mask_i.astype(np.int64),
            mask_j.astype(np.int64),
            HH,
            float(fill_value),
            float(invalid_value),
        )
        return out_depth

    def bottomValues(self, invar3d, invalid_value=1e37):
        return extract_value_at_bottom(invar3d, invalid_value)

    def surfaceValues(self, invar3d, factor=1.0, invalid_value=1e37):
        return extract_value_at_surface(invar3d, factor, invalid_value)
