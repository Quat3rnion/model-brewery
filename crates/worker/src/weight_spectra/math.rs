use super::*;
use std::f64::consts::TAU;

const TOP: usize = 32;
const WIDTH: usize = TOP + 8;
pub(super) const MAX_AXIS: usize = 1 << 17;
const TILE_CELLS: usize = 1 << 16;

pub(super) fn measure(
    rows: usize,
    cols: usize,
    mut read: impl FnMut(usize, usize) -> Result<DMatrix<f64>>,
) -> Result<Metrics> {
    let mut energy = 0.;
    let (method, s) = if rows.min(cols) <= 128 && rows * cols <= TILE_CELLS {
        let w = read(0, rows)?;
        energy = w.norm_squared();
        ("exact", singular_values(w)?)
    } else {
        // Seeded Gaussian range finder with one subspace iteration (Halko et al.,
        // SIAM Review 2011). Stream row tiles; never materialize W or W^T.
        let width = WIDTH.min(rows).min(cols);
        let rows_per_tile = (TILE_CELLS / cols).max(1);
        let mut tiles = |f: &mut dyn FnMut(usize, DMatrix<f64>)| -> Result<()> {
            for offset in (0..rows).step_by(rows_per_tile) {
                f(offset, read(offset, rows_per_tile.min(rows - offset))?);
            }
            Ok(())
        };
        let mut state = 0xa3c5_74f2_691b_08e7_u64;
        let omega = DMatrix::from_fn(cols, width, |_, _| {
            let u = uniform(&mut state);
            let v = uniform(&mut state);
            (-2. * u.ln()).sqrt() * (TAU * v).cos()
        });
        let mut y = DMatrix::zeros(rows, width);
        tiles(&mut |i, w| {
            energy += w.norm_squared();
            apply(i, &w, &omega, &mut y);
        })?;
        drop(omega);
        orthonormalize(&mut y);
        let mut z = DMatrix::zeros(cols, width);
        tiles(&mut |i, w| transpose_apply(i, &w, &y, &mut z))?;
        orthonormalize(&mut z);
        tiles(&mut |i, w| apply(i, &w, &z, &mut y))?;
        orthonormalize(&mut y);
        z.fill(0.);
        tiles(&mut |i, w| transpose_apply(i, &w, &y, &mut z))?;
        let mut s = singular_values(z)?;
        s.truncate(TOP);
        ("randomized_top_32", s)
    };
    let total: f64 = s.iter().sum();
    let effective_rank = if total == 0. {
        0.
    } else {
        (-s.iter()
            .filter(|&&x| x > 0.)
            .map(|x| {
                let p = x / total;
                p * p.ln()
            })
            .sum::<f64>())
        .exp()
    };
    let fit = alpha_fit(&s);
    Ok(Metrics {
        method,
        effective_rank,
        stable_rank: if s[0] == 0. {
            0.
        } else {
            energy / s[0].powi(2)
        },
        captured_energy_fraction: (energy > 0.)
            .then(|| (s.iter().map(|x| x.powi(2)).sum::<f64>() / energy).min(1.)),
        singular_values: s,
        alpha: fit.map(|f| f.1),
        alpha_tail_count: fit.map(|f| f.2),
        alpha_xmin: fit.map(|f| f.3),
        alpha_ks: fit.map(|f| f.0),
    })
}

fn apply(i: usize, w: &DMatrix<f64>, basis: &DMatrix<f64>, out: &mut DMatrix<f64>) {
    out.rows_mut(i, w.nrows()).gemm(1., w, basis, 0.);
}

fn transpose_apply(i: usize, w: &DMatrix<f64>, basis: &DMatrix<f64>, out: &mut DMatrix<f64>) {
    out.gemm(1., &w.transpose(), &basis.rows(i, w.nrows()), 1.);
}

fn orthonormalize(matrix: &mut DMatrix<f64>) {
    // Two modified Gram-Schmidt passes. Keep dependent columns zero.
    let n = matrix.nrows();
    let scale = matrix.norm();
    for i in 0..matrix.ncols() {
        let (previous, rest) = matrix.as_mut_slice().split_at_mut(i * n);
        let current = &mut rest[..n];
        for _ in 0..2 {
            for p in previous.chunks_exact(n) {
                let dot: f64 = p.iter().zip(current.iter()).map(|(a, b)| a * b).sum();
                for (v, p) in current.iter_mut().zip(p) {
                    *v -= dot * p;
                }
            }
        }
        let norm = current.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= scale * 1e-13 {
            current.fill(0.);
        } else {
            current.iter_mut().for_each(|x| *x /= norm);
        }
    }
}

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((x ^ (x >> 31)) >> 12) as f64 * 2f64.powi(-52) + 2f64.powi(-53)
}

fn singular_values(w: DMatrix<f64>) -> Result<Vec<f64>> {
    let svd = w
        .try_svd(false, false, f64::EPSILON * 5., 8192)
        .context("SVD did not converge")?;
    let s = svd.singular_values.as_slice().to_vec();
    ensure!(s.iter().all(|x| x.is_finite()), "non-finite singular value");
    Ok(s)
}

fn alpha_fit(s: &[f64]) -> Option<(f64, f64, usize, f64)> {
    // Martin & Mahoney, JMLR 22:1-73 (2021), Sec. 4, https://jmlr.org/papers/v22/20-410.html
    // Continuous CSN fit to eigenvalues s^2: choose xmin by minimum KS.
    // Exclude numerical zeros and require at least eight tail samples.
    let logs: Vec<_> = s
        .iter()
        .rev()
        .filter(|&&x| x > s[0] * 1e-12)
        .map(|x| 2. * x.ln())
        .collect();
    let mut best = None;
    for i in 0..logs.len().saturating_sub(7) {
        if i > 0 && logs[i] == logs[i - 1] {
            continue;
        }
        let tail = &logs[i..];
        let n = tail.len() as f64;
        let mean = tail.iter().map(|x| x - tail[0]).sum::<f64>() / n;
        if mean <= 1e-12 {
            continue;
        }
        let ks = tail
            .iter()
            .enumerate()
            .map(|(j, x)| {
                let cdf = -(-(x - tail[0]) / mean).exp_m1();
                let j = j as f64;
                (j / n - cdf).abs().max(((j + 1.) / n - cdf).abs())
            })
            .fold(0., f64::max);
        if best.is_none_or(|(d, _, _, _)| ks < d) {
            best = Some((ks, 1. + 1. / mean, tail.len(), tail[0].exp()));
        }
    }
    best
}
