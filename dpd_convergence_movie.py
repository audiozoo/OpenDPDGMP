"""
Movie showing how the PA output PSD evolves with each iterative DPD update.

The Indirect Learning Architecture (ILA) is applied iteratively:
  Iteration 0: PA output with no DPD (baseline)
  Iteration k: Apply DPD(w_k) → PA → re-identify w_{k+1} from new I/O pair

Each frame plots the PSD of the PA output at that iteration, all normalized
to 0 dB peak.  Saved as an MP4 movie.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import welch, firwin, kaiserord
from scipy.signal.windows import blackmanharris


# ===================================================================
# Core functions (copied from gmp_dpd_standalone.py for self-containment)
# ===================================================================

def _delay(x, d):
    N = len(x)
    out = np.zeros(N, dtype=np.complex128)
    if 0 <= d < N:
        out[d:] = x[:N - d]
    elif d < 0 and -d < N:
        out[:N + d] = x[-d:]
    return out


def build_gmp_basis(x, Ka, La, Kb, Lb, Mb, Kc, Lc, Mc):
    N = len(x)
    n_cols = Ka * La + Kb * Lb * Mb + Kc * Lc * Mc
    Phi = np.zeros((N, n_cols), dtype=np.complex128)
    col = 0
    for k in range(Ka):
        for q in range(La):
            xq = _delay(x, q)
            Phi[:, col] = xq * np.abs(xq) ** k
            col += 1
    for k in range(1, Kb + 1):
        for q in range(Lb):
            for l in range(1, Mb + 1):
                xq = _delay(x, q)
                xql = _delay(x, q + l)
                Phi[:, col] = xq * np.abs(xql) ** k
                col += 1
    for k in range(1, Kc + 1):
        for q in range(Lc):
            for l in range(1, Mc + 1):
                xq = _delay(x, q)
                xql = _delay(x, q - l)
                Phi[:, col] = xq * np.abs(xql) ** k
                col += 1
    return Phi


@dataclass
class GMPConfig:
    Ka: int = 5; La: int = 4
    Kb: int = 3; Lb: int = 4; Mb: int = 2
    Kc: int = 3; Lc: int = 4; Mc: int = 1
    @property
    def n_coeffs(self):
        return self.Ka * self.La + self.Kb * self.Lb * self.Mb + self.Kc * self.Lc * self.Mc


def iq_to_complex(iq):
    return iq[:, 0] + 1j * iq[:, 1]

def complex_to_iq(c):
    return np.column_stack([c.real, c.imag])

def compute_target_gain(pa_in, pa_out):
    return np.max(np.abs(iq_to_complex(pa_out))) / np.max(np.abs(iq_to_complex(pa_in)))

def identify_gmp(pa_in_iq, pa_out_iq, cfg, target_gain):
    x_c = iq_to_complex(pa_in_iq)
    z_c = iq_to_complex(pa_out_iq)
    Phi = build_gmp_basis(z_c / target_gain, cfg.Ka, cfg.La, cfg.Kb, cfg.Lb,
                          cfg.Mb, cfg.Kc, cfg.Lc, cfg.Mc)
    w, _, _, _ = np.linalg.lstsq(Phi, x_c, rcond=None)
    return w

def apply_dpd(input_iq, w, cfg):
    x_c = iq_to_complex(input_iq)
    Phi = build_gmp_basis(x_c, cfg.Ka, cfg.La, cfg.Kb, cfg.Lb,
                          cfg.Mb, cfg.Kc, cfg.Lc, cfg.Mc)
    return complex_to_iq(Phi @ w)

def nmse_db(pred, ref):
    err = pred - ref
    return 10 * np.log10(np.mean(err[:, 0]**2 + err[:, 1]**2) /
                         np.mean(ref[:, 0]**2 + ref[:, 1]**2))

def aclr_db(signal_iq, fs, channel_bw, adjacent_offset=None, nperseg=8192):
    """ACLR (lower, upper) in dB. Higher is better."""
    if adjacent_offset is None:
        adjacent_offset = channel_bw
    c = iq_to_complex(signal_iq) if signal_iq.ndim == 2 else signal_iq
    f, psd = welch(c, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                   return_onesided=False, scaling='density')
    df = f[1] - f[0]
    def band_power(f_lo, f_hi):
        return np.sum(psd[(f >= f_lo) & (f < f_hi)]) * df
    p_main  = band_power(-channel_bw / 2, channel_bw / 2)
    p_lower = band_power(-adjacent_offset - channel_bw / 2,
                         -adjacent_offset + channel_bw / 2)
    p_upper = band_power( adjacent_offset - channel_bw / 2,
                          adjacent_offset + channel_bw / 2)
    lo = 10 * np.log10(p_main / p_lower) if p_lower > 0 else np.inf
    hi = 10 * np.log10(p_main / p_upper) if p_upper > 0 else np.inf
    return lo, hi


# ===================================================================
# PA model
# ===================================================================

PA_COEFFS = {
    (1, 0):  1.0513 + 0.0904j,
    (1, 1): -0.0680 - 0.0023j,
    (1, 2):  0.0289 - 0.0054j,
    (3, 0): -0.0542 - 0.2900j,
    (3, 1):  0.2234 + 0.2317j,
    (3, 2): -0.0621 - 0.0932j,
    (5, 0): -0.9657 - 0.7028j,
    (5, 1): -0.2451 - 0.3735j,
    (5, 2):  0.1229 + 0.1508j,
}

def memory_polynomial_pa(x_iq, coeffs=PA_COEFFS, K=5, Q=2):
    z = iq_to_complex(x_iq)
    N = len(z)
    y = np.zeros(N, dtype=np.complex128)
    for k in range(1, K + 1, 2):
        for q in range(Q + 1):
            c_kq = coeffs.get((k, q), 0.0)
            if c_kq == 0.0:
                continue
            zq = _delay(z, q)
            y += c_kq * zq * np.abs(zq) ** (k - 1)
    return complex_to_iq(y)


# ===================================================================
# WCDMA signal generation
# ===================================================================

def generate_wcdma(n_samples, n_codes=16, chip_rate=3.84e6, fs=61.44e6, sf=16):
    oversample = int(round(fs / chip_rate))
    n_chips = n_samples // oversample + sf

    def _ovsf(sf_val, code_idx):
        h = np.array([[1]])
        while h.shape[0] < sf_val:
            h = np.block([[h, h], [h, -h]])
        return h[code_idx % sf_val]

    composite = np.zeros(n_chips, dtype=np.complex128)
    n_sym = n_chips // sf
    for c in range(n_codes):
        qpsk = (np.sign(np.random.randn(n_sym))
                + 1j * np.sign(np.random.randn(n_sym))) / np.sqrt(2)
        spread = np.repeat(qpsk, sf) * np.tile(_ovsf(sf, c + 1), n_sym)
        composite[:len(spread)] += spread
    composite /= np.sqrt(n_codes)

    upsampled = np.zeros(n_chips * oversample, dtype=np.complex128)
    upsampled[::oversample] = composite

    alpha = 0.22
    span = 12
    t = np.arange(-span * oversample, span * oversample + 1) / oversample
    eps = 1e-12
    h_rrc = np.where(
        np.abs(t) < eps, 1.0 - alpha + 4.0 * alpha / np.pi,
        np.where(
            np.abs(np.abs(t) - 1.0 / (4.0 * alpha)) < eps,
            alpha / np.sqrt(2) * ((1 + 2/np.pi) * np.sin(np.pi/(4*alpha))
                                  + (1 - 2/np.pi) * np.cos(np.pi/(4*alpha))),
            (np.sin(np.pi * t * (1 - alpha))
             + 4 * alpha * t * np.cos(np.pi * t * (1 + alpha))
            ) / (np.pi * t * (1 - (4 * alpha * t)**2 + eps))
        )
    )
    h_rrc /= np.sqrt(np.sum(h_rrc**2))
    h_rrc *= oversample

    out = np.convolve(upsampled, h_rrc, mode='full')
    rrc_out = out[len(h_rrc)//2 : len(h_rrc)//2 + n_samples]

    pb = chip_rate * (1 + alpha) / 2
    sb = pb + 0.5e6
    numtaps, beta = kaiserord(80, (sb - pb) / (fs / 2))
    if numtaps % 2 == 0:
        numtaps += 1
    h_chan = firwin(numtaps, (pb + sb) / 2, window=('kaiser', beta), fs=fs)
    chan_out = np.convolve(rrc_out, h_chan, mode='full')
    return chan_out[len(h_chan)//2 : len(h_chan)//2 + n_samples]


def generate_multicarrier_wcdma(n_samples, n_carriers=3, carrier_spacing=5e6,
                                n_codes=16, chip_rate=3.84e6, fs=61.44e6, sf=16):
    composite = np.zeros(n_samples, dtype=np.complex128)
    t = np.arange(n_samples) / fs
    f_centres = (np.arange(n_carriers) - (n_carriers - 1) / 2.0) * carrier_spacing
    for fc in f_centres:
        carrier = generate_wcdma(n_samples, n_codes=n_codes,
                                 chip_rate=chip_rate, fs=fs, sf=sf)
        composite += carrier * np.exp(1j * 2 * np.pi * fc * t)
    composite /= np.sqrt(n_carriers)
    return composite


# ===================================================================
# Main — iterative DPD and movie generation
# ===================================================================

def main():
    np.random.seed(42)

    # Signal parameters
    chip_rate = 3.84e6
    fs = 61.44e6
    n_carriers = 3
    carrier_spacing = 5e6
    carrier_bw_mhz = 5
    N = 1_000_000
    n_iterations = 10

    print(f"Generating {n_carriers}-carrier WCDMA signal ({N} samples)...")
    x_c = generate_multicarrier_wcdma(N, n_carriers=n_carriers,
                                       carrier_spacing=carrier_spacing, fs=fs)
    target_rms = 0.22
    x_c *= target_rms / np.sqrt(np.mean(np.abs(x_c)**2))
    x_iq = complex_to_iq(x_c)

    def pa(sig):
        return memory_polynomial_pa(sig)

    # Baseline: no DPD
    y_no_dpd = pa(x_iq)
    target_gain = compute_target_gain(x_iq, y_no_dpd)
    ideal_output = target_gain * x_iq

    cfg = GMPConfig(Ka=7, La=5, Kb=3, Lb=5, Mb=2, Kc=3, Lc=5, Mc=1)

    # PSD computation
    nperseg = 8192
    noverlap = nperseg * 3 // 4
    psd_window = blackmanharris(nperseg)

    def compute_psd_db(iq):
        c = iq_to_complex(iq) if iq.ndim == 2 else iq
        f, p = welch(c, fs=fs, nperseg=nperseg, window=psd_window,
                     noverlap=noverlap, return_onesided=False, scaling='density')
        idx = np.argsort(f)
        return f[idx] / 1e6, 10 * np.log10(p[idx] / np.max(p[idx]))

    # Compute PSDs for each iteration
    f_mhz, psd_ideal = compute_psd_db(ideal_output)
    _, psd_no_dpd = compute_psd_db(y_no_dpd)

    channel_bw_hz = n_carriers * carrier_bw_mhz * 1e6

    # frames: (iteration, psd_db, nmse, aclr_lower, aclr_upper)
    nmse_0 = nmse_db(y_no_dpd, ideal_output)
    aclr_lo_0, aclr_hi_0 = aclr_db(y_no_dpd, fs, channel_bw_hz)
    frames = [(0, psd_no_dpd, nmse_0, aclr_lo_0, aclr_hi_0)]
    print(f"  Iteration 0 (no DPD): NMSE = {nmse_0:+.2f} dB, "
          f"ACLR = {aclr_lo_0:.1f} / {aclr_hi_0:.1f} dB")

    # First identification from raw PA I/O
    w = identify_gmp(x_iq, y_no_dpd, cfg, target_gain)

    for it in range(1, n_iterations + 1):
        x_dpd = apply_dpd(x_iq, w, cfg)
        y_dpd = pa(x_dpd)
        nmse = nmse_db(y_dpd, ideal_output)
        aclr_lo, aclr_hi = aclr_db(y_dpd, fs, channel_bw_hz)
        _, psd = compute_psd_db(y_dpd)
        frames.append((it, psd, nmse, aclr_lo, aclr_hi))
        print(f"  Iteration {it}: NMSE = {nmse:+.2f} dB, "
              f"ACLR = {aclr_lo:.1f} / {aclr_hi:.1f} dB")

        w = identify_gmp(x_dpd, y_dpd, cfg, target_gain)

    # ---------------------------------------------------------------
    # Build movie  (2 subplots: PSD + ACLR vs iteration)
    # ---------------------------------------------------------------
    print(f"\nRendering movie ({len(frames)} frames)...")

    all_iters    = [f[0] for f in frames]
    all_aclr_lo  = [f[3] for f in frames]
    all_aclr_hi  = [f[4] for f in frames]
    aclr_min = min(min(all_aclr_lo), min(all_aclr_hi))
    aclr_max = max(max(all_aclr_lo), max(all_aclr_hi))

    fig, (ax_psd, ax_aclr) = plt.subplots(1, 2, figsize=(18, 6),
                                           gridspec_kw={'width_ratios': [2, 1]})

    # --- Left: PSD plot ---
    ax_psd.plot(f_mhz, psd_ideal, 'k--', lw=1.2, label='Ideal (linear PA)')
    ax_psd.plot(f_mhz, psd_no_dpd, color='#1f77b4', lw=0.6, alpha=0.4,
                label='No DPD (reference)')
    line_current, = ax_psd.plot([], [], color='#d62728', lw=1.2,
                                label='Current iteration')

    for ci in range(n_carriers):
        fc = (ci - (n_carriers - 1) / 2.0) * carrier_spacing / 1e6
        lo, hi = fc - carrier_bw_mhz / 2, fc + carrier_bw_mhz / 2
        lbl = f'{carrier_bw_mhz} MHz carrier' if ci == 0 else None
        ax_psd.axvspan(lo, hi, color='green', alpha=0.06, label=lbl)

    ax_psd.set_xlim(-20, 20)
    ax_psd.set_ylim(-80, 5)
    ax_psd.set_xlabel('Frequency (MHz)', fontsize=12)
    ax_psd.set_ylabel('PSD (dB, normalized)', fontsize=12)
    ax_psd.set_title(f'{n_carriers}× WCDMA {carrier_bw_mhz} MHz — GMP DPD',
                     fontsize=13)
    ax_psd.legend(loc='lower center', fontsize=9, ncol=2)
    ax_psd.grid(True, alpha=0.3)

    # Text annotation box on the PSD plot
    annot_text = ax_psd.text(0.02, 0.97, '', transform=ax_psd.transAxes,
                             fontsize=11, verticalalignment='top',
                             fontfamily='monospace',
                             bbox=dict(boxstyle='round,pad=0.4',
                                       facecolor='white', alpha=0.85,
                                       edgecolor='gray'))

    # --- Right: ACLR vs iteration ---
    line_aclr_lo, = ax_aclr.plot([], [], 'o-', color='#2ca02c', lw=2,
                                  markersize=7, label='ACLR lower')
    line_aclr_hi, = ax_aclr.plot([], [], 's-', color='#ff7f0e', lw=2,
                                  markersize=7, label='ACLR upper')
    ax_aclr.axhline(45, color='gray', ls=':', lw=1.5, label='3GPP spec (45 dB)')
    ax_aclr.set_xlim(-0.5, n_iterations + 0.5)
    ax_aclr.set_ylim(aclr_min - 5, aclr_max + 5)
    ax_aclr.set_xlabel('ILA Iteration', fontsize=12)
    ax_aclr.set_ylabel('ACLR (dB)', fontsize=12)
    ax_aclr.set_title('ACLR Convergence', fontsize=13)
    ax_aclr.legend(loc='lower right', fontsize=10)
    ax_aclr.grid(True, alpha=0.3)
    ax_aclr.set_xticks(range(0, n_iterations + 1))

    # Marker for current iteration on ACLR plot
    marker_lo, = ax_aclr.plot([], [], 'o', color='#d62728', markersize=12,
                               zorder=5, markeredgecolor='black', markeredgewidth=1.5)
    marker_hi, = ax_aclr.plot([], [], 's', color='#d62728', markersize=12,
                               zorder=5, markeredgecolor='black', markeredgewidth=1.5)

    fig.tight_layout()

    def init():
        line_current.set_data([], [])
        line_aclr_lo.set_data([], [])
        line_aclr_hi.set_data([], [])
        marker_lo.set_data([], [])
        marker_hi.set_data([], [])
        annot_text.set_text('')
        return (line_current, line_aclr_lo, line_aclr_hi,
                marker_lo, marker_hi, annot_text)

    def update(frame_idx):
        it, psd, nmse, aclr_lo, aclr_hi = frames[frame_idx]

        # PSD curve
        line_current.set_data(f_mhz, psd)

        # Annotation
        label = "No DPD" if it == 0 else f"ILA iter {it}"
        annot_text.set_text(
            f'Iteration: {it}\n'
            f'NMSE:       {nmse:+.2f} dB\n'
            f'ACLR lower: {aclr_lo:.1f} dB\n'
            f'ACLR upper: {aclr_hi:.1f} dB'
        )

        # ACLR history up to current frame
        iters_so_far = all_iters[:frame_idx + 1]
        lo_so_far    = all_aclr_lo[:frame_idx + 1]
        hi_so_far    = all_aclr_hi[:frame_idx + 1]
        line_aclr_lo.set_data(iters_so_far, lo_so_far)
        line_aclr_hi.set_data(iters_so_far, hi_so_far)

        # Highlight current point
        marker_lo.set_data([it], [aclr_lo])
        marker_hi.set_data([it], [aclr_hi])

        return (line_current, line_aclr_lo, line_aclr_hi,
                marker_lo, marker_hi, annot_text)

    anim = FuncAnimation(fig, update, frames=len(frames),
                         init_func=init, blit=False, repeat=False)

    output_file = 'dpd_convergence.gif'
    writer = PillowWriter(fps=1)
    anim.save(output_file, writer=writer, dpi=150)
    print(f"Movie saved to {output_file}")

    # Also save a static summary PNG with all iterations overlaid
    fig2, (ax2_psd, ax2_aclr) = plt.subplots(
        1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [2, 1]})

    ax2_psd.plot(f_mhz, psd_ideal, 'k--', lw=1.2, label='Ideal')
    cmap = plt.cm.coolwarm
    for i, (it, psd, nmse, aclr_lo, aclr_hi) in enumerate(frames):
        color = cmap(i / max(len(frames) - 1, 1))
        alpha_val = 0.4 if i == 0 else 0.8
        lw = 0.7 if i == 0 else 1.0
        ax2_psd.plot(f_mhz, psd, color=color, lw=lw, alpha=alpha_val,
                     label=f'Iter {it} ({nmse:+.1f} dB)')
    for ci in range(n_carriers):
        fc = (ci - (n_carriers - 1) / 2.0) * carrier_spacing / 1e6
        lo, hi = fc - carrier_bw_mhz / 2, fc + carrier_bw_mhz / 2
        ax2_psd.axvspan(lo, hi, color='green', alpha=0.04)
    ax2_psd.set_xlim(-20, 20)
    ax2_psd.set_ylim(-80, 5)
    ax2_psd.set_xlabel('Frequency (MHz)', fontsize=12)
    ax2_psd.set_ylabel('PSD (dB, normalized)', fontsize=12)
    ax2_psd.set_title('PSD — all iterations', fontsize=13)
    ax2_psd.legend(loc='lower left', fontsize=8, ncol=2)
    ax2_psd.grid(True, alpha=0.3)

    ax2_aclr.plot(all_iters, all_aclr_lo, 'o-', color='#2ca02c', lw=2,
                  markersize=7, label='ACLR lower')
    ax2_aclr.plot(all_iters, all_aclr_hi, 's-', color='#ff7f0e', lw=2,
                  markersize=7, label='ACLR upper')
    ax2_aclr.axhline(45, color='gray', ls=':', lw=1.5, label='3GPP spec (45 dB)')
    ax2_aclr.set_xlim(-0.5, n_iterations + 0.5)
    ax2_aclr.set_ylim(aclr_min - 5, aclr_max + 5)
    ax2_aclr.set_xlabel('ILA Iteration', fontsize=12)
    ax2_aclr.set_ylabel('ACLR (dB)', fontsize=12)
    ax2_aclr.set_title('ACLR Convergence', fontsize=13)
    ax2_aclr.legend(loc='lower right', fontsize=10)
    ax2_aclr.grid(True, alpha=0.3)
    ax2_aclr.set_xticks(range(0, n_iterations + 1))

    fig2.tight_layout()
    fig2.savefig('dpd_convergence_summary.png', dpi=150)
    print("Summary plot saved to dpd_convergence_summary.png")

    plt.close('all')


if __name__ == '__main__':
    main()
