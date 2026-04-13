"""
Block-based GMP DPD with coefficient update guard.

V2 of the block-based script: if the coefficient NMSE between the
candidate update and the current coefficients exceeds 0 dB (i.e. the
change is larger than the coefficients themselves), the update is
rejected and the previous coefficients are kept.  This prevents
catastrophic coefficient jumps from poorly conditioned blocks.

The guard is applied *after* the damping blend:
  w_candidate = α·w_new + (1-α)·w_old
  coeff_nmse  = ||w_candidate - w_old||² / ||w_old||²
  if coeff_nmse > 0 dB:  keep w_old   (reject)
  else:                   accept w_candidate
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, firwin, kaiserord
from scipy.signal.windows import blackmanharris


# ===================================================================
# Core GMP functions
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
# Main — block-based DPD with coefficient update guard
# ===================================================================

def main():
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Signal parameters
    # ------------------------------------------------------------------
    chip_rate       = 3.84e6
    fs              = 61.44e6
    n_carriers      = 3
    carrier_spacing = 5e6
    carrier_bw_mhz  = 5
    bw_mhz          = n_carriers * carrier_bw_mhz  # 15 MHz aggregate
    channel_bw_hz   = bw_mhz * 1e6

    BLOCK_SIZE      = 16384
    N_SIGNAL        = 1_000_000
    n_blocks        = N_SIGNAL // BLOCK_SIZE

    target_rms      = 0.22
    damping         = 0.5   # coefficient blend factor: w = α·w_new + (1-α)·w_old
    COEFF_NMSE_THRESHOLD = 0.0  # dB — reject updates above this

    pa_K, pa_Q = 5, 2
    cfg = GMPConfig(Ka=7, La=5, Kb=3, Lb=5, Mb=2, Kc=3, Lc=5, Mc=1)

    print(f"Block-based GMP DPD identification (v2 — with update guard)")
    print(f"  Signal: {n_carriers}× WCDMA {carrier_bw_mhz} MHz, "
          f"fs={fs/1e6:.2f} MHz")
    print(f"  Block size: {BLOCK_SIZE} samples "
          f"({BLOCK_SIZE/fs*1e3:.2f} ms)")
    print(f"  Total signal: {N_SIGNAL} samples → {n_blocks} blocks")
    print(f"  GMP config: {cfg} ({cfg.n_coeffs} coefficients)")
    print(f"  Damping α = {damping}")
    print(f"  Coefficient NMSE guard: reject if > {COEFF_NMSE_THRESHOLD:.0f} dB")
    print()

    # ------------------------------------------------------------------
    # Generate full signal
    # ------------------------------------------------------------------
    print("Generating signal...", flush=True)
    x_full_c = generate_multicarrier_wcdma(
        N_SIGNAL, n_carriers=n_carriers, carrier_spacing=carrier_spacing, fs=fs)
    x_full_c *= target_rms / np.sqrt(np.mean(np.abs(x_full_c)**2))
    x_full_iq = complex_to_iq(x_full_c)

    def pa(sig):
        return memory_polynomial_pa(sig)

    # Estimate target gain from the first block through the PA
    y0 = pa(x_full_iq[:BLOCK_SIZE])
    target_gain = compute_target_gain(x_full_iq[:BLOCK_SIZE], y0)
    print(f"  Target gain G = {target_gain:.4f}")

    ideal_full = target_gain * x_full_iq

    # No-DPD baseline on entire signal
    y_no_dpd_full = pa(x_full_iq)
    nmse_baseline = nmse_db(y_no_dpd_full, ideal_full)
    aclr_lo_base, aclr_hi_base = aclr_db(y_no_dpd_full, fs, channel_bw_hz)
    print(f"  Baseline (no DPD): NMSE = {nmse_baseline:+.2f} dB, "
          f"ACLR = {aclr_lo_base:.1f} / {aclr_hi_base:.1f} dB")
    print()

    # ------------------------------------------------------------------
    # Block-based processing with update guard
    # ------------------------------------------------------------------
    w = None
    w_prev = None
    block_metrics = []    # (block_idx, nmse, aclr_lo, aclr_hi)
    coeff_nmse_list = []  # (block_idx, coeff_nmse, accepted)
    n_rejected = 0

    for b in range(n_blocks):
        s = b * BLOCK_SIZE
        e = s + BLOCK_SIZE
        x_block = x_full_iq[s:e]
        ideal_block = ideal_full[s:e]

        if w is None:
            # First block: no DPD, just identify from raw PA I/O
            y_block = pa(x_block)
            w_new = identify_gmp(x_block, y_block, cfg, target_gain)
            w = w_new
            accepted = True
        else:
            # Apply current DPD, then pass through PA
            x_dpd_block = apply_dpd(x_block, w, cfg)
            y_block = pa(x_dpd_block)

            # Re-identify and damp
            w_raw = identify_gmp(x_dpd_block, y_block, cfg, target_gain)
            w_candidate = damping * w_raw + (1 - damping) * w

            # Guard: check coefficient NMSE before accepting
            delta = w_candidate - w
            coeff_nmse = 10.0 * np.log10(
                np.sum(np.abs(delta)**2) / np.sum(np.abs(w)**2))

            if coeff_nmse <= COEFF_NMSE_THRESHOLD:
                w = w_candidate
                accepted = True
            else:
                accepted = False
                n_rejected += 1

            coeff_nmse_list.append((b, coeff_nmse, accepted))

        # Track coefficient change vs w_prev (for the plot — always recorded)
        if w_prev is not None and len(coeff_nmse_list) == 0:
            delta_prev = w - w_prev
            cn = 10.0 * np.log10(
                np.sum(np.abs(delta_prev)**2) / np.sum(np.abs(w_prev)**2))
            coeff_nmse_list.append((b, cn, True))
        w_prev = w.copy()

        # Measure metrics on this block using current coefficients
        x_eval_dpd = apply_dpd(x_block, w, cfg)
        y_eval = pa(x_eval_dpd)
        nmse_val = nmse_db(y_eval, ideal_block)
        aclr_lo, aclr_hi = aclr_db(y_eval, fs, channel_bw_hz,
                                    nperseg=min(4096, BLOCK_SIZE))
        block_metrics.append((b, nmse_val, aclr_lo, aclr_hi))

        if b % 10 == 0 or b == n_blocks - 1:
            coeff_str = ""
            if coeff_nmse_list:
                _, cn_val, acc = coeff_nmse_list[-1]
                tag = "" if acc else " REJECTED"
                coeff_str = f", Δw = {cn_val:+.1f} dB{tag}"
            print(f"  Block {b:3d}/{n_blocks}: "
                  f"NMSE = {nmse_val:+.2f} dB, "
                  f"ACLR = {aclr_lo:.1f} / {aclr_hi:.1f} dB"
                  f"{coeff_str}")

    print(f"\n  Updates rejected: {n_rejected}/{n_blocks - 1} "
          f"({100*n_rejected/(n_blocks - 1):.1f}%)")

    # ------------------------------------------------------------------
    # Final evaluation on the entire signal with converged coefficients
    # ------------------------------------------------------------------
    print("\nFinal evaluation (full signal with converged coefficients)...")
    x_dpd_final = apply_dpd(x_full_iq, w, cfg)
    y_dpd_final = pa(x_dpd_final)
    nmse_final = nmse_db(y_dpd_final, ideal_full)
    aclr_lo_final, aclr_hi_final = aclr_db(y_dpd_final, fs, channel_bw_hz)

    print(f"\n{'='*65}")
    print(f"  {'Metric':<28s} {'No DPD':>10s} {'Block DPD':>10s} {'Ideal':>10s}")
    print(f"  {'-'*61}")
    print(f"  {'NMSE (dB)':<28s} {nmse_baseline:>+10.2f} {nmse_final:>+10.2f} {'—':>10s}")

    aclr_lo_ideal, aclr_hi_ideal = aclr_db(ideal_full, fs, channel_bw_hz)
    print(f"  {'ACLR lower (dB)':<28s} {aclr_lo_base:>10.1f} {aclr_lo_final:>10.1f} {aclr_lo_ideal:>10.1f}")
    print(f"  {'ACLR upper (dB)':<28s} {aclr_hi_base:>10.1f} {aclr_hi_final:>10.1f} {aclr_hi_ideal:>10.1f}")
    print(f"  {'-'*61}")
    print(f"  NMSE improvement: {nmse_baseline - nmse_final:.2f} dB")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # Plots: 2×2 — PSD, NMSE vs block, Coeff NMSE vs block, ACLR vs block
    # ------------------------------------------------------------------
    nperseg_psd = 8192
    noverlap_psd = nperseg_psd * 3 // 4
    psd_window = blackmanharris(nperseg_psd)

    def compute_psd_db(iq):
        c = iq_to_complex(iq) if iq.ndim == 2 else iq
        f, p = welch(c, fs=fs, nperseg=nperseg_psd, window=psd_window,
                     noverlap=noverlap_psd, return_onesided=False, scaling='density')
        idx = np.argsort(f)
        return f[idx] / 1e6, 10 * np.log10(p[idx] / np.max(p[idx]))

    f_mhz, psd_ideal   = compute_psd_db(ideal_full)
    _,     psd_no_dpd   = compute_psd_db(y_no_dpd_full)
    _,     psd_dpd_final = compute_psd_db(y_dpd_final)

    blocks_arr  = np.array([m[0] for m in block_metrics])
    nmse_arr    = np.array([m[1] for m in block_metrics])
    aclr_lo_arr = np.array([m[2] for m in block_metrics])
    aclr_hi_arr = np.array([m[3] for m in block_metrics])

    coeff_blocks_arr = np.array([m[0] for m in coeff_nmse_list])
    coeff_nmse_arr   = np.array([m[1] for m in coeff_nmse_list])
    coeff_accepted   = np.array([m[2] for m in coeff_nmse_list])

    fig, ((ax_psd, ax_nmse), (ax_coeff, ax_aclr)) = plt.subplots(
        2, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [2, 1]})

    # --- Top-left: PSD ---
    ax_psd.plot(f_mhz, psd_ideal,     'k--', lw=1.2, label='Ideal (linear PA)')
    ax_psd.plot(f_mhz, psd_no_dpd,    color='#1f77b4', lw=0.8, alpha=0.6,
                label='PA output — no DPD')
    ax_psd.plot(f_mhz, psd_dpd_final, color='#d62728', lw=1.0,
                label='PA output — block DPD (final)')

    for ci in range(n_carriers):
        fc = (ci - (n_carriers - 1) / 2.0) * carrier_spacing / 1e6
        lo, hi = fc - carrier_bw_mhz / 2, fc + carrier_bw_mhz / 2
        lbl = f'{carrier_bw_mhz} MHz carrier' if ci == 0 else None
        ax_psd.axvspan(lo, hi, color='green', alpha=0.06, label=lbl)

    ax_psd.set_xlim(-20, 20)
    ax_psd.set_ylim(-80, 5)
    ax_psd.set_xlabel('Frequency (MHz)', fontsize=12)
    ax_psd.set_ylabel('PSD (dB, normalized)', fontsize=12)
    ax_psd.set_title(f'{n_carriers}× WCDMA {carrier_bw_mhz} MHz — '
                     f'Block DPD v2 ({BLOCK_SIZE} samples/block, '
                     f'guard > {COEFF_NMSE_THRESHOLD:.0f} dB)',
                     fontsize=11)
    ax_psd.legend(loc='lower center', fontsize=9, ncol=2)
    ax_psd.grid(True, alpha=0.3)

    # --- Top-right: NMSE vs block ---
    ax_nmse.plot(blocks_arr, nmse_arr, '-', color='#9467bd', lw=1.2)
    ax_nmse.axhline(nmse_baseline, color='#1f77b4', ls=':', lw=1.5,
                    label=f'No DPD ({nmse_baseline:+.1f} dB)')
    ax_nmse.set_xlabel('Block index', fontsize=12)
    ax_nmse.set_ylabel('NMSE (dB)', fontsize=12)
    ax_nmse.set_title('Output NMSE Convergence', fontsize=13)
    ax_nmse.legend(loc='upper right', fontsize=10)
    ax_nmse.grid(True, alpha=0.3)

    # --- Bottom-left: Coefficient NMSE vs block (accepted / rejected) ---
    acc_mask = coeff_accepted
    rej_mask = ~coeff_accepted

    ax_coeff.plot(coeff_blocks_arr, coeff_nmse_arr, '-', color='#e377c2',
                  lw=0.8, alpha=0.5)
    if np.any(acc_mask):
        ax_coeff.scatter(coeff_blocks_arr[acc_mask], coeff_nmse_arr[acc_mask],
                         c='#2ca02c', s=25, zorder=3, label='Accepted')
    if np.any(rej_mask):
        ax_coeff.scatter(coeff_blocks_arr[rej_mask], coeff_nmse_arr[rej_mask],
                         c='#d62728', s=35, marker='x', linewidths=2,
                         zorder=3, label='Rejected')

    ax_coeff.axhline(COEFF_NMSE_THRESHOLD, color='gray', ls='--', lw=1.5,
                     label=f'Threshold ({COEFF_NMSE_THRESHOLD:.0f} dB)')
    ax_coeff.set_xlabel('Block index', fontsize=12)
    ax_coeff.set_ylabel('Coefficient NMSE (dB)', fontsize=12)
    ax_coeff.set_title('Coefficient Change: '
                       r'$10\log_{10}\left(\Vert\mathbf{w}_k - \mathbf{w}_{k-1}\Vert^2'
                       r' / \Vert\mathbf{w}_{k-1}\Vert^2\right)$',
                       fontsize=11)
    ax_coeff.legend(loc='upper right', fontsize=9)
    ax_coeff.grid(True, alpha=0.3)

    # --- Bottom-right: ACLR vs block ---
    ax_aclr.plot(blocks_arr, aclr_lo_arr, '-', color='#2ca02c', lw=1.2,
                 label='ACLR lower')
    ax_aclr.plot(blocks_arr, aclr_hi_arr, '-', color='#ff7f0e', lw=1.2,
                 label='ACLR upper')
    ax_aclr.axhline(45, color='gray', ls=':', lw=1.5, label='3GPP spec (45 dB)')
    ax_aclr.axhline(aclr_lo_base, color='#1f77b4', ls=':', lw=1,
                    alpha=0.7, label=f'No DPD ({aclr_lo_base:.0f} dB)')
    ax_aclr.set_xlabel('Block index', fontsize=12)
    ax_aclr.set_ylabel('ACLR (dB)', fontsize=12)
    ax_aclr.set_title('ACLR Convergence', fontsize=13)
    ax_aclr.legend(loc='lower right', fontsize=9)
    ax_aclr.grid(True, alpha=0.3)

    fig.tight_layout()
    output_file = 'gmp_dpd_block_v2_convergence.png'
    fig.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")
    plt.close('all')


if __name__ == '__main__':
    main()
