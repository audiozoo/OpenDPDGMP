"""
Block-based GMP DPD — 5G NR 100 MHz OFDM, multi-rate architecture.

Signal chain:
  1. Generate OFDM at fs_sig = 122.88 MHz  (4096-pt FFT × 30 kHz SCS)
  2. Upsample ×4 → fs_dpd = 491.52 MHz    (DPD + PA processing)
  3. Upsample ×2 → fs_psd = 983.04 MHz    (PSD visualisation, ±150 MHz)

DPD engine: ILA with damping (α=0.3) and coefficient-NMSE guard.
"""

import numpy as np
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import welch, firwin, kaiserord
from scipy.signal.windows import blackmanharris


# ===================================================================
# Utilities
# ===================================================================

def upsample_fft(x, factor):
    """Upsample a complex baseband signal by *factor* via spectral
    zero-padding (FFT → zero-insert → IFFT)."""
    N = len(x)
    X = np.fft.fft(x)
    N_up = N * factor
    X_up = np.zeros(N_up, dtype=np.complex128)
    half = N // 2
    X_up[:half] = X[:half]
    X_up[N_up - half:] = X[half:]
    return np.fft.ifft(X_up) * factor


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
# 5G NR OFDM signal generation at 122.88 MHz (4096-pt FFT)
# ===================================================================

NR_PRB_TABLE_SCS30 = {
    5: 11, 10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 40: 106,
    50: 133, 60: 162, 70: 189, 80: 217, 90: 245, 100: 273,
}


def generate_nr_ofdm(n_samples, bw_mhz=100, scs_khz=30, fs=122.88e6):
    """
    Single-carrier 5G NR OFDM signal at the native DAC rate.

    At fs = 122.88 MHz with SCS = 30 kHz the FFT size is 4096.
    273 PRBs (3276 active subcarriers) for 100 MHz channel BW
    per 3GPP TS 38.104 Table 5.3.2-1.
    """
    scs = scs_khz * 1e3
    n_fft = int(round(fs / scs))                     # 4096
    n_prb = NR_PRB_TABLE_SCS30[int(bw_mhz)]
    n_sc = n_prb * 12                                # 3276

    cp_normal = int(round(144 * n_fft / 2048))       # 288
    cp_extended = int(round(160 * n_fft / 2048))      # 320

    symbols_per_slot = 14
    n_ext_per_slot = 2

    samples_per_slot = (n_ext_per_slot * (n_fft + cp_extended)
                        + (symbols_per_slot - n_ext_per_slot) * (n_fft + cp_normal))

    n_slots = n_samples // samples_per_slot + 2
    buf = np.zeros(n_slots * samples_per_slot, dtype=np.complex128)

    half_sc = n_sc // 2
    pos = 0
    for _slot in range(n_slots):
        for sym_idx in range(symbols_per_slot):
            cp_len = cp_extended if sym_idx in (0, 7) else cp_normal

            data = (np.sign(np.random.randn(n_sc))
                    + 1j * np.sign(np.random.randn(n_sc))) / np.sqrt(2)

            freq = np.zeros(n_fft, dtype=np.complex128)
            freq[1:half_sc + 1] = data[:half_sc]
            freq[n_fft - half_sc:] = data[half_sc:]

            td = np.fft.ifft(freq) * np.sqrt(n_fft)
            sym_with_cp = np.concatenate([td[-cp_len:], td])

            end = pos + len(sym_with_cp)
            if end <= len(buf):
                buf[pos:end] = sym_with_cp
            pos = end

    buf = buf[:n_samples]

    # Channel filter: –80 dB stopband at the channel edge
    occupied_bw = n_sc * scs                         # 98.28 MHz
    pb = occupied_bw / 2 + scs                       # 49.17 MHz
    sb = bw_mhz * 1e6 / 2                            # 50 MHz
    transition = (sb - pb) / (fs / 2)
    numtaps, beta = kaiserord(80, transition)
    if numtaps % 2 == 0:
        numtaps += 1
    h_chan = firwin(numtaps, (pb + sb) / 2, window=('kaiser', beta), fs=fs)
    filtered = np.convolve(buf, h_chan, mode='full')
    return filtered[len(h_chan) // 2: len(h_chan) // 2 + n_samples]


# ===================================================================
# Main — multi-rate block-based DPD
# ===================================================================

def main():
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Sample rates
    # ------------------------------------------------------------------
    scs_khz     = 30
    bw_mhz      = 100
    fs_sig      = 122.88e6           # signal generation: 4096-FFT × 30 kHz
    fs_dpd      = 491.52e6           # DPD + PA processing (×4)
    fs_psd      = 983.04e6           # PSD visualisation   (×2 from fs_dpd)
    UP_SIG_DPD  = int(fs_dpd / fs_sig)   # 4
    UP_DPD_PSD  = int(fs_psd / fs_dpd)   # 2

    channel_bw_hz   = bw_mhz * 1e6
    adjacent_offset = channel_bw_hz

    n_prb = NR_PRB_TABLE_SCS30[bw_mhz]
    n_sc  = n_prb * 12

    # ------------------------------------------------------------------
    # Signal & DPD parameters
    # ------------------------------------------------------------------
    N_SIG       = 500_000            # samples at fs_sig (4.07 ms)
    N_DPD       = N_SIG * UP_SIG_DPD # 2 000 000 at fs_dpd
    BLOCK_SIZE  = 32768              # at fs_dpd — 0.067 ms ≈ ~2 OFDM symbols
    n_blocks    = N_DPD // BLOCK_SIZE

    target_rms  = 0.15
    damping     = 0.3
    COEFF_NMSE_THRESHOLD = -3.0

    cfg = GMPConfig(Ka=9, La=7, Kb=5, Lb=7, Mb=2, Kc=5, Lc=7, Mc=1)

    print("Block-based GMP DPD — NR 100 MHz, multi-rate architecture")
    print(f"  Signal gen : fs_sig = {fs_sig/1e6:.2f} MHz  "
          f"(4096-FFT, {N_SIG} samples, {N_SIG/fs_sig*1e3:.2f} ms)")
    print(f"  DPD / PA   : fs_dpd = {fs_dpd/1e6:.2f} MHz  "
          f"(×{UP_SIG_DPD} upsample, {N_DPD} samples)")
    print(f"  PSD display: fs_psd = {fs_psd/1e6:.2f} MHz  "
          f"(×{UP_DPD_PSD} upsample)")
    print(f"  Signal: {n_prb} PRBs ({n_sc} sc), "
          f"SCS={scs_khz} kHz, {bw_mhz} MHz channel")
    print(f"  Block size: {BLOCK_SIZE} samples at fs_dpd "
          f"({BLOCK_SIZE/fs_dpd*1e3:.2f} ms)")
    print(f"  Blocks: {n_blocks}")
    print(f"  GMP: {cfg} ({cfg.n_coeffs} coefficients)")
    print(f"  Damping α = {damping},  guard > {COEFF_NMSE_THRESHOLD:.0f} dB")
    print()

    # ------------------------------------------------------------------
    # Generate OFDM at fs_sig, upsample to fs_dpd
    # ------------------------------------------------------------------
    print("Generating NR 100 MHz OFDM at "
          f"{fs_sig/1e6:.2f} MHz ...", flush=True)
    x_sig_c = generate_nr_ofdm(N_SIG, bw_mhz=bw_mhz, scs_khz=scs_khz,
                                fs=fs_sig)

    print(f"Upsampling ×{UP_SIG_DPD} → {fs_dpd/1e6:.2f} MHz ...",
          flush=True)
    x_dpd_c = upsample_fft(x_sig_c, UP_SIG_DPD)
    x_dpd_c *= target_rms / np.sqrt(np.mean(np.abs(x_dpd_c)**2))
    x_full_iq = complex_to_iq(x_dpd_c)          # all processing uses this

    def pa(sig):
        return memory_polynomial_pa(sig)

    y0 = pa(x_full_iq[:BLOCK_SIZE])
    target_gain = compute_target_gain(x_full_iq[:BLOCK_SIZE], y0)
    print(f"  Target gain G = {target_gain:.4f}")

    ideal_full = target_gain * x_full_iq

    y_no_dpd_full = pa(x_full_iq)
    nmse_baseline = nmse_db(y_no_dpd_full, ideal_full)
    aclr_lo_base, aclr_hi_base = aclr_db(y_no_dpd_full, fs_dpd,
                                          channel_bw_hz, adjacent_offset)
    print(f"  Baseline (no DPD): NMSE = {nmse_baseline:+.2f} dB, "
          f"ACLR = {aclr_lo_base:.1f} / {aclr_hi_base:.1f} dB")
    print()

    # ------------------------------------------------------------------
    # Filter frequency response plot
    # ------------------------------------------------------------------
    print("Plotting filter frequency responses ...", flush=True)

    scs_hz = scs_khz * 1e3
    n_fft_sig = int(round(fs_sig / scs_hz))          # 4096
    n_sc_active = n_prb * 12                          # 3276
    half_sc = n_sc_active // 2
    N_disp = n_fft_sig * 8                            # smooth display

    # (1) OFDM spectral shape: unit-amplitude on all active subcarriers,
    #     IFFT → zero-padded FFT to reveal the sinc sidelobes.
    sym_freq = np.zeros(n_fft_sig, dtype=np.complex128)
    sym_freq[1:half_sc + 1] = 1.0
    sym_freq[n_fft_sig - half_sc:] = 1.0
    sym_td = np.fft.ifft(sym_freq) * np.sqrt(n_fft_sig)
    H_ofdm = np.fft.fftshift(np.fft.fft(sym_td, n=N_disp))
    H_ofdm_db = 20 * np.log10(np.abs(H_ofdm) / np.max(np.abs(H_ofdm)) + 1e-15)
    f_disp = np.linspace(-fs_sig / 2, fs_sig / 2, N_disp,
                          endpoint=False) / 1e6

    # (2) Channel filter (Kaiser FIR) — re-derive coefficients for the plot
    occupied_bw = n_sc_active * scs_hz
    pb_filt = occupied_bw / 2 + scs_hz
    sb_filt = bw_mhz * 1e6 / 2
    transition_filt = (sb_filt - pb_filt) / (fs_sig / 2)
    numtaps_filt, beta_filt = kaiserord(80, transition_filt)
    if numtaps_filt % 2 == 0:
        numtaps_filt += 1
    h_chan = firwin(numtaps_filt, (pb_filt + sb_filt) / 2,
                    window=('kaiser', beta_filt), fs=fs_sig)
    H_chan = np.fft.fftshift(np.fft.fft(h_chan, n=N_disp))
    H_chan_db = 20 * np.log10(np.abs(H_chan) / np.max(np.abs(H_chan)) + 1e-15)

    fig_filt, (ax_ofdm, ax_chan) = plt.subplots(2, 1, figsize=(14, 10),
                                                 sharex=True)
    # Subplot 1 — OFDM spectral shape
    ax_ofdm.plot(f_disp, H_ofdm_db, color='#1f77b4', lw=1.0)
    ax_ofdm.axvspan(-occupied_bw / 2e6, occupied_bw / 2e6,
                     color='green', alpha=0.08,
                     label=f'Occupied BW ({occupied_bw/1e6:.2f} MHz)')
    ax_ofdm.axvline(-bw_mhz / 2, color='red', ls='--', lw=0.8,
                     label=f'Channel edge (±{bw_mhz/2:.0f} MHz)')
    ax_ofdm.axvline( bw_mhz / 2, color='red', ls='--', lw=0.8)
    ax_ofdm.set_ylim(-100, 5)
    ax_ofdm.set_ylabel('Magnitude (dB)', fontsize=12)
    ax_ofdm.set_title(
        f'OFDM Subcarrier Allocation Spectral Shape\n'
        f'({n_sc_active} active sc / {n_fft_sig}-pt FFT, '
        f'SCS = {scs_khz} kHz, fs = {fs_sig/1e6:.2f} MHz)',
        fontsize=12)
    ax_ofdm.legend(loc='upper right', fontsize=10)
    ax_ofdm.grid(True, alpha=0.3)

    # Subplot 2 — Channel filter
    ax_chan.plot(f_disp, H_chan_db, color='#d62728', lw=1.0)
    ax_chan.axvline(-pb_filt / 1e6, color='#2ca02c', ls=':', lw=0.8,
                    label=f'Passband edge (±{pb_filt/1e6:.2f} MHz)')
    ax_chan.axvline( pb_filt / 1e6, color='#2ca02c', ls=':', lw=0.8)
    ax_chan.axvline(-sb_filt / 1e6, color='red', ls='--', lw=0.8,
                    label=f'Stopband edge (±{sb_filt/1e6:.0f} MHz)')
    ax_chan.axvline( sb_filt / 1e6, color='red', ls='--', lw=0.8)
    ax_chan.axhline(-80, color='gray', ls=':', lw=0.8,
                    label='–80 dB design target')
    ax_chan.set_ylim(-100, 5)
    ax_chan.set_xlabel('Frequency (MHz)', fontsize=12)
    ax_chan.set_ylabel('Magnitude (dB)', fontsize=12)
    ax_chan.set_title(
        f'Channel Filter (Kaiser FIR, {numtaps_filt} taps, '
        f'β = {beta_filt:.2f})',
        fontsize=12)
    ax_chan.legend(loc='upper right', fontsize=10)
    ax_chan.grid(True, alpha=0.3)

    fig_filt.tight_layout()
    fig_filt.savefig('gmp_dpd_block_nr100_v3_filters.png', dpi=150)
    print(f"  Filter plot saved to gmp_dpd_block_nr100_v3_filters.png")
    plt.close(fig_filt)

    # ------------------------------------------------------------------
    # PSD helper — upsample from fs_dpd to fs_psd before Welch
    # ------------------------------------------------------------------
    nperseg_psd = 16384
    noverlap_psd = nperseg_psd * 3 // 4
    psd_window = blackmanharris(nperseg_psd)

    def compute_psd(iq_at_dpd):
        """Return (f_MHz, psd_linear) computed at fs_psd."""
        c = iq_to_complex(iq_at_dpd) if iq_at_dpd.ndim == 2 else iq_at_dpd
        c_up = upsample_fft(c, UP_DPD_PSD)
        f, p = welch(c_up, fs=fs_psd, nperseg=nperseg_psd, window=psd_window,
                     noverlap=noverlap_psd, return_onesided=False,
                     scaling='density')
        idx = np.argsort(f)
        return f[idx] / 1e6, p[idx]

    print("Computing baseline PSDs (upsampled to "
          f"{fs_psd/1e6:.2f} MHz) ...", flush=True)
    f_mhz_psd, psd_ideal_raw = compute_psd(ideal_full)
    _, psd_no_dpd_raw = compute_psd(y_no_dpd_full)

    # ------------------------------------------------------------------
    # Block-based DPD loop (all at fs_dpd)
    # ------------------------------------------------------------------
    w = None
    w_prev = None
    block_metrics = []
    coeff_nmse_list = []
    block_psd_list = []
    n_rejected = 0
    coeff_update_idx = 0

    for b in range(n_blocks):
        s = b * BLOCK_SIZE
        e = s + BLOCK_SIZE
        x_block = x_full_iq[s:e]
        ideal_block = ideal_full[s:e]

        if w is None:
            y_block = pa(x_block)
            w_new = identify_gmp(x_block, y_block, cfg, target_gain)
            w = w_new
            accepted = True
            coeff_update_idx += 1
        else:
            x_dpd_block = apply_dpd(x_block, w, cfg)
            y_block = pa(x_dpd_block)

            w_raw = identify_gmp(x_dpd_block, y_block, cfg, target_gain)
            w_candidate = damping * w_raw + (1 - damping) * w

            delta = w_candidate - w
            coeff_nmse = 10.0 * np.log10(
                np.sum(np.abs(delta)**2) / np.sum(np.abs(w)**2))

            if coeff_nmse <= COEFF_NMSE_THRESHOLD:
                w = w_candidate
                accepted = True
                coeff_update_idx += 1
            else:
                accepted = False
                n_rejected += 1

            coeff_nmse_list.append((b, coeff_nmse, accepted))

        if w_prev is not None and len(coeff_nmse_list) == 0:
            delta_prev = w - w_prev
            cn = 10.0 * np.log10(
                np.sum(np.abs(delta_prev)**2) / np.sum(np.abs(w_prev)**2))
            coeff_nmse_list.append((b, cn, True))
        w_prev = w.copy()

        # Per-block metrics (at fs_dpd)
        x_eval_dpd = apply_dpd(x_block, w, cfg)
        y_eval = pa(x_eval_dpd)
        nmse_val = nmse_db(y_eval, ideal_block)
        aclr_lo, aclr_hi = aclr_db(y_eval, fs_dpd, channel_bw_hz,
                                    adjacent_offset,
                                    nperseg=min(8192, BLOCK_SIZE))
        block_metrics.append((b, nmse_val, aclr_lo, aclr_hi, accepted))

        # Full-signal PSD snapshot (upsample to fs_psd)
        x_dpd_cur = apply_dpd(x_full_iq, w, cfg)
        y_dpd_cur = pa(x_dpd_cur)
        _, psd_cur_raw = compute_psd(y_dpd_cur)
        block_psd_list.append(
            10 * np.log10(psd_cur_raw / np.max(psd_cur_raw)))

        if b % 5 == 0 or b == n_blocks - 1:
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
    # Final evaluation
    # ------------------------------------------------------------------
    print("\nFinal evaluation (full signal, converged coefficients)...")
    x_dpd_final = apply_dpd(x_full_iq, w, cfg)
    y_dpd_final = pa(x_dpd_final)
    nmse_final = nmse_db(y_dpd_final, ideal_full)
    aclr_lo_final, aclr_hi_final = aclr_db(y_dpd_final, fs_dpd,
                                            channel_bw_hz, adjacent_offset)
    aclr_lo_ideal, aclr_hi_ideal = aclr_db(ideal_full, fs_dpd,
                                            channel_bw_hz, adjacent_offset)

    print(f"\n{'='*65}")
    print(f"  {'Metric':<28s} {'No DPD':>10s} {'Block DPD':>10s} {'Ideal':>10s}")
    print(f"  {'-'*61}")
    print(f"  {'NMSE (dB)':<28s} {nmse_baseline:>+10.2f} {nmse_final:>+10.2f} {'—':>10s}")
    print(f"  {'ACLR lower (dB)':<28s} {aclr_lo_base:>10.1f} {aclr_lo_final:>10.1f} {aclr_lo_ideal:>10.1f}")
    print(f"  {'ACLR upper (dB)':<28s} {aclr_hi_base:>10.1f} {aclr_hi_final:>10.1f} {aclr_hi_ideal:>10.1f}")
    print(f"  {'-'*61}")
    print(f"  NMSE improvement: {nmse_baseline - nmse_final:.2f} dB")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # Prepare arrays for plotting
    # ------------------------------------------------------------------
    psd_ideal_db  = 10 * np.log10(psd_ideal_raw / np.max(psd_ideal_raw))
    psd_no_dpd_db = 10 * np.log10(psd_no_dpd_raw / np.max(psd_no_dpd_raw))

    blocks_arr   = np.array([m[0] for m in block_metrics])
    nmse_arr     = np.array([m[1] for m in block_metrics])
    aclr_lo_arr  = np.array([m[2] for m in block_metrics])
    aclr_hi_arr  = np.array([m[3] for m in block_metrics])
    accepted_arr = np.array([m[4] for m in block_metrics])

    coeff_blocks_arr = np.array([m[0] for m in coeff_nmse_list])
    coeff_nmse_arr   = np.array([m[1] for m in coeff_nmse_list])
    coeff_accepted   = np.array([m[2] for m in coeff_nmse_list])

    acc_blocks  = blocks_arr[accepted_arr]
    acc_aclr_lo = aclr_lo_arr[accepted_arr]
    acc_aclr_hi = aclr_hi_arr[accepted_arr]

    nmse_ylim  = (min(nmse_arr.min(), nmse_baseline) - 5,
                  max(nmse_arr.max(), nmse_baseline) + 5)
    coeff_ylim = (coeff_nmse_arr.min() - 3, max(coeff_nmse_arr.max(), 5) + 3)
    aclr_lo_min = min(aclr_lo_arr.min(), aclr_hi_arr.min(), aclr_lo_base)
    aclr_hi_max = max(aclr_lo_arr.max(), aclr_hi_arr.max())
    aclr_ylim  = (aclr_lo_min - 5, aclr_hi_max + 5)

    psd_xlim = (-150, 150)

    # ------------------------------------------------------------------
    # Static summary PNG
    # ------------------------------------------------------------------
    fig_s = plt.figure(figsize=(22, 12))
    gs_s = fig_s.add_gridspec(2, 3, width_ratios=[2, 1, 1])
    ax = {}
    ax['psd']      = fig_s.add_subplot(gs_s[0, 0])
    ax['nmse']     = fig_s.add_subplot(gs_s[0, 1])
    ax['coeff']    = fig_s.add_subplot(gs_s[0, 2])
    ax['aclr']     = fig_s.add_subplot(gs_s[1, 0])
    ax['aclr_acc'] = fig_s.add_subplot(gs_s[1, 1:])

    ax['psd'].plot(f_mhz_psd, psd_ideal_db, 'k--', lw=1.2,
                   label='Ideal (linear PA)')
    ax['psd'].plot(f_mhz_psd, psd_no_dpd_db, color='#1f77b4', lw=0.8,
                   alpha=0.6, label='PA output — no DPD')
    ax['psd'].plot(f_mhz_psd, block_psd_list[-1], color='#d62728', lw=1.0,
                   label='PA output — block DPD (final)')
    ax['psd'].axvspan(-bw_mhz / 2, bw_mhz / 2,
                      color='green', alpha=0.06,
                      label=f'{bw_mhz} MHz channel')
    ax['psd'].axvspan(-adjacent_offset / 1e6 - bw_mhz / 2,
                      -adjacent_offset / 1e6 + bw_mhz / 2,
                      color='orange', alpha=0.04, label='Adjacent channel')
    ax['psd'].axvspan( adjacent_offset / 1e6 - bw_mhz / 2,
                       adjacent_offset / 1e6 + bw_mhz / 2,
                      color='orange', alpha=0.04)
    ax['psd'].set_xlim(*psd_xlim); ax['psd'].set_ylim(-100, 5)
    ax['psd'].set_xlabel('Frequency (MHz)', fontsize=12)
    ax['psd'].set_ylabel('PSD (dB, normalized)', fontsize=12)
    ax['psd'].set_title(
        f'NR {bw_mhz} MHz SCS={scs_khz} kHz — PSD at '
        f'{fs_psd/1e6:.0f} MHz\n'
        f'(sig {fs_sig/1e6:.0f} → DPD {fs_dpd/1e6:.0f} → '
        f'PSD {fs_psd/1e6:.0f} MHz)',
        fontsize=11)
    ax['psd'].legend(loc='lower center', fontsize=9, ncol=2)
    ax['psd'].grid(True, alpha=0.3)

    ax['nmse'].plot(blocks_arr, nmse_arr, '-', color='#9467bd', lw=1.2)
    ax['nmse'].axhline(nmse_baseline, color='#1f77b4', ls=':', lw=1.5,
                       label=f'No DPD ({nmse_baseline:+.1f} dB)')
    ax['nmse'].set_xlabel('Block index', fontsize=12)
    ax['nmse'].set_ylabel('NMSE (dB)', fontsize=12)
    ax['nmse'].set_title('Output NMSE Convergence', fontsize=13)
    ax['nmse'].legend(loc='upper right', fontsize=10)
    ax['nmse'].grid(True, alpha=0.3)

    acc_mask = coeff_accepted; rej_mask = ~coeff_accepted
    ax['coeff'].plot(coeff_blocks_arr, coeff_nmse_arr, '-', color='#e377c2',
                     lw=0.8, alpha=0.5)
    if np.any(acc_mask):
        ax['coeff'].scatter(coeff_blocks_arr[acc_mask],
                            coeff_nmse_arr[acc_mask],
                            c='#2ca02c', s=25, zorder=3, label='Accepted')
    if np.any(rej_mask):
        ax['coeff'].scatter(coeff_blocks_arr[rej_mask],
                            coeff_nmse_arr[rej_mask],
                            c='#d62728', s=35, marker='x', linewidths=2,
                            zorder=3, label='Rejected')
    ax['coeff'].axhline(COEFF_NMSE_THRESHOLD, color='gray', ls='--', lw=1.5,
                        label=f'Threshold ({COEFF_NMSE_THRESHOLD:.0f} dB)')
    ax['coeff'].set_xlabel('Block index', fontsize=12)
    ax['coeff'].set_ylabel('Coefficient NMSE (dB)', fontsize=12)
    ax['coeff'].set_title('Coefficient Change', fontsize=13)
    ax['coeff'].legend(loc='upper right', fontsize=9)
    ax['coeff'].grid(True, alpha=0.3)

    ax['aclr'].plot(blocks_arr, aclr_lo_arr, '-', color='#2ca02c', lw=1.2,
                    label='ACLR lower')
    ax['aclr'].plot(blocks_arr, aclr_hi_arr, '-', color='#ff7f0e', lw=1.2,
                    label='ACLR upper')
    ax['aclr'].axhline(45, color='gray', ls=':', lw=1.5,
                       label='3GPP NR spec (45 dB)')
    ax['aclr'].axhline(aclr_lo_base, color='#1f77b4', ls=':', lw=1,
                       alpha=0.7, label=f'No DPD ({aclr_lo_base:.0f} dB)')
    ax['aclr'].set_xlabel('Block index', fontsize=12)
    ax['aclr'].set_ylabel('ACLR (dB)', fontsize=12)
    ax['aclr'].set_title('ACLR Convergence (all blocks)', fontsize=13)
    ax['aclr'].legend(loc='lower right', fontsize=9)
    ax['aclr'].grid(True, alpha=0.3)

    update_idx = np.arange(len(acc_blocks))
    ax['aclr_acc'].plot(update_idx, acc_aclr_lo, 'o-', color='#2ca02c',
                        lw=1.2, markersize=4, label='ACLR lower')
    ax['aclr_acc'].plot(update_idx, acc_aclr_hi, 's-', color='#ff7f0e',
                        lw=1.2, markersize=4, label='ACLR upper')
    ax['aclr_acc'].axhline(45, color='gray', ls=':', lw=1.5,
                           label='3GPP NR spec (45 dB)')
    ax['aclr_acc'].axhline(aclr_lo_base, color='#1f77b4', ls=':', lw=1,
                           alpha=0.7,
                           label=f'No DPD ({aclr_lo_base:.0f} dB)')
    ax['aclr_acc'].set_xlabel('Accepted update index', fontsize=12)
    ax['aclr_acc'].set_ylabel('ACLR (dB)', fontsize=12)
    ax['aclr_acc'].set_title(f'ACLR — accepted updates only '
                             f'({len(acc_blocks)}/{n_blocks})', fontsize=13)
    ax['aclr_acc'].legend(loc='lower right', fontsize=9)
    ax['aclr_acc'].grid(True, alpha=0.3)

    fig_s.tight_layout()
    fig_s.savefig('gmp_dpd_block_nr100_v3_convergence.png', dpi=150)
    print(f"\nStatic plot saved to gmp_dpd_block_nr100_v3_convergence.png")
    plt.close(fig_s)

    # ------------------------------------------------------------------
    # Animated GIF
    # ------------------------------------------------------------------
    print(f"Rendering animation ({n_blocks} frames)...")

    fig_a = plt.figure(figsize=(22, 12))
    gs_a = fig_a.add_gridspec(2, 3, width_ratios=[2, 1, 1])
    a_psd      = fig_a.add_subplot(gs_a[0, 0])
    a_nmse     = fig_a.add_subplot(gs_a[0, 1])
    a_coeff    = fig_a.add_subplot(gs_a[0, 2])
    a_aclr     = fig_a.add_subplot(gs_a[1, 0])
    a_aclr_acc = fig_a.add_subplot(gs_a[1, 1:])

    a_psd.plot(f_mhz_psd, psd_ideal_db, 'k--', lw=1.2,
               label='Ideal (linear PA)')
    a_psd.plot(f_mhz_psd, psd_no_dpd_db, color='#1f77b4', lw=0.8,
               alpha=0.4, label='PA output — no DPD')
    line_psd, = a_psd.plot([], [], color='#d62728', lw=1.2,
                           label='Current block DPD')
    a_psd.axvspan(-bw_mhz / 2, bw_mhz / 2,
                  color='green', alpha=0.06, label=f'{bw_mhz} MHz ch.')
    a_psd.axvspan(-adjacent_offset / 1e6 - bw_mhz / 2,
                  -adjacent_offset / 1e6 + bw_mhz / 2,
                  color='orange', alpha=0.04, label='Adjacent ch.')
    a_psd.axvspan( adjacent_offset / 1e6 - bw_mhz / 2,
                   adjacent_offset / 1e6 + bw_mhz / 2,
                  color='orange', alpha=0.04)
    a_psd.set_xlim(*psd_xlim); a_psd.set_ylim(-100, 5)
    a_psd.set_xlabel('Frequency (MHz)', fontsize=12)
    a_psd.set_ylabel('PSD (dB, normalized)', fontsize=12)
    a_psd.legend(loc='lower center', fontsize=8, ncol=2)
    a_psd.grid(True, alpha=0.3)

    psd_annot = a_psd.text(0.02, 0.97, '', transform=a_psd.transAxes,
                           fontsize=10, verticalalignment='top',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=0.4',
                                     facecolor='white', alpha=0.85,
                                     edgecolor='gray'))

    a_nmse.axhline(nmse_baseline, color='#1f77b4', ls=':', lw=1.5,
                   label=f'No DPD ({nmse_baseline:+.1f} dB)')
    line_nmse, = a_nmse.plot([], [], '-', color='#9467bd', lw=1.2)
    marker_nmse, = a_nmse.plot([], [], 'o', color='#d62728', markersize=8,
                               zorder=5)
    a_nmse.set_xlim(-0.5, n_blocks - 0.5); a_nmse.set_ylim(*nmse_ylim)
    a_nmse.set_xlabel('Block index', fontsize=12)
    a_nmse.set_ylabel('NMSE (dB)', fontsize=12)
    a_nmse.set_title('Output NMSE Convergence', fontsize=13)
    a_nmse.legend(loc='upper right', fontsize=10)
    a_nmse.grid(True, alpha=0.3)

    a_coeff.axhline(COEFF_NMSE_THRESHOLD, color='gray', ls='--', lw=1.5,
                    label=f'Threshold ({COEFF_NMSE_THRESHOLD:.0f} dB)')
    line_coeff, = a_coeff.plot([], [], '-', color='#e377c2', lw=0.8,
                               alpha=0.5)
    scat_coeff_acc = a_coeff.scatter([], [], c='#2ca02c', s=25, zorder=3,
                                     label='Accepted')
    scat_coeff_rej = a_coeff.scatter([], [], c='#d62728', s=35, marker='x',
                                     linewidths=2, zorder=3, label='Rejected')
    a_coeff.set_xlim(-0.5, n_blocks - 0.5); a_coeff.set_ylim(*coeff_ylim)
    a_coeff.set_xlabel('Block index', fontsize=12)
    a_coeff.set_ylabel('Coefficient NMSE (dB)', fontsize=12)
    a_coeff.set_title('Coefficient Change', fontsize=13)
    a_coeff.legend(loc='upper right', fontsize=9)
    a_coeff.grid(True, alpha=0.3)

    a_aclr.axhline(45, color='gray', ls=':', lw=1.5,
                   label='3GPP NR spec (45 dB)')
    a_aclr.axhline(aclr_lo_base, color='#1f77b4', ls=':', lw=1, alpha=0.7,
                   label=f'No DPD ({aclr_lo_base:.0f} dB)')
    line_aclr_lo, = a_aclr.plot([], [], '-', color='#2ca02c', lw=1.2,
                                label='ACLR lower')
    line_aclr_hi, = a_aclr.plot([], [], '-', color='#ff7f0e', lw=1.2,
                                label='ACLR upper')
    a_aclr.set_xlim(-0.5, n_blocks - 0.5); a_aclr.set_ylim(*aclr_ylim)
    a_aclr.set_xlabel('Block index', fontsize=12)
    a_aclr.set_ylabel('ACLR (dB)', fontsize=12)
    a_aclr.set_title('ACLR Convergence (all blocks)', fontsize=13)
    a_aclr.legend(loc='lower right', fontsize=9)
    a_aclr.grid(True, alpha=0.3)

    a_aclr_acc.axhline(45, color='gray', ls=':', lw=1.5,
                       label='3GPP NR spec (45 dB)')
    a_aclr_acc.axhline(aclr_lo_base, color='#1f77b4', ls=':', lw=1,
                       alpha=0.7, label=f'No DPD ({aclr_lo_base:.0f} dB)')
    line_acc_lo, = a_aclr_acc.plot([], [], 'o-', color='#2ca02c', lw=1.2,
                                   markersize=4, label='ACLR lower')
    line_acc_hi, = a_aclr_acc.plot([], [], 's-', color='#ff7f0e', lw=1.2,
                                   markersize=4, label='ACLR upper')
    n_accepted_total = int(accepted_arr.sum())
    a_aclr_acc.set_xlim(-0.5, n_accepted_total + 0.5)
    a_aclr_acc.set_ylim(*aclr_ylim)
    a_aclr_acc.set_xlabel('Accepted update index', fontsize=12)
    a_aclr_acc.set_ylabel('ACLR (dB)', fontsize=12)
    a_aclr_acc.set_title('ACLR — accepted updates only', fontsize=13)
    a_aclr_acc.legend(loc='lower right', fontsize=9)
    a_aclr_acc.grid(True, alpha=0.3)

    fig_a.tight_layout()

    def update(frame_idx):
        b = frame_idx
        line_psd.set_data(f_mhz_psd, block_psd_list[b])
        acc_tag = "" if accepted_arr[b] else "  [REJECTED]"
        psd_annot.set_text(
            f'Block: {blocks_arr[b]:.0f}{acc_tag}\n'
            f'NMSE:       {nmse_arr[b]:+.2f} dB\n'
            f'ACLR lower: {aclr_lo_arr[b]:.1f} dB\n'
            f'ACLR upper: {aclr_hi_arr[b]:.1f} dB'
        )
        a_psd.set_title(
            f'NR {bw_mhz} MHz SCS={scs_khz} kHz — '
            f'Block {blocks_arr[b]:.0f}/{n_blocks}  '
            f'(PSD @ {fs_psd/1e6:.0f} MHz)',
            fontsize=11)

        line_nmse.set_data(blocks_arr[:b+1], nmse_arr[:b+1])
        marker_nmse.set_data([blocks_arr[b]], [nmse_arr[b]])

        n_coeff = min(b, len(coeff_blocks_arr))
        if n_coeff > 0:
            line_coeff.set_data(coeff_blocks_arr[:n_coeff],
                                coeff_nmse_arr[:n_coeff])
            c_acc = coeff_accepted[:n_coeff]
            c_rej = ~c_acc
            if np.any(c_acc):
                scat_coeff_acc.set_offsets(
                    np.column_stack([coeff_blocks_arr[:n_coeff][c_acc],
                                    coeff_nmse_arr[:n_coeff][c_acc]]))
            if np.any(c_rej):
                scat_coeff_rej.set_offsets(
                    np.column_stack([coeff_blocks_arr[:n_coeff][c_rej],
                                    coeff_nmse_arr[:n_coeff][c_rej]]))

        line_aclr_lo.set_data(blocks_arr[:b+1], aclr_lo_arr[:b+1])
        line_aclr_hi.set_data(blocks_arr[:b+1], aclr_hi_arr[:b+1])

        n_acc_so_far = int(accepted_arr[:b+1].sum())
        if n_acc_so_far > 0:
            idx_acc = np.arange(n_acc_so_far)
            line_acc_lo.set_data(idx_acc, acc_aclr_lo[:n_acc_so_far])
            line_acc_hi.set_data(idx_acc, acc_aclr_hi[:n_acc_so_far])

        return (line_psd, psd_annot, line_nmse, marker_nmse,
                line_coeff, scat_coeff_acc, scat_coeff_rej,
                line_aclr_lo, line_aclr_hi, line_acc_lo, line_acc_hi)

    anim = FuncAnimation(fig_a, update, frames=n_blocks,
                         blit=False, repeat=False)

    gif_file = 'gmp_dpd_block_nr100_v3_convergence.gif'
    writer = PillowWriter(fps=4)
    anim.save(gif_file, writer=writer, dpi=120)
    print(f"Animation saved to {gif_file}")
    plt.close('all')


if __name__ == '__main__':
    main()
