"""
Generate a 3-carrier WCDMA signal and plot its PSD over -20 to +20 MHz.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal.windows import blackmanharris


# ---------------------------------------------------------------------------
# WCDMA signal generation
# ---------------------------------------------------------------------------

def generate_wcdma(n_samples, n_codes=8, chip_rate=3.84e6, fs=61.44e6, sf=16):
    """Generate a single-carrier WCDMA baseband signal."""
    oversample = int(round(fs / chip_rate))
    n_chips = n_samples // oversample + sf

    def _ovsf(sf_val, code_idx):
        h = np.array([[1]])
        while h.shape[0] < sf_val:
            h = np.block([[h, h], [h, -h]])
        return h[code_idx % sf_val]

    composite = np.zeros(n_chips, dtype=np.complex128)
    n_symbols_per_code = n_chips // sf

    for c in range(n_codes):
        qpsk = (np.sign(np.random.randn(n_symbols_per_code))
                + 1j * np.sign(np.random.randn(n_symbols_per_code))) / np.sqrt(2)
        spread = np.repeat(qpsk, sf) * np.tile(_ovsf(sf, c + 1), n_symbols_per_code)
        composite[:len(spread)] += spread

    composite /= np.sqrt(n_codes)

    upsampled = np.zeros(n_chips * oversample, dtype=np.complex128)
    upsampled[::oversample] = composite

    alpha = 0.22
    span_syms = 12
    t_rrc = np.arange(-span_syms * oversample, span_syms * oversample + 1) / oversample
    eps = 1e-12
    h_rrc = np.where(
        np.abs(t_rrc) < eps,
        1.0 - alpha + 4.0 * alpha / np.pi,
        np.where(
            np.abs(np.abs(t_rrc) - 1.0 / (4.0 * alpha)) < eps,
            alpha / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            ),
            (np.sin(np.pi * t_rrc * (1 - alpha))
             + 4 * alpha * t_rrc * np.cos(np.pi * t_rrc * (1 + alpha))
            ) / (np.pi * t_rrc * (1 - (4 * alpha * t_rrc) ** 2 + eps))
        )
    )
    h_rrc /= np.sqrt(np.sum(h_rrc ** 2))
    h_rrc *= oversample

    filtered = np.convolve(upsampled, h_rrc, mode='full')
    start = len(h_rrc) // 2
    return filtered[start : start + n_samples]


def generate_multicarrier_wcdma(n_samples, n_carriers=3, carrier_spacing=5e6,
                                n_codes=8, chip_rate=3.84e6, fs=61.44e6, sf=16):
    """Generate a multi-carrier WCDMA baseband signal."""
    composite = np.zeros(n_samples, dtype=np.complex128)
    t = np.arange(n_samples) / fs
    f_centres = (np.arange(n_carriers) - (n_carriers - 1) / 2.0) * carrier_spacing

    for fc in f_centres:
        carrier = generate_wcdma(n_samples, n_codes=n_codes,
                                 chip_rate=chip_rate, fs=fs, sf=sf)
        composite += carrier * np.exp(1j * 2 * np.pi * fc * t)

    composite /= np.sqrt(n_carriers)
    return composite


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    chip_rate = 3.84e6
    fs = 61.44e6
    n_carriers = 3
    carrier_spacing = 5e6
    carrier_bw_mhz = 5
    n_codes = 16
    N = 1_000_000

    print(f"Generating {n_carriers}-carrier WCDMA signal: "
          f"{carrier_bw_mhz} MHz/carrier, {chip_rate/1e6:.2f} Mcps, "
          f"{n_codes} codes/carrier, fs={fs/1e6:.2f} MHz, {N} samples")

    signal = generate_multicarrier_wcdma(
        N, n_carriers=n_carriers, carrier_spacing=carrier_spacing,
        n_codes=n_codes, chip_rate=chip_rate, fs=fs)

    papr_db = 10 * np.log10(np.max(np.abs(signal)**2) / np.mean(np.abs(signal)**2))
    print(f"PAPR: {papr_db:.1f} dB, duration: {N/fs*1e3:.2f} ms")

    # PSD via Welch with Blackman-Harris window
    nperseg = 8192
    noverlap = nperseg * 3 // 4
    window = blackmanharris(nperseg)

    f, psd = welch(signal, fs=fs, nperseg=nperseg, window=window,
                   noverlap=noverlap, return_onesided=False, scaling='density')
    idx = np.argsort(f)
    f_mhz = f[idx] / 1e6
    psd_sorted = psd[idx]
    psd_db = 10 * np.log10(psd_sorted / np.max(psd_sorted))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(f_mhz, psd_db, color='#1f77b4', linewidth=1.0)

    for ci in range(n_carriers):
        fc = (ci - (n_carriers - 1) / 2.0) * carrier_spacing / 1e6
        lo, hi = fc - carrier_bw_mhz / 2, fc + carrier_bw_mhz / 2
        lbl = f'{carrier_bw_mhz} MHz carrier' if ci == 0 else None
        ax.axvspan(lo, hi, color='green', alpha=0.06, label=lbl)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-80, 5)
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('PSD (dB, normalized)', fontsize=12)
    ax.set_title(f'{n_carriers}× WCDMA {carrier_bw_mhz} MHz · '
                 f'{chip_rate/1e6:.2f} Mcps · {n_codes} codes/carrier · '
                 f'fs={fs/1e6:.2f} MHz',
                 fontsize=13)
    ax.legend(loc='lower center', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('wcdma_3carrier_psd.png', dpi=150)
    print(f"Plot saved to wcdma_3carrier_psd.png")
    plt.show()


if __name__ == '__main__':
    main()
