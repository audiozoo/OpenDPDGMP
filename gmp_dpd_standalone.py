"""
Standalone GMP (Generalized Memory Polynomial) Digital Pre-Distortion.

Implements the Indirect Learning Architecture (ILA) with closed-form
least-squares identification, following Morgan et al., IEEE TSP 2006.

The ILA works in two phases:
  1. IDENTIFY: Given paired (PA_input, PA_output) training data, build a
     polynomial basis from the normalized PA output and solve for the
     postdistorter coefficients via least squares.
  2. APPLY: Use those same coefficients as a predistorter on new input
     signals before they enter the PA.

No external dependencies beyond NumPy (and optionally Matplotlib for plots).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# GMP Basis Construction
# ---------------------------------------------------------------------------

def _delay(x: np.ndarray, d: int) -> np.ndarray:
    """Delay complex signal *x* by *d* samples (zero-padded)."""
    N = len(x)
    out = np.zeros(N, dtype=np.complex128)
    if 0 <= d < N:
        out[d:] = x[:N - d]
    elif d < 0 and -d < N:
        out[:N + d] = x[-d:]
    return out


def build_gmp_basis(
    x: np.ndarray,
    Ka: int, La: int,
    Kb: int, Lb: int, Mb: int,
    Kc: int, Lc: int, Mc: int,
) -> np.ndarray:
    """
    Build the Generalized Memory Polynomial basis matrix.

    Three term groups (Morgan et al., IEEE TSP 2006):

        Aligned:  x(n-q) |x(n-q)|^k        k=0..Ka-1, q=0..La-1
        Lagging:  x(n-q) |x(n-q-l)|^k       k=1..Kb,   q=0..Lb-1, l=1..Mb
        Leading:  x(n-q) |x(n-q+l)|^k       k=1..Kc,   q=0..Lc-1, l=1..Mc

    Parameters
    ----------
    x : complex ndarray, shape (N,)
    Ka, La : aligned nonlinearity orders and memory depth
    Kb, Lb, Mb : lagging orders, memory, cross-term depth
    Kc, Lc, Mc : leading orders, memory, cross-term depth

    Returns
    -------
    Phi : complex ndarray, shape (N, n_coeffs)
        where n_coeffs = Ka*La + Kb*Lb*Mb + Kc*Lc*Mc
    """
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


def build_mp_basis(x: np.ndarray, K: int, Q: int) -> np.ndarray:
    """
    Build a Memory Polynomial basis matrix (GMP aligned-only subset).

    Basis:  x(n-q) |x(n-q)|^k,   k=0..K-1, q=0..Q-1

    Parameters
    ----------
    x : complex ndarray, shape (N,)
    K : nonlinearity orders
    Q : memory depth (delay taps)

    Returns
    -------
    Phi : complex ndarray, shape (N, K*Q)
    """
    return build_gmp_basis(x, Ka=K, La=Q, Kb=0, Lb=0, Mb=0, Kc=0, Lc=0, Mc=0)


# ---------------------------------------------------------------------------
# GMP DPD Configuration
# ---------------------------------------------------------------------------

@dataclass
class GMPConfig:
    """Parameters for the GMP basis."""
    Ka: int = 5
    La: int = 4
    Kb: int = 3
    Lb: int = 4
    Mb: int = 2
    Kc: int = 3
    Lc: int = 4
    Mc: int = 1

    @property
    def n_coeffs(self) -> int:
        return self.Ka * self.La + self.Kb * self.Lb * self.Mb + self.Kc * self.Lc * self.Mc


# ---------------------------------------------------------------------------
# ILA Identification & Application
# ---------------------------------------------------------------------------

def iq_to_complex(iq: np.ndarray) -> np.ndarray:
    """Convert (N,2) real I/Q array to (N,) complex."""
    return iq[:, 0] + 1j * iq[:, 1]


def complex_to_iq(c: np.ndarray) -> np.ndarray:
    """Convert (N,) complex to (N,2) real I/Q array."""
    return np.column_stack([c.real, c.imag])


def compute_target_gain(pa_input: np.ndarray, pa_output: np.ndarray) -> float:
    """
    Estimate the small-signal linear gain of the PA.

    Uses the ratio of peak output amplitude to peak input amplitude,
    matching the OpenDPD convention.

    Parameters
    ----------
    pa_input  : (N,2) I/Q array — signal fed into the PA
    pa_output : (N,2) I/Q array — signal measured at PA output
    """
    amp_in = np.sqrt(pa_input[:, 0]**2 + pa_input[:, 1]**2)
    amp_out = np.sqrt(pa_output[:, 0]**2 + pa_output[:, 1]**2)
    return np.max(amp_out) / np.max(amp_in)


def identify_gmp_coefficients(
    pa_input_iq: np.ndarray,
    pa_output_iq: np.ndarray,
    cfg: GMPConfig,
    target_gain: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Identify GMP pre-distorter coefficients using the Indirect Learning
    Architecture (ILA) with closed-form least squares.

    ILA procedure:
      1. Normalize PA output by the linear target gain:  z_norm = z / G
      2. Build GMP basis Phi from z_norm  (this is the *postdistorter* input)
      3. Solve  x = Phi w  for w via least squares  (postdistorter coeffs)
      4. The same w is then used as the *predistorter*

    Parameters
    ----------
    pa_input_iq  : (N, 2) float — PA input I/Q training data
    pa_output_iq : (N, 2) float — PA output I/Q training data
    cfg          : GMPConfig with basis dimensions
    target_gain  : linear gain G.  Estimated from data if None.

    Returns
    -------
    w            : complex coefficient vector, shape (n_coeffs,)
    target_gain  : the linear gain that was used
    """
    if target_gain is None:
        target_gain = compute_target_gain(pa_input_iq, pa_output_iq)

    x_c = iq_to_complex(pa_input_iq)
    z_c = iq_to_complex(pa_output_iq)

    z_norm = z_c / target_gain

    Phi = build_gmp_basis(z_norm, cfg.Ka, cfg.La, cfg.Kb, cfg.Lb, cfg.Mb,
                          cfg.Kc, cfg.Lc, cfg.Mc)

    w, _, rank, _ = np.linalg.lstsq(Phi, x_c, rcond=None)
    print(f"[identify] coefficients: {cfg.n_coeffs} complex "
          f"({2*cfg.n_coeffs} real),  LS rank: {rank}")
    return w, target_gain


def apply_gmp_predistortion(
    input_iq: np.ndarray,
    w: np.ndarray,
    cfg: GMPConfig,
) -> np.ndarray:
    """
    Apply GMP pre-distortion to an input signal.

    Parameters
    ----------
    input_iq : (N, 2) float — clean transmit I/Q signal
    w        : complex coefficient vector from identify_gmp_coefficients
    cfg      : same GMPConfig used during identification

    Returns
    -------
    predistorted_iq : (N, 2) float — signal to feed into the PA
    """
    x_c = iq_to_complex(input_iq)
    Phi = build_gmp_basis(x_c, cfg.Ka, cfg.La, cfg.Kb, cfg.Lb, cfg.Mb,
                          cfg.Kc, cfg.Lc, cfg.Mc)
    x_dpd_c = Phi @ w
    return complex_to_iq(x_dpd_c)


# ---------------------------------------------------------------------------
# Quality Metrics
# ---------------------------------------------------------------------------

def nmse_db(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Normalized Mean Squared Error in dB.  Inputs are (N,2) I/Q arrays."""
    err = prediction - reference
    mse = np.mean(err[:, 0]**2 + err[:, 1]**2)
    pwr = np.mean(reference[:, 0]**2 + reference[:, 1]**2)
    return 10.0 * np.log10(mse / pwr)


# ---------------------------------------------------------------------------
# Odd-order Memory Polynomial PA Model  (Eq. 11)
#
#   y(n) = sum_{k=1,3,...,K}  sum_{q=0}^{Q}  c_{k,q}  z(n-q) |z(n-q)|^{k-1}
#
# ---------------------------------------------------------------------------

# PA coefficients c_{k,q} — random complex values (K=5, Q=2, odd k only)
_rng_pa = np.random.RandomState(7)
PA_COEFFS = {}
for _k in (1, 3, 5):
    for _q in range(3):
        mag = {1: 1.0, 3: 0.3, 5: 0.8}[_k] * np.exp(-0.4 * _q)
        PA_COEFFS[(_k, _q)] = mag * np.exp(1j * _rng_pa.uniform(-np.pi, np.pi))
del _rng_pa, _k, _q, mag


def memory_polynomial_pa(
    x_iq: np.ndarray,
    coeffs: dict = PA_COEFFS,
    K: int = 5,
    Q: int = 2,
) -> np.ndarray:
    """
    Simulate a PA using the odd-order Memory Polynomial model.

        y(n) = sum_{k=1,3,...,K}  sum_{q=0}^{Q}  c_{k,q}  z(n-q) |z(n-q)|^{k-1}

    Parameters
    ----------
    x_iq   : (N, 2) I/Q input signal  (z in the equation)
    coeffs : dict mapping (k, q) → complex coefficient c_{k,q}
    K      : maximum nonlinearity order (odd)
    Q      : maximum memory depth

    Returns
    -------
    y_iq : (N, 2) I/Q output signal
    """
    z = iq_to_complex(x_iq)
    N = len(z)
    y = np.zeros(N, dtype=np.complex128)

    for k in range(1, K + 1, 2):                    # odd k: 1, 3, 5
        for q in range(Q + 1):                       # q: 0, 1, 2
            c_kq = coeffs.get((k, q), 0.0)
            if c_kq == 0.0:
                continue
            zq = _delay(z, q)
            y += c_kq * zq * np.abs(zq) ** (k - 1)

    return complex_to_iq(y)


# ---------------------------------------------------------------------------
# 5G NR OFDM Signal Generator
# ---------------------------------------------------------------------------

def generate_64qam(n: int) -> np.ndarray:
    """Generate *n* random 64-QAM symbols (unit average power)."""
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    i = np.random.choice(levels, n)
    q = np.random.choice(levels, n)
    return (i + 1j * q) / np.sqrt(42.0)


def generate_5g_nr_ofdm(
    n_symbols: int,
    n_rb: int = 51,
    scs: float = 30e3,
    fs: float = 122.88e6,
) -> np.ndarray:
    """
    Generate a 5G NR-like OFDM time-domain signal (complex baseband).

    Parameters
    ----------
    n_symbols : number of OFDM symbols to generate
    n_rb      : resource blocks (51 for 20 MHz BW with 30 kHz SCS, TS 38.104)
    scs       : subcarrier spacing [Hz]
    fs        : sampling rate [Hz]

    Returns
    -------
    signal : complex ndarray — time-domain baseband samples
    """
    n_fft = int(fs / scs)                       # 4096
    n_sc = n_rb * 12                             # 612 active subcarriers
    n_sc_half = n_sc // 2                        # 306 per side

    # Normal CP lengths per slot (14 symbols) at 122.88 MHz, μ=1
    # Symbols 0 and 7 carry extended CP; total CP per slot = 4096 samples
    cp_per_slot = [320, 288, 288, 288, 288, 288, 288,
                   320, 288, 288, 288, 288, 288, 288]

    parts = []
    for s in range(n_symbols):
        data = generate_64qam(n_sc)

        freq = np.zeros(n_fft, dtype=np.complex128)
        freq[1 : n_sc_half + 1]      = data[n_sc_half:]    # positive freqs
        freq[n_fft - n_sc_half :]     = data[:n_sc_half]    # negative freqs

        td = np.fft.ifft(freq) * np.sqrt(n_fft)

        cp_len = cp_per_slot[s % 14]
        parts.append(np.concatenate([td[-cp_len:], td]))

    return np.concatenate(parts)


# ===================================================================
# Example / Demo  —  5G NR 20 MHz, SCS = 30 kHz
# ===================================================================

def main():
    np.random.seed(42)

    # ==================================================================
    # 5G NR signal parameters
    # ==================================================================
    scs       = 30e3        # subcarrier spacing
    bw_mhz    = 20          # channel bandwidth
    n_rb      = 51          # RBs for 20 MHz @ 30 kHz SCS  (TS 38.104)
    fs        = 122.88e6    # sampling rate  → ±61.44 MHz PSD span
    n_fft     = int(fs / scs)   # 4096

    n_train_symbols = 200   # ~870 k samples
    n_test_symbols  = 50    # ~218 k samples

    print(f"5G NR signal: BW={bw_mhz} MHz, SCS={scs/1e3:.0f} kHz, "
          f"N_RB={n_rb}, N_sc={n_rb*12}, N_FFT={n_fft}, fs={fs/1e6:.2f} MHz")

    # ------------------------------------------------------------------
    # 1. Generate OFDM signals
    # ------------------------------------------------------------------
    x_train_c = generate_5g_nr_ofdm(n_train_symbols, n_rb=n_rb, scs=scs, fs=fs)
    x_test_c  = generate_5g_nr_ofdm(n_test_symbols,  n_rb=n_rb, scs=scs, fs=fs)

    # Scale signal so the PA operates in mild compression.
    # The PA coefficients (Eq. 12) are designed for signals with peak
    # amplitude near 0.5, where the k=3 and k=5 terms contribute visibly.
    target_rms = 0.15
    x_train_c *= target_rms / np.sqrt(np.mean(np.abs(x_train_c)**2))
    x_test_c  *= target_rms / np.sqrt(np.mean(np.abs(x_test_c)**2))

    x_train = complex_to_iq(x_train_c)
    x_test  = complex_to_iq(x_test_c)

    papr_db = 10 * np.log10(np.max(np.abs(x_test_c)**2) /
                             np.mean(np.abs(x_test_c)**2))
    print(f"Test signal PAPR: {papr_db:.1f} dB  "
          f"({len(x_test_c)} samples, {len(x_test_c)/fs*1e3:.2f} ms)")

    # ------------------------------------------------------------------
    # 2. Memory Polynomial PA  (Eq. 11, K=5 odd-only, Q=2)
    #
    #   y(n) = sum_{k=1,3,5} sum_{q=0}^{2} c_{k,q} z(n-q)|z(n-q)|^{k-1}
    #
    # ------------------------------------------------------------------
    pa_K = 5
    pa_Q = 2
    n_pa_coeffs = len(PA_COEFFS)
    print(f"PA model: Memory Polynomial (Eq. 11), K={pa_K} (odd only), Q={pa_Q}")
    print(f"  Coefficients: {n_pa_coeffs} complex ({2*n_pa_coeffs} real)")
    for (k, q), c in sorted(PA_COEFFS.items()):
        print(f"    c_{{{k},{q}}} = {c.real:+.4f} {c.imag:+.4f}j")

    def pa(x_iq):
        return memory_polynomial_pa(x_iq, PA_COEFFS, K=pa_K, Q=pa_Q)

    y_train       = pa(x_train)
    y_test_no_dpd = pa(x_test)

    # ------------------------------------------------------------------
    # 3. Identify GMP DPD (ILA, closed-form least squares)
    # ------------------------------------------------------------------
    cfg = GMPConfig(Ka=7, La=5, Kb=3, Lb=5, Mb=2, Kc=3, Lc=5, Mc=1)
    w, target_gain = identify_gmp_coefficients(x_train, y_train, cfg)

    print(f"[info] target gain G = {target_gain:.4f}")
    print(f"[info] GMP config: {cfg}")

    # ------------------------------------------------------------------
    # 4. Apply DPD → PA
    # ------------------------------------------------------------------
    x_test_dpd      = apply_gmp_predistortion(x_test, w, cfg)
    y_test_with_dpd  = pa(x_test_dpd)

    ideal_output = target_gain * x_test

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    nmse_before = nmse_db(y_test_no_dpd, ideal_output)
    nmse_after  = nmse_db(y_test_with_dpd, ideal_output)

    print(f"\n{'='*50}")
    print(f"  NMSE without DPD : {nmse_before:+.2f} dB")
    print(f"  NMSE with GMP DPD: {nmse_after:+.2f} dB")
    print(f"  Improvement      : {nmse_before - nmse_after:.2f} dB")
    print(f"{'='*50}")

    # ------------------------------------------------------------------
    # 6. PSD Plot  (frequency axis: −61.44 … +61.44 MHz)
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.signal import welch

        nperseg_psd = n_fft  # one full OFDM symbol → Δf = SCS = 30 kHz

        def compute_psd(iq):
            c = iq_to_complex(iq) if iq.ndim == 2 else iq
            f, p = welch(c, fs=fs, nperseg=nperseg_psd,
                         noverlap=nperseg_psd // 2,
                         return_onesided=False, scaling='density')
            idx = np.argsort(f)
            return f[idx], p[idx]

        f_hz, psd_input   = compute_psd(ideal_output)
        _,    psd_no_dpd  = compute_psd(y_test_no_dpd)
        _,    psd_with_dpd = compute_psd(y_test_with_dpd)

        f_mhz = f_hz / 1e6
        ref_power = np.max(psd_input)

        psd_input_db   = 10 * np.log10(psd_input    / ref_power)
        psd_no_dpd_db  = 10 * np.log10(psd_no_dpd   / ref_power)
        psd_with_dpd_db = 10 * np.log10(psd_with_dpd / ref_power)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(f_mhz, psd_input_db,    color='black',    linewidth=1.2,
                linestyle='--', label='Input (ideal linear output)')
        ax.plot(f_mhz, psd_no_dpd_db,   color='#1f77b4',  linewidth=1.0,
                label='PA output — no DPD')
        ax.plot(f_mhz, psd_with_dpd_db, color='#d62728',  linewidth=1.0,
                label='PA output — with GMP DPD')

        # Channel bandwidth shading
        ax.axvspan(-bw_mhz / 2, bw_mhz / 2, color='green', alpha=0.06,
                   label=f'{bw_mhz} MHz channel')

        ax.set_xlim(-61.44, 61.44)
        ax.set_ylim(-80, 5)
        ax.set_xlabel('Frequency (MHz)', fontsize=12)
        ax.set_ylabel('PSD (dB, normalized)', fontsize=12)
        ax.set_title(f'5G NR {bw_mhz} MHz · SCS {scs/1e3:.0f} kHz · '
                     f'MP PA (K={pa_K} odd, Q={pa_Q}) · '
                     f'GMP DPD (Ka={cfg.Ka} La={cfg.La} Kb={cfg.Kb} '
                     f'Lb={cfg.Lb} Mb={cfg.Mb} Kc={cfg.Kc} Lc={cfg.Lc} '
                     f'Mc={cfg.Mc})',
                     fontsize=11)
        ax.legend(loc='lower center', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig('gmp_dpd_5gnr_psd.png', dpi=150)
        print(f"\nPSD plot saved to gmp_dpd_5gnr_psd.png")
        plt.show()

    except ImportError:
        print("\n(matplotlib/scipy not available — skipping plots)")


if __name__ == '__main__':
    main()
