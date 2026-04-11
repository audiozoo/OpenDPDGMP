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


def aclr_db(
    signal_iq: np.ndarray,
    fs: float,
    channel_bw: float,
    adjacent_offset: Optional[float] = None,
    nperseg: int = 4096,
) -> Tuple[float, float]:
    """
    Compute Adjacent Channel Leakage Ratio (ACLR) per 3GPP TS 38.104.

    ACLR = 10 log10( P_channel / P_adjacent )

    Measured separately for the lower and upper adjacent channels.

    Parameters
    ----------
    signal_iq      : (N, 2) I/Q signal to measure
    fs             : sampling rate [Hz]
    channel_bw     : channel bandwidth [Hz] (e.g. 20e6)
    adjacent_offset: center-to-center spacing to adjacent channel [Hz].
                     Defaults to channel_bw (the standard NR definition).
    nperseg        : Welch segment length

    Returns
    -------
    aclr_lower : ACLR for the lower adjacent channel [dB] (positive = good)
    aclr_upper : ACLR for the upper adjacent channel [dB] (positive = good)
    """
    from scipy.signal import welch as _welch

    if adjacent_offset is None:
        adjacent_offset = channel_bw

    c = iq_to_complex(signal_iq) if signal_iq.ndim == 2 else signal_iq
    f, psd = _welch(c, fs=fs, nperseg=nperseg,
                    noverlap=nperseg // 2,
                    return_onesided=False, scaling='density')

    df = f[1] - f[0]

    def band_power(f_lo, f_hi):
        mask = (f >= f_lo) & (f < f_hi)
        return np.sum(psd[mask]) * df

    p_main  = band_power(-channel_bw / 2, channel_bw / 2)
    p_lower = band_power(-adjacent_offset - channel_bw / 2,
                         -adjacent_offset + channel_bw / 2)
    p_upper = band_power( adjacent_offset - channel_bw / 2,
                          adjacent_offset + channel_bw / 2)

    aclr_lower = 10.0 * np.log10(p_main / p_lower) if p_lower > 0 else np.inf
    aclr_upper = 10.0 * np.log10(p_main / p_upper) if p_upper > 0 else np.inf
    return aclr_lower, aclr_upper


# ---------------------------------------------------------------------------
# Odd-order Memory Polynomial PA Model  (Eq. 11)
#
#   y(n) = sum_{k=1,3,...,K}  sum_{q=0}^{Q}  c_{k,q}  z(n-q) |z(n-q)|^{k-1}
#
# ---------------------------------------------------------------------------

# PA coefficients c_{k,q} from Eq. 12  (K=5, Q=2, odd k only)
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
# WCDMA Signal Generator
# ---------------------------------------------------------------------------

def generate_wcdma(
    n_samples: int,
    n_codes: int = 8,
    chip_rate: float = 3.84e6,
    fs: float = 30.72e6,
    sf: int = 16,
) -> np.ndarray:
    """
    Generate a WCDMA-like downlink baseband signal.

    Sums multiple OVSF-spread QPSK channels, each at a given spreading
    factor, then upsamples to the target sampling rate using an RRC filter.

    Parameters
    ----------
    n_samples : desired number of output samples at rate *fs*
    n_codes   : number of simultaneously active channelisation codes
    chip_rate : WCDMA chip rate [Hz] (3.84 Mcps)
    fs        : output sampling rate [Hz]
    sf        : spreading factor per code channel

    Returns
    -------
    signal : complex ndarray, shape (n_samples,) — baseband at rate fs
    """
    oversample = int(round(fs / chip_rate))  # 8 for 30.72 MHz
    n_chips = n_samples // oversample + sf

    # OVSF codes (Walsh-Hadamard rows)
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

    # Upsample by inserting zeros then apply RRC pulse-shaping filter
    upsampled = np.zeros(n_chips * oversample, dtype=np.complex128)
    upsampled[::oversample] = composite

    # RRC filter (roll-off 0.22 per 3GPP, 12-symbol span)
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
    rrc_out = filtered[start : start + n_samples]

    # Channel FIR filter: -80 dB sidelobes, passband = occupied BW
    from scipy.signal import firwin, kaiserord
    passband_edge = chip_rate * (1 + alpha) / 2
    stopband_edge = passband_edge + 0.5e6
    transition_width = stopband_edge - passband_edge
    numtaps, beta = kaiserord(80, transition_width / (fs / 2))
    if numtaps % 2 == 0:
        numtaps += 1
    cutoff = (passband_edge + stopband_edge) / 2
    h_chan = firwin(numtaps, cutoff, window=('kaiser', beta), fs=fs)
    chan_filtered = np.convolve(rrc_out, h_chan, mode='full')
    start2 = len(h_chan) // 2
    return chan_filtered[start2 : start2 + n_samples]


def generate_multicarrier_wcdma(
    n_samples: int,
    n_carriers: int = 3,
    carrier_spacing: float = 5e6,
    n_codes: int = 8,
    chip_rate: float = 3.84e6,
    fs: float = 61.44e6,
    sf: int = 16,
) -> np.ndarray:
    """
    Generate a multi-carrier WCDMA baseband signal.

    Each carrier is an independent WCDMA signal frequency-shifted to its
    centre frequency.  Carriers are placed symmetrically around DC with
    *carrier_spacing* between adjacent centres.

    Parameters
    ----------
    n_samples       : desired output length at rate *fs*
    n_carriers      : number of WCDMA carriers
    carrier_spacing : centre-to-centre spacing [Hz] (typically 5 MHz)
    n_codes         : active codes per carrier
    chip_rate       : WCDMA chip rate [Hz]
    fs              : output sampling rate [Hz]
    sf              : spreading factor per code

    Returns
    -------
    signal : complex ndarray, shape (n_samples,)
    """
    composite = np.zeros(n_samples, dtype=np.complex128)
    t = np.arange(n_samples) / fs

    f_centres = (np.arange(n_carriers) - (n_carriers - 1) / 2.0) * carrier_spacing

    for i, fc in enumerate(f_centres):
        carrier = generate_wcdma(n_samples, n_codes=n_codes,
                                 chip_rate=chip_rate, fs=fs, sf=sf)
        composite += carrier * np.exp(1j * 2 * np.pi * fc * t)

    composite /= np.sqrt(n_carriers)
    return composite


# ===================================================================
# Example / Demo  —  3× WCDMA carriers (15 MHz aggregate)
# ===================================================================

def main():
    np.random.seed(42)

    # ==================================================================
    # Multi-carrier WCDMA signal parameters
    # ==================================================================
    chip_rate       = 3.84e6    # 3.84 Mcps
    carrier_bw_mhz  = 5         # per-carrier BW
    n_carriers      = 3
    carrier_spacing = 5e6       # 5 MHz centre-to-centre
    bw_mhz          = n_carriers * carrier_bw_mhz   # 15 MHz aggregate
    fs              = 61.44e6   # sampling rate → ±30.72 MHz Nyquist
    n_codes         = 16        # active codes per carrier (full SF)

    N_train = 1_000_000
    N_test  = 1_000_000

    print(f"WCDMA signal: {n_carriers}× carriers, {carrier_bw_mhz} MHz each, "
          f"aggregate BW={bw_mhz} MHz, chip rate={chip_rate/1e6:.2f} Mcps, "
          f"{n_codes} codes/carrier, fs={fs/1e6:.2f} MHz")

    # ------------------------------------------------------------------
    # 1. Generate multi-carrier WCDMA signals
    # ------------------------------------------------------------------
    x_train_c = generate_multicarrier_wcdma(
        N_train, n_carriers=n_carriers, carrier_spacing=carrier_spacing,
        n_codes=n_codes, chip_rate=chip_rate, fs=fs)
    x_test_c = generate_multicarrier_wcdma(
        N_test, n_carriers=n_carriers, carrier_spacing=carrier_spacing,
        n_codes=n_codes, chip_rate=chip_rate, fs=fs)

    # Scale signal to drive the PA into moderate compression.
    target_rms = 0.22
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

    channel_bw_hz = bw_mhz * 1e6
    nperseg_metric = 8192
    aclr_lo_no,  aclr_hi_no  = aclr_db(y_test_no_dpd,  fs, channel_bw_hz, nperseg=nperseg_metric)
    aclr_lo_dpd, aclr_hi_dpd = aclr_db(y_test_with_dpd, fs, channel_bw_hz, nperseg=nperseg_metric)
    aclr_lo_ideal, aclr_hi_ideal = aclr_db(ideal_output, fs, channel_bw_hz, nperseg=nperseg_metric)

    print(f"\n{'='*60}")
    print(f"  {'Metric':<28s} {'No DPD':>10s} {'GMP DPD':>10s} {'Ideal':>10s}")
    print(f"  {'-'*56}")
    print(f"  {'NMSE (dB)':<28s} {nmse_before:>+10.2f} {nmse_after:>+10.2f} {'—':>10s}")
    print(f"  {'ACLR lower (dB)':<28s} {aclr_lo_no:>10.2f} {aclr_lo_dpd:>10.2f} {aclr_lo_ideal:>10.2f}")
    print(f"  {'ACLR upper (dB)':<28s} {aclr_hi_no:>10.2f} {aclr_hi_dpd:>10.2f} {aclr_hi_ideal:>10.2f}")
    print(f"  {'-'*56}")
    print(f"  NMSE improvement: {nmse_before - nmse_after:.2f} dB")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 6. PSD Plot  (frequency axis: −20 … +20 MHz)
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.signal import welch

        from scipy.signal.windows import blackmanharris
        nperseg_psd = 8192
        noverlap_psd = nperseg_psd * 3 // 4   # 75% overlap
        psd_window = blackmanharris(nperseg_psd)

        def compute_psd(iq):
            c = iq_to_complex(iq) if iq.ndim == 2 else iq
            f, p = welch(c, fs=fs, nperseg=nperseg_psd, window=psd_window,
                         noverlap=noverlap_psd,
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
                     f'{chip_rate/1e6:.2f} Mcps · '
                     f'{n_codes} codes/carrier · '
                     f'MP PA (K={pa_K} odd, Q={pa_Q}) · '
                     f'GMP DPD (Ka={cfg.Ka} La={cfg.La} Kb={cfg.Kb} '
                     f'Lb={cfg.Lb} Mb={cfg.Mb} Kc={cfg.Kc} Lc={cfg.Lc} '
                     f'Mc={cfg.Mc})',
                     fontsize=11)
        ax.legend(loc='lower center', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig('gmp_dpd_wcdma_psd.png', dpi=150)
        print(f"\nPSD plot saved to gmp_dpd_wcdma_psd.png")
        plt.show()

    except ImportError:
        print("\n(matplotlib/scipy not available — skipping plots)")


if __name__ == '__main__':
    main()
