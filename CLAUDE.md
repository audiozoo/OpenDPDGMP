# OpenDPDGMP — Claude Code Instructions

## Git & GitHub Workflow

This project uses Git for version control. **All work must be committed and pushed to GitHub regularly** so that progress is never lost and changes can be reverted at any time.

### Rules

- After completing any meaningful unit of work (new feature, bug fix, refactor, config change), commit it immediately.
- Write clean, descriptive commit messages in the imperative mood (e.g. "Add user authentication", "Fix off-by-one error in score calculation").
- Push to the `main` branch on GitHub after every commit (or logical batch of commits).
- Never let uncommitted changes accumulate — commit early, commit often.
- Before starting a new task, ensure the working tree is clean (`git status`).

### Commit Message Format

```
<short summary in imperative mood> (50 chars or less)

<optional body: explain what and why, not how>
```

Examples of good commit messages:
- `Initialize project structure`
- `Add CLAUDE.md with project conventions`
- `Fix calculation error in damage multiplier`

### GitHub Repository

- Remote: `https://github.com/audiozoo/OpenDPDGMP`
- Branch: `main`

### Workflow Steps

1. Make changes to files.
2. Stage relevant files: `git add <files>`
3. Commit with a clean message: `git commit -m "..."`
4. Push: `git push origin main`

This ensures there is always a saved, recoverable state of the project on GitHub.

---

## Project Summary

This project implements **GMP (Generalized Memory Polynomial) Digital Pre-Distortion (DPD)** for power amplifier linearization, using the **Indirect Learning Architecture (ILA)** with closed-form least-squares coefficient identification.

### Origin

The project started by cloning the [lab-emi/OpenDPD](https://github.com/lab-emi/OpenDPD) repository and studying its GMP implementation. Two GMP implementations exist in that repo: a PyTorch neural network version (`backbones/gmp.py`) and a NumPy closed-form version (`benchmark_volterra_qr.py`). This project builds on the classical closed-form approach.

### Scripts

| File | Purpose |
|------|---------|
| `gmp_dpd_standalone.py` | Main DPD simulation — signal generation, PA model, GMP ILA identification, DPD application, metrics (NMSE + ACLR), and PSD plotting |
| `wcdma_3carrier_psd.py` | Standalone 3-carrier WCDMA signal generator with PSD visualization |
| `dpd_convergence_movie.py` | V1: Animated GIF showing PSD and ACLR evolution across iterative ILA updates (exhibits limit-cycle oscillation) |
| `dpd_convergence_movie_v2.py` | V2: Same as V1 but with coefficient damping (α=0.5) to suppress the ILA oscillation |

### Key Components

#### GMP Basis Construction (`build_gmp_basis`)
Implements the full 3-component GMP basis matrix per Morgan et al. (IEEE TSP 2006):
- **Aligned**: `x(n-q) |x(n-q)|^k` — standard memory polynomial terms
- **Lagging**: `x(n-q) |x(n-q-l)|^k` — cross-terms with past envelope samples
- **Leading**: `x(n-q) |x(n-q+l)|^k` — cross-terms with future envelope samples

#### PA Model (`memory_polynomial_pa`)
Odd-order memory polynomial PA with coefficients from Eq. 12 of the reference paper:
- K=5 (odd orders only: 1, 3, 5), Q=2 (memory depth)
- 9 complex coefficients (18 real parameters)
- Coefficients: `c_{k,q}` for k ∈ {1,3,5}, q ∈ {0,1,2}

#### ILA Identification (`identify_gmp_coefficients`)
1. Normalize PA output by target gain: `z_norm = z / G`
2. Build GMP basis from `z_norm` (postdistorter input)
3. Solve `x = Phi · w` via `np.linalg.lstsq` (postdistorter coefficients)
4. Apply same `w` as predistorter: `x_dpd = Phi(x) · w`

#### WCDMA Signal Generation
- `generate_wcdma`: Single-carrier WCDMA with OVSF spreading (Walsh-Hadamard codes), QPSK modulation, RRC pulse shaping (α=0.22), and -80 dB channel FIR filter (Kaiser window)
- `generate_multicarrier_wcdma`: Frequency-shifts multiple independent carriers to create a multi-carrier signal
- Parameters: 3.84 Mcps chip rate, SF=16, 16 codes/carrier (full code space for flat in-band spectrum)

#### Metrics
- **NMSE**: Normalized Mean Squared Error comparing PA output to ideal linear output (G × input)
- **ACLR**: Adjacent Channel Leakage Ratio — ratio of in-band power to adjacent-band power. 3GPP WCDMA spec requires ≥45 dB at ±5 MHz offset (TS 25.104)

### Signal & Simulation Parameters

Current configuration in `gmp_dpd_standalone.py`:
- **Signal**: 3× WCDMA carriers, 5 MHz each, 15 MHz aggregate bandwidth
- **Sampling rate**: 61.44 MHz
- **Codes per carrier**: 16 (full SF=16 for flat in-band spectrum)
- **Drive level**: target_rms = 0.22 (moderate compression)
- **DPD config**: Ka=7, La=5, Kb=3, Lb=5, Mb=2, Kc=3, Lc=5, Mc=1 (80 complex coefficients)
- **PSD**: Welch method, nperseg=8192, 75% overlap, Blackman-Harris window, frequency axis -20 to +20 MHz

### Key Findings

1. **WCDMA signal flatness**: Using all SF codes (n_codes = SF = 16) is essential for a flat in-band spectrum. Using fewer codes creates spectral ripple from the Walsh-Hadamard code structure.

2. **Channel FIR filter**: An -80 dB Kaiser-windowed FIR filter after the RRC pulse shaper provides clean out-of-band suppression, improving the ideal ACLR floor from ~63 dB to ~72 dB.

3. **ILA convergence**: The standard iterative ILA exhibits a limit-cycle oscillation where NMSE alternates between ~-40 dB (odd iterations) and ~-55 dB (even iterations). This occurs because re-identifying coefficients from a nearly-linear PA output causes overfitting/overcorrection.

4. **Coefficient damping**: Blending new and old coefficients (`w = 0.5·w_new + 0.5·w_old`) suppresses the oscillation and produces smoother convergence with all iterations above the 3GPP 45 dB ACLR spec.

5. **PA drive level sensitivity**: The DPD breaks down at high drive levels (target_rms > ~0.25 for this PA model) because the PA operates in deep compression where the polynomial model cannot adequately represent the inverse. Moderate compression (target_rms ≈ 0.22) gives the best DPD results.

### Typical Results (3× WCDMA, target_rms=0.22)

| Metric | No DPD | GMP DPD | Ideal |
|--------|--------|---------|-------|
| NMSE (dB) | -7.7 | -55.5 | — |
| ACLR lower (dB) | 37.4 | 61.7 | 72.7 |
| ACLR upper (dB) | 35.6 | 60.8 | 71.8 |
