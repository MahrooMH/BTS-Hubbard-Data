# Data Availability

- `BTSHubResults.csv` — Hubbard-model energies across interaction strengths (energy-column suffixes indicate lattice sizes, e.g., `6` = `6×1`, `23` = `2×3`) and opposite-spin density correlations.
- `BTSHub.zip` — Archive containing the six calculation scripts listed below.
- `BTSHub/UBTSHub_analytic.jl` — Optimizes unrestricted BTS orbitals and amplitudes using analytic amplitude gradients.
- `BTSHub/UDOCIfromUBTS.jl` — Calculates DOCI energies with fixed BTS-optimized orbitals.
- `BTSHub/HubbardAGPUnFastV2.jl` — Optimizes unrestricted AGP wave functions for Hubbard lattices.
- `BTSHub/HubbardBTSUnFastV2.jl` — Optimizes unrestricted BTS wave functions for Hubbard lattices.
- `BTSHub/NiupNjdown_BTS_DOCI_pCCD.py` — Computes opposite-spin site-density correlations for BTS, DOCI, and pCCD.
- `BTSHub/upccdfrombts_with_state_saving.py` — Solves pCCD amplitudes in frozen BTS orbitals and saves the resulting states.
