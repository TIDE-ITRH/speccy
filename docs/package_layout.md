
## Main functions:

### High-level
 - Main speccy object w/ regularly-spaced data: data that knows its dimension and property (ala xarray)
 - ACF<->PSD object: object that knows conversion tricks between common or generic ACF and PSD
 - debiased whittle: function to call whittle with a bias spectrum
 - dwelch: function for debiased Welch
 - generate random outputs
 - Joe Guinness (Lachy to investigate)

### Low-level 

- A bunch of ACF and PSD in closed form
- function to go discrete ACF-->PSD (Bochner) [done]
- function to go discrete PSD-->ACF (inverse Bochner) [done]
- periodogram our way (current thinking: always two-sided)
- whittle likelihood return
- function to compute bias spectrum from an ACF [done]
