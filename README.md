This Python script performs the analysis for the T(τ) relation
presented by Ball (RNAAS 2021).  It

1. downloads the data from [Trampedach et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014MNRAS.442..805T),
2. caches it to `TtauFeH0.dat` wherever the script is run,
3. fits the new formula,
4. reports some information and
5. plots Figure 1.

The script requires [NumPy](https://numpy.org/),
[SciPy](https://scipy.org/),
[Matplotlib](https://matplotlib.org/)
and an Internet connection to download the data the first time the
script is run.
