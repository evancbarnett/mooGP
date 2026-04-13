Example usage:

python -m experiments.run \
  --functions borehole \
  --methods LCGP OILMM \
  --ns 100 500 \
  --ps 4 8 \
  --q 3 \
  --reps 10 \
  --results-dir results-oilmm-lcgp-only \
  --base-seed 123

**Ensure that the backslash is the last character on each line or else the terminal command will no work.**  

  Default Values:
   - Function: "borehole"
   - methods : ["MOOGP", "MOGP", "LCGP", "OILMM", "PUQ"]
   - Training sizes (--ns): [50, 100, 250, 1000, 2500]
   - Output dimension (--ps): [10, 20, 50]
   - Replications (--reps): 5
   - Testing size (--n-test): [250]
   - Latent dimension (q): 5
   - Maximum number of optimizer iterations: 1000
   - Jitter: 1e-6
   - Noise added to each output column: 0.05 * Output Variance
   -  