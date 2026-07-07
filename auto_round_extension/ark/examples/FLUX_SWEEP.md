# FLUX Sweep

Use the helper script below to sweep one dense baseline plus sparse `topk=1.0..0.1`
with the default FLUX generation settings (`1024x1024`, `50` steps, guidance `3.5`)
and the `q_tile=256`, `sparse_q_block_tokens=256`, `sparse_k_block_tokens=64` kernel.

```bash
cd auto_round_extension/ark/examples
FLUX_SWEEP_PYTHON=/home/yiliu4/workspace/auto-round-py/.venv/bin/python \
FLUX_MODEL=/home/yiliu4/workspace/models/black-forest-labs/FLUX.1-dev \
FLUX_SWEEP_DEVICES=0,1,2,3,4,5,6,7 \
bash run_flux_sweep.sh
```

Results are written into `auto_round_extension/ark/examples/results/flux_sweep_defaultsteps_<timestamp>/`
with:

- `summary.csv` for the full sweep summary
- `commands.txt` for the exact per-run commands
- one `.png` and one `.log` per config
