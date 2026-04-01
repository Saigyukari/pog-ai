# gpt2claude

- `train_bc.py` now implements the roadmap BC launcher instead of the earlier smoke-test wrapper.
- Verified locally on a 128-record subset: 1 epoch completed and wrote `/tmp/pog_bc_subset/epoch_001.pkl`.
- In this environment JAX could not initialize CUDA, so validation ran on CPU fallback rather than multi-GPU `pmap`.
