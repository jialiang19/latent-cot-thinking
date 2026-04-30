# Do Latent-CoT Models Think Step-by-Step?
### A Mechanistic Study on Sequential Reasoning Tasks

This is the official repository for the paper: **"Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks"**.

This work investigates the internal reasoning mechanisms of Latent Chain-of-Thought models using mechanistic interpretability techniques (Logit Lens, Patching, and Probing) on sequential reasoning tasks.

## Setup

To recreate the conda environment:

```bash
conda env create -f environment.yml
```

## Reproducing the Results

The data and model checkpoints required to reproduce the results are provided.

If you are only interested in the interpretability results, you can directly proceed to **Step 4: Interpretability Notebooks**. To reproduce the full workflow, follow **Steps 1–4** in order.

We provide checkpointed models for both `mod = 50` and `mod = 41`:

- `mod = 50`: an example of the composite-modulus setting described in the paper.
- `mod = 41`: an example of the prime-modulus setting described in the paper.

### Note on Sequence Length

A 2-hop problem, as defined in the paper, has a total sequence length of `3`.

More generally, an `n`-hop problem has a total sequence length of `n + 1`.

## 1. Data Generation

**File:** `src/data/data_processing.py`

Defines the polynomial iteration task: *f(x, y) = x · y + 1 (mod M)*. Key classes:

- `Polynomial` — generates raw polynomial sequences with chain-of-thought
- `Polynomial_CODI` — same dynamics but inserts an explicit `[ANS]` token before the final answer, matching the CODI training format

**How to generate data:**

Run `notebook/Data-Generation.ipynb`. It loops over sequence lengths and modular bases, calling `generate_datafiles()` to write `.npy` shards to:

```
data/polynomial_data/seq_{total_seqs}/num_{n_data_per_seq}/mod_{mod}/codi/
```

## 2. Training

**Entry point:** `train.py`

Uses HuggingFace `Trainer` with the `CODI` model defined in `src/model.py`. Configuration is parsed via `HfArgumentParser` from three dataclasses:

- `ModelArguments` — architecture (layers, heads, embedding dim), checkpoint path
- `DataArguments` — dataset name, sequence lengths, polynomial mod, data counts
- `TrainingArguments` — extends HF `TrainingArguments` with CODI-specific fields (`num_latent`, `use_prj`, `prj_dim`, distillation/CE loss factors, `fix_attn_mask`, etc.)

**How to run:**

```bash
# Single run: GPU 0, total_seqs=4, mod=50
bash script/polynomial_L3H2_seq_x_mod50.sh 0 4

# Batch multiple seq lengths on GPU 0
bash script/run_polynomial_L3H2_seq_x_mod50.sh
```

Checkpoints are saved to `save/transformer_tiny_polynomial/Layer3Head2_totalseq_{N}_mod{M}/transformer_tiny/ep_{epochs}/lr_{lr}/seed_{seed}/`.

## 3. Evaluation

**Entry point:** `test.py`

Loads a trained checkpoint and evaluates teacher/student accuracy on train and test splits, broken down by sequence length. Writes results to `test_results.csv` alongside the checkpoint.

**How to run:**

```bash
# Single run: GPU 0, total_seqs=4, mod=50
bash script/test/L3H2_seq_x_mod50.sh 0 4

# Loop over all seq lengths
bash script/test/run_L3H2_seq_x_mod50.sh
```

## 4. Interpretability Notebooks

All notebooks are in `notebook/` and use the `HookedTransformer`-based CODI model from `src/model_interp.py`.

| Notebook | Purpose |
|---|---|
| `activation_patching_logit_lens.ipynb` | **Logit lens**: applies the unembedding matrix to intermediate hidden states at each layer/phase to see what tokens the model predicts at each position. **Activation patching**: replaces activations from a corrupted run with clean-run activations to localize causal components. |
| `attn_final.ipynb` | **Attention analysis**: extracts attention patterns across encoder, latent, and decoder phases, stitches them into composite attention maps, and saves per-layer/per-head PDF figures. |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{liang2026latent,
  title={Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks},
  author={Liang, Jia and Pan, Liangming},
  journal={arXiv preprint arXiv:2602.00449},
  year={2026}
}
