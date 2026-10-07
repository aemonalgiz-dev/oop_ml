# Inspectable modern-model operations

These NumPy objects support the modern-model teaching lessons. They are also
exported from `oop_ml`. They require no model downloads or hosted services.

| Object | Implemented behavior | Scope |
| --- | --- | --- |
| `CausalLanguageModel` | Embeddings, positions, a pre-normalized attention block, feed-forward branch, vocabulary logits, bounded generation | Seeded inference reference, no full-decoder training or KV cache |
| `TokenDistribution` | Stable temperature scaling, top-p filtering, greedy selection and seeded sampling | One vocabulary row |
| `TokenCrossEntropy` | Masked mean loss, per-token probabilities and loss, exact logit gradient, perplexity | Objective and scoring, no optimizer |
| `LinearAutoencoder` | Encode, decode, measure reconstruction error | Caller-supplied linear matrices |
| `DiagonalGaussian` | Reparameterized sample and KL against a standard normal | VAE latent operation, no encoder or decoder training |
| `AdversarialGaussian` | Alternating discriminator and non-saturating generator updates | A trainable scalar shift and sigmoid discriminator |
| `GaussianDiffusion` | Gaussian forward schedule, direct noising, clean estimate from supplied noise prediction | No trained predictor or reverse sampler |
| `LowRankAdapter` | Factored weight correction, merged forward result, parameter counts | Caller-supplied factors, no adapter training |
| `TermOverlapRetriever` | Word-count cosine search with source indices | Retrieval baseline, no language-model call |

The decoder's causal attention prevents changed future tokens from changing
earlier outputs. Its parameters are initialized once and reused as the prefix
grows. A generated end token or context/length bound stops generation.

Loss masks select target predictions, not prompt visibility. The derivative
is averaged over included targets only; an all-excluded mask is rejected.

The adversarial example updates the discriminator with generated values fixed,
then updates the generator through the newly updated, fixed discriminator.
Both losses are recorded separately. It cannot represent a multimodal family.

The diffusion recovery method accepts a predicted noise array. Supplying the
true forward noise is an oracle correctness check, not a generative algorithm.

Run the focused numerical and dependency checks from the SDK root:

```powershell
.venv/Scripts/python.exe -m pytest test/numpy/modern test/test_layering.py -q
```

Tests include causal prefix invariance, sampling normalization and threshold
handling, finite-difference loss and GAN gradients, reconstruction information
loss, exact-noise recovery, LoRA merge equivalence, and missing retrieval.
