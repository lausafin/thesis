# Characterizing Side Effects of Activation Steering

Hetianyu Huang, Lau Safin — KU bachelor's thesis, June 2026

We apply Contrastive Activation Addition to Llama-3.1-8B-Instruct across 36 behavioral datasets, sweeping nine steering multipliers from −2.0 to +2.0. Side effects are scored with a 19-dimension ordinal taxonomy and a two-pass LLM-as-judge protocol. The clearest primary claim is steering asymmetry: negative and positive multipliers produce unequal collateral effects.

## Documentation

- [REPRODUCTION.md](REPRODUCTION.md) — reproduce tables and figures from archived evaluation data
- [experiment/README.md](experiment/README.md) — optional full GPU/API pipeline (CAA generation + judging)

## Build PDF

Run `python scripts/reproduce.py` first, then:

```bash
cd thesis && latexmk -pdf main.tex
```

See [REPRODUCTION.md § Compile the PDF](REPRODUCTION.md#compile-the-pdf).

## Third-party assets

- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (Hugging Face)
- MWE prompts from [anthropics/evals](https://github.com/anthropics/evals) (CC-BY-4.0)

No `LICENSE` file is included in this repository.
