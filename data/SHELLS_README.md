# DopplerIANN Shell Data

Shell datasets can be generated with the scripts in `data_generators/`.

Detailed instructions for each generator script are available in `data_generators/data_generator_README.md`.

A full paper-scale shell set (all period combinations, 10 realizations with different random phases, and both temperature and flux) requires about **17 GB**.

For lighter workflows, smaller subsets are usually enough:

- **One realization (temperature + flux):** about **1.7 GB**.
- **Diagnostics without detection maps:** one realization is typically sufficient.
- **Single modality only (temperature or flux):** about **0.8 GB** (roughly half the size), enough to train one model in that configuration or test shell files with pretrained models.
