# Interpreting DeepShape

Pipeline that computes DeepLIFT (Shrikumar et al. 2017) attributions with respect to a trained DeepShape model's predictions, extracts motifs using TF-MoDISco (modisco-lite), and analyzes co-occurrences of a sequence and shape motif.

First, make sure you install the modified version of tfmodisco-lite (original: https://github.com/jmschrei/tfmodisco-lite/tree/main), which includes an option to reverse sequences without complementing in order to handle quaternized shape features, by running the following commands:
```
cd tfmodisco-lite-mod
pip install .
```

## 1. Compute DeepLIFT attributions
Usage:
`python -u run_deeplift.py /path/to/config/deeplift.yml`
The config specifies a trained DeepShape model checkpoint, a set of positive sequences to compute DeepLIFT attributions for, a set of background sequences for reference, the index of the output task attributions are computed with respect to, and other parameters. See `config/deeplift.yml` for a template.

Outputs `encodings.npz`, which stores a `(n_seqs, 9, seq_len)` numpy array encoding the inputted positive sequences with the first 4 features corresponding to one-hot sequence features and the last 5 corresponding to EP, HelT, MGW, ProT, and Roll, and `attributions.npz`, which stores an array of attributions which is of the same size as `encodings.npz`. These are output to the specified output directory.

## 2. Run TF-MoDISco
Usage:
`python -u run_tfmodisco.py /path/to/config/tfmodisco.yml`
The config specifies the paths to `attributions.npz` and `encodings.npz` output by `run_deeplift.py`, the output directory, and the maximum total number of seqlets to consider for TF-MoDISco.

## 3. Analyze co-occurrences of a sequence and a shape motif
After extracting motifs with TF-MoDISco, you can analyze co-occurrences between a particular pair of sequence and shape motifs using `analyze_cooccurrences.ipynb`.
