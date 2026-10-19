# Interpret DeepShape

Install `requirements-workflows.txt`, then install the included TF-MoDISco modification from the repository root:

```bash
python -m pip install -r requirements-workflows.txt
python -m pip install ./interpret/tfmodisco-lite-mod
```

## Check DeepLIFT with the included example

```bash
python examples/quickstart.py
python interpret/run_deeplift.py config/deeplift_quickstart.yml
```

The first command downloads and verifies the public checkpoint and checks pretrained CPU predictions against reference values. The second exercises the original genome-reader attribution pipeline using the miniature genome fixture.

## Genome-scale attributions

Prepare the FASTA and shape HDF5 described in `data/README.md`, set `DEEPSHAPE_DATA` and `DEEPSHAPE_OUTPUT` to absolute paths, and edit `config/deeplift.yml` to identify the positive/background BED files and zero-based target index. Then run:

```bash
python interpret/run_deeplift.py config/deeplift.yml
```

Inputs and attributions are saved as `encodings.npz` and `attributions.npz`, respectively, each with one array (`arr_0`) of shape `(sequences, 9, 1000)`. Channels are A, C, G, T, EP, HelT, MGW, ProT, Roll. The normalized genome reader must be paired with `deepshape_no_normalize.py` to avoid normalizing twice. The published feature list determines the target index; 504 is HepG2 MafK.

## Sequence and shape motif discovery

```bash
python interpret/run_tfmodisco.py config/tfmodisco.yml
```

This config points to the attribution files and specifies the output directory and maximum seqlet count. Use enough biological examples for motif discovery; the tiny installation fixture does not provide a meaningful motif catalog. Shape channels are quartile-encoded and analyzed with `--nofliprc`, which reverses without complementing the encoded shape categories. A failed command stops the workflow and preserves the intermediate files for diagnosis.

`analyze_cooccurrences.ipynb` is the exploratory notebook for inspecting a selected sequence–shape motif pair. Its analysis paths and selected motifs must be set for the dataset being analyzed; it is not an automatic continuation of the small installation example.
