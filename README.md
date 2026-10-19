# DeepShape

![DeepShape Model](DeepShape.png)

bioRxiv pre-print [here](https://www.biorxiv.org/content/10.1101/2025.04.01.646034v1)

DeepShape is a deep convolutional neural network designed to predict molecular phenotypes from DNA sequences. Unlike traditional models that rely solely on one-hot encoded DNA sequences, DeepShape integrates DNA structural attributes indicative of local shape: minor groove width (MGW), helical twist (HelT), propeller twist (ProT), roll, and electrostatic potential (EP). This combination enhances the interpretability of the model and helps identify regulatory patterns that are not apparent from sequence information alone.

DeepShape is built upon DeeperDeepSEA, a PyTorch-based deep learning model originally designed to predict chromatin features from DNA sequence alone as implemented in [Selene](https://www.nature.com/articles/s41592-019-0360-8).

## Setup

From a fresh clone, run the following commands from the repository root. The environment includes Python 3.12, CPU PyTorch, and a C compiler for installing Selene and pytabix.

```bash
git clone https://github.com/ni-lab/DeepShape.git
cd DeepShape
conda env create -f environment.yaml
conda activate dnashapeenv
python -m pip install ./interpret/tfmodisco-lite-mod
```

`environment.yaml` installs the dependencies in `requirements.txt`, including those for training and interpretation. Installation and the checks below were tested on Linux x86-64 with Python 3.12. For GPU training, replace CPU PyTorch with the appropriate [CUDA build](https://pytorch.org/get-started/locally/).

A short CPU check runs the original model on synthetic input; it does not require genome data or a trained checkpoint:

```bash
python - <<'PYTHON'
import torch
from model.deepshape import DeepShape

torch.set_num_threads(2)
model = DeepShape(sequence_length=1000, n_targets=919).eval()
with torch.no_grad():
    predictions = model(torch.rand(2, 9, 1000))
assert predictions.shape == (2, 919) and torch.isfinite(predictions).all()
print("DeepShape CPU check passed:", tuple(predictions.shape))
PYTHON
python utils/run_deepshape.py --help
modisco motifs --help
```

## Running DeepShape

The `utils` directory holds essential scripts and helper files needed to run DeepShape. Ensure the following are present in `utils`:

- `run_deepshape.py`: The main script to run DeepShape.
- `shape_fasta.py`: A helper script for processing FASTA files.
- `genome_shape_hdf5`: Directory containing helper scripts for processing genome shape data.
- `intervals_sampler_hdf5`: Directory containing helper scripts for sampling.

The `model` directory contains the DeepShape model implementation:

- `deepshape.py`

### Prepare the Configuration File

Replace `/ABSOLUTE/PATH` in `config/train_deepshape.yml` with your local paths. Required external inputs are the [hg19 FASTA](https://www.encodeproject.org/files/male.hg19/@@download/male.hg19.fasta.gz), [DNA shape HDF5](https://zenodo.org/records/13334120/files/hg19.EP_HelT_MGW_ProT_Roll.hdf5.gz) (decompress before use), and a sorted, bgzip-compressed, tabix-indexed target BED matching the 919 features in `data/distinct_features.txt`. The full target BED must be prepared separately; the single-target example in `data/README.md` is not the 919-target training dataset. The default training configuration requires a GPU.

The [pretrained checkpoint](https://huggingface.co/ryankeivanfar/DeepShape/resolve/8fb6e29695b9728f45e8f97c74ca59197f9cf589/deepshape.pth.tar) and [interpretation instructions](interpret/README.md) are available separately. When running interpretation from the repository root, use `PYTHONPATH="$PWD/utils${PYTHONPATH:+:$PYTHONPATH}" python interpret/run_deeplift.py config/deeplift.yml`, after setting local paths in that configuration.

### Run the DeepShape Model

To run the DeepShape model using the `run_deepshape.py` script, execute the following command in your terminal:

```bash
python utils/run_deepshape.py /ABSOLUTE/PATH/config/train_deepshape.yml
```


