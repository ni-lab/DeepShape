# Data files and model weights

## Included runnable example

`examples/quickstart/` contains 16 real hg19 sequence/shape inputs derived from the supplied HepG2 MafK positive and background interval files. The fixture includes a miniature FASTA, raw shape HDF5 with its normalization statistics, pre-encoded inputs, one-track labels, and reference predictions. Its [manifest](../examples/quickstart/manifest.json) records the original coordinates, channel order, target index, source resources, and checksums. Run the commands in the main README without downloading genome-scale inputs.

## Genome-scale resources

| Resource | Location / purpose |
| --- | --- |
| Pretrained 919-output checkpoint | [Hugging Face](https://huggingface.co/ryankeivanfar/DeepShape) and [Zenodo](https://zenodo.org/records/13334119). The quickstart downloads a pinned revision and verifies SHA-256. |
| Five-feature shape HDF5 | [Zenodo](https://zenodo.org/records/13334119), `hg19.EP_HelT_MGW_ProT_Roll.hdf5.gz` (about 14.5 GB compressed; allow space for decompression). |
| hg19 sequence | [ENCODE male.hg19](https://www.encodeproject.org/files/male.hg19/@@download/male.hg19.fasta.gz). Decompress to `male.hg19.fasta`; pyfaidx creates the `.fai` index. |
| Ordered 919-task list | Included as `data/distinct_features.txt`; preserve line order. |
| Sampling intervals | Included as `data/TF_intervals.txt`. |
| 919-task target labels | A coordinate-sorted, BGZF-compressed, tabix-indexed BED with columns `chrom, start, end, feature`, where every feature matches an entry in the ordered 919-task list. This large merged BED is not included in this repository; prepare it from the benchmark peak annotations before using the full training template. |

`resources.json` contains exact URLs, published checksums where supplied, and the checkpoint SHA-256. Validate downloads before use. The feature order in the supplied list belongs to the published checkpoint; do not substitute an alphabetically sorted list.

Example shape download:

```bash
mkdir -p data/genome
curl -L --fail https://zenodo.org/records/13334120/files/hg19.EP_HelT_MGW_ProT_Roll.hdf5.gz -o data/genome/hg19.EP_HelT_MGW_ProT_Roll.hdf5.gz
# Verify MD5 ff4546fc0a47f62df52949092c22d74c, then decompress:
gunzip data/genome/hg19.EP_HelT_MGW_ProT_Roll.hdf5.gz
```

Set `DEEPSHAPE_DATA` to the directory containing the full FASTA, uncompressed HDF5, and indexed target BED. `config/train_deepshape.yml` names the expected files.

## Single-track data versus the full benchmark

The [Selene quickstart archive](https://zenodo.org/records/1443558/files/selene_quickstart.tar.gz) is a **single-feature CTCF tutorial**, not the full 919-task DeepShape dataset. Using that archive unchanged with a 919-output configuration produces a label/output mismatch.

A one-track run needs `n_targets: 1`, a one-line feature list, matching BED labels, and `sample_negative: True`. The bundled `train_quickstart.yml` demonstrates those settings with miniature-coordinate MafK data. A new single-track model is trained from initialization; it cannot load the 919-output checkpoint unchanged.
