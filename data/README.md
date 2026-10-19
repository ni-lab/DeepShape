# Download and format the data

The data and formatting follow the same instructions as outlined in [Selene](https://github.com/FunctionLab/selene/blob/master/tutorials/getting_started_with_selene/getting_started_with_selene.ipynb). (The **"Shortcut"** below contains all the formatted data in a `.tar.gz`.)

In this tutorial, we will go through a single-feature example with data from the ENCODE Uniform TFBS composite track. These are the transcription factor datasets that were used to train [Zhou and Troyanskaya's (2015) DeepSEA model](https://www.nature.com/articles/nmeth.3547).

We can download the measurements for transcription factor CTCF in cell type GM12878 by running:

```bash
wget http://hgdownload.cse.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/wgEncodeAwgTfbsUtaGm12878CtcfUniPk.narrowPeak.gz
```

and format the data with:

```bash
bgzip -d wgEncodeAwgTfbsUtaGm12878CtcfUniPk.narrowPeak.gz

cut -f 1-3 wgEncodeAwgTfbsUtaGm12878CtcfUniPk.narrowPeak > GM12878_CTCF.bed

sed -i "s/$/\tGM12878|CTCF|None/" GM12878_CTCF.bed

sort -k1V -k2n -k3n GM12878_CTCF.bed > sorted_GM12878_CTCF.bed
```
The formatted BED file should contain 4 columns, in order: chromosome, start, end, feature. We do not support strand-specific data at this time.

In this example, we will use the `ParallelIntervalsSampler` class for partitioning and sampling the data. The intervals sampler requires that we pass in an intervals file with 3 columns: chrom, start, end. This intervals file determines where in the genome we sample our data. There is a provided intervals file (`TF_intervals.txt`) with the regions in the original DeepSEA dataset that contained at least 1 transcription factor (TF).

It also requires that we tabix-index the dataset BED file for fast querying of targets in genomic regions.

```bash
bgzip -c sorted_GM12878_CTCF.bed > sorted_GM12878_CTCF.bed.gz

tabix -p bed sorted_GM12878_CTCF.bed.gz
```
The sampler classes are used to partition your dataset into training/testing/validation sets and will draw examples from the appropriate partitions during the training/evaluation process.

These require that you have a file containing the distinct genomic features that the model predicts. Note that when we refer to a model's "features", we are referring to the genomic features that it predicts (i.e. they are the same as classes, labels, or targets that a deep learning model predicts).

```bash
cut -f 4 sorted_GM12878_CTCF.bed | sort -u > distinct_features.txt
```
Finally, we must download the hg19 FASTA file:
```bash
wget https://www.encodeproject.org/files/male.hg19/@@download/male.hg19.fasta.gz

bgzip -d male.hg19.fasta.gz
```

## SHORTCUT: download all formatted data from Zenodo record

Download the compressed data from here:

```bash
wget https://zenodo.org/record/1443558/files/selene_quickstart.tar.gz
```
Extract it and `mv` all files from the extracted directory `selene_quickstart_tutorial` to the current directory:
```bash
tar -xzvf selene_quickstart.tar.gz
mv selene_quickstart_tutorial/* .
```
