"""
This module provides the `IntervalsSampler` class and supporting
methods.
"""
from collections import namedtuple
import logging
import random
import os
from multiprocessing import Pool, Value, Lock
from multiprocessing.sharedctypes import RawArray

import numpy as np

from selene_sdk.samplers.online_sampler import OnlineSampler
from selene_sdk.utils.utils import get_indices_and_probabilities

# initialization function for parallel processes
def init(l, h_n_s, n_s, n_c, s, t, i, self_):
    global lock, n_samples_drawn, n_cache_accesses
    global sequences, targets, hyp_n_samples_drawn
    global intervals, self
    lock = l
    hyp_n_samples_drawn = h_n_s
    n_samples_drawn = n_s
    n_cache_accesses = n_c
    sequences = s
    targets = t
    intervals = i
    self = self_
    self.reference_sequence = self.reference_sequence.make_gs()

# This function will be executed by each worker in the pool
def worker_fn(tup):
    #print("WORKER FN STARTING")
    batch_size, cache_access_limit, mode, seq_chunk_size, = tup
    while True:
        with lock:
            if (hyp_n_samples_drawn.value == batch_size or
                n_cache_accesses.value == cache_access_limit):
                break
            #print(f"cache_access_limit: {cache_access_limit}, n_cache_accesses: {n_cache_accesses.value}, pid: {os.getpid()}")
            sample_index = self._randcache[mode]["sample_next"] + n_cache_accesses.value
            idx = n_samples_drawn.value
            n_cache_accesses.value += 1
            hyp_n_samples_drawn.value += 1

        rand_interval_index = self._randcache[mode]["cache_indices"][sample_index]
        interval_info = self.sample_from_intervals[rand_interval_index]
        interval_length = self.interval_lengths[rand_interval_index]
        chrom = interval_info[0]
        position = int(interval_info[1] + random.uniform(0, 1) * interval_length)
        #print(f"rii: {rand_interval_index}, chrom: {chrom}, pos: {position}, pid: {os.getpid()}")
        
        retrieve_output = self._retrieve(chrom, position)
        if not retrieve_output:
            with lock:
                hyp_n_samples_drawn.value -= 1
            continue
        seq, seq_targets, interval = retrieve_output
        with lock:
            idx = n_samples_drawn.value
            n_samples_drawn.value += 1
        
        np_sequences = np.frombuffer(sequences, dtype=np.float32)
        np_targets = np.frombuffer(targets, dtype=int)
        np_intervals = np.frombuffer(intervals, dtype=int)
        np_sequences[idx*seq_chunk_size:(idx+1)*(seq_chunk_size)] = seq.flatten()
        np_targets[idx*self.n_features:(idx+1)*self.n_features] = seq_targets 
        np_intervals[idx*4:(idx+1)*4] = interval

logger = logging.getLogger(__name__)

SampleIndices = namedtuple(
    "SampleIndices", ["indices", "weights"])
"""
A tuple containing the indices for some samples, and a weight to
allot to each index when randomly drawing from them.

Parameters
----------
indices : list(int)
    The numeric index of each sample.
weights : list(float)
    The amount of weight assigned to each sample.

Attributes
----------
indices : list(int)
    The numeric index of each sample.
weights : list(float)
    The amount of weight assigned to each sample.

"""


# @TODO: Extend this class to work with stranded data.
class ParallelIntervalsSampler(OnlineSampler):
    """
    Draws samples from pre-specified windows in the reference sequence.

    Parameters
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        A reference sequence from which to create examples.
    target_path : str
        Path to tabix-indexed, compressed BED file (`*.bed.gz`) of genomic
        coordinates mapped to the genomic features we want to predict.
    features : list(str)
        List of distinct features that we aim to predict.
    intervals_path : str
        The path to the file that contains the intervals to sample from.
        In this file, each interval should occur on a separate line.
    sample_negative : bool, optional
        Default is `False`. This tells the sampler whether negative
        examples (i.e. with no positive labels) should be drawn when
        generating samples. If `True`, both negative and positive
        samples will be drawn. If `False`, only samples with at least
        one positive label will be drawn.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    validation_holdout : list(str) or float, optional
        Default is `['chr6', 'chr7']`. Holdout can be regional or
        proportional. If regional, expects a list (e.g. `['X', 'Y']`).
        Regions must match those specified in the first column of the
        tabix-indexed BED file. If proportional, specify a percentage
        between (0.0, 1.0). Typically 0.10 or 0.20.
    test_holdout : list(str) or float, optional
        Default is `['chr8', 'chr9']`. See documentation for
        `validation_holdout` for additional information.
    sequence_length : int, optional
        Default is 1000. Model is trained on sequences of `sequence_length`
        where genomic features are annotated to the center regions of
        these sequences.
    center_bin_to_predict : int, optional
        Default is 200. Query the tabix-indexed file for a region of
        length `center_bin_to_predict`.
    feature_thresholds : float [0.0, 1.0] or None, optional
         Default is 0.5. The `feature_threshold` to pass to the
        `GenomicFeatures` object.
    mode : {'train', 'validate', 'test'}
        Default is `'train'`. The mode to run the sampler in.
    save_datasets : list of str
        Default is `["test"]`. The list of modes for which we should
        save the sampled data to file.
    output_dir : str or None, optional
        Default is None. The path to the directory where we should
        save sampled examples for a mode. If `save_datasets` is
        a non-empty list, `output_dir` must be specified. If
        the path in `output_dir` does not exist it will be created
        automatically.

    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        The reference sequence that examples are created from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features that we
        would like to predict.
    sample_from_intervals : list(tuple(str, int, int))
        A list of coordinates that specify the intervals we can draw
        samples from.
    interval_lengths : list(int)
        A list of the lengths of the intervals that we can draw samples
        from. The probability that we will draw a sample from an
        interval is a function of that interval's length and the length
        of all other intervals.
    sample_negative : bool
        Whether negative examples (i.e. with no positive label) should
        be drawn when generating samples. If `True`, both negative and
        positive samples will be drawn. If `False`, only samples with at
        least one positive label will be drawn.
    validation_holdout : list(str) or float
        The samples to hold out for validating model performance. These
        can be "regional" or "proportional". If regional, this is a list
        of region names (e.g. `['chrX', 'chrY']`). These Regions must
        match those specified in the first column of the tabix-indexed
        BED file. If proportional, this is the fraction of total samples
        that will be held out.
    test_holdout : list(str) or float
        The samples to hold out for testing model performance. See the
        documentation for `validation_holdout` for more details.
    sequence_length : int
        The length of the sequences to  train the model on.
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.

    """

    def __init__(self,
                 reference_sequence,
                 target_path,
                 features,
                 intervals_path,
                 sample_negative=False,
                 seed=436,
                 validation_holdout=['chr6', 'chr7'],
                 test_holdout=['chr8', 'chr9'],
                 sequence_length=1000,
                 center_bin_to_predict=200,
                 feature_thresholds=0.5,
                 mode="train",
                 save_datasets=["test"],
                 output_dir=None):
        """
        Constructs a new `IntervalsSampler` object.
        """
        super(ParallelIntervalsSampler, self).__init__(
            reference_sequence,
            target_path,
            features,
            seed=seed,
            validation_holdout=validation_holdout,
            test_holdout=test_holdout,
            sequence_length=sequence_length,
            center_bin_to_predict=center_bin_to_predict,
            feature_thresholds=feature_thresholds,
            mode=mode,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self._sample_from_mode = {}
        self._randcache = {}
        for mode in self.modes:
            self._sample_from_mode[mode] = None
            self._randcache[mode] = {"cache_indices": None, "sample_next": 0}

        self.sample_from_intervals = []
        self.interval_lengths = []

        if self._holdout_type == "chromosome":
            self._partition_dataset_chromosome(intervals_path)
        else:
            self._partition_dataset_proportion(intervals_path)

        for mode in self.modes:
            self._update_randcache(mode=mode)

        self.sample_negative = sample_negative

    def _partition_dataset_proportion(self, intervals_path):
        """
        When holdout sets are created by randomly sampling a proportion
        of the data, this method is used to divide the data into
        train/test/validate subsets.

        Parameters
        ----------
        intervals_path : str
            The path to the file that contains the intervals to sample
            from. In this file, each interval should occur on a separate
            line.

        """
        with open(intervals_path, 'r') as file_handle:
            for line in file_handle:
                cols = line.strip().split('\t')
                chrom = cols[0]
                start = int(cols[1])
                end = int(cols[2])
                self.sample_from_intervals.append((chrom, start, end))
                self.interval_lengths.append(end - start)
        n_intervals = len(self.sample_from_intervals)

        # all indices in the intervals list are shuffled
        select_indices = list(range(n_intervals))
        np.random.shuffle(select_indices)

        # the first section of indices is used as the validation set
        n_indices_validate = int(n_intervals * self.validation_holdout)
        val_indices, val_weights = get_indices_and_probabilities(
            self.interval_lengths, select_indices[:n_indices_validate])
        self._sample_from_mode["validate"] = SampleIndices(
            val_indices, val_weights)

        if self.test_holdout:
            # if applicable, the second section of indices is used as the
            # test set
            n_indices_test = int(n_intervals * self.test_holdout)
            test_indices_end = n_indices_test + n_indices_validate
            test_indices, test_weights = get_indices_and_probabilities(
                self.interval_lengths,
                select_indices[n_indices_validate:test_indices_end])
            self._sample_from_mode["test"] = SampleIndices(
                test_indices, test_weights)

            # remaining indices are for the training set
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[test_indices_end:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)
        else:
            # remaining indices are for the training set
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[n_indices_validate:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)

    def _partition_dataset_chromosome(self, intervals_path):
        """
        When holdout sets are created by selecting all samples from a
        specified region (e.g. a chromosome) this method is used to
        divide the data into train/test/validate subsets.

        Parameters
        ----------
        intervals_path : str
            The path to the file that contains the intervals to sample
            from. In this file, each interval should occur on a separate
            line.

        """
        for mode in self.modes:
            self._sample_from_mode[mode] = SampleIndices([], [])
        self.chroms = []
        with open(intervals_path, 'r') as file_handle:
            for index, line in enumerate(file_handle):
                cols = line.strip().split('\t')
                chrom = cols[0]
                start = int(cols[1])
                end = int(cols[2])
                if chrom in self.validation_holdout:
                    self._sample_from_mode["validate"].indices.append(
                        index)
                elif self.test_holdout and chrom in self.test_holdout:
                    self._sample_from_mode["test"].indices.append(
                        index)
                else:
                    self._sample_from_mode["train"].indices.append(
                        index)
                self.sample_from_intervals.append((chrom, start, end))
                self.interval_lengths.append(end - start)
                if chrom not in self.chroms:
                    self.chroms.append(chrom)
        self.chrom_to_i = {chrom: i for i, chrom in enumerate(self.chroms)}

        for mode in self.modes:
            sample_indices = self._sample_from_mode[mode].indices
            indices, weights = get_indices_and_probabilities(
                self.interval_lengths, sample_indices)
            self._sample_from_mode[mode] = \
                self._sample_from_mode[mode]._replace(
                    indices=indices, weights=weights)

    # modified _retrieve method

    def _retrieve(self, chrom, position):
        """
        Retrieves samples around a position in the `reference_sequence`.
        Parameters
        ----------
        chrom : str
            The name of the region (e.g. "chrX", "YFP")
        position : int
            The position in the query region that we will search around
            for samples.
        Returns
        -------
        retrieved_seq, retrieved_targets : \
        tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence centered at the query position, as well as a list
            of samples within this region that met the filtering
            standards.
        """
        bin_start = position - self._start_radius
        bin_end = position + self._end_radius
        retrieved_targets = self.target.get_feature_data(
            chrom, bin_start, bin_end)
        if not self.sample_negative and np.sum(retrieved_targets) == 0:
            logger.info("No features found in region surrounding "
                        "region \"{0}\" position {1}. Sampling again.".format(
                chrom, position))
            return None

        window_start = position - self._start_window_radius
        window_end = position + self._end_window_radius
        strand = '+'#self.STRAND_SIDES[random.randint(0, 1)]
        retrieved_seq = \
            self.reference_sequence.get_encoding_from_coords(
                chrom, window_start, window_end, strand)
        if retrieved_seq.shape[0] == 0:
            logger.info("Full sequence centered at region \"{0}\" position "
                        "{1} could not be retrieved. Sampling again.".format(
                chrom, position))
            return None
        elif np.sum(retrieved_seq[:, :4]) / float(retrieved_seq.shape[0]) < 0.60:
            logger.info("Over 30% of the bases in the sequence centered "
                        "at region \"{0}\" position {1} are ambiguous ('N'). "
                        "Sampling again.".format(chrom, position))
            return None

        #if self.mode in self._save_datasets:
        #    feature_indices = ';'.join(
        #        [str(f) for f in np.nonzero(retrieved_targets)[0]])
        #    self._save_datasets[self.mode].append(
        #        [chrom,
        #         window_start,
        #         window_end,
        #         strand,
        #         feature_indices])
        #    if len(self._save_datasets[self.mode]) > 200000:
        #        self.save_dataset_to_file(self.mode)
        chrom_i = self.chrom_to_i[chrom]
        if strand == '+':
            strand_i = 1
        elif strand == '-':
            strand_i = 0
        return (retrieved_seq, retrieved_targets, (chrom_i, window_start, window_end, strand_i))

    def _update_randcache(self, mode=None):
        """
        Updates the cache of indices of intervals. This allows us
        to randomly sample from our data without having to use a
        fixed-point approach or keeping all labels in memory.

        Parameters
        ----------
        mode : str or None, optional
            Default is `None`. The mode that these samples should be
            used for. See `selene_sdk.samplers.IntervalsSampler.modes` for
            more information.

        """
        if not mode:
            mode = self.mode
        self._randcache[mode]["cache_indices"] = np.random.choice(
            self._sample_from_mode[mode].indices,
            size=len(self._sample_from_mode[mode].indices),
            replace=True,
            p=self._sample_from_mode[mode].weights)
        self._randcache[mode]["sample_next"] = 0

    def sample(self, batch_size=1, mode=None):
        """
        Randomly draws a mini-batch of examples and their corresponding
        labels.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The number of examples to include in the
            mini-batch.
        mode : str, optional
            Default is None. The operating mode that the object should run in.
            If None, will use the current mode `self.mode`.

        Returns
        -------
        sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence examples and their corresponding labels. The
            shape of `sequences` will be
            :math:`B \\times L \\times N`, where :math:`B` is
            `batch_size`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` will be :math:`B \\times F`,
            where :math:`F` is the number of features.

        """
        mode = mode if mode else self.mode
        lock = Lock()
        n_samples_drawn = Value('i', 0)
        n_cache_accesses = Value('i', 0)
        hyp_n_samples_drawn = Value('i', 0)
        # Initialize the sequences and targets arrays
        enc_len = 4 + self.reference_sequence.n_shape_features
        sequences = RawArray('f', batch_size * self.sequence_length * enc_len) # 'f' for float32
        targets = RawArray('l', batch_size * self.n_features) # 'l' for long
        intervals = RawArray('l', batch_size * 4)
        seq_chunk_size = self.sequence_length * enc_len 
        
        cache_access_limit = len(self._sample_from_mode[mode].indices) \
                                - self._randcache[mode]["sample_next"]
       
        initargs = (lock, hyp_n_samples_drawn, n_samples_drawn, n_cache_accesses, sequences, targets, intervals, self)
        n_procs = 8
        while n_samples_drawn.value < batch_size:
            with Pool(n_procs, initializer=init, initargs=initargs) as pool:
                tup = (batch_size, cache_access_limit, mode, seq_chunk_size)
                r = pool.map(worker_fn, [tup for _ in range(n_procs)])
            if n_cache_accesses.value == cache_access_limit:
                self._update_randcache()
                n_cache_accesses.value = 0
                #print(f"Updated randcache! sample_next: {self._randcache[mode]['sample_next']}")
                cache_access_limit = len(self._sample_from_mode[mode].indices)

        self._randcache[mode]["sample_next"] += n_cache_accesses.value

        sequences = np.frombuffer(sequences, dtype=np.float32)
        sequences = sequences.reshape(batch_size, self.sequence_length, enc_len)
        targets = np.frombuffer(targets, dtype=int).reshape(batch_size, self.n_features)
        intervals = np.frombuffer(intervals, dtype=int).reshape(batch_size, 4)

        if self.mode in self._save_datasets:
            for j in range(batch_size):
                feature_indices = ';'.join(
                    [str(f) for f in np.nonzero(targets[j])[0]])
                chrom_i, window_start, window_end, strand_i = intervals[j]
                chrom = self.chroms[chrom_i]
                if strand_i == 0:
                    strand = '-'
                if strand_i == 1:
                    strand = '+'
                self._save_datasets[self.mode].append(
                    [chrom,
                     window_start,
                     window_end,
                     strand,
                     feature_indices])
                if len(self._save_datasets[self.mode]) > 200000:
                    self.save_dataset_to_file(self.mode)

        return (sequences, targets)
