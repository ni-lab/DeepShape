import numpy as np
from selene_sdk.sequences import Genome
import h5py

def load_from_file(shape_feature_file):
    if 'hdf5' in shape_feature_file:
        return h5py.File(shape_feature_file)

class GenomeShape(Genome):
    def __init__(self, input_path, shape_feature_files=None, blacklist_regions=None,
                 bases_order=None, init_unpicklable=False, multi_feat_file=None):
        """
        multi_feat_file: Optional[str]
            An hdf5 file containing multiple features, or None. If multi_feat_file is not None, then shape_feature_files
            should be None.
        """
        super().__init__(input_path, blacklist_regions, bases_order, init_unpicklable)
        self.multi_feat_file = multi_feat_file 
        self.shape_feature_files = shape_feature_files if shape_feature_files else {}

        self.shape_features = {}
        if multi_feat_file:
            assert not shape_feature_files
            self.load_multi_feat_file(multi_feat_file) 
        if shape_feature_files:
            self.load_shape_features(shape_feature_files)

    def load_shape_features(self, shape_feature_files):
        for feature_name, feature_file in shape_feature_files.items():
            shape_feature = load_from_file(feature_file)
            self.shape_features[feature_name] = shape_feature

    def load_multi_feat_file(self, multi_feat_file):
        self.shape_features = h5py.File(multi_feat_file)        

    @property
    def n_shape_features(self):
        if self.multi_feat_file:
            return len(self.shape_features['shape_feat_names'])
        return len(self.shape_features)

    def set_shape_features(self, shape_features):
        self.shape_features = shape_features

    def get_shape_features(self):
        return self.shape_features

    def get_shape_feature_file(self, shape_feature):
        return self.shape_feature_files.get(shape_feature, None)

    def get_shape_feature(self, shape_feature_name, chrom, start, end, strand='+', pad=False):
        #print(f"Shape features: {self.shape_features}")
        #print(f"Shape feature files: {self.shape_feature_files}")
        assert start >= 0
        shape_feature = self.shape_features.get(shape_feature_name)
        if shape_feature:
            #print(f"Shape Feature fasta for {shape_feature_name}: {shape_feature}")
            if strand == "-":
                if shape_feature_name in ['HelT', 'Roll']:
                    if start != 0:
                        start, end = start - 1, end - 1
                result = shape_feature[chrom][start:end][::-1]
                # manually pad beginning
                if start == 0 and shape_feature_name in ['HelT', 'Roll']:
                    val0 = shape_feature[chrom][0]
                    result = np.concatenate((result[1:], np.float32([val0])))
            else:
                result = shape_feature[chrom][start:end]
                if shape_feature_name in ['HelT', 'Roll']:
                    # manually pad end
                    if end > len(shape_feature[chrom]) and start < len(shape_feature[chrom]):
                        valend = shape_feature[chrom][len(shape_feature[chrom]) - 1]
                        result = np.concatenate((result, np.float32([valend])))
            #print(f"Shape Feature fasta result: {result}")
            return result
        else:
            raise KeyError(f"No such shape feature: {shape_feature_name}")

    def get_shape_features_from_coords(self, chrom, start, end, strand='+', pad=False):
        if self.multi_feat_file:
            if strand == '-':
                raise ValueError("Strand is -, not supported")
            shape_feats = self.shape_features[chrom][start:end]
            stats = self.shape_features['stats']
            mins = stats[:, 1]
            maxs = stats[:, 2]
            return (shape_feats - mins) / (maxs - mins)
        shape_features = []
        for shape_feature_name in self.shape_features:
            shape_feature_encoding = self.get_shape_feature(shape_feature_name, chrom, start, end, strand, pad)
            # Ensure the shape_feature_encoding is a 2D array
            shape_feature_encoding = np.array(shape_feature_encoding).reshape(-1, 1)
            #print(f"{shape_feature_name} {shape_feature_encoding.shape} {chrom} {start} {end}")
            shape_features.append(shape_feature_encoding)
        #print(f"Shapes of shape_features before concatenation: {[f.shape for f in shape_features]}")
        return np.concatenate(shape_features, axis=1) if shape_features else None

    def get_encoding_from_coords(self, chrom, start, end, strand='+', pad=False):
        sequence_encoding = Genome.get_encoding_from_coords(self, chrom, start, end, strand, pad)
        shape_features = self.get_shape_features_from_coords(chrom, start, end, strand, pad)
        if len(sequence_encoding) > 0 and shape_features is not None:
            return np.concatenate([sequence_encoding, shape_features], axis=1)
        elif len(sequence_encoding) > 0: # corresponds to having no shape features
            return sequence_encoding
        else:
            return np.empty((0, 4 + self.n_shape_features), dtype=np.float32)

    def get_encoding(self, region, strand='+', pad=False):
        sequence_encoding = super().get_encoding(region, strand, pad)
        shape_features = self.get_shape_features_from_coords(region.chrom, region.start, region.end, strand, pad)
        if shape_features is not None:
            return np.concatenate([sequence_encoding, shape_features], axis=1)
        else:
            return sequence_encoding
