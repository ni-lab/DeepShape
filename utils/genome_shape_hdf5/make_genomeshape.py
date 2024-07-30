from .genomeshape import GenomeShape

class GenomeShapeMaker:
    """Used as the reference sequence when the sampler is ParallelIntervalsSampler.
    File handles are not serializable, so a GenomeShape instance which has handles
    to genome and shape fasta files cannot be given to multiprocessing Processes.
    This class is used so Processes can be given the sampler. Each Process then 
    instantiates their own GenomeShape with the appropriate file handles using make_gs.
    """
    def __init__(self, input_path, shape_feature_files=None, blacklist_regions=None, bases_order=None, init_unpicklable=False, multi_feat_file=None):
        self.init_kwargs = {
            'input_path': input_path,
            'shape_feature_files': shape_feature_files,
            'blacklist_regions': blacklist_regions,
            'bases_order': bases_order,
            'init_unpicklable': init_unpicklable,
            'multi_feat_file': multi_feat_file,
        }
        gs = GenomeShape(**self.init_kwargs)
        self.n_shape_features = gs.n_shape_features

    def make_gs(self):
        return GenomeShape(**self.init_kwargs)
