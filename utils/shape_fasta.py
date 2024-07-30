import pyfaidx
import numpy as np

"""Reads/accesses a genome numerical feature stored as a fasta file where
each entry is n=4 characters denoting the feature value at that position.
Just like a standard Fasta file except n times as long (n chars per position
instead of one) and requires this custom class to conveniently retrieve
the feature value at the ith genomic position (chars n*i to n*i+n converted
to float)
"""

class ShapeFastaRecord:
    def __init__(self, fasta_record):
        self.record = fasta_record
        assert len(self.record) % 4 == 0

    def __getitem__(self, key):
        n = 4
        if isinstance(key, slice):
            # Validate slice indices are integers
            if isinstance(key.start, str):
                start = int(key.start)
            else:
                start = key.start
            if isinstance(key.stop, str):
                stop = int(key.stop)
            else:
                stop = key.stop

            string_of_vals = self.record[4*start:4*stop]
            return np.float32([str(string_of_vals[i:i+4]) for i in range(0, len(string_of_vals), 4)])
        else:
            # Validate key is integer
            if isinstance(key, str):
                key = int(key)
            return np.float32(str(self.record[4*key:4*key+4]))

    def __len__(self):
        return len(self.record) // 4


class ShapeFasta:
    def __init__(self, path):
        self.fasta = pyfaidx.Fasta(path)

    def __getitem__(self, key):
        # Check if key is a string (chromosome name)
        if isinstance(key, str):
            return ShapeFastaRecord(self.fasta[key])
        else:
            raise TypeError("Invalid key type. Must be a string representing a chromosome.")

    def __len__(self):
        return len(self.fasta)

