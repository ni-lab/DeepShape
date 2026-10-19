"""Create the indexed miniature training labels from the bundled example."""
from pathlib import Path
import pysam
root=Path(__file__).resolve().parent/'quickstart'
output=root/'labels.bed.gz'
pysam.tabix_compress(str(root/'labels.bed'),str(output),force=True)
pysam.tabix_index(str(output),preset='bed',force=True)
print(output)
