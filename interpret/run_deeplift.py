from captum.attr import DeepLiftShap
import importlib
import logging
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
import yaml

"""Usage:
run_deeplift.py /path/to/config.yaml
"""

logging.basicConfig(level=logging.INFO)
config_file = sys.argv[1]
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

def instantiate_class(object_config):
    module_path = object_config["path"]
    module_name = os.path.basename(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    object_class = getattr(module, object_config["class"])
    return object_class(**object_config["kwargs"])

# Load the model and checkpoint
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = instantiate_class(config["model"])
model.to(device)
checkpoint = torch.load(config["checkpoint"], map_location=device)
raw_state_dict = checkpoint["state_dict"]
def rm_module_prefix(string):
    # If checkpoint is from a model wrapped with DataParallel,
    # remove the module. prefix from the weight name.
    if string.startswith("module."):
        return string[7:]
    return string
state_dict = {rm_module_prefix(k): v for k, v in raw_state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
logging.info("Model and checkpoint loaded successfully.")

# Load positive and background sequences, and their encodings
def read_bed(bedfile):
    intervals = []
    with open(bedfile, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            line[1] = int(line[1])
            line[2] = int(line[2])
            intervals.append(line)
    return intervals
intervals = read_bed(config["positive_data_bed"])
background_intervals = read_bed(config["background_data_bed"])
sequence = instantiate_class(config["sequence"]) 
def encodings_from_intervals(intervals, sequence):
    encodings = []
    for interval in intervals:
        chrom, start, end = interval[:3]
        encoding = sequence.get_encoding_from_coords(chrom, start, end, "+")
        encodings.append(encoding.T) # encoding.T.shape = (n_feats, seq_len)
    encodings = np.stack(encodings) # (n_seqs, n_feats, seq_len)
    return encodings
encodings = encodings_from_intervals(intervals, sequence)
np.random.seed(config["seed"])
def get_random_encodings(intervals, sequence, n_seqs):
    idxs = np.random.choice(len(intervals), n_seqs, replace=False)
    intervals = [intervals[i] for i in idxs] 
    return encodings_from_intervals(intervals, sequence)
n_background = config["n_background"]
background_encodings = get_random_encodings(
    background_intervals, sequence, n_background
)
logging.info("Positive intervals and encodings loaded, background intervals ready.")

# Run DeepLIFT on positive sequences, averaging over attributions from
# random background seqs per positive seq, using captum's DeepLiftShap
logging.info(f"Running DeepLIFT averaged over {n_background} refs (DeepLiftShap)")
attributions = []
dl = DeepLiftShap(model)
progress_bar = tqdm(encodings,
                    file=open(os.devnull, "w"),
                    total=len(encodings))
for encoding in progress_bar:
    x = torch.Tensor(encoding)[None].to(device)
    x_background = torch.Tensor(background_encodings).to(device)
    attribution = dl.attribute(x, x_background, target=config["target"])
    # ^attribution is (1, n_feats, seq_len)
    attributions.append(attribution[0].detach().cpu().numpy())
    if progress_bar.n % 5 == 0:
        logging.info(str(progress_bar))
attributions = np.stack(attributions)

# Save encodings and attributions to directory
output_dir = config["output_dir"]
os.makedirs(output_dir, exist_ok=True)
logging.info(f"Directory {output_dir} created.")
np.savez(f"{output_dir}/encodings.npz", encodings)
np.savez(f"{output_dir}/attributions.npz", attributions)
logging.info(f"Encodings and attributions saved to {output_dir}!")
