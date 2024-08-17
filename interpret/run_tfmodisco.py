import datetime
import logging
import numpy as np
import os
import sys
import time
import yaml

def format_time(seconds):
    time_object = datetime.timedelta(seconds=seconds)
    return str(time_object)

logging.basicConfig(level=logging.INFO)
config_file = sys.argv[1]
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

encodings = np.load(config["encodings"])["arr_0"]
attributions = np.load(config["attributions"])["arr_0"]
output_dir = config["output_dir"]
os.makedirs(output_dir, exist_ok=True)
logging.info(f"Created output_dir {output_dir}.")
max_seqlets = config["max_seqlets"]

# Run TF-MoDISco on sequence inputs / attributions
seq_onehot = encodings[:, :4]
seq_attribs = attributions[:, :4]
onehot_path = f"{output_dir}/seq_onehot.npz"
attribs_path = f"{output_dir}/seq_attribs.npz"
np.savez(onehot_path, seq_onehot)
np.savez(attribs_path, seq_attribs)
modisco_out_path = f"{output_dir}/seq_results.h5"
command = (f"modisco motifs -s '{onehot_path}' "
           f"-a '{attribs_path}' "
           f"-n {max_seqlets} " 
           f"-o '{modisco_out_path}' "
           "-v")
logging.info(f"Running {command}")
time1 = time.time()
os.system(command)
runtime = format_time(time.time() - time1)
logging.info(f"Ran TF-MoDISco for sequence features in {runtime}.")
os.remove(onehot_path)
os.remove(attribs_path)


# Run TF-MoDISco on shape inputs / attributions
shape_feats = ["EP", "HelT", "MGW", "ProT", "Roll"]
for feat_i, feat in enumerate(shape_feats):
    # Quaternize shape feature based on its quartiles
    feat_encoding = encodings[:, 4+feat_i] # (n_seqs, seq_len)
    feat_vals = feat_encoding.flatten() # (n_seqs*seq_len,)
    feat_quartiles = np.quantile(feat_vals, [0.25, 0.5, 0.75]) # (3,)
    n_seqs, seq_len = feat_encoding.shape
    feat_onehot = np.zeros((n_seqs, 4, seq_len), dtype=np.float32)
    feat_n_greater = (feat_encoding[:, :, None] >= feat_quartiles
                     ).sum(axis=2)[:, None] # (n_seqs, 1, seq_len)
    np.put_along_axis(feat_onehot, feat_n_greater, 1, axis=1)
    assert (feat_onehot.sum(axis=1) == 1).all()
    feat_attribs = attributions[:, 4+feat_i] # (n_seqs, seq_len)
    feat_onehot_attribs = feat_onehot * feat_attribs[:, None]
    logging.info(f"Generated quaternized {feat} features.")
    onehot_path = f"{output_dir}/{feat}_onehot.npz"
    attribs_path = f"{output_dir}/{feat}_attribs.npz"
    np.savez(onehot_path, feat_onehot)
    np.savez(attribs_path, feat_onehot_attribs)
    modisco_out_path = f"{output_dir}/{feat}_results.h5"
    command = (f"modisco motifs -s '{onehot_path}' "
               f"-a '{attribs_path}' "
               f"-n {max_seqlets} " 
               f"-o '{modisco_out_path}' "
               "--nofliprc "
               "-v")
    logging.info(f"Running {command}")
    time1 = time.time()
    os.system(command)
    runtime = format_time(time.time() - time1)
    logging.info(f"Ran TF-MoDISco for {feat} in {runtime}.")
    os.remove(onehot_path)
    os.remove(attribs_path)
