import numpy as np
import torch

def compute_window_loglikelihoods_from_kde(shape_data, kde_grid, kde_scores):
    """Compute log-likelihood of each window in shape data where the
    distribution at each position is represented as a vector of log
    densities corresponding to the grid values.

    Parameters
    ----------
    shape_data : np.ndarray
        (n_seqs, seq_len) array of shape data
    kde_grid : np.ndarray
        (grid_len,) array of increasing grid values
    kde_scores : np.ndarray 
        (motif_len, grid_len) log densities per grid value and motif
        position

    Returns
    -------
    np.ndarray
        (n_seqs, seq_len - motif_len + 1) array where the (i, j)th
        entry is the log likelihood of window [i, j:j+motif_len]
    """

    # (n_seqs, seq_len, grid_len)
    grid_val_is_below = kde_grid[None, None] <= shape_data[:, :, None]
    # (n_seqs, seq_len)
    highest_grid_index_below = grid_val_is_below.sum(axis=2) - 1
    # If this causes an error, then grid minimum is not low enough
    # (n_seqs, seq_len)
    highest_grid_val_below = kde_grid[highest_grid_index_below]
    # If this causes an error, then grid maximum is not high enough
    lowest_grid_val_above = kde_grid[highest_grid_index_below + 1]

    dist_to_highest_below = shape_data - highest_grid_val_below
    dist_to_lowest_above = lowest_grid_val_above - shape_data
    highest_below_weight = dist_to_lowest_above / (
        dist_to_lowest_above + dist_to_highest_below)
    lowest_above_weight = 1 - highest_below_weight

    # (n_seqs, seq_len, grid_len - 1) 
    grid_loc_onehot = np.zeros_like(grid_val_is_below[:, :, :-1])
    np.put_along_axis(
        grid_loc_onehot, highest_grid_index_below[:, :, None], 1, axis=2
    )
    # (n_seqs, seq_len, grid_len - 1)
    grid_loc_hbwhot = grid_loc_onehot * highest_below_weight[:, :, None]
    grid_loc_lawhot = grid_loc_onehot * lowest_above_weight[:, :, None]
    # (n_seqs, grid_len - 1, seq_len)
    grid_loc_hbwhot_tensor = torch.Tensor(grid_loc_hbwhot).transpose(1, 2)
    grid_loc_lawhot_tensor = torch.Tensor(grid_loc_lawhot).transpose(1, 2)
    # (1, grid_len, motif_len)
    kde_scores_tensor = torch.Tensor(kde_scores.T)[None]
    
    highest_below_ll_contributions = torch.nn.functional.conv1d( 
        grid_loc_hbwhot_tensor, kde_scores_tensor[:, :-1], bias=None
    ).squeeze() # (n_seqs, seq_len - motif_len + 1)
    
    lowest_above_ll_contributions = torch.nn.functional.conv1d( 
        grid_loc_lawhot_tensor, kde_scores_tensor[:, 1:], bias=None
    ).squeeze() # (n_seqs, seq_len - motif_len + 1)

    window_lls = (
        highest_below_ll_contributions + lowest_above_ll_contributions
    ).numpy()
    return window_lls

def compute_shape_motif_llrs(
    shape_data, background_kde_grid, background_kde_scores,
    motif_kde_grid, motif_kde_scores
):
    """Compute log likelihood ratio scores of shape motif on shape data.

    Parameters
    ----------
    shape data : np.ndarray
        (n_seqs, seq_len) array of shape data for a set of sequences
    background_kde_grid : np.ndarray
        (background_grid_len,) array of feature values background log
        density is evaluated at
    background_kde_scores : np.ndarray
        (background_grid_len,) background log densities
    motif_kde_grid : np.ndarray
        (motif_kde_grid_len,) array of feature values motif log density
        is evaluated at
    motif_kde_scores : np.ndarray
        (motif_len, motif_kde_grid_len) position-wise log densities

    Returns
    -------
    np.ndarray
        (2, n_seqs, seq_len - motif_len + 1) array of log(motif likelihood
        / background likelihood) at each position, + and - (rc)
    """
    motif_len = motif_kde_scores.shape[0]
    background_grid_len = background_kde_scores.shape[0]
    broadcast_shape = (motif_len, background_grid_len)
    background_kde_scores_broadcasted = np.copy(
        np.broadcast_to(background_kde_scores[None], broadcast_shape)
    )
    background_window_lls = compute_window_loglikelihoods_from_kde(
        shape_data, background_kde_grid, background_kde_scores_broadcasted
    )
    motif_window_lls = compute_window_loglikelihoods_from_kde(
        shape_data, motif_kde_grid, motif_kde_scores
    )
    rc_motif_window_lls = compute_window_loglikelihoods_from_kde(
        shape_data, motif_kde_grid, motif_kde_scores[::-1].copy()
    )
    motif_window_llrs = motif_window_lls - background_window_lls
    rc_motif_window_llrs = rc_motif_window_lls - background_window_lls
    return np.stack((motif_window_llrs, rc_motif_window_llrs))

def compute_seq_motif_llrs(seq_data, pwm):
    """Computes log-likelihood ratio scores of sequence motif on
    sequence data using position weight matrix.

    Parameters
    ----------
    seq_data : np.ndarray
        (n_seqs, 4, seq_len) array of DNA one-hot encodings
    pwm : np.ndarray
        (motif_len, 4) position weight matrix of motif

    Returns
    -------
    np.ndarray
        (2, n_seqs, seq_len - motif_len + 1) log likelihood ratios of
        each window of each input sequence, + and - (rc)
    """
    seq_tensor = torch.Tensor(seq_data) 
    # (1, 4, motif_len)
    pwm_tensor = torch.Tensor(pwm).T[None]
    motif_llrs = torch.nn.functional.conv1d(seq_tensor, pwm_tensor)
    rc_motif_llrs = torch.nn.functional.conv1d(
        seq_tensor, torch.flip(pwm_tensor, dims=[1,2])
    )
    motif_and_rc_llrs = np.stack(
        (motif_llrs.squeeze().numpy(), rc_motif_llrs.squeeze().numpy())
    )
    return motif_and_rc_llrs

def call_occurrences(motif_llrs, threshold, interval=None):
    """Call occurrences of motif according to window log likelihood
    ratios and threshold.

    Parameters
    ----------
    motif_llrs : np.ndarray
        (2, n_seqs, n_positions)
    threshold : float
    interval : tuple or None
        (start, end) 

    Returns
    -------
    np.ndarray
        (2, n_seqs, n_positions) binary matrix of occurrence positions
    """
    binary_occurrence_vecs = []
    n_positions = motif_llrs.shape[2]
    for i in range(motif_llrs.shape[1]):
        binary_occurrence_vec = np.zeros((2, n_positions))
        llrs_i = motif_llrs[:, i].copy()
        llrs_i[llrs_i <= threshold] = -np.inf
        while True:
            argmax = np.argmax(llrs_i)
            strand, pos = argmax // n_positions, argmax % n_positions
            maxval = llrs_i[strand, pos]
            if maxval == -np.inf:
                break
            binary_occurrence_vec[strand, pos] = 1
            if interval is None:
                l_idx = max(pos - 15, 0)
                r_idx = min(pos + 15, n_positions)
                llrs_i[:, l_idx:r_idx] = -np.inf
            else:
                length = interval[1] - interval[0]
                start = interval[0]
                rev_start = 30 - interval[1]
                if strand == 0:
                    cur_start = start
                else:
                    cur_start = rev_start
                l_idx = max(pos + start - cur_start - length + 1, 0)
                r_idx = min(pos + start - cur_start + length, n_positions)
                llrs_i[0, l_idx:r_idx] = -np.inf
                l_idx = max(pos + rev_start - cur_start - length + 1, 0)
                r_idx = min(pos + rev_start - cur_start + length, n_positions)
                llrs_i[1, l_idx:r_idx] = -np.inf
        binary_occurrence_vecs.append(binary_occurrence_vec)
    return np.stack(binary_occurrence_vecs, axis=1)
