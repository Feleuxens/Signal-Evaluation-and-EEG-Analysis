from glob import glob

import mne
import mne_bids
import numpy as np
from mne import read_epochs


def get_subjectlist(bids_root="../data/"):
    """Get list of subject IDs from BIDS dataset."""

    subject_list = mne_bids.get_entity_vals(bids_root, entity_key="subject")
    subject_list = [f"{int(s):03d}" for s in subject_list]  # zero-pad to 3 digits

    return subject_list


def evoke_channels(epochs):
    desc_random = "random"
    desc_regular = "regular"

    # Select epochs by condition name (string key, not integer ID!)
    epochs_random = epochs[desc_random]
    epochs_regular = epochs[desc_regular]

    # Average across trials (mean)
    evoked_random = epochs_random.average()
    evoked_regular = epochs_regular.average()

    return evoked_random, evoked_regular


def average_channel(channel, bids_root="../data/"):
    """
        Load all processed epoch files and compute the grand average for channel PO7,
        separately for random and regular conditions.

        Parameters
        ----------
        channel : str
        bids_root : str
            Path to the BIDS root directory.

        Returns
        -------
        data_random : np.ndarray
            Grand average waveform for random condition in µV (n_times,).
        data_regular : np.ndarray
            Grand average waveform for regular condition in µV (n_times,).
        times : np.ndarray
            Time vector in seconds.
        """
    processed_dir = f"{bids_root}processed/"
    epoch_files = sorted(glob(f"{processed_dir}sub-*_epochs.fif"))

    if not epoch_files:
        print(f"No processed epoch files found in {processed_dir}")
        return None, None, None

    print(f"Found {len(epoch_files)} processed epoch file(s)")

    evokeds_random = []
    evokeds_regular = []
    times = None

    for fpath in epoch_files:
        subject_id = fpath.split("sub-")[-1].split("_epochs")[0]
        print(f"  Loading subject {subject_id}...")

        epochs = read_epochs(fpath, preload=True)

        # Check if channel exists
        if channel not in epochs.ch_names:
            print(
                f"    WARNING: {channel} not found in subject {subject_id}, skipping."
            )
            continue

        # Create evoked responses for each condition
        evoked_random, evoked_regular = evoke_channels(epochs)
        evoked_diff = mne.combine_evoked(
            [evoked_regular, evoked_random],
            weights=[1, -1]
        )

        # Get channel channel index and extract data
        channel_idx = evoked_random.ch_names.index(channel)

        evokeds_random.append(evoked_random.get_data()[channel_idx, :])
        evokeds_regular.append(evoked_regular.get_data()[channel_idx, :])

        times = evoked_random.times  # Same for all subjects

    if not evokeds_random:
        print(f"No valid subjects with {channel} channel found.")
        return None, None, None

    # Stack and compute mean across subjects
    n_subjects = len(evokeds_random)
    evokeds_random = np.array(evokeds_random)  # shape: (n_subjects, n_times)
    evokeds_regular = np.array(evokeds_regular)

    data_random = np.mean(evokeds_random, axis=0) * 1e6  # Convert to µV
    data_regular = np.mean(evokeds_regular, axis=0) * 1e6

    print(f"\nGrand average computed from {n_subjects} subject(s)")
    print(f"  Time range: {times[0]:.3f} to {times[-1]:.3f} s")
    print(f"  Number of time points: {len(times)}")

    return data_random, data_regular, times, n_subjects, evoked_diff


def pairwise_average(arr1, arr2):
    assert len(arr1) == len(arr2)

    result = []
    for i in range(0, len(arr1)):
        result.append((arr1[i] + arr2[i])/2)
    return result