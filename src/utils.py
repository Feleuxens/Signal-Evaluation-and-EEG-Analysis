import mne_bids


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
