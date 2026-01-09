from mne.io import read_raw_fif
from mne import read_epochs
from os import mkdir
from os.path import isdir, isfile

from plots import generate_plots, average_channel
from utils import get_subjectlist
from analyze_subject import analyze_subject


def main(bids_root="../data/"):
    # path where to save the datasets.
    if not isdir(bids_root + "processed"):
        mkdir(bids_root + "processed")

    subjects = get_subjectlist(bids_root)
    print(f"Subjects: {subjects}")

    i = input("One subject (1) or all (2) or calculate mean of all PO7/PO8 (3): ")
    if i.lower() == "1":
        subject = input("Subject ID: ")
        process_one_subject(subject, bids_root)
    elif i.lower() == "2":
        for subject in subjects:
            analyze_and_save(subject, bids_root)
    elif i.lower() == "3":
        average_channel("PO7")
        average_channel("PO8")


def process_one_subject(subject_id, bids_root):
    epochs = None
    raw = None
    ica = None

    if isfile(f"{bids_root}processed/sub-{subject_id}_raw.fif"):
        print("Subject already processed")
        i = input("Use existing data?: Y/N ")
        if i.lower() == "y":
            epochs = read_epochs(
                f"{bids_root}processed/sub-{subject_id}_epochs.fif", preload=True
            )
            raw = read_raw_fif(
                f"{bids_root}processed/sub-{subject_id}_raw.fif", preload=True
            )
        else:
            epochs, raw, ica = analyze_and_save(subject_id, bids_root)
    generate_plots(epochs, raw, ica)


def analyze_and_save(subject_id, bids_root):
    print("Running new processing pipeline for subject " + subject_id)

    epochs, raw, ica = analyze_subject(subject_id, bids_root)

    epochs.save(f"{bids_root}processed/sub-{subject_id}_epochs.fif", overwrite=True)
    raw.save(f"{bids_root}processed/sub-{subject_id}_raw.fif", overwrite=True)
    return epochs, raw, ica


if __name__ == "__main__":
    main()
