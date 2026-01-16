from time import time

import json
from mne.io import read_raw_fif
from mne import read_epochs
from os import mkdir
from os.path import isdir, isfile

from plots import generate_plots, plot_channel, plot_topomap
from utils import get_subjectlist, average_channel, pairwise_average, pipeline_statistics
from analyze_subject import analyze_subject


def main(bids_root="../data/"):
    # path where to save the datasets.
    if not isdir(bids_root + "processed"):
        mkdir(bids_root + "processed")

    subjects = get_subjectlist(bids_root)
    print(f"Subjects: {subjects}")

    i = input("One subject (1) or all (2) or calculate mean of all PO7/PO8 (3) or Topomap (4) or Pipeline Statistics (5): ")
    if i.lower() == "1":
        subject = input("Subject ID: ")
        process_one_subject(subject, bids_root)
    elif i.lower() == "2":
        time_start = time()
        failed = []
        for subject in subjects:
            try:
                analyze_and_save(subject, bids_root)
            except Exception as e:
                failed.append({subject: e})
        total_time = time() - time_start
        print(f"\nElapsed time: {total_time} seconds\n")
        print(f"{len(failed)} subjects failed")
        print(failed)
    elif i.lower() == "3":
        data_random_po7, data_regular_po7, times_po7, n_subjects, evoked_diff_po7 = average_channel("PO7")
        plot_channel("PO7", data_random_po7, data_regular_po7, times_po7, n_subjects)
        data_random_po8, data_regular_po8, times_po8, n_subjects, evoked_diff_po8 = average_channel("PO8")
        plot_channel("PO8", data_random_po8, data_regular_po8, times_po8, n_subjects)
        data_random_both = pairwise_average(data_random_po7, data_random_po8)
        data_regular_both = pairwise_average(data_regular_po7, data_regular_po8)
        plot_channel("PO7+PO8", data_random_both, data_regular_both, times_po7, n_subjects)
    elif i.lower() == "4":
        data_random_po7, data_regular_po7, times_po7, n_subjects, evoked_diff_po7 = average_channel("PO7")
        plot_topomap(evoked_diff_po7)
    elif i.lower() == "5":
        pipeline_statistics()



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
            print("Running pipeline for subject " + subject_id)
            epochs, raw, ica = analyze_and_save(subject_id, bids_root)
    else:
        print("Running pipeline for subject " + subject_id)
        epochs, raw, ica = analyze_and_save(subject_id, bids_root)
    generate_plots(epochs, raw, ica)


def analyze_and_save(subject_id, bids_root):
    print("Running new processing pipeline for subject " + subject_id)

    epochs, raw, ica, pipeline_stats = analyze_subject(subject_id, bids_root)

    epochs.save(f"{bids_root}processed/sub-{subject_id}_epochs.fif", overwrite=True)
    raw.save(f"{bids_root}processed/sub-{subject_id}_raw.fif", overwrite=True)
    if ica is not None:
        ica.save(f"{bids_root}processed/sub-{subject_id}_ica.fif", overwrite=True)

    with open(f"{bids_root}processed/sub-{subject_id}_meta.txt", "w") as f:
        f.write(json.dumps(pipeline_stats))
    return epochs, raw, ica


if __name__ == "__main__":
    main()
