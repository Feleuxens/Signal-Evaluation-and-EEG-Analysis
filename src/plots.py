from glob import glob
import matplotlib.pyplot as plt
from mne import read_epochs, Evoked
import mne
import numpy as np

from utils import evoke_channels

BASELINE = (-0.25, 0.0)  # baseline correction period


def generate_plots(epochs, raw, ica):
    power_spectral_density_plot(raw, epochs, 0, 64)

    if ica is not None:
        ica_topography_plot(ica, raw)

    one_channel_erp_plot(raw, epochs, BASELINE)

    all_channel_erp_plot(epochs, BASELINE)

    # unprocessed_vs_processed_plot(raw_unprocessed, raw)

    butterfly_plot(epochs)


def power_spectral_density_plot(raw, epochs, fmin, fmax):
    # Sensor-level PSD (Welch) — full-band and zoomed alpha/beta
    fig_psd = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=2048).plot(
        average=True, picks="eeg", show=False
    )
    fig_psd.suptitle(f"Sensor PSD ({fmin}-{fmax} Hz)")
    plt.show()


def ica_topography_plot(ica, raw):
    if ica is None:
        return

    # drop EXG from raw copy, then fit ICA
    raw_no_exg = raw.copy().drop_channels(
        [c for c in raw.ch_names if c.startswith("EXG")]
    )

    ica.fit(raw_no_exg)

    comp_picks = list(range(min(20, ica.n_components_)))
    ica.plot_components(picks=comp_picks, inst=raw_no_exg)
    # ica.plot_properties(raw_no_exg, picks=comp_picks)

    # # 1) ensure comp picks are component indices (0..n_components-1)
    # n_comp = ica.n_components_
    # comp_picks = list(range(min(20, n_comp)))  # e.g., first 20 components
    #
    # # 2) create an inst whose info has EXG removed (used only for plotting)
    # exclude_exg = [ch for ch in raw.ch_names if ch.startswith("EXG")]
    # inst_plot = raw.copy()
    # if exclude_exg:
    #     inst_plot.drop_channels(exclude_exg)
    #
    # # 3) sanity checks
    # print("n_comp", n_comp)
    # print("comp_picks", comp_picks)
    # print("EXG removed:", exclude_exg)
    #
    # print("ica.n_components_", getattr(ica, "n_components_", None))
    # print("ica.ch_names", getattr(ica, "ch_names", None))
    # print("inst_plot.info['chs'] count", len(inst_plot.info["chs"]))
    # print(
    #     "overlapping channels present:",
    #     [ch for ch in inst_plot.ch_names if ch.startswith("EXG")],
    # )
    #
    # # 4) plot using component indices and cleaned inst
    # ica.plot_components(picks=comp_picks, inst=inst_plot)
    # ica.plot_properties(inst_plot, picks=comp_picks)
    #
    # # picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, meg=False)
    # # picks = picks_eeg[: min(len(picks_eeg), ica.n_components_)]
    # # ica.plot_components(picks=picks, inst=raw)
    # # ica.plot_properties(raw, picks=picks)


def one_channel_erp_plot(raw, epochs, baseline):
    evoked_random, evoked_regular = evoke_channels(epochs)

    # choose channel
    preferred = "PO7"
    if preferred not in raw.ch_names:
        # pick a posterior channel if available
        for ch in ["POz", "Oz", "P3", "P4", "O1", "O2"]:
            if ch in raw.ch_names:
                preferred = ch
                break
    # get single-channel evoked (random vs regular)
    evoked_ch_random = evoked_random.copy().pick([preferred])
    evoked_ch_regular = evoked_regular.copy().pick([preferred])
    plt.figure()
    plt.plot(
        evoked_ch_random.times,
        evoked_ch_random.data.T * 1e6,
        label=f"Random — {preferred}",
    )
    plt.plot(
        evoked_ch_regular.times,
        evoked_ch_regular.data.T * 1e6,
        label=f"Regular — {preferred}",
    )
    plt.axvspan(baseline[0], baseline[1], color="gray", alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"ERP at {preferred}")
    plt.legend()
    plt.show()


def all_channel_erp_plot(epochs, baseline):
    evoked_random, evoked_regular = evoke_channels(epochs)

    # For type checking reasons
    if type(evoked_random) is not mne.evoked.EvokedArray:
        raise ValueError("evoked_random is not an instance of mne.evoked.EvokedArray")

    if type(evoked_regular) is not mne.evoked.EvokedArray:
        raise ValueError("evoked_regular is not an instance of mne.evoked.EvokedArray")

    # simple channel-mean time series plot
    times = evoked_random.times
    mean_random = evoked_random.data.mean(axis=0)
    mean_regular = evoked_regular.data.mean(axis=0)

    plt.plot(times, mean_random * 1e6, label="Random")  # convert to µV if data in V
    plt.plot(times, mean_regular * 1e6, label="Regular")
    plt.axvspan(baseline[0], baseline[1], color="gray", alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.title("Mean across channels — Random vs Regular")
    plt.show()


def unprocessed_vs_processed_plot(raw_unprocessed, raw):
    # # Quick overlay of unprocessed vs processed PSD
    # fig = plt.figure(figsize=(8, 4))
    # raw_unprocessed.compute_psd(method="welch", fmin=1, fmax=40, n_fft=2048).plot(
    #     average=True, picks="eeg", show=False
    # )
    # raw_processed.compute_psd(method="welch", fmin=1, fmax=40, n_fft=2048).plot(
    #     average=True, picks="eeg", show=False
    # )
    # plt.suptitle(f"PSD: unprocessed (first) vs processed (second)")
    # plt.show()

    # ensure raw_unprocessed saved before filters/ASR/ICA
    # raw_unprocessed = raw.copy()  # add this right after raw.load_data()

    # select up to 10 channels (change list if desired)
    candidates = ["Fp1", "Fp2", "AF7", "AF8", "Fz", "Cz", "Pz", "Oz", "O1", "O2"]
    picked = [c for c in candidates if c in raw.ch_names]
    if len(picked) < 10:
        eeg_chs = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
        for ch in eeg_chs:
            if ch not in picked:
                picked.append(ch)
            if len(picked) == 10:
                break
    picked = picked[:10]

    # time window (s)
    tstart, duration = 100.0, 10.0

    for i in range(10):
        tmin = tstart + i * duration
        sf = raw.info["sfreq"]
        start_idx = int(tmin * sf)
        stop_idx = int((tmin + duration) * sf)
        times = np.linspace(tmin, tmin + duration, stop_idx - start_idx)

        # get data (channels x times)
        data_un = (
            raw_unprocessed.copy().pick(picked).get_data(start=start_idx, stop=stop_idx)
        )

        data_proc = raw.copy().pick(picked).get_data(start=start_idx, stop=stop_idx)

        # scale to µV
        data_un_uV = data_un * 1e6
        data_proc_uV = data_proc * 1e6

        # compute demeaned, unit-range signals per channel
        def norm(x):
            x = x - x.mean(axis=1, keepdims=True)
            rng = x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True)
            rng[rng == 0] = 1.0
            return x / rng

        un = norm(data_un_uV)
        pr = norm(data_proc_uV)

        spacing = 1.5  # small spacing after normalization
        offsets = np.arange(len(picked))[::-1] * spacing

        plt.figure(figsize=(12, 6))
        for i, ch in enumerate(picked):
            off = offsets[i]
            plt.plot(times, un[i] + off, color="k", linewidth=0.6)
            plt.plot(times, pr[i] + off, color="r", linewidth=0.8)
            plt.text(times[0] - duration * 0.01, off, ch, va="center", fontsize=9)
        plt.yticks([])
        plt.xlabel("Time (s)")
        plt.title("Normalized: unprocessed (black) vs processed (red)")
        plt.show()

    # # compute offsets for stacking
    # max_amp = np.max(np.abs(np.concatenate([data_un_uV, data_proc_uV])))
    # spacing = max_amp * 3  # vertical spacing between channels
    # offsets = np.arange(len(picked))[::-1] * spacing  # top channel first
    #
    # plt.figure(figsize=(12, 6))
    # for i, ch in enumerate(picked):
    #     off = offsets[i]
    #     plt.plot(
    #         times, data_un_uV[i] + off, color="k", linewidth=0.6
    #     )  # unprocessed black
    #     plt.plot(
    #         times, data_proc_uV[i] + off, color="r", linewidth=0.8
    #     )  # processed red
    #     plt.text(times[0] - duration * 0.01, off, ch, va="center", fontsize=9)
    #
    # plt.xlim(times[0], times[-1])
    # plt.xlabel("Time (s)")
    # plt.yticks([])  # hide numeric y ticks
    # plt.title("Raw: unprocessed (black) vs processed (red) — selected channels")
    # plt.tight_layout()
    # plt.show()
    #


def butterfly_plot(epochs):
    evoked_random, evoked_regular = evoke_channels(epochs)

    # Butterfly plot for evoked difference or single condition
    evoked_random.plot(
        spatial_colors=False,
        show=True,
        time_unit="s",
        titles=f"Butterfly — Random",
    )
    # Alternatively plot difference
    evoked_diff = mne.combine_evoked([evoked_random, evoked_regular], weights=[1, -1])
    evoked_diff.plot(
        spatial_colors=False,
        show=True,
        time_unit="s",
        titles=f"Butterfly — Random-Regular diff",
    )


def plot_channel(channel, data_random, data_regular, times, n_subjects, bids_root="../data/"):
    processed_dir = f"{bids_root}processed/"

    plt.figure(figsize=(10, 5))
    plt.plot(times * 1000, data_random, "r-", linewidth=2, label="Random")
    plt.plot(times * 1000, data_regular, "b-", linewidth=2, label="Regular")
    plt.axhline(0, color="k", linestyle="--", linewidth=0.5)
    plt.axvline(0, color="k", linestyle="--", linewidth=0.5)
    plt.yticks([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"Grand Average ERP at {channel} (n={n_subjects} subjects)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{processed_dir}grand_average_{channel}.png", dpi=150)
    plt.show()

    return data_random, data_regular, times


def plot_topomap(evoked_diff: Evoked):
    evoked_diff = evoked_diff.copy().drop_channels(
        [c for c in evoked_diff.ch_names if c.startswith("EXG")]
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    time_windows = {
        "100–130 ms": (0.100, 0.130),
        "170–200 ms": (0.170, 0.200),
        "300–1000 ms": (0.300, 1.000),
    }

    for ax, (title, (tmin, tmax)) in zip(axes, time_windows.items()):
        evoked_diff.plot_topomap(
            times=[(tmin + tmax) / 2],
            average=tmax - tmin,
            axes=ax,
            show=False,
            colorbar=False,
            vlim=(3, -3),
            cmap="RdYlBu_r",
        )
        ax.set_title(title)

    fig.colorbar(axes[-1].images[0], ax=axes, shrink=0.6, label="µV")
    plt.show()