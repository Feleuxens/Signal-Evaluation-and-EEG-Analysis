from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne.channels import make_standard_montage
from mne import pick_types, create_info
import mne
import numpy as np
from mne.io.edf.edf import RawEDF

ICA_N_COMPONENTS = 0.99  # Variance explained or number of components


def _run_ica(raw: RawEDF) -> tuple[RawEDF, ICA | None]:

    # TODO: https://mne.tools/mne-icalabel/stable/generated/examples/00_iclabel.html

    return raw, None


def run_ica(raw: RawEDF, n_components=ICA_N_COMPONENTS) -> tuple[RawEDF, ICA | None, int]:
    # require at least two EEG channels
    picks_eeg = pick_types(raw.info, eeg=True, meg=False, exclude="bads")
    if len(picks_eeg) < 2:
        print("Not enough EEG channels for ICA.")
        return raw, None, 0

    ica = ICA(
        n_components=n_components,
        method="infomax",
        random_state=42,
        max_iter="auto",
    )
    ica.fit(raw, picks=picks_eeg)

    # find EOG components using any channels already marked as 'eog' or named EXG*
    ch_types = raw.get_channel_types()
    eog_chs = [ch for ch, t in zip(raw.ch_names, ch_types) if t == "eog"]
    if not eog_chs:
        # fall back to EXG labels if they exist (do not change channel types or interpolate)
        eog_chs = [ch for ch in raw.ch_names if ch.upper().startswith("EXG")]

    eog_inds = []
    if eog_chs:
        try:
            # use first available EOG/EXG channel to create EOG epochs
            eog_epochs = create_eog_epochs(
                raw, ch_name=eog_chs[0], reject_by_annotation=True, preload=True
            )
            eog_inds, scores = ica.find_bads_eog(eog_epochs)
        except Exception:
            # fallback correlation method
            try:
                sources = ica.get_sources(raw).get_data()
                eog_data = raw.get_data(picks=[eog_chs[0]]).ravel()
                scores = np.array(
                    [
                        np.corrcoef(sources[i], eog_data)[0, 1]
                        for i in range(sources.shape[0])
                    ]
                )
                eog_inds = list(np.where(np.abs(scores) > 0.3)[0])
            except Exception:
                eog_inds = []

    ecg_inds = []
    if True:
        try:
            ecg_epochs = create_ecg_epochs(raw, reject_by_annotation=True, preload=True)
            ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs)
        except Exception:
            ecg_inds = []

    to_remove = sorted(set(eog_inds + ecg_inds))
    if to_remove:
        ica.exclude = to_remove
        ica.apply(raw)  # applies ICA to raw in-place

    number_excluded_components = len(getattr(ica, "exclude", []))
    print(
        "ICA cleaning applied. Excluded components:",
        getattr(ica, "exclude", []),
    )

    return raw, ica, number_excluded_components


# def _ensure_eog_channels(raw):
#     """
#     Ensure there are EOG channels in raw. If none exist, create virtual EOGs
#     from frontal EEG pairs (best-effort) and add them as channels of type 'eog'.
#     Returns list of created EOG channel names (may be empty if existing EOGs present).
#     """
#     candidates = ["EOG", "HEOG", "VEOG", "eog"]
#     present = [
#         ch for ch in raw.ch_names if any(c.lower() in ch.lower() for c in candidates)
#     ]
#     if present:
#         return present
#
#     picks = pick_types(raw.info, eeg=True, meg=False, exclude="bads")
#     ch_names = [raw.ch_names[p] for p in picks]
#     sfreq = raw.info["sfreq"]
#     created = []
#
#     # Horizontal and vertical candidate pairs (best-effort)
#     horiz_pairs = [("Fp1", "Fp2"), ("F7", "F8"), ("F9", "F10")]
#     vert_pairs = [("Fp1", "Oz"), ("Fp2", "Oz")]
#
#     # Try to create a horizontal VEOG/HEOG from first matching pair
#     for a, b in horiz_pairs:
#         if a in ch_names and b in ch_names:
#             data = raw.get_data(picks=[a]) - raw.get_data(picks=[b])
#             name = "HEOG_virtual"
#             info_new = create_info([name], sfreq, ch_types="eog")
#             raw.add_channels([mne.io.RawArray(data, info_new)], force_update_info=True)
#             created.append(name)
#             break
#
#     # Try to create a vertical EOG if not created already
#     if not created:
#         for a, b in vert_pairs:
#             if a in ch_names and b in ch_names:
#                 data = raw.get_data(picks=[a]) - raw.get_data(picks=[b])
#                 name = "VEOG_virtual"
#                 info_new = create_info([name], sfreq, ch_types="eog")
#                 raw.add_channels(
#                     [mne.io.RawArray(data, info_new)], force_update_info=True
#                 )
#                 created.append(name)
#                 break
#
#     # Final fallback: use Fp1 minus average of occipitals if available
#     if not created:
#         for a in ["Fp1", "Fp2"]:
#             if a in ch_names:
#                 occ = [ch for ch in ["Oz", "O1", "O2"] if ch in ch_names]
#                 if occ:
#                     data = raw.get_data(picks=[a]) - raw.get_data(picks=[occ[0]])
#                     name = "VEOG_virtual"
#                     info_new = create_info([name], sfreq, ch_types="eog")
#                     raw.add_channels(
#                         [mne.io.RawArray(data, info_new)], force_update_info=True
#                     )
#                     created.append(name)
#                     break
#
#     return created
#
#
# def _run_ica(raw, n_components=ICA_N_COMPONENTS):
#     picks_eeg = pick_types(raw.info, eeg=True, meg=False, exclude="bads")
#     if len(picks_eeg) < 2:
#         raise RuntimeError("Not enough EEG channels for ICA.")
#
#     ica = ICA(
#         method="fastica",
#         random_state=42,
#         max_iter="auto",
#     )
#     ica.fit(raw, picks=picks_eeg)
#
#     return raw, ica
#
#
# def run_ica_and_interpolate(
#     raw, n_components=0.99, method="fastica", random_state=42, reject_ecg=False
# ):
#     """
#     Run ICA on raw, remove EOG (and optionally ECG) components, then safely interpolate bad channels.
#
#     Parameters:
#       raw: mne.io.Raw
#       n_components: float or int for ICA
#       method: ICA method ('fastica', 'picard', etc.)
#       random_state: RNG seed
#       reject_ecg: bool — try to find and remove ECG components if True
#
#     Returns:
#       raw_clean: Raw after ICA cleaning and interpolation
#       ica: fitted ICA instance
#     """
#
#     # Ensure there is a montage for interpolation later (if absent, set a standard one)
#     if not raw.info.get("dig"):
#         try:
#             raw.set_montage(make_standard_montage("standard_1020"), on_missing="ignore")
#         except Exception:
#             pass
#
#     picks_eeg = pick_types(raw.info, eeg=True, meg=False, exclude="bads")
#     if len(picks_eeg) < 2:
#         raise RuntimeError("Not enough EEG channels for ICA.")
#
#     ica = ICA(
#         method=method,
#         random_state=random_state,
#         max_iter="auto",
#     )
#     ica.fit(raw, picks=picks_eeg)
#
#     # Ensure EOG channels exist or create virtual ones
#     eog_chs = _ensure_eog_channels(raw)
#     # Determine EOG channel names present now
#     eog_found = [
#         ch
#         for ch in raw.ch_names
#         if raw.get_channel_types()[raw.ch_names.index(ch)] == "eog" or ch in eog_chs
#     ]
#
#     eog_inds = []
#     # Try detection via create_eog_epochs + find_bads_eog
#     if eog_found:
#         try:
#             # use the first EOG channel found for epoch creation
#             eog_epochs = create_eog_epochs(
#                 raw,
#                 picks=picks_eeg,
#                 ch_name=eog_found[0],
#                 reject_by_annotation=True,
#                 preload=True,
#             )
#             eog_inds, _ = ica.find_bads_eog(eog_epochs)
#         except Exception:
#             # fallback: correlate ICA components with EOG timeseries
#             try:
#                 sources = ica.get_sources(raw).get_data()
#                 eog_data = raw.get_data(picks=[eog_found[0]]).ravel()
#                 scores = np.array(
#                     [
#                         np.corrcoef(sources[i], eog_data)[0, 1]
#                         for i in range(sources.shape[0])
#                     ]
#                 )
#                 thresh = 0.3
#                 eog_inds = list(np.where(np.abs(scores) > thresh)[0])
#             except Exception:
#                 eog_inds = []
#
#     ecg_inds = []
#     if reject_ecg:
#         try:
#             ecg_epochs = create_ecg_epochs(
#                 raw, picks=picks_eeg, reject_by_annotation=True, preload=True
#             )
#             ecg_inds, _ = ica.find_bads_ecg(ecg_epochs)
#         except Exception:
#             # if no ECG channel or detection fails, skip
#             ecg_inds = []
#
#     to_remove = sorted(set(eog_inds + ecg_inds))
#     if to_remove:
#         ica.exclude = to_remove
#         ica.apply(raw)
#
#     # Interpolate bad channels safely
#     # try:
#     #     raw.interpolate_bads(reset_bads=False, mode="accurate")
#     # except Exception:
#     #     # fallback: set a standard montage then try again
#     #     try:
#     #         raw.set_montage(make_standard_montage("standard_1020"), on_missing="ignore")
#     #         raw.interpolate_bads(reset_bads=False, mode="accurate")
#     #     except Exception as e:
#     #         print("Interpolation failed:", e)
#
#     return raw, ica
