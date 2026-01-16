from mne.io.edf.edf import RawEDF


def interpolate_bad_channels(raw: RawEDF) -> RawEDF:
    """Interpolate bad channels in the raw data."""

    if raw.info["bads"]:
        try:
            # Ensure montage is set for interpolation
            if not raw.info.get("dig"):
                raw.set_montage("standard_1020", on_missing="ignore")

            raw.interpolate_bads(reset_bads=False, mode="accurate")

            print(f"Interpolated bad channels: {raw.info['bads']}")

        except Exception as e:
            print(f"Interpolation failed: {e}")

        # finally:
        return raw
    return raw
