from tfbind8_data import download_tfbind8, load_tfbind8


transcription_factor = "SIX6_REF_R1"
local_data_dir = "data/design_bench_data"

if main := __name__ == "__main__":
    download_tfbind8(
    transcription_factor=transcription_factor,
    local_dir=local_data_dir,
    )

    x, y = load_tfbind8(
    transcription_factor=transcription_factor,
    local_dir=local_data_dir,
    )

    # Lightweight replacement for a design-bench task object.
    task = {
    "name": f"tf_bind_8-{transcription_factor}",
    "x": x,
    "y": y,
    }

    print(f"x shape: {x.shape}, dtype={x.dtype}")
    print(f"y shape: {y.shape}, dtype={y.dtype}")
