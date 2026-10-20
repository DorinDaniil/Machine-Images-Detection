"""One CSV format for every dataset in the project.

Columns
    path          image file, absolute or relative to `data_root`
    label         0 = real, 1 = generated or edited
    mask_path     binary PNG (255 = generated region) for partial fakes, else empty
    generator     e.g. gpt-image-2, nano-banana-2, flux.2, sdxl-vae-redecode; empty for real
    content_type  photo | art | graphics | text | screenshot | biomed | other
    source        dataset name the row came from
    split         train | val | calib | test
"""

from pathlib import Path

import pandas as pd

COLUMNS = ["path", "label", "mask_path", "generator", "content_type", "source", "split"]
CONTENT_TYPES = ("photo", "art", "graphics", "text", "screenshot", "biomed", "other")
SPLITS = ("train", "val", "calib", "test")


def read_manifest(path: str | Path, data_root: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["label"] = df["label"].astype(int)

    bad_types = set(df["content_type"]) - set(CONTENT_TYPES)
    if bad_types:
        raise ValueError(f"{path}: unknown content_type values {bad_types}")
    bad_splits = set(df["split"]) - set(SPLITS)
    if bad_splits:
        raise ValueError(f"{path}: unknown split values {bad_splits}")

    if data_root is not None:
        root = Path(data_root)
        for col in ("path", "mask_path"):
            df[col] = [str(root / p) if p and not Path(p).is_absolute() else p for p in df[col]]
    return df


def write_manifest(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(path, index=False)


def filter_manifest(df: pd.DataFrame, **conditions) -> pd.DataFrame:
    """filter_manifest(df, split="test", label=0, content_type=["photo", "art"])."""
    mask = pd.Series(True, index=df.index)
    for column, value in conditions.items():
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple, set)) else [value]
        mask &= df[column].isin(values)
    return df[mask].reset_index(drop=True)


def summarize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Row counts per source, content type and label."""
    return (
        df.groupby(["source", "content_type", "label"]).size().rename("n").reset_index()
    )
