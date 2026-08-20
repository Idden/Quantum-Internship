"""
Data export utilities for quantum battery test artifacts.

Functions for saving arrays (NPZ), metadata (JSON), and tabular
data (CSV) alongside test visualizations.
"""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        return super().default(obj)


def save_array_data(output_path: Path, **arrays: np.ndarray) -> Path:
    """
    Save one or more arrays to a compressed NPZ file.

    Args:
        output_path: Destination path (should end in .npz).
        **arrays: keyword=array pairs to store.

    Returns:
        Path written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def save_metadata(output_path: Path, metadata: Dict[str, Any]) -> Path:
    """
    Save a metadata dictionary to a JSON file with numpy type conversion.

    Args:
        output_path: Destination path (should end in .json).
        metadata: Arbitrary dict — numpy scalars/arrays are auto-converted.

    Returns:
        Path written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, cls=_NumpyEncoder)
    return output_path


def save_csv_table(
    output_path: Path,
    columns: Dict[str, Sequence],
    header: Optional[str] = None,
) -> Path:
    """
    Save columnar data to a CSV file.

    Args:
        output_path: Destination path (should end in .csv).
        columns: ``{column_name: values}`` dict.  All value sequences
                 must have the same length.
        header: Optional comment line written before the CSV header.

    Returns:
        Path written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns.keys())
    rows = list(zip(*(columns[n] for n in names)))

    with open(output_path, 'w', newline='') as f:
        if header:
            f.write(f'# {header}\n')
        writer = csv.writer(f)
        writer.writerow(names)
        for row in rows:
            writer.writerow(
                [float(v) if isinstance(v, (np.floating, float)) else v for v in row]
            )
    return output_path
