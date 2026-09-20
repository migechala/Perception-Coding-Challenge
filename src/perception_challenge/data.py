import csv
import re
from pathlib import Path

import numpy as np


def index_files(directory: Path, suffix: str) -> dict[int, Path]:
    result = {}
    for path in sorted(directory.glob(f'*{suffix}')):
        match = re.search(r'(\d+)$', path.stem)
        if match is None:
            raise ValueError(f'Missing numeric frame ID: {path}')
        frame = int(match[1])
        if frame in result:
            raise ValueError(f'Duplicate frame ID {frame}: {path}')
        result[frame] = path
    return result


def load_boxes(path: Path) -> dict[int, tuple[int, int, int, int]]:
    result = {}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = ('frame', 'x1', 'y1', 'x2', 'y2')
        if not set(columns).issubset(reader.fieldnames or []):
            columns = ('frame_id', 'x_min', 'y_min', 'x_max', 'y_max')
        for row in reader:
            frame, *box = (int(row[key]) for key in columns)
            if frame in result:
                raise ValueError(f'Duplicate CSV frame {frame}')
            result[frame] = tuple(box)
    return result


def valid_box(box, width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def load_xyz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        key = 'points' if 'points' in archive else 'xyz'
        if key not in archive:
            raise ValueError(f'No points or xyz array: {path}')
        values = archive[key]
        if values.ndim != 3 or values.shape[2] not in (3, 4):
            raise ValueError(f'Unexpected XYZ shape {values.shape}: {path}')
        return values[..., :3]


def sample_box(xyz: np.ndarray, box, radius: int = 3, min_points: int = 3, max_range: float = 100.):
    if radius < 0 or min_points < 1 or not np.isfinite(max_range) or max_range <= 0:
        raise ValueError('Invalid sampling parameters')
    h, w = xyz.shape[:2]
    if not valid_box(box, w, h):
        return None
    x1, y1, x2, y2 = box
    u, v = min(round((x1+x2)/2), x2-1), min(round((y1+y2)/2), y2-1)
    patch = xyz[max(y1,v-radius):min(y2,v+radius+1),
                max(x1,u-radius):min(x2,u+radius+1)].reshape(-1, 3)
    ranges = np.linalg.norm(patch, axis=1)
    good = np.isfinite(patch).all(axis=1) & (patch[:,0] > 0) & (ranges < max_range)
    return np.median(patch[good], axis=0) if good.sum() >= min_points else None
