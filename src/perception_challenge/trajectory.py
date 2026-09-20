import numpy as np


def estimate(frames, measurements, fps=30., y_sign=1., max_gap=8):
    frames = np.asarray(frames)
    measured = np.asarray(measurements, dtype=float)
    if fps <= 0 or not np.isfinite(fps) or len(frames) < 2:
        raise ValueError('Need positive finite FPS and at least two frames')
    if frames.ndim != 1 or not np.isfinite(frames).all() or np.any(frames != frames.astype(int)):
        raise ValueError('Frame IDs must be finite integers')
    if measured.shape != (len(frames), 3):
        raise ValueError('Measurements must have shape (number of frames, 3)')
    if y_sign not in (-1, 1) or max_gap < 1 or int(max_gap) != max_gap:
        raise ValueError('Invalid Y sign or maximum gap')
    frames = frames.astype(int)
    if np.any(np.diff(frames) <= 0):
        raise ValueError('Frame IDs must be strictly increasing')
    good = np.isfinite(measured).all(axis=1) & (measured[:, 0] > 0) & (np.linalg.norm(measured, axis=1) < 100)
    q = measured[:, :2] * [1, y_sign]
    supported = good.copy()
    for i in np.flatnonzero(supported):
        neighbors = supported & (abs(frames-frames[i]) <= 8)
        neighbors[i] = False
        if neighbors.sum() >= 3 and np.linalg.norm(q[i]-np.median(q[neighbors], axis=0)) > 2:
            good[i] = False
    if good.sum() < 2:
        raise ValueError('Not enough valid traffic-light observations')
    first = np.flatnonzero(good)[0]
    angle = np.arctan2(q[first, 1], q[first, 0])
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    raw = -q @ rotation.T
    ids = np.arange(frames[0], frames[-1]+1)
    path = np.full((len(ids), 2), np.nan)
    kept = frames[good]
    smooth = np.array([np.median(raw[good & (abs(frames-f) <= 3)], axis=0) for f in kept])
    smooth[0] = raw[first]
    for f, p in zip(kept, smooth):
        path[f-ids[0]] = p
    for i, gap in enumerate(np.diff(kept)):
        if gap <= max_gap:
            lo, hi = kept[i]-ids[0], kept[i+1]-ids[0]
            path[lo:hi+1] = np.linspace(smooth[i], smooth[i+1], gap+1)
    return ids, raw, path, good, rotation, int(frames[first])
