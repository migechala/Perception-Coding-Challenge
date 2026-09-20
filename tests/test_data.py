from pathlib import Path

import numpy as np
import pytest

from perception_challenge.data import index_files, load_boxes, load_xyz, sample_box


def test_sparse_ids_not_positions(tmp_path):
    for name in ['left000298.png', 'left000201.png']:
        (tmp_path / name).touch()
    assert list(index_files(tmp_path, '.png')) == [201, 298]
    (tmp_path / 'other201.png').touch()
    with pytest.raises(ValueError, match='Duplicate'):
        index_files(tmp_path, '.png')


@pytest.mark.parametrize('key,channels', [('points', 3), ('xyz', 4)])
def test_archive_formats(tmp_path, key, channels):
    path = tmp_path / 'depth.npz'
    np.savez(path, **{key: np.ones((8, 9, channels), dtype=np.float32)})
    assert load_xyz(path).shape == (8, 9, 3)


def test_robust_patch():
    xyz = np.full((10, 10, 3), [12., -2., 3.])
    xyz[4, 4] = np.nan
    xyz[4, 5] = np.inf
    xyz[5, 4] = 0
    xyz[5, 5] = 1000
    np.testing.assert_allclose(sample_box(xyz, (2, 2, 8, 8)), [12, -2, 3])
    assert sample_box(xyz, (0, 0, 0, 0)) is None
    assert sample_box(xyz, (-1, 0, 3, 3)) is None
    assert sample_box(np.zeros_like(xyz), (2, 2, 8, 8)) is None


def test_csv(tmp_path):
    path = tmp_path / 'boxes.csv'
    path.write_text('frame,x1,y1,x2,y2\n201,1,2,3,4\n')
    assert load_boxes(path) == {201: (1, 2, 3, 4)}


def test_patch_rejects_out_of_range_and_behind_camera():
    xyz = np.full((7,7,3), [150.,0.,1.])
    xyz[:2] = [12.,1.,2.]
    np.testing.assert_allclose(sample_box(xyz,(0,0,7,7)), [12,1,2])
    xyz[:] = [-10,1,2]
    assert sample_box(xyz,(0,0,7,7)) is None
