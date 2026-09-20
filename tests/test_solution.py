import numpy as np
from PIL import Image, ImageDraw

from perception_challenge.trajectory import estimate
from perception_challenge.objects import associate, detect, light_state


def test_trajectory_sign_anchor_spike_and_gaps():
    frames = np.array([10,11,12,13,14,15,16,30,31])
    points = np.array([[20-(f-10)*.1, 2., 3.] for f in frames])
    points[3] = [70,2,3]
    ids, raw, path, good, rotation, anchor = estimate(frames,points,max_gap=3)
    assert anchor == 10 and not good[3]
    np.testing.assert_allclose(path[0],[-np.hypot(20,2),0],atol=1e-12)
    assert path[6,0] > path[0,0]
    assert np.isfinite(path[3]).all() and np.isnan(path[7:20]).all()
    np.testing.assert_allclose(raw[0]+rotation@points[0,:2],0,atol=1e-12)
    mirrored = points.copy(); mirrored[:,1] *= -1
    np.testing.assert_allclose(estimate(frames,mirrored,y_sign=-1,max_gap=3)[2],path)


def test_association_is_one_to_one_and_expires():
    tracks = {1:dict(kind='barrel',world=[0.,0.],frame=1)}
    ds = [dict(kind='barrel',world=[.1,0.]),dict(kind='barrel',world=[.2,0.])]
    nxt = associate(ds,tracks,2,2)
    assert [d['track_id'] for d in ds] == [1,2] and nxt == 3
    ds = [dict(kind='barrel',world=[.1,0.])]
    assert associate(ds,tracks,30,nxt) == 4 and ds[0]['track_id'] == 3


def test_color_shapes_and_light_unknown():
    image = Image.new('RGB',(480,300),'gray')
    draw = ImageDraw.Draw(image)
    draw.rectangle((300,180,314,186),fill=(240,140,25))
    draw.rectangle((300,192,314,198),fill=(240,140,25))
    draw.rectangle((185,150,223,154),fill='ivory')
    kinds = [kind for kind,_ in detect(image)]
    assert kinds.count('barrel') == 1 and 'golf_cart' in kinds
    assert light_state(image,(0,0,0,0)) == 'unknown'
    assert light_state(Image.new('RGB',(30,60),'cyan'),(0,0,30,60)) == 'green'


def test_roof_shadow_does_not_move_cart_box():
    image = Image.new('RGB', (480,300), 'gray')
    draw = ImageDraw.Draw(image)
    draw.rectangle((190,149,234,153), fill='ivory')
    original = next(box for kind,box in detect(image) if kind == 'golf_cart')
    draw.line((209,149,209,153), fill='gray', width=1)
    shadowed = next(box for kind,box in detect(image) if kind == 'golf_cart')
    assert original == shadowed


def test_estimate_input_validation_and_missing_ends():
    import pytest
    with pytest.raises(ValueError, match='shape'):
        estimate([1,2], [[1,2],[3,4]])
    with pytest.raises(ValueError, match='finite integers'):
        estimate([1,2.5], [[1,2,3],[3,4,5]])
    with pytest.raises(ValueError, match='maximum gap'):
        estimate([1,2], [[1,2,3],[3,4,5]], max_gap=0)
    _,_,path,good,_,anchor = estimate([1,2,3,4], [[np.nan]*3,[10,0,2],[9,0,2],[np.nan]*3])
    assert anchor == 2 and good.tolist() == [False,True,True,False]
    assert np.isnan(path[[0,-1]]).all()


def test_cart_smoothing_respects_elapsed_frames():
    def track_at(frames):
        tracks = {}
        nxt = associate([dict(kind='golf_cart',world=[0.,0.])],tracks,1,1)
        for frame in frames:
            ds = [dict(kind='golf_cart',world=[1.,0.])]
            nxt = associate(ds,tracks,frame,nxt)
        assert ds[0]['raw_world'] == [1.,0.]
        return ds[0]['world']
    np.testing.assert_allclose(track_at([6]), track_at([2,3,4,5,6]))
