import numpy as np
from PIL import Image, ImageFilter


def components(mask):
    mask = mask.copy()
    for y, x in zip(*np.nonzero(mask)):
        if not mask[y, x]:
            continue
        stack, pixels = [(y, x)], []
        mask[y, x] = False
        while stack:
            v, u = stack.pop()
            pixels.append((u, v))
            for a, b in ((v-1,u), (v+1,u), (v,u-1), (v,u+1)):
                if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a,b]:
                    mask[a,b] = False
                    stack.append((a,b))
        if len(pixels) >= 6:
            p = np.array(pixels)
            yield (*p.min(axis=0), *(p.max(axis=0)+1)), len(pixels)


def detect(image):
    small = image.resize((480, 300))
    hsv = np.asarray(small.convert('HSV')) / 255.
    h, s, v = hsv.transpose(2,0,1)
    yy, xx = np.indices(h.shape)
    orange = (h > .035) & (h < .115) & (s > .5) & (v > .42) & (yy > 155) & (xx > 250)
    detections = []
    for (x1,y1,x2,y2), area in components(orange):
        w, ht = x2-x1, y2-y1
        if 7 <= w <= 28 and 3 <= ht <= 14 and 1.1 < w/ht < 5 and area/(w*ht) > .55:
            if any(abs((x1+x2)/2-(b[0]+b[2])/2) < w for _, b in detections):
                continue
            detections.append(('barrel', (x1, y1, x2, min(300, y1+round(w*1.9)))))
    cream = (s < .22) & (v > .65) & (yy > 140) & (yy < 164) & (xx > 170) & (xx < 260)
    
    cream = np.asarray(Image.fromarray(cream).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3)))
    roofs = [(box, area) for box, area in components(cream) if 23 < box[2]-box[0] < 80]
    if roofs:
        (left,top,right,_), _ = max(roofs, key=lambda item: item[1])
    
        columns = np.flatnonzero(cream[top:top+4, left:right].any(axis=0)) + left
        x1,x2 = columns[0],columns[-1]+1
        w = x2-x1
        detections.append(('golf_cart', (max(0,x1-4), top, min(480,x2+4), min(300,top+round(w*1.8)))))
    scale = np.array([image.width/480, image.height/300]*2)
    return [(kind, tuple(np.rint(np.array(box)*scale).astype(int))) for kind, box in detections]


def light_state(image, box):
    x1,y1,x2,y2 = box
    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
        return 'unknown'
    crop = image.crop(box)

    crop = crop.crop((crop.width//4, 0, 3*crop.width//4, crop.height))
    h,s,v = (np.asarray(crop.convert('HSV'))/255.).transpose(2,0,1)
    bright = (s > .45) & (v > .65)
    scores = [np.sum(bright & ((h < .04)|(h > .95))),
              np.sum(bright & (h > .1) & (h < .18)),
              np.sum(bright & (h > .3) & (h < .56))]
    return ['red','yellow','green'][int(np.argmax(scores))] if max(scores) >= 5 else 'unknown'


def associate(detections, tracks, frame, next_id, max_age=15, gate=2.):

    candidates = []
    for i, d in enumerate(detections):
        for tid, old in tracks.items():
            distance = np.linalg.norm(np.array(d['world'])-old['world'])
            if d['kind'] == old['kind'] and frame-old['frame'] <= max_age and distance <= gate:
                candidates.append((distance, i, tid))
    matches, used = {}, set()
    for _, i, tid in sorted(candidates):
        if i not in matches and tid not in used:
            matches[i] = tid
            used.add(tid)
    for i, d in enumerate(detections):
        if i not in matches:
            matches[i] = next_id
            next_id += 1
        d['track_id'] = matches[i]
        previous = tracks.get(matches[i], {})
        d['raw_world'] = d['world']
        history = (previous.get('history', []) + [d['world']])[-30:]
        if d['kind'] == 'barrel':
            d['world'] = np.median(history, axis=0).tolist()
        elif previous:

            alpha = 1 - np.exp(-(frame-previous['frame'])/5.)
            d['world'] = (np.array(previous['world']) + alpha*(np.array(d['world'])-previous['world'])).tolist()
        tracks[matches[i]] = dict(d, frame=frame, history=history)
    return next_id
