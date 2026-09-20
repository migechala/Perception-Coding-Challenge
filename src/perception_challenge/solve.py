import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from PIL import Image, ImageDraw

from .data import index_files, load_boxes, load_xyz, sample_box
from .trajectory import estimate
from .objects import detect, light_state, associate


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def run():
    dataset = Path('dataset')
    output = Path('outputs')
    fps = 30.
    if not shutil.which('ffmpeg'):
        raise ValueError('Install ffmpeg to encode the required MP4 outputs')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive and finite')
    output.mkdir(parents=True, exist_ok=True)
    rgb = index_files(dataset/'rgb', '.png')
    depth = index_files(dataset/'xyz', '.npz')
    boxes = load_boxes(dataset/'bbox_light.csv')
    frames = sorted(depth)
    measurements = []
    for frame in frames:
        xyz = load_xyz(depth[frame])
        point = sample_box(xyz, boxes.get(frame, (0,0,0,0)))
        measurements.append(point if point is not None else [np.nan]*3)
    ids, raw, path, good, rotation, anchor = estimate(frames, measurements, fps, 1, 8)
    write_csv(output/'measurements.csv', ['frame','X','Y','Z','raw_world_x','raw_world_y','accepted'],
              ([f,*m,*p,int(g)] for f,m,p,g in zip(frames,measurements,raw,good)))
    write_csv(output/'trajectory.csv', ['frame','seconds_from_anchor','x_m','y_m','source'],
              ([int(f),(f-anchor)/fps,*p,'observed' if f in np.array(frames)[good] else 'interpolated' if np.isfinite(p).all() else 'missing'] for f,p in zip(ids,path)))
    print(f'Part A: {good.sum()}/{len(frames)} accepted measurements; anchor {anchor}', flush=True)
    observations, tracks, next_id = {}, {}, 1
    for frame in sorted(rgb.keys() & depth.keys()):
        ego = path[frame-ids[0]]
        image = Image.open(rgb[frame]).convert('RGB')
        xyz = load_xyz(depth[frame])
        if xyz.shape[:2] != (image.height,image.width):
            raise ValueError(f'RGB/XYZ dimensions differ at {frame}')
        detections = []
        for kind, box in detect(image):
            sample_region = box
            if kind == 'golf_cart':
                x1,y1,x2,y2 = box
                sample_region = (x1,y1,x2,min(y2,y1+max(7,round(image.height/75))))
            point = sample_box(xyz, sample_region)
            if point is None or not 0 < point[0] < 60 or np.linalg.norm(point) >= 100:
                continue
            local = point[:2]
            world = ego + rotation @ local
            if np.isfinite(world).all():
                detections.append(dict(kind=kind, box=list(map(int,box)), sample_box=list(map(int,sample_region)), ego=local.tolist(), world=world.tolist()))
        next_id = associate(detections, tracks, frame, next_id)
        state = light_state(image, boxes.get(frame,(0,0,0,0)))
        observations[frame] = dict(state=state, detections=detections)
        if frame in [min(rgb), sorted(rgb)[len(rgb)//2], max(rgb)]:
            draw = ImageDraw.Draw(image)
            for d in detections:
                color = 'orange' if d['kind']=='barrel' else 'cyan'
                draw.rectangle(d['box'],outline=color,width=4)
                b = d['sample_box']
                u,v = (b[0]+b[2])//2,(b[1]+b[3])//2
                draw.ellipse((u-5,v-5,u+5,v+5), fill='red')
                draw.text(tuple(d['box'][:2]), f"{d['kind']} #{d['track_id']}",fill='black',stroke_width=1,stroke_fill='white')
            image.thumbnail((960,600))
            image.save(output/f'objects_{frame:06d}.png')
    (output/'objects.json').write_text(json.dumps(observations,indent=2)+'\n')
    counts = {kind: sum(d['kind']==kind for obs in observations.values() for d in obs['detections']) for kind in ['barrel','golf_cart']}
    stability = {}
    for tid in sorted(tracks):
        seen = [(f,d) for f,o in observations.items() for d in o['detections'] if d['track_id'] == tid]
        positions = np.array([d['raw_world'] for _,d in seen])
        residuals = np.linalg.norm(positions-np.median(positions,axis=0),axis=1)
        stability[str(tid)] = dict(kind=seen[0][1]['kind'], observations=len(seen),
                                  raw_spread_p95_m=float(np.percentile(residuals,95)))
    summary = dict(track_spread=stability, anchor_frame=anchor, fps_assumed=fps, y_sign=1,
                   accepted_measurements=int(good.sum()), xyz_frames=len(frames), rgb_frames=len(rgb), detections=counts,
                   assumption='Constant heading; origin under light; first available valid frame defines X. Original frame zero unavailable.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    render(Path('.'), ids, frames, raw, path, observations, anchor, fps)
    print(f'Part B detections: {counts}. Plots/videos: project root; diagnostics: {output}', flush=True)


def render(output, ids, frames, raw, path, observations, anchor, fps):
    fig, ax = plt.subplots(figsize=(9,4), layout='constrained')
    raw_points = ax.scatter(raw[:,0],raw[:,1],s=8,color='silver',label='Raw light measurements')
    line, = ax.plot(path[:,0],path[:,1],color='tab:blue',label='Filtered ego trajectory')
    valid = path[np.isfinite(path).all(axis=1)]
    ax.scatter(*valid[0],marker='s',color='green',label=f'Start (frame {anchor})')
    end = ax.scatter(*valid[-1],marker='x',color='red',label='End')
    ax.scatter(0,0,marker='*',s=160,color='gold',edgecolor='black',label='Reference light')
    current, = ax.plot([],[],'o',color='tab:blue')
    ax.set(xlabel='World X (m)',ylabel='World Y, left (m)',title='Ego trajectory — constant-heading approximation')
    ax.margins(y=.3)
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(output/'trajectory.png',dpi=160)
    ax.set_xlim(ax.get_xlim()); ax.set_ylim(ax.get_ylim())
    title = ax.set_title('')
    writer = FFMpegWriter(fps=fps,codec='libx264',extra_args=['-pix_fmt','yuv420p'])
    with writer.saving(fig,str(output/'trajectory.mp4'),dpi=100):
        for i,frame in enumerate(ids):
            raw_points.set_offsets(raw[np.asarray(frames) <= frame])
            end.set_visible(frame == ids[-1])
            line.set_data(path[:i+1,0],path[:i+1,1])
            current.set_data([path[i,0]],[path[i,1]])
            status = '' if np.isfinite(path[i]).all() else ' — no estimate (gap)'
            title.set_text(f'Constant-heading approximation | assumed {fps:g} fps\nFrame {frame} | {(frame-anchor)/fps:.2f} s{status}')
            writer.grab_frame()
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,6),layout='constrained')
    ego_path, = ax.plot(path[:,0],path[:,1],color='lightgray',label='Ego path')
    ego, = ax.plot([],[],'o',color='tab:blue',label='Ego')
    light = ax.scatter([0],[0],marker='*',s=180,color='gray',edgecolor='black',label='Light state')
    barrels = ax.scatter([],[],marker='s',s=55,color='darkorange',label='Barrels')
    cart = ax.scatter([],[],marker='D',s=65,color='purple',label='Golf cart')
    obj_points = [d['world'] for obs in observations.values() for d in obs['detections']]
    all_points = np.vstack([valid, [[0,0]], np.array(obj_points).reshape(-1,2)])
    lo,hi = all_points.min(0)-2,all_points.max(0)+2
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World X (m)',ylabel='World Y, left (m)')
    ax.set_aspect('equal'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    labels = []
    trail, = ax.plot([], [], color='purple', alpha=.5, linewidth=1)
    with writer.saving(fig,str(output/'enhanced_bev.mp4'),dpi=100):
        for i,frame in enumerate(ids):
            ego_path.set_data(path[:i+1,0],path[:i+1,1])
            ego.set_data([path[i,0]],[path[i,1]])
            obs = observations.get(int(frame),dict(state='unknown',detections=[]))
            light.set_facecolor({'unknown':'gray','green':'limegreen','yellow':'gold','red':'red'}[obs['state']])
            for artist,kind in [(barrels,'barrel'),(cart,'golf_cart')]:
                artist.set_offsets(np.array([d['world'] for d in obs['detections'] if d['kind']==kind]).reshape(-1,2))
            cart_ids = {d['track_id'] for d in obs['detections'] if d['kind'] == 'golf_cart'}
            recent = [d['world'] for f,o in observations.items() if frame-15 <= f <= frame
                      for d in o['detections'] if d['track_id'] in cart_ids]
            trail_points = np.array(recent).reshape(-1,2) if obs['detections'] else np.empty((0,2))
            trail.set_data(trail_points[:,0],trail_points[:,1])
            for label in labels:
                label.remove()
            labels = [ax.annotate(str(d['track_id']),d['world'],xytext=(5,5),textcoords='offset points',fontsize=7) for d in obs['detections']]
            ax.set_title(f'Frame {frame} | {(frame-anchor)/fps:.2f} s | constant heading\n'+('RGB detections' if frame in observations else 'No RGB — objects unobserved'))
            if frame == max(observations,default=int(ids[-1])):
                fig.savefig(output/'enhanced_bev.png',dpi=160)
            writer.grab_frame()
    plt.close(fig)
