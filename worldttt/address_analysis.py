"""Approximate raw Q/K address diagnostic; reports invalid pairs explicitly."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def address_scores(query, keys, correspondence, seed=0):
    """Query [tokens, heads, dim] against first-visit keys at matched positions."""
    if query.shape[1:] != keys.shape[1:] or correspondence.numel() != query.shape[0]:
        raise ValueError('Query, key and correspondence dimensions differ')
    q = F.normalize(query.float(), dim=-1).transpose(0, 1)
    k = F.normalize(keys.float(), dim=-1).transpose(0, 1)
    similarity = q @ k.transpose(1, 2)
    target = correspondence.long().view(1, -1, 1).expand(q.shape[0], -1, 1)
    rank = 1 + (similarity > similarity.gather(2, target)).sum(dim=2)
    perm = torch.randperm(k.shape[1], generator=torch.Generator().manual_seed(seed))
    shuffled = similarity[:, :, perm]
    shuffled_rank = 1 + (shuffled > shuffled.gather(2, target)).sum(dim=2)
    return dict(top1=float((rank == 1).float().mean()), mrr=float((1 / rank.float()).mean()),
                shuffled_top1=float((shuffled_rank == 1).float().mean()),
                shuffled_mrr=float((1 / shuffled_rank.float()).mean()), tokens=query.shape[0])


def pose_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    distance = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    relative = a[:3, :3].T @ b[:3, :3]
    angle = math.degrees(math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    return distance, angle


def video_frame(capture, index):
    import cv2
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f'Cannot decode frame {index}')
    return frame


def alignment(first, returned):
    """Find first→return image homography using independent visual evidence."""
    import cv2
    orb = cv2.ORB_create(nfeatures=2000)
    a, da = orb.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), None)
    b, db = orb.detectAndCompute(cv2.cvtColor(returned, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance]
    if len(good) < 20:
        return None, len(good)
    src = np.float32([a[m.queryIdx].pt for m in good])
    dst = np.float32([b[m.trainIdx].pt for m in good])
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.)
    inliers = int(mask.sum()) if mask is not None else 0
    if matrix is None or inliers < 15 or inliers / len(good) < .3 or abs(np.linalg.det(matrix)) < 1e-8:
        return None, inliers
    return matrix, inliers


def token_centers(record, image_shape):
    grid_h, grid_w = record['token_grid']
    ids = record['spatial_indices'].numpy()
    height, width = image_shape[:2]
    return np.column_stack((((ids % grid_w) + .5) * width / grid_w,
                            ((ids // grid_w) + .5) * height / grid_h)).astype(np.float32)


def correspondence(first, returned, homography, image_shape):
    import cv2
    first_points = token_centers(first, image_shape)
    return_points = token_centers(returned, image_shape)
    projected = cv2.perspectiveTransform(return_points[None], np.linalg.inv(homography))[0]
    distances = np.linalg.norm(projected[:, None] - first_points[None], axis=-1)
    match = distances.argmin(axis=1)
    # Do not claim a spatial match farther than one feature-cell diagonal.
    grid_h, grid_w = first['token_grid']
    radius = 1.5 * math.hypot(image_shape[1] / grid_w, image_shape[0] / grid_h)
    valid = distances[np.arange(len(match)), match] <= radius
    return torch.as_tensor(match, dtype=torch.long), valid


def analyze(run, max_distance=.5, max_angle=15.):
    import cv2
    run = Path(run)
    case = json.loads((run / 'case.json').read_text(encoding='utf-8'))
    trace = torch.load(run / 'address_trace.pt', map_location='cpu', weights_only=True)
    records = {(r['layer'], r['latent_frame']): r for r in trace['records']}
    cap = cv2.VideoCapture(str(run / 'video_generated.mp4'))
    if not cap.isOpened():
        raise ValueError('Cannot open generated video for address analysis')
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    try:
        ordered_pairs = sorted(case['revisit_pairs'], key=lambda p: p.get('quality_score', 999))
        for pair_idx, pair in enumerate(ordered_pairs):
            a, b = int(pair['frame_a']), int(pair['frame_b'])
            base = dict(frame_a=a, frame_b=b)
            if pair_idx >= 8:
                rows.append(dict(base, valid=False, reason='trace_budget'))
                continue
            fa, fb = round(a / 8), round(b / 8)
            first = records.get((trace['layers'][0], fa))
            returned = records.get((trace['layers'][0], fb))
            if first is None or returned is None:
                rows.append(dict(base, valid=False, reason='missing_trace_record'))
                continue
            distance, angle = pose_difference(first['camera'], returned['camera'])
            base.update(commanded_distance=distance, commanded_angle_deg=angle)
            if distance > max_distance or angle > max_angle:
                rows.append(dict(base, valid=False, reason='camera_pose_far'))
                continue
            va, vb = round(a * fps / 16), round(b * fps / 16)
            if va >= count or vb >= count:
                rows.append(dict(base, valid=False, reason='video_too_short'))
                continue
            image_a, image_b = video_frame(cap, va), video_frame(cap, vb)
            transform, inliers = alignment(image_a, image_b)
            if transform is None:
                rows.append(dict(base, valid=False, reason='image_alignment_failed', inliers=inliers))
                continue
            matches, valid_tokens = correspondence(first, returned, transform, image_a.shape)
            if valid_tokens.sum() < 4:
                rows.append(dict(base, valid=False, reason='spatial_coverage_low',
                                 matched_tokens=int(valid_tokens.sum())))
                continue
            for layer in trace['layers']:
                key = records.get((layer, fa))
                query = records.get((layer, fb))
                if key is None or query is None:
                    rows.append(dict(base, layer=layer, valid=False, reason='missing_layer_record'))
                    continue
                score = address_scores(query['q'][valid_tokens], key['k'], matches[valid_tokens],
                                       seed=3407 + pair_idx + layer)
                rows.append(dict(base, layer=layer, valid=True, inliers=inliers, **score))
    finally:
        cap.release()
    report = dict(case_id=case['id'], pairs=len(case['revisit_pairs']),
                  valid_pairs=len({(r['frame_a'], r['frame_b']) for r in rows if r['valid']}),
                  rows=rows, diagnostic='raw pre-shortconv/RoPE Q/K; approximate image-aligned correspondence')
    (run / 'address_analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    report = analyze(args.run)
    print(json.dumps(dict(case_id=report['case_id'], valid_pairs=report['valid_pairs'], pairs=report['pairs'])))


if __name__ == '__main__':
    main()
