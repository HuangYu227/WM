"""Within-video gap-matched nonreturn controls; not an official R2M score."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .address_analysis import pose_difference, video_frame


def choose_control(camera, first, returned, gap_tolerance=16):
    gap = returned - first
    return_distance, return_angle = pose_difference(camera[first], camera[returned])
    candidates = []
    for a in range(8, len(camera), 8):
        for delta in range(-gap_tolerance, gap_tolerance + 1, 8):
            b = a + gap + delta
            if b >= len(camera) or abs(a - first) <= 16:
                continue
            distance, angle = pose_difference(camera[a], camera[b])
            if distance >= max(.5, 2 * return_distance) or angle >= max(15., 2 * return_angle):
                candidates.append((abs(delta), abs(a - first), a, b))
    return None if not candidates else min(candidates)[2:]


def image_metrics(a, b):
    from tools.metrics.sana_wm.eval_unified import compute_ssim, compute_lpips
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    psnr = 100. if mse == 0 else 10 * math.log10(255**2 / mse)
    return dict(psnr=psnr, ssim=compute_ssim(a, b), lpips=compute_lpips(a, b))


def latent_metrics(latent, first, second):
    if latent.ndim == 5:
        latent = latent[0]
    if latent.ndim != 4:
        raise ValueError('Expected Stage-1 latent with C,T,H,W axes')
    a, b = latent[:, round(first / 8)].float(), latent[:, round(second / 8)].float()
    return dict(mse=float((a - b).square().mean()),
                cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)))


def nonreturn_segment(video, first, second, fps):
    """Sparse local frame change and sharpness proxies across a control interval."""
    import cv2
    changes, sharpness = [], []
    for frame in np.linspace(first, second - 1, 8, dtype=int):
        index = round(int(frame) * fps / 16)
        a = cv2.cvtColor(video_frame(video, index), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(video_frame(video, index + 1), cv2.COLOR_BGR2GRAY)
        changes.append(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()))
        sharpness.append(float(cv2.Laplacian(a, cv2.CV_32F).var()))
    return dict(frame_change_mean=float(np.mean(changes)),
                laplacian_variance_mean=float(np.mean(sharpness)),
                sampled_pairs=len(changes))


def analyze(run):
    import cv2
    run = Path(run)
    case = json.loads((run / 'case.json').read_text(encoding='utf-8'))
    camera = np.load(case['camera'], allow_pickle=False)
    latent = torch.load(run / 'latent.pt', map_location='cpu', weights_only=True)
    video = cv2.VideoCapture(str(run / 'video_generated.mp4'))
    if not video.isOpened():
        raise ValueError('Cannot open generated video')
    fps = float(video.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        video.release()
        raise ValueError('Generated video has an invalid frame rate')
    count = 0
    while video.read()[0]:
        count += 1
    expected = int(case.get('num_frames', len(camera)))
    validation = dict(decoded_frames=count, expected_frames=expected,
                      complete=count == expected, fps=fps)
    rows = []
    try:
        for pair in case.get('revisit_pairs', []):
            a, b = int(pair['frame_a']), int(pair['frame_b'])
            control = pair['control'] if 'control' in pair else choose_control(camera, a, b)
            revisit_valid = 0 <= a < b and round(b * fps / 16) < count
            control_valid = (revisit_valid and control is not None and len(control) == 2
                             and 0 < control[0] < control[1]
                             and round(control[1] * fps / 16) < count)
            row = dict(revisit=[a, b], control=control, revisit_valid=revisit_valid,
                       valid=control_valid, control_metrics=None, control_latent=None,
                       nonreturn_segment=None)
            if revisit_valid:
                first, returned = [cv2.cvtColor(video_frame(video, round(i * fps / 16)), cv2.COLOR_BGR2RGB)
                                   for i in (a, b)]
                row['revisit_metrics'] = image_metrics(first, returned)
                row['revisit_latent'] = latent_metrics(latent, a, b)
            if control_valid:
                images = [cv2.cvtColor(video_frame(video, round(i * fps / 16)), cv2.COLOR_BGR2RGB)
                          for i in control]
                row['control_metrics'] = image_metrics(*images)
                row['control_latent'] = latent_metrics(latent, *control)
                row['nonreturn_segment'] = nonreturn_segment(video, *control, fps)
            else:
                row['reason'] = ('video_too_short_or_invalid_revisit' if not revisit_valid
                                 else 'no_pose_distinct_gap_match_or_short_video')
            rows.append(row)
    finally:
        video.release()
    result = dict(case_id=case['id'], label='relative revisit analysis, not official R2M-Bench',
                  video_validation=validation,
                  total_pairs=len(rows), valid_revisit_pairs=sum(r['revisit_valid'] for r in rows),
                  valid_control_pairs=sum(r['valid'] for r in rows), rows=rows)
    (run / 'relative_revisit.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
