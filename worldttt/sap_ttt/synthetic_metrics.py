"""Instance and prompt-hidden sticker scores for short procedural return videos."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .scene import ProceduralScene


def _sticker_predictions(scene: ProceduralScene, image: np.ndarray, frame: int):
    rendered = scene.render(frame)
    if image.shape != rendered['rgb'].shape:
        raise ValueError('Generated frame/renderer dimensions differ')
    result = {}
    for identity, center in enumerate(scene.centers, start=1):
        world = rendered['world']
        u = (world[..., 0] - center + 1.05) / 2.1
        v = (world[..., 1] + .95) / 1.9
        patch = ((rendered['instance'] == identity) &
                 (np.abs(u - .5) < .3) & (np.abs(v - .15) < .13))
        column = np.clip(((u - .2) / .6 * 20).astype(int), 0, 19)
        bits = np.array(list(map(int, f'{scene.sticker_codes[identity - 1]:020b}')))
        predicted = []
        for j in range(20):
            selected = image[patch & (column == j)]
            if len(selected) == 0:
                predicted.append(-1)
            else:
                predicted.append(int(selected[:, 0].mean() > selected[:, 2].mean()))
        result[identity] = {'truth': bits.tolist(), 'prediction': predicted}
    return result


def sticker_bit_accuracy(scene: ProceduralScene, image: np.ndarray, frame: int):
    predictions = _sticker_predictions(scene, image, frame)
    valid = correct = 0
    rows = {}
    for identity, row in predictions.items():
        visible = [i for i, bit in enumerate(row['prediction']) if bit >= 0]
        if visible:
            accuracy = sum(row['prediction'][i] == row['truth'][i] for i in visible) / len(visible)
            rows[str(identity)] = {'accuracy': accuracy, 'valid_bits': len(visible)}
            valid += len(visible)
            correct += sum(row['prediction'][i] == row['truth'][i] for i in visible)
    return {'mean_accuracy': None if not valid else correct / valid,
            'valid_bits': valid, 'per_instance': rows}


def _psnr(a, b, mask):
    if not mask.any():
        return None
    mse = float(np.mean((a[mask].astype(np.float32) - b[mask].astype(np.float32)) ** 2))
    return 100. if mse == 0 else 10 * math.log10(255**2 / mse)


def analyze(run):
    import cv2
    run = Path(run)
    case = json.loads((run / 'case.json').read_text(encoding='utf-8'))
    video = cv2.VideoCapture(str(run / 'video_generated.mp4'))
    if not video.isOpened():
        raise ValueError('Cannot decode generated procedural video')
    frames = {}
    count = 0
    while True:
        ok, image = video.read()
        if not ok:
            break
        if count in (8, 80, 24, 96):
            frames[count] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        count += 1
    video.release()
    if count != 97 or len(frames) != 4:
        raise ValueError(f'Incomplete procedural video: {count}/97 decoded frames')
    height, width = frames[8].shape[:2]
    scene = ProceduralScene(case['seed'], height, width)
    reference = scene.render(8)['instance']
    per_instance = {str(i): _psnr(frames[8], frames[80], reference == i) for i in (1, 2)}
    first = sticker_bit_accuracy(scene, frames[8], 8)
    return_score = sticker_bit_accuracy(scene, frames[80], 80)
    old_bits = _sticker_predictions(scene, frames[8], 8)
    new_bits = _sticker_predictions(scene, frames[80], 80)
    agreement = []
    for identity in (1, 2):
        a, b = old_bits[identity]['prediction'], new_bits[identity]['prediction']
        agreement += [x == y for x, y in zip(a, b) if x >= 0 and y >= 0]
    result = {'case_id': case['id'], 'mode': json.loads((run / 'metrics.json').read_text())['mode'],
              'frames': count, 'pair': [8, 80], 'control': [24, 96],
              'instance_psnr': per_instance,
              'sticker_first': first, 'sticker_return': return_score,
              'sticker_bit_agreement': None if not agreement else sum(agreement) / len(agreement),
              'camera_following_verified': False,
              'label': 'procedural short return; not official 60-second benchmark'}
    (run / 'sap_synthetic_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate one procedural SAP-TTT video')
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
