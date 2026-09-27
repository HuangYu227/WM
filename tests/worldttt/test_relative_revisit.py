import numpy as np
import torch


def test_control_has_similar_gap_and_distinct_pose():
    from worldttt.relative_revisit import choose_control
    camera = np.broadcast_to(np.eye(4), (961, 4, 4)).copy()
    camera[:, 0, 3] = np.arange(961) / 100
    camera[320, 0, 3] = camera[20, 0, 3]
    pair = choose_control(camera, 20, 320)
    assert pair is not None
    assert pair[0] > 0
    assert abs((pair[1] - pair[0]) - 300) <= 16
    assert np.linalg.norm(camera[pair[0], :3, 3] - camera[pair[1], :3, 3]) > .5


def test_stage1_latent_pair_uses_matching_8_frame_indices():
    from worldttt.relative_revisit import latent_metrics
    latent = torch.zeros(1, 2, 121, 2, 2)
    latent[:, :, 40] = 1
    score = latent_metrics(latent, 16, 320)
    assert score['mse'] == 1.


def test_revisit_without_nonreturn_control_is_still_scored(monkeypatch, tmp_path):
    import json
    import sys
    from types import SimpleNamespace
    from worldttt import relative_revisit

    camera = np.broadcast_to(np.eye(4), (961, 4, 4)).copy()
    np.save(tmp_path / 'camera.npy', camera)
    torch.save(torch.zeros(1, 2, 121, 2, 2), tmp_path / 'latent.pt')
    (tmp_path / 'case.json').write_text(json.dumps({
        'id': 'scene', 'camera': str(tmp_path / 'camera.npy'),
        'revisit_pairs': [{'frame_a': 16, 'frame_b': 320, 'control': None}]}))

    class Capture:
        cursor = 0
        def isOpened(self):
            return True
        def get(self, prop):
            return 16 if prop == 1 else 961
        def read(self):
            if self.cursor >= 961:
                return False, None
            self.cursor += 1
            return True, np.zeros((2, 2, 3), dtype=np.uint8)
        def release(self):
            pass

    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(
        VideoCapture=lambda path: Capture(), CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2,
        COLOR_BGR2RGB=3, cvtColor=lambda image, code: image))
    monkeypatch.setattr(relative_revisit, 'video_frame', lambda video, index: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(relative_revisit, 'image_metrics', lambda a, b: {'psnr': 30., 'ssim': .9, 'lpips': .1})
    result = relative_revisit.analyze(tmp_path)
    row = result['rows'][0]
    assert result['video_validation']['complete'] is True
    assert result['video_validation']['decoded_frames'] == 961
    assert row['revisit_valid'] is True
    assert row['valid'] is False
    assert row['revisit_metrics']['lpips'] == .1
    assert row['control_metrics'] is None


def test_nonreturn_segment_records_local_motion_and_sharpness(monkeypatch):
    import sys
    from types import SimpleNamespace
    from worldttt import relative_revisit

    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(
        COLOR_BGR2GRAY=1, CV_32F=2,
        cvtColor=lambda frame, code: frame[..., 0],
        Laplacian=lambda gray, dtype: gray.astype(np.float32)))
    monkeypatch.setattr(relative_revisit, 'video_frame', lambda video, index: np.full(
        (4, 4, 3), index % 255, dtype=np.uint8))
    metrics = relative_revisit.nonreturn_segment(object(), 32, 336, 16.)
    assert metrics['frame_change_mean'] > 0
    assert metrics['sampled_pairs'] == 8
