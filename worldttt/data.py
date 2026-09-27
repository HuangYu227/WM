"""Strict wrapper around native SANA zip latents; never uses its retry fallback."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def validate_manifest(rows):
    scenes, keys = {}, set()
    for row in rows:
        key, scene, split = row['key'], row['scene_id'], row['split']
        if not key or not scene or split not in {'train', 'val', 'test'}:
            raise ValueError('Every clip needs key, original scene_id, and train/val/test split')
        if key in keys:
            raise ValueError(f'Duplicate clip key: {key}')
        if scene in scenes and scenes[scene] != split:
            raise ValueError(f'Original scene crosses splits: {scene}')
        keys.add(key)
        scenes[scene] = split
    if not rows:
        raise ValueError('Empty scene manifest')
    return rows


def validate_episode(sample):
    z, camera, plucker = (sample[k] for k in ('latent', 'camera', 'plucker'))
    if z.ndim != 4 or z.shape[1] < 10:
        raise ValueError('Episode requires >=10 latent frames (4 + 3 + 3)')
    if camera.shape != (z.shape[1], 20):
        raise ValueError('Invalid camera shape')
    if plucker.shape != (48, z.shape[1], *z.shape[2:]):
        raise ValueError('Invalid Plucker shape')
    for name in ('latent', 'camera', 'plucker'):
        if not torch.isfinite(sample[name]).all():
            raise ValueError(f'Nonfinite {name}: {sample["key"]}')
    if not (camera[:, 16:18] > 0).all():
        raise ValueError('Camera focal lengths must be positive')
    poses = camera[:, :16].reshape(-1, 4, 4)
    if not torch.allclose(poses[:, 3], poses.new_tensor([0, 0, 0, 1]).expand_as(poses[:, 3]), atol=1e-4):
        raise ValueError('Invalid homogeneous camera pose')
    if not sample['prompt'].strip():
        raise ValueError('Missing caption')


def require_requested_frames(latent, frames, key):
    if latent.ndim != 4 or latent.shape[1] < frames:
        raise ValueError(f'{key}: requested {frames} contiguous latents, found {latent.shape[1]}')


def require_vae_stride(native):
    if native.vae_time_stride != 8:
        raise ValueError('GRAIL chunk Plucker channels require VAE temporal stride 8')


def native_frame_limit(frames):
    # Native num_frames caps both streams. This raw-camera horizon is always
    # longer than the requested latent horizon, so neither is truncated short.
    return 1 + 8 * (frames - 1)


class EpisodeDataset(Dataset):
    def __init__(self, native_config, manifest, split, frames=10):
        from diffusion.data.datasets.video.sana_wm_zip_latent_data import SanaWMZipLatentDataset
        if frames < 10:
            raise ValueError('Episode must contain at least 10 latent frames')
        options = dict(native_config, num_frames=native_frame_limit(frames), return_chunk_plucker=True,
                       data_repeat=1, shuffle_dataset=False, sort_dataset=True)
        self.native = SanaWMZipLatentDataset(**options)
        require_vae_stride(self.native)
        self.frames = frames
        rows = validate_manifest([json.loads(s) for s in Path(manifest).read_text(encoding='utf-8').splitlines() if s.strip()])
        lookup = {f'{x["dataset_name"]}/{x["key"]}': i for i, x in enumerate(self.native.dataset)}
        if len(lookup) != len(self.native.dataset):
            raise ValueError('Duplicate native dataset/key across ZIPs; assign unique dataset identities')
        self.rows = [r for r in rows if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No episodes in split {split}')
        self.indices = []
        for row in self.rows:
            if row['key'] not in lookup:
                raise ValueError(f'Manifest clip absent from native dataset: {row["key"]}')
            i = lookup[row['key']]
            item = self.native.dataset[i]
            sidecar = self.native.load_camera_sidecar(item['camera_npz'])
            if sidecar is None or item['key'] not in sidecar['ids'].tolist():
                raise ValueError(f'Missing measured camera metadata: {row["key"]}')
            cam_idx = sidecar['ids'].tolist().index(item['key'])
            start, count = map(int, sidecar['ranges'][cam_idx])
            needed_pixels = native_frame_limit(frames)
            if count < needed_pixels or start < 0 or start + count > len(sidecar['pose']):
                raise ValueError(f'Need >={needed_pixels} valid camera frames: {row["key"]}')
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        z, prompt, _, _, _, _, camera, plucker = self.native.getdata(self.indices[index])
        require_requested_frames(z, self.frames, self.rows[index]['key'])
        sample = dict(latent=z[:, :self.frames], camera=camera[:self.frames], plucker=plucker[:, :self.frames],
                      prompt=prompt, key=self.rows[index]['key'])
        validate_episode(sample)
        return sample
