import hashlib
from pathlib import Path

import pytest
import torch


def test_checkpoint_hash_and_generation_context(tmp_path):
    from worldttt.provenance import file_sha256
    from worldttt.grail_entry import generation_context
    file = tmp_path / 'base.pt'
    file.write_bytes(b'base checkpoint')
    assert file_sha256(file) == hashlib.sha256(b'base checkpoint').hexdigest()
    from types import SimpleNamespace
    owner = SimpleNamespace(model=SimpleNamespace(worldttt_grail_controller=SimpleNamespace(mode='online')))
    @generation_context
    def generate(self):
        return torch.is_inference_mode_enabled(), torch.is_grad_enabled()
    with torch.inference_mode():
        assert generate(owner) == (False, False)
    owner.model.worldttt_grail_controller.mode = 'off'
    assert generate(owner) == (True, False)


def test_active_mode_requires_adapter_before_mount():
    from worldttt.grail_entry import mount_grail
    with pytest.raises(ValueError, match='adapter'):
        mount_grail(None, None, mode='online')
    assert mount_grail(None, None, mode='off') is None


def test_production_config_uses_streaming_cache_not_bidir_parent():
    from worldttt.grail_entry import configure_grail_sana
    from types import SimpleNamespace
    config = SimpleNamespace(model=SimpleNamespace(chunk_size=3, chunk_split_strategy='first_chunk_plus_one',
                              softmax_every_n=4))
    configure_grail_sana(config)
    assert config.model.model == 'SanaMSVideoCamCtrlStreaming_1600M_P1_D20'
    assert config.model.camctrl_type is None
    assert config.model.ffn_type == 'CachedGLUMBConvTemp'
