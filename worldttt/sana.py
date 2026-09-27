"""Linux/CUDA SANA integration. Heavy dependencies are loaded only by commands."""
import copy
import time

import torch


def fixture_to_device(fixture, device, dtype):
    """Match SANA's model-input dtype while retaining mask and metadata types."""
    model_inputs = {'latent', 'text', 'camera', 'plucker'}
    return {key: value.to(device=device, dtype=dtype if key in model_inputs else value.dtype)
            if isinstance(value, torch.Tensor) else value
            for key, value in fixture.items()}


def load_config(path, cached=True):
    import pyrallis
    from inference_video_scripts.wm.inference_sana_wm import InferenceConfig
    with open(path, encoding='utf-8') as f:
        config = pyrallis.load(InferenceConfig, f)
    if config.model.chunk_size != 3 or config.model.chunk_split_strategy != 'first_chunk_plus_one':
        raise ValueError('v1 requires ordinary causal chunks: first 4, then 3 frames')
    if config.vae.vae_type != 'LTX2VAE_diffusers':
        raise ValueError('Use the ordinary LTX2 VAE, not streaming causal VAE')
    config.model.use_autograd_kernel = True
    if cached:
        config.model.model = 'SanaMSVideoCamCtrlStreaming_1600M_P1_D20'
        # The parent constructor otherwise overrides the cached camera class.
        config.model.camctrl_type = None
        config.model.attn_type = 'BidirectionalGDNTriton'
        config.model.ffn_type = 'CachedGLUMBConvTemp'
        config.model.pos_embed_type = 'casual_wan_rope'
    return config


def build_backbone(config, checkpoint, device, dtype):
    import diffusion.model.nets  # register architectures
    from diffusion.model.builder import build_model
    from diffusion.model.nets import sana_multi_scale_video_camctrl as camctrl_model
    from diffusion.utils.camctrl_config import model_video_camctrl_init_config
    from tools.download import find_model
    kwargs = model_video_camctrl_init_config(config, latent_size=config.model.image_size // config.vae.vae_stride[-1])
    model = build_model(config.model.model, use_fp32_attention=config.model.fp32_attention, **kwargs)
    # Match the backend that prepared the text mask in the video model.
    # Its SDPA default differs from CrossAttention's xFormers default.
    for block in model.blocks:
        if hasattr(block.cross_attn, 'set_use_xformers'):
            block.cross_attn.set_use_xformers(camctrl_model._xformers_available)
            if block.cross_attn.use_xformers != camctrl_model._xformers_available:
                raise RuntimeError('SANA video and cross-attention selected different text-mask backends')
    state = find_model(str(checkpoint))
    state = state.get('generator', state)
    state = state.get('state_dict', state)
    state = {k.removeprefix('model.'): v for k, v in state.items()}
    state.pop('pos_embed', None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) - {'pos_embed'} or unexpected:
        raise ValueError(f'Backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    model.requires_grad_(False).eval().to(device=device, dtype=dtype)
    if 'Streaming' in config.model.model:
        for i, block in enumerate(model.blocks):
            expected = 'CachedSoftmax' if (i + 1) % config.model.softmax_every_n == 0 else 'CachedChunkCausalGDN'
            if not type(block.attn).__name__.startswith(expected):
                raise ValueError(f'Wrong cached attention at layer {i+1}: {type(block.attn).__name__}')
    return model


def make_pipeline(config, checkpoint, device, training=False, refiner=None):
    from inference_video_scripts.wm.inference_sana_wm import SanaWMPipeline
    class Pipeline(SanaWMPipeline):
        stage1_seconds = 0.

        def _build_model(self, model_path):
            self.model = build_backbone(self.config, model_path, self.device, self.weight_dtype)

        def _build_vae(self):
            if training:
                self.vae = None  # cached-latent training never encodes/decodes pixels
            else:
                super()._build_vae()
                if self.config.vae.vae_type == 'LTX2VAE_diffusers':
                    # The upstream 96/64-frame tiles exceed L20 decode memory
                    # for a 961-frame benchmark. Keep the same VAE weights.
                    self.vae.tile_sample_min_num_frames = 24
                    self.vae.tile_sample_stride_num_frames = 16

        def _prepare_stage1_nvfp4(self):
            pass  # v1 uses the specified unquantized BF16 backbone

        def _sample_stage1(self, *args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            original_forward = self.model.forward_long
            latent = None
            try:
                latent = super()._sample_stage1(*args, **kwargs)
                if path := getattr(self, 'stage1_latent_path', None):
                    torch.save(latent.detach().cpu(), path)
            finally:
                # Native sampler temporarily patches forward_long to slice camera
                # tensors. Avoid retaining previous episode's closure on reuse.
                self.model.forward_long = original_forward
                torch.cuda.synchronize()
                self.stage1_seconds = time.perf_counter() - started
            if callback := getattr(self, 'stage1_complete', None):
                callback()
            return latent

        def _decode_with_sana_vae(self, sana_latent):
            if self.refiner_settings is not None or self.config.vae.vae_type != 'LTX2VAE_diffusers':
                return super()._decode_with_sana_vae(sana_latent)
            from .vae_stream import decode_ltx2_video

            # Stage-1 has finished. Keep only one small VAE tile on GPU;
            # Diffusers otherwise retains every decoded tile until the end.
            if getattr(self, 'model', None) is not None:
                self.model.to('cpu')
            torch.cuda.empty_cache()
            self.vae.enable_tiling(tile_sample_min_height=384, tile_sample_min_width=384,
                                   tile_sample_stride_height=320, tile_sample_stride_width=320)
            if self.offload_vae:
                self.vae.to(self.device)
            self.logger.info('[sana-vae] streamed decode of %d latent frames', sana_latent.shape[2])
            started = time.perf_counter()
            try:
                video = decode_ltx2_video(self.vae, sana_latent, self.device)
            finally:
                if self.offload_vae:
                    self.vae.to('cpu')
                torch.cuda.empty_cache()
            self.logger.info('[timing] streamed vae decode: %.3fs (%d frames)',
                             time.perf_counter() - started, len(video))
            return video

        @torch.no_grad()
        def generate(self, *args, **kwargs):
            with torch.inference_mode(False):
                return SanaWMPipeline.generate.__wrapped__(self, *args, **kwargs)
    pipeline = Pipeline(config, checkpoint, device=device, refiner=refiner,
                        offload_vae=True, offload_text_encoder=True, offload_refiner=True)
    pipeline.text_encoder.requires_grad_(False).eval()
    return pipeline


def make_fixture(sample, pipeline, device):
    with torch.no_grad():
        text, mask, _, _ = pipeline._encode_prompts(sample['prompt'], '')
    fixture = dict(latent=sample['latent'][None], camera=sample['camera'][None],
                   plucker=sample['plucker'][None], text=text.detach(), mask=mask,
                   episode_id=sample['key'])
    return fixture_to_device(fixture, device, pipeline.weight_dtype)
