"""Single GPU / torchrun DDP outer loop; only adapter parameters are optimized."""
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from .data import EpisodeDataset
from .episode import EpisodeModel
from .memory import TTTConfig
from .runtime import WorldTTTController
from .sana import load_config, make_fixture, make_pipeline


def validation_indices(size, limit, seed, rank=0, world=1):
    """One fixed subset, sharded without DistributedSampler's duplicate padding."""
    if size < 1 or limit < 1 or not 0 <= rank < world:
        raise ValueError('Validation needs samples, a positive limit, and a valid rank')
    indices = random.Random(seed).sample(range(size), min(size, limit))
    return indices[rank::world]


def validate(module, data, pipeline, device, indices, seed, rank=0, world=1,
             show_progress=False, histories=('real',)):
    """Independent future-query MSE for each requested support-history source."""
    if not histories or any(h not in {'real', 'generated'} for h in histories) or len(set(histories)) != len(histories):
        raise ValueError('histories must contain unique real/generated entries')
    ctl = module.controller
    saved_state, saved_metrics = ctl.state, ctl.metrics
    py_state, np_state = random.getstate(), np.random.get_state()
    cuda_devices = [device.index] if device.type == 'cuda' else []
    totals = torch.zeros(len(histories) + 1, dtype=torch.float64, device=device)
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            for index in tqdm(indices, desc='Validation (rank 0 shard)', leave=False,
                              disable=not show_progress or rank != 0):
                sample_seed = seed + index
                random.seed(sample_seed)
                np.random.seed(sample_seed % (2**32))
                torch.random.default_generator.manual_seed(sample_seed)
                if cuda_devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(sample_seed)
                batch = make_fixture(data[index], pipeline, device)
                for slot, history in enumerate(histories):
                    random.seed(sample_seed)
                    np.random.seed(sample_seed % (2**32))
                    torch.random.default_generator.manual_seed(sample_seed)
                    if cuda_devices:
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(sample_seed)
                    loss = module(**batch, generated=history == 'generated',
                                  seed=sample_seed, meta_grad=False)
                    totals[slot] += loss.detach().double()
                totals[-1] += 1
        if world > 1:
            dist.all_reduce(totals)
        if totals[-1] < 1 or not torch.isfinite(totals).all():
            raise FloatingPointError('Empty or nonfinite validation; checkpoint not selected')
        result = {('loss' if h == 'real' else 'generated_loss'): float(totals[i] / totals[-1])
                  for i, h in enumerate(histories)}
        return dict(result, samples=int(totals[-1]))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        ctl.state, ctl.metrics = saved_state, saved_metrics


def save_checkpoints(controller, output, step, settings, val_loss, best_loss):
    """Atomically replace only last/best adapter files; retain inference compatibility."""
    if val_loss is not None and not math.isfinite(val_loss):
        raise FloatingPointError('Refusing checkpoint selection with nonfinite validation')
    improved = val_loss is not None and val_loss < best_loss
    best_loss = val_loss if improved else best_loss
    extra = dict(step=step, settings=settings, val_loss=val_loss,
                 best_val_loss=best_loss if math.isfinite(best_loss) else None,
                 selection_metric=settings.get('selection_metric', 'real_prefix_future_query_flow_mse'),
                 checkpoint_kind='adapter_only')
    for name in ('last', 'best') if improved else ('last',):
        temporary = output / f'{name}.tmp.pt'
        try:
            controller.save_checkpoint(temporary, extra=extra)
            temporary.replace(output / f'{name}.pt')
        finally:
            temporary.unlink(missing_ok=True)
    return best_loss


def train(settings, output, adapter=None):
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank, local_rank = int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0))
    if not torch.cuda.is_available():
        raise RuntimeError('SANA training requires the Linux CUDA environment; run CPU unit tests separately')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    accumulation = settings.get('gradient_accumulation', 1)
    max_steps = settings.get('max_steps', 1000)
    save_every = settings.get('save_every', 100)
    val_every = settings.get('val_every', save_every)
    val_limit = settings.get('val_max_samples', 16)
    val_seed = settings.get('val_seed', 12345)
    histories = tuple(settings.get('validation_histories', ['real']))
    selection_key = settings.get('selection_key', 'loss')
    valid_keys = {'loss' if h == 'real' else 'generated_loss' for h in histories}
    if selection_key not in valid_keys:
        raise ValueError('selection_key must be present in validation_histories')
    if min(accumulation, max_steps, save_every, val_every, val_limit) < 1:
        raise ValueError('Training, saving and validation intervals/counts must be positive')
    output = Path(output)
    if any((output / name).exists() for name in ('last.pt', 'best.pt')):
        raise FileExistsError('Use a fresh output directory; --adapter is a warm start, not resume')
    if world > 1:
        dist.init_process_group('nccl')
    seed = settings.get('seed', 3407)
    torch.manual_seed(seed)
    random.seed(seed + rank)
    config = load_config(settings['sana_config'])
    pipe = make_pipeline(config, settings['base_checkpoint'], device, training=True)
    ctl = WorldTTTController(pipe.model, TTTConfig(**settings['ttt']))
    ctl.base_checkpoint = settings['base_checkpoint']
    if not ctl.memories:
        raise ValueError('off is an inference baseline; there are no parameters to train')
    if adapter:
        ctl.load_checkpoint(adapter)
    data = EpisodeDataset(settings['data'], settings['manifest'], 'train')
    validation_data = EpisodeDataset(settings['data'], settings['manifest'], 'val')
    val_indices = validation_indices(len(validation_data), val_limit, val_seed, rank, world)
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, seed=seed)
    loader = DataLoader(data, batch_size=None, sampler=sampler, num_workers=settings.get('workers', 0))
    module = EpisodeModel(pipe.model, ctl, settings.get('steps', 50), config.scheduler.inference_flow_shift)
    wrapped = DistributedDataParallel(module, device_ids=[local_rank], find_unused_parameters=False) if world > 1 else module
    parameters = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings.get('outer_lr', 1e-4), weight_decay=0.01)
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / 'resolved.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    if not len(loader):
        raise ValueError('Nonempty training data required')
    warmup = settings.get('real_prefix_steps', max_steps // 2)
    step = micro = epoch = 0
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best_loss, val_loss = float('inf'), None
    step_loss = torch.zeros((), device=device)
    progress = tqdm(total=max_steps, desc='WorldTTT train', unit='step', disable=rank != 0,
                    dynamic_ncols=True, mininterval=1.0)
    try:
        while step < max_steps:
            sampler.set_epoch(epoch)
            for sample in loader:
                # Phase 2 independently selects real/generated prefixes at p=.5.
                generated = step >= warmup and random.random() < .5
                batch = make_fixture(sample, pipe, device)
                synchronize = (micro + 1) % accumulation == 0
                sync_context = nullcontext() if world == 1 or synchronize else wrapped.no_sync()
                with sync_context:
                    loss = wrapped(**batch, generated=generated, seed=seed + rank * 1000003 + micro)
                    finite = torch.isfinite(loss).to(torch.int32)
                    if world > 1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite:
                        raise FloatingPointError('Nonfinite query loss; no optimizer step committed')
                    (loss / accumulation).backward()
                step_loss += loss.detach() / accumulation
                micro += 1
                if synchronize:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if world > 1:
                        dist.all_reduce(step_loss)
                    mean_loss = float(step_loss / world)
                    step_loss.zero_()
                    do_val = step % val_every == 0 or step == max_steps
                    current_val = None
                    if do_val:
                        # Bypass the DDP wrapper: ranks may have unequal val counts.
                        result = validate(module, validation_data, pipe, device, val_indices,
                                          val_seed, rank, world, show_progress=True,
                                          histories=histories)
                        current_val = val_loss = result[selection_key]
                        if rank == 0:
                            with (output / 'val.jsonl').open('a', encoding='utf-8') as f:
                                f.write(json.dumps(dict(step=step, **result, seed=val_seed,
                                    selection_key=selection_key), allow_nan=False) + '\n')
                    if rank == 0:
                        if do_val or step % save_every == 0 or step == max_steps:
                            best_loss = save_checkpoints(ctl, output, step, settings, current_val, best_loss)
                        row = dict(step=step, loss=mean_loss, gradient_norm=float(norm),
                                   generated_prefix=generated, elapsed_seconds=time.perf_counter() - started,
                                   peak_bytes=torch.cuda.max_memory_allocated(), updates=ctl.metrics,
                                   learning_rate=optimizer.param_groups[0]['lr'], val_loss=current_val,
                                   best_val_loss=best_loss if math.isfinite(best_loss) else None)
                        with (output / 'train.jsonl').open('a', encoding='utf-8') as f:
                            f.write(json.dumps(row, allow_nan=False) + '\n')
                        progress.set_postfix(loss=f'{mean_loss:.4g}',
                            val='-' if val_loss is None else f'{val_loss:.4g}',
                            best='-' if not math.isfinite(best_loss) else f'{best_loss:.4g}',
                            lr=f'{row["learning_rate"]:.2g}', mem=f'{row["peak_bytes"] / 2**30:.1f}GiB',
                            refresh=False)
                        progress.update(1)
                    if step >= max_steps:
                        break
            epoch += 1
    finally:
        progress.close()
        if world > 1:
            dist.destroy_process_group()
