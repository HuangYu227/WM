"""Addressability, persistence, and short differentiable delayed episodes."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .memory import SapMemory, SapMemoryState


def retrieval_metrics(queries, keys, positive, valid=None):
    if queries.ndim != 2 or keys.ndim != 2 or queries.shape[1] != keys.shape[1]:
        raise ValueError('Queries and keys must be flat vectors in one address space')
    if positive.shape != (len(queries),) or (positive < 0).any() or (positive >= len(keys)).any():
        raise ValueError('Positive key indices are invalid')
    if valid is None:
        valid = torch.ones(len(queries), dtype=torch.bool, device=queries.device)
    if valid.shape != positive.shape:
        raise ValueError('Validity mask shape mismatch')
    coverage = float(valid.float().mean())
    if not valid.any():
        return dict(coverage=coverage, valid_queries=0, top1=None, top5=None, margin=None, mean_rank=None)
    q = F.normalize(queries[valid].float(), dim=-1)
    k = F.normalize(keys.float(), dim=-1)
    labels = positive[valid]
    similarity = q @ k.T
    correct = similarity.gather(1, labels[:, None]).squeeze(1)
    other = similarity.scatter(1, labels[:, None], float('-inf')).max(1).values
    rank = 1 + (similarity > correct[:, None]).sum(1)
    return dict(coverage=coverage, valid_queries=int(valid.sum()),
                top1=float((rank == 1).float().mean()),
                top5=float((rank <= min(5, len(keys))).float().mean()),
                margin=float((correct - other).mean()), mean_rank=float(rank.float().mean()))


def delayed_episode_loss(address, memory: SapMemory, supports: list[dict], query: dict,
                         positive: torch.Tensor, *, mode='selective_geometry',
                         delayed_weight=1., address_weight=.1, write_weight=.01):
    """A/B/C are completed; A' uses only its current noisy available features.

    ``positive`` is supervision only. It selects the historical A target after
    query encoding and never becomes an input to ``address.query``.
    """
    if len(supports) != 3:
        raise ValueError('Expected completed supports A, B, C')
    batch, count, _ = query['visual'].shape
    if batch != 1 or positive.shape != (count,):
        raise ValueError('Probe uses one independent scene per episode')
    state = SapMemoryState.new(memory, 'meta', training=True)
    old_key = old_value = None
    candidates = []
    write_losses = []
    for index, support in enumerate(supports):
        key = address.write(support['visual'], support['text'], support['text_mask'],
                            support.get('rays'), mode=mode)
        value = address.value_for(support.get('post', support['visual'])).detach()
        if index == 0:
            old_key, old_value = key, value
        candidates.append(key)
        new_weight, loss = memory.update(state.weight, key, value, create_graph=True)
        state.weight = new_weight
        write_losses.append(loss)
    current_query = address.query(query['visual'], query['text'], query['text_mask'],
                                  query.get('rays'), query['sigma'], mode=mode)
    if (positive < 0).any() or (positive >= old_key.shape[1]).any():
        raise ValueError('A prime correspondence does not point into A')
    target = old_value[:, positive].detach()
    retrieved = memory.read(state, current_query)
    delayed = F.mse_loss(retrieved, target)
    old_error = F.mse_loss(memory.read(state, old_key), old_value)
    similarities = torch.cat([current_query @ keys.transpose(1, 2) for keys in candidates], -1)[0]
    address_loss = F.cross_entropy(similarities / .07, positive)
    total = delayed_weight * delayed + address_weight * address_loss + write_weight * torch.stack(write_losses).mean()
    return total, dict(delayed_mse=float(delayed.detach()), old_key_mse=float(old_error.detach()),
                       address_ce=float(address_loss.detach()),
                       new_write_objective=float(write_losses[-1].detach()))


def forgetting_curve(memory: SapMemory, supports: list[tuple[torch.Tensor, torch.Tensor]],
                     *, replay='none', replay_capacity=64, replay_weight=.1, seed=3407,
                     return_state=False):
    """Evaluate every old write after every subsequent write; no oracle keys enter generation."""
    if replay not in {'none', 'anchor', 'random'} or replay_capacity < 1:
        raise ValueError('Invalid bounded replay setting')
    state = SapMemoryState.new(memory, 'probe', batch=supports[0][0].shape[0])
    curve = []
    bank_k = bank_v = None
    seen = 0
    rng = torch.Generator().manual_seed(seed)
    for chunk, (keys, values) in enumerate(supports):
        old = None if bank_k is None else (bank_k, bank_v)
        result = state.commit(memory, chunk, keys, values, replay=old,
                              replay_weight=replay_weight if replay != 'none' else 0.)
        if not result['committed']:
            raise FloatingPointError(result['reason'])
        with torch.no_grad():
            errors = [float(F.mse_loss(memory.read(state, k), v.float())) for k, v in supports[:chunk + 1]]
        if replay != 'none' and (replay == 'random' or chunk == 0):
            if bank_k is None:
                bank_k, bank_v = keys[:, :0].clone(), values[:, :0].clone()
            for index in range(keys.shape[1]):
                if bank_k.shape[1] < replay_capacity:
                    bank_k = torch.cat((bank_k, keys[:, index:index + 1].detach()), 1)
                    bank_v = torch.cat((bank_v, values[:, index:index + 1].detach()), 1)
                elif replay == 'random':
                    slot = int(torch.randint(seen + 1, (1,), generator=rng))
                    if slot < replay_capacity:
                        bank_k[:, slot] = keys[:, index].detach()
                        bank_v[:, slot] = values[:, index].detach()
                seen += 1
        curve.append({'chunk': chunk, 'old_key_mse': errors, 'new_write_mse': errors[-1],
                      'state_bytes': state.weight.numel() * state.weight.element_size(),
                      'replay_bytes': 0 if bank_k is None else bank_k.numel() * bank_k.element_size()
                      + bank_v.numel() * bank_v.element_size()})
    return (curve, state) if return_state else curve


def delayed_supervision_loss(query, old_keys, old_values, readout, old_token_indices,
                             positive, valid, old_instance, query_instance,
                             *, max_queries=256, temperature=.07, exact_weight=0.):
    """Use labels only to score already-computed noisy queries against committed A."""
    if query.shape[0] != 1 or old_keys.shape[0] != 1 or readout.shape != query.shape:
        raise ValueError('Delayed probe expects one scene with aligned query/readout tensors')
    if positive.shape != valid.shape or positive.numel() != query.shape[1]:
        raise ValueError('Query correspondence length mismatch')
    mapping = torch.full((old_instance.numel(),), -1, dtype=torch.long, device=query.device)
    old_token_indices = old_token_indices.to(query.device)
    mapping[old_token_indices] = torch.arange(len(old_token_indices), device=query.device)
    positive = positive.to(query.device)
    valid = valid.to(query.device).bool() & (positive >= 0)
    candidate = positive.clamp_min(0)
    valid &= mapping[candidate] >= 0
    positions = torch.where(valid)[0][:max_queries]
    if len(positions) == 0:
        raise ValueError('No A-to-A-prime correspondence survived support sampling')
    correct = mapping[positive[positions]]
    target = old_values[:, correct].detach()
    delayed = F.mse_loss(readout[:, positions].float(), target.float())
    scores = query[0, positions].float() @ old_keys[0].float().T / temperature
    old_labels = old_instance.to(query.device).flatten()[old_token_indices]
    new_labels = query_instance.to(query.device).flatten()[positions]
    positives = (new_labels[:, None] == old_labels[None]) & (new_labels[:, None] > 0)
    if not positives.any(dim=1).all():
        raise ValueError('Matched query has no positive instance in old support')
    address = (torch.logsumexp(scores, dim=1) -
               torch.logsumexp(scores.masked_fill(~positives, float('-inf')), dim=1)).mean()
    exact = F.cross_entropy(scores, correct)
    return delayed + .1 * address + exact_weight * exact, {'matched_queries': len(positions),
                                    'delayed_mse': float(delayed.detach()),
                                    'address_ce': float(address.detach()),
                                    'exact_ce': float(exact.detach())}
