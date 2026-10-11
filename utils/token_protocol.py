"""Explicit Token corruption and validity, independent of checkpoint RNG."""
import torch
import torch.nn.functional as F


def validity(text, audio, vision, audio_lengths=None, vision_lengths=None):
    vt = text[:, 1].bool() if text.ndim == 3 and text.shape[1] == 3 else torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
    result = [vt]
    for value, lengths in [(audio, audio_lengths), (vision, vision_lengths)]:
        if lengths is None:
            lengths = torch.full((value.shape[0],), value.shape[1], device=value.device)
        result.append(torch.arange(value.shape[1], device=value.device)[None] < lengths.to(value.device)[:, None])
    return result


def align_validity(valid, target_length):
    return torch.stack([F.adaptive_max_pool1d(v.float()[:, None], target_length)[:, 0].bool() for v in valid], dim=1)


def keep_mask(valid, rate, generator=None, protect_edges=False):
    eligible = valid.clone()
    if protect_edges:
        indices = torch.arange(valid.shape[-1], device=valid.device).expand_as(valid)
        first = indices.masked_fill(~valid, valid.shape[-1]).amin(dim=-1)
        last = indices.masked_fill(~valid, -1).amax(dim=-1)
        eligible &= (indices != first[..., None]) & (indices != last[..., None])
    missing = (torch.rand(valid.shape, device=valid.device, generator=generator) < rate) & eligible
    return (~missing).float(), eligible


def native_corruption(text, audio, vision, valid, rate, generator, unk_id):
    keeps, eligible = [], []
    for i, v in enumerate(valid):
        k, e = keep_mask(v, rate, generator, protect_edges=i == 0)
        keeps.append(k); eligible.append(e)
    tt = text.clone()
    if tt.ndim == 3 and tt.shape[1] == 3:
        tt[:, 0] = torch.where(keeps[0].bool(), tt[:, 0], unk_id)
    else:
        tt *= keeps[0][..., None]
    return (tt, audio * keeps[1][..., None], vision * keeps[2][..., None]), keeps, eligible


def aligned_availability(keeps, target_length, valid):
    # Availability fractions use valid native positions as the denominator;
    # padding must not inflate a partially observed pooling window.
    result=[]
    for keep,mask in zip(keeps,valid):
        numerator=F.adaptive_avg_pool1d((keep*mask.float())[:,None],target_length)[:,0]
        denominator=F.adaptive_avg_pool1d(mask.float()[:,None],target_length)[:,0]
        # Empty windows are neutral dropout placeholders, not missing events.
        result.append(torch.where(denominator>0,numerator/denominator.clamp_min(1e-8),torch.ones_like(numerator)))
    return torch.stack(result,dim=1)


def feature_corruption_masks(valid, target_length, rate, generator):
    return keep_mask(align_validity(valid, target_length), rate, generator, protect_edges=True)
