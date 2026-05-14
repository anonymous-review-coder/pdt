import hashlib

import torch


def _batch_seed(base_seed: int, pred_len: int, corruption_type: str, batch_index: int) -> int:
    key = f"{base_seed}:{pred_len}:{corruption_type}:{batch_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)


def maybe_apply_test_corruption(batch_x, args, batch_index: int):
    corruption_type = getattr(args, "test_corruption_type", "none")
    if corruption_type in {"", "none", None}:
        return batch_x

    rate = float(getattr(args, "test_corruption_rate", 0.05))
    amp = float(getattr(args, "test_corruption_amp", 3.0))
    seed = int(getattr(args, "test_corruption_seed", 2023))
    segment_len = int(getattr(args, "test_corruption_segment_len", 4))

    if not 0 <= rate <= 1:
        raise ValueError(f"test_corruption_rate must be in [0, 1], got {rate}.")
    if amp < 0:
        raise ValueError(f"test_corruption_amp must be non-negative, got {amp}.")
    if segment_len <= 0:
        raise ValueError(f"test_corruption_segment_len must be positive, got {segment_len}.")

    generator = torch.Generator(device=batch_x.device)
    generator.manual_seed(_batch_seed(seed, int(args.pred_len), str(corruption_type), batch_index))

    if corruption_type == "spike":
        mask = torch.rand(batch_x.shape, device=batch_x.device, generator=generator) < rate
        signs = torch.where(
            torch.rand(batch_x.shape, device=batch_x.device, generator=generator) < 0.5,
            -1.0,
            1.0,
        )
        return batch_x + mask.to(batch_x.dtype) * signs.to(batch_x.dtype) * amp

    if corruption_type == "segment":
        corrupted = batch_x.clone()
        batch_size, seq_len, channels = corrupted.shape
        active_len = min(segment_len, seq_len)
        target_positions = int(round(rate * batch_size * seq_len * channels))
        num_segments = max(1, (target_positions + active_len - 1) // active_len)

        batch_indices = torch.randint(batch_size, (num_segments,), device=batch_x.device, generator=generator)
        channel_indices = torch.randint(channels, (num_segments,), device=batch_x.device, generator=generator)
        max_start = max(seq_len - active_len + 1, 1)
        start_indices = torch.randint(max_start, (num_segments,), device=batch_x.device, generator=generator)
        signs = torch.where(
            torch.rand((num_segments,), device=batch_x.device, generator=generator) < 0.5,
            -1.0,
            1.0,
        ).to(corrupted.dtype)

        for b_idx, start_idx, c_idx, sign in zip(batch_indices, start_indices, channel_indices, signs):
            start = int(start_idx.item())
            stop = min(start + active_len, seq_len)
            corrupted[int(b_idx.item()), start:stop, int(c_idx.item())] += sign * amp
        return corrupted

    raise ValueError(f"Unsupported test_corruption_type: {corruption_type}.")
