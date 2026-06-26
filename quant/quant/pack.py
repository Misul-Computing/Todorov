"""Bit-packed storage + dequant (workstream 4).

Replaces the "overwrite weights in place" trick with real packed storage:
b-bit indices packed into a byte array + FP16 scales per group.  The dequant
path reconstructs weights on load by looking up codebook values, multiplying by
scales, and applying the inverse Hadamard rotation.

API:
  pack(W, bits, group_size, method) -> PackedTensor
  unpack(packed) -> Wq  (bit-exact match to quant_dequant output)
  save_model(model, path, bits, group_size, method, assignment=None)
  load_model(path) -> model with quantized weights restored

Bit-packing: b-bit indices packed into a uint8 byte array.  For b=3, 8 indices
per 3 bytes; for b=2, 4 per byte; for b=4, 2 per byte; for b=8, 1 per byte.
General case: pack sequentially into a bit stream.
"""
import torch
import torch.nn as nn

from .rotate import fwht, next_pow2, signs_for
from .codebook import lloyd_max_gaussian


def _adjust_group(d_in, G):
    while d_in % G and G > 1:
        G //= 2
    return G


def pack_indices(indices_flat, bits):
    """Pack a 1-D LongTensor of b-bit indices into a uint8 byte array.

    Indices are packed LSB-first into a bit stream.  Returns (bytes, n_indices).
    """
    n = indices_flat.numel()
    arr = indices_flat.to(torch.uint8).cpu().numpy()
    total_bits = n * bits
    n_bytes = (total_bits + 7) // 8
    out = bytearray(n_bytes)
    for i in range(n):
        val = int(arr[i])
        bit_pos = i * bits
        for b in range(bits):
            if val & (1 << b):
                byte_idx = (bit_pos + b) // 8
                bit_idx = (bit_pos + b) % 8
                out[byte_idx] |= (1 << bit_idx)
    return bytes(out), n


def unpack_indices(packed_bytes, n_indices, bits):
    """Unpack b-bit indices from a uint8 byte array.  Returns 1-D LongTensor."""
    out = torch.zeros(n_indices, dtype=torch.long)
    for i in range(n_indices):
        val = 0
        bit_pos = i * bits
        for b in range(bits):
            byte_idx = (bit_pos + b) // 8
            bit_idx = (bit_pos + b) % 8
            if packed_bytes[byte_idx] & (1 << bit_idx):
                val |= (1 << b)
        out[i] = val
    return out


class PackedTensor:
    """Packed quantized tensor: b-bit indices + FP16 scales + metadata."""
    def __init__(self, indices_bytes, n_indices, scales, shape, bits,
                 method, group_size, sign_seed=None, centroids=None,
                 dtype=torch.float16):
        self.indices_bytes = indices_bytes
        self.n_indices = n_indices
        self.scales = scales          # FP16 tensor of per-row or per-group scales
        self.shape = shape            # (d_out, d_in)
        self.bits = bits
        self.method = method
        self.group_size = group_size
        self.sign_seed = sign_seed    # for fullrot methods
        self.centroids = centroids    # the codebook (small, stored once per model)
        self.dtype = dtype            # original weight dtype for bit-exact restore

    def storage_bytes(self):
        return len(self.indices_bytes) + self.scales.numel() * 2

    def bpw(self):
        return (self.bits * self.shape[0] * self.shape[1] +
                self.scales.numel() * 16) / (self.shape[0] * self.shape[1])


def pack(W, bits, group_size=128, method="fullrot_whlm", centroids=None,
         sign_seed=42, fullrot_chunk=8192):
    """Quantize W and return a PackedTensor with the indices + scales.

    The indices are in the rotated domain (for fullrot methods) or the original
    domain (for rtn).  unpack() reverses the process.
    """
    d_out, d_in = W.shape
    Wf = W.float()
    device = W.device

    if method == "rtn":
        G = _adjust_group(d_in, group_size)
        blocks = Wf.reshape(d_out, d_in // G, G)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / (2 ** (bits - 1) - 1)
        q_vals = (blocks / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        indices = (q_vals + 2 ** (bits - 1)).to(torch.long)
        scales = scale.squeeze(-1)  # float32 for bit-exactness
        idx_bytes, n = pack_indices(indices.reshape(-1), bits)
        return PackedTensor(idx_bytes, n, scales, (d_out, d_in), bits,
                            method, G, dtype=W.dtype)

    elif method == "fullrot_whlm":
        if centroids is None:
            centroids = lloyd_max_gaussian(bits, device)
        npad = next_pow2(d_in)
        s = signs_for(npad, device)
        bounds = (centroids[1:] + centroids[:-1]) / 2
        all_indices = []
        all_scales = []
        for i in range(0, d_out, fullrot_chunk):
            Wc = Wf[i:i + fullrot_chunk]
            Wp = torch.zeros(Wc.shape[0], npad, device=device)
            Wp[:, :d_in] = Wc
            Wr = fwht(Wp * s)
            sc = Wr.std(1, keepdim=True).clamp(min=1e-8)
            idx = torch.bucketize(Wr / sc, bounds)
            all_indices.append(idx.cpu())
            all_scales.append(sc.squeeze(1).cpu())
        indices = torch.cat(all_indices, dim=0)  # [d_out, npad]
        scales = torch.cat(all_scales, dim=0)  # [d_out] float32 for bit-exactness
        idx_bytes, n = pack_indices(indices.reshape(-1), bits)
        return PackedTensor(idx_bytes, n, scales, (d_out, d_in), bits,
                            method, group_size, sign_seed=sign_seed,
                            centroids=centroids.cpu(), dtype=W.dtype)

    else:
        raise ValueError(f"pack not implemented for method {method!r}")


def unpack(packed, device="cpu", fullrot_chunk=8192):
    """Reconstruct the dequantized weight from a PackedTensor."""
    d_out, d_in = packed.shape
    bits = packed.bits
    method = packed.method
    dtype = packed.dtype

    if method == "rtn":
        G = packed.group_size
        indices = unpack_indices(packed.indices_bytes, packed.n_indices, bits)
        indices = indices.reshape(d_out, d_in // G, G).to(device)
        q_vals = indices - 2 ** (bits - 1)
        scales = packed.scales.to(device).unsqueeze(-1).float()
        deq = q_vals.float() * scales
        return deq.reshape(d_out, d_in).to(dtype)

    elif method == "fullrot_whlm":
        npad = next_pow2(d_in)
        s = signs_for(npad, device)
        centroids = packed.centroids.to(device)
        indices = unpack_indices(packed.indices_bytes, packed.n_indices, bits)
        indices = indices.reshape(d_out, npad).to(device)
        scales = packed.scales.to(device).unsqueeze(1).float()  # upcast for exact match
        q = centroids[indices] * scales
        out = torch.empty(d_out, d_in, dtype=dtype, device=device)
        for i in range(0, d_out, fullrot_chunk):
            qc = q[i:i + fullrot_chunk]
            deq = (fwht(qc) * s)[:, :d_in]
            out[i:i + fullrot_chunk] = deq.to(dtype)
        return out

    else:
        raise ValueError(f"unpack not implemented for method {method!r}")


def save_model(model, path, bits, group_size=128, method="fullrot_whlm",
               assignment=None):
    """Pack every nn.Linear and save to a single file.

    assignment: optional {name: bits} dict for mixed-precision (picker).
    Tensors not in assignment or assigned 16-bit are saved as FP16.
    """
    device = next(model.parameters()).device
    packed_layers = {}
    fp16_layers = {}
    centroids_cache = {}

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        b = bits
        if assignment is not None:
            b = assignment.get(name, 16)
        if b >= 16:
            fp16_layers[name] = mod.weight.data.detach().cpu().clone()
            continue
        if b not in centroids_cache:
            centroids_cache[b] = lloyd_max_gaussian(b, device)
        pt = pack(mod.weight.data, b, group_size, method,
                  centroids=centroids_cache[b])
        packed_layers[name] = pt

    torch.save({
        "packed": packed_layers,
        "fp16": fp16_layers,
        "model_class": model.__class__.__name__,
        "method": method,
        "group_size": group_size,
    }, path)


def load_model(path, model, device="cuda"):
    """Load packed weights into an existing model (same architecture)."""
    data = torch.load(path, map_location=device, weights_only=False)
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name in data["packed"]:
            pt = data["packed"][name]
            Wq = unpack(pt, device=device)
            with torch.no_grad():
                mod.weight.data.copy_(Wq)
        elif name in data["fp16"]:
            with torch.no_grad():
                mod.weight.data.copy_(data["fp16"][name].to(device))
    return model
