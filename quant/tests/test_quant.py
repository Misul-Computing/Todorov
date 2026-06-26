"""Unit tests for the quant/ package.  Pure CPU, no model download.

Run: pytest tests/test_quant.py -q
"""
import torch

from quant import rotate, codebook, quantize


def test_fwht_involututive():
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    assert torch.allclose(rotate.fwht(rotate.fwht(x)), x, atol=1e-4)


def test_fwht_orthonormal():
    H = rotate.fwht(torch.eye(8))
    assert torch.allclose(H @ H.T, torch.eye(8), atol=1e-5)
    assert torch.allclose(H, H.T, atol=1e-5)


def test_hadamard_matrix_orthonormal():
    H = rotate.hadamard_matrix(64, "cpu")
    assert torch.allclose(H @ H.T, torch.eye(64), atol=1e-5)


def test_next_pow2():
    assert rotate.next_pow2(1) == 1
    assert rotate.next_pow2(1536) == 2048
    assert rotate.next_pow2(2048) == 2048


def test_signs_deterministic():
    a = rotate.signs_for(2048, "cpu")
    b = rotate.signs_for(2048, "cpu")
    assert torch.equal(a, b)
    assert set(a.unique().tolist()) == {-1.0, 1.0}


def test_lloyd_max_sorted_and_finite():
    c = codebook.lloyd_max_gaussian(3, "cpu", n_samples=20000, iters=10)
    assert c.shape == (8,)
    assert torch.all(c.diff() > 0)          # sorted ascending
    assert torch.all(torch.isfinite(c))


def test_lloyd_max_2bit_near_optimal():
    # For N(0,1), 4-level Lloyd-Max centroids are approximately +-0.4599, +-1.5102
    c = codebook.lloyd_max_gaussian(2, "cpu", n_samples=200000, iters=60)
    c = c.tolist()
    assert abs(abs(c[0]) - 1.5102) < 0.05
    assert abs(abs(c[1]) - 0.4599) < 0.05


def test_quant_codebook_std_preserves_scale():
    blocks = torch.randn(3, 128) * 5.0
    c = codebook.lloyd_max_gaussian(4, "cpu", n_samples=50000, iters=20)
    q = codebook.quant_codebook_std(blocks, c)
    assert q.shape == blocks.shape
    # per-block std of reconstruction should be in the right ballpark
    assert (q.std(-1) - blocks.std(-1)).abs().mean() < 1.0


def test_quant_uniform_sym_round_trip_zero_error():
    blocks = torch.linspace(-7, 7, 15).unsqueeze(0).float()
    q = codebook.quant_uniform_sym(blocks, 4)
    assert torch.allclose(q, blocks, atol=1e-4)   # integers in [-7,7] quantize exactly


def test_quant_dequant_shape_dtype_methods():
    W = (torch.randn(64, 128) * 0.02).to(torch.bfloat16)
    c = codebook.lloyd_max_gaussian(3, "cpu", n_samples=20000, iters=10)
    nf4 = codebook.NF4.to("cpu")
    for method in ("rtn", "whlm", "whrtn", "fullrot_whlm", "fullrot_whrtn", "nf4"):
        Wq, err, bpw, G = quantize.quant_dequant(
            W, 3 if method != "nf4" else 4, group_size=128, method=method,
            centroids=c, nf4=nf4)
        assert Wq.shape == W.shape, method
        assert Wq.dtype == W.dtype, method
        assert 0.0 <= err <= 2.0, method
        assert bpw > 0, method


def test_quant_dequant_fullrot_recon_better_than_rtn_at_low_bits():
    # full-dim rotation should Gaussianize and beat plain RTN at 2-bit on outlier data
    torch.manual_seed(1)
    W = torch.randn(32, 256) * 0.02
    W[:, ::32] += torch.randn(32, 8) * 0.3      # inject outlier channels
    c = codebook.lloyd_max_gaussian(2, "cpu", n_samples=100000, iters=40)
    _, e_rtn, _, _ = quantize.quant_dequant(W, 2, 128, "rtn")
    _, e_full, _, _ = quantize.quant_dequant(W, 2, 128, "fullrot_whlm", centroids=c)
    assert e_full < e_rtn


def test_snapshot_restore_round_trip():
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 16))
    snap = quantize.snapshot(model)
    orig = {n: m.weight.data.clone() for n, m in model.named_modules()
            if isinstance(m, nn.Linear)}
    # mutate
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.add_(1.0)
    quantize.restore(model, snap)
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            assert torch.equal(m.weight.data, orig[n])


def test_skip_names_protects_tensor():
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 16))
    snap = quantize.snapshot(model)
    before = dict(model.named_modules())["1"].weight.data.clone()
    quantize.quantize_model_inplace(model, 2, 128, "rtn", skip_names=["1"],
                                    verbose=False)
    after = dict(model.named_modules())["1"].weight.data
    assert torch.equal(after, before)            # protected
    quantize.restore(model, snap)


def test_picker_sensitivity_table_keys_and_monotonicity():
    """Sensitivity should be 0 at 16-bit and decrease as bits increase."""
    import torch.nn as nn
    from quant import picker
    model = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 32))
    # fake activations: uniform energy per channel
    acts = {}
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            acts[n] = torch.ones(m.weight.shape[1])
    table, numel = picker.sensitivity_table(model, acts, bits_grid=(2, 3, 4, 16),
                                            group_size=32, method="rtn")
    assert set(table.keys()) == set(acts.keys())
    for n in table:
        assert table[n][16] == 0.0
        # sensitivity at 2-bit should be >= sensitivity at 4-bit
        assert table[n][2] >= table[n][4] - 1e-9


def test_picker_assign_budget_respects_target():
    """Assignment avg BPW should be <= target (with small overshoot tolerance)."""
    from quant import picker
    table = {"a": {2: 100.0, 3: 50.0, 4: 10.0, 8: 1.0, 16: 0.0},
             "b": {2: 200.0, 3: 80.0, 4: 20.0, 8: 2.0, 16: 0.0}}
    numel = {"a": 1000, "b": 1000}
    assignment = picker.assign_budget(table, numel, target_avg_bpw=3.0,
                                      bits_grid=(2, 3, 4, 8, 16), group_size=128)
    avg, hist = picker.assignment_stats(assignment, numel, 128)
    assert avg <= 3.0 * 1.05  # within 5% of target
    # the more sensitive tensor 'b' should get >= bits than 'a'
    assert assignment["b"] >= assignment["a"]


def test_picker_assign_budget_greedy_picks_best_ratio():
    """At a tight budget, the greedy should upgrade the tensor with the best
    sensitivity-reduction-per-bit first."""
    from quant import picker
    # 'b' has much higher sensitivity reduction from 2->3 than 'a'
    table = {"a": {2: 100.0, 3: 99.0, 4: 98.0, 8: 97.0, 16: 0.0},
             "b": {2: 100.0, 3: 10.0, 4: 5.0, 8: 1.0, 16: 0.0}}
    numel = {"a": 1000, "b": 1000}
    # budget for ~2.5 bpw -> enough to upgrade one tensor from 2->3
    assignment = picker.assign_budget(table, numel, target_avg_bpw=2.6,
                                      bits_grid=(2, 3, 4, 8, 16), group_size=128)
    # 'b' should be upgraded before 'a' (90 units reduction vs 1)
    assert assignment["b"] >= 3


def test_picker_floor_prevents_2bit_when_budget_allows():
    """With floor=3 and a budget that affords 3-bit for all, no tensor should
    be at 2-bit.  3-bit at group=128 costs 3.125 bpw, so the target must be
    >= 3.125 to afford the floor for all tensors."""
    from quant import picker
    table = {"a": {2: 100.0, 3: 50.0, 4: 10.0, 8: 1.0, 16: 0.0},
             "b": {2: 200.0, 3: 80.0, 4: 20.0, 8: 2.0, 16: 0.0}}
    numel = {"a": 1000, "b": 1000}
    # 3.2 bpw affords 3-bit for all (3.125 * 2000 = 6250 < 3.2 * 2000 = 6400)
    assignment = picker.assign_budget(table, numel, target_avg_bpw=3.2,
                                      bits_grid=(2, 3, 4, 8, 16), group_size=128,
                                      floor=3)
    assert all(b >= 3 for b in assignment.values())


def test_random_assignment_respects_budget():
    """Random assignment avg BPW should be <= target (with tolerance)."""
    from quant import picker
    table = {"a": {2: 100.0, 3: 50.0, 4: 10.0, 8: 1.0, 16: 0.0},
             "b": {2: 200.0, 3: 80.0, 4: 20.0, 8: 2.0, 16: 0.0},
             "c": {2: 50.0, 3: 30.0, 4: 5.0, 8: 0.5, 16: 0.0}}
    numel = {"a": 1000, "b": 1000, "c": 1000}
    assignment = picker.random_assignment(table, numel, target_avg_bpw=3.0,
                                          bits_grid=(2, 3, 4, 8, 16),
                                          group_size=128, seed=42)
    avg, _ = picker.assignment_stats(assignment, numel, 128)
    assert avg <= 3.0 * 1.05


def test_reasoning_module_api():
    """The reasoning module exposes the expected eval functions."""
    from quant import reasoning
    for fn in ('_loglikelihood', '_score_choices', 'eval_hellaswag',
               'eval_arc_challenge', 'eval_piqa', 'eval_all'):
        assert hasattr(reasoning, fn), f"missing {fn}"


def test_pack_unpack_indices_round_trip():
    """Bit-packed indices must round-trip exactly for all bit widths."""
    from quant import pack as P
    for bits in (2, 3, 4, 8):
        idx = torch.randint(0, 2 ** bits, (1000,))
        packed, n = P.pack_indices(idx, bits)
        recovered = P.unpack_indices(packed, n, bits)
        assert torch.equal(idx, recovered), f"round-trip failed for bits={bits}"


def test_pack_unpack_rtn_bit_exact():
    """pack(W) then unpack must match quant_dequant(W) exactly for rtn."""
    from quant import pack as P
    from quant import quantize as Q
    W = (torch.randn(32, 128) * 0.02).to(torch.bfloat16)
    Wq_ref, _, _, _ = Q.quant_dequant(W, 3, 128, "rtn")
    pt = P.pack(W, 3, 128, "rtn")
    Wq = P.unpack(pt)
    assert torch.equal(Wq.to(torch.float32), Wq_ref.to(torch.float32)), \
        "rtn pack/unpack not bit-exact"


def test_pack_unpack_fullrot_whlm_bit_exact():
    """pack(W) then unpack must match quant_dequant(W) exactly for fullrot_whlm."""
    from quant import pack as P
    from quant import quantize as Q
    from quant.codebook import lloyd_max_gaussian
    W = (torch.randn(16, 128) * 0.02).to(torch.bfloat16)
    c = lloyd_max_gaussian(3, "cpu", n_samples=20000, iters=10)
    Wq_ref, _, _, _ = Q.quant_dequant(W, 3, 128, "fullrot_whlm", centroids=c)
    pt = P.pack(W, 3, 128, "fullrot_whlm", centroids=c)
    Wq = P.unpack(pt)
    assert torch.equal(Wq.to(torch.float32), Wq_ref.to(torch.float32)), \
        "fullrot_whlm pack/unpack not bit-exact"


def test_pack_storage_bytes_smaller_than_fp16():
    """Packed storage must be smaller than the FP16 weight."""
    from quant import pack as P
    W = (torch.randn(64, 256) * 0.02).to(torch.float16)
    pt = P.pack(W, 3, 128, "rtn")
    fp16_bytes = W.numel() * 2
    assert pt.storage_bytes() < fp16_bytes, \
        f"packed {pt.storage_bytes()} >= fp16 {fp16_bytes}"


def test_vq_d4_lattice_points_valid():
    """D4 lattice points must have integer coords with even sum."""
    from quant import vq
    pts = vq.d4_lattice_points(radius=2)
    assert pts.shape[1] == 4
    # all coordinates are integers
    assert torch.all(pts == pts.floor())
    # all sums are even
    sums = pts.sum(dim=1)
    assert torch.all(sums % 2 == 0)


def test_vq_e8_lattice_points_valid():
    """E8 lattice points must be in Z^8 or (Z+0.5)^8 with even sum."""
    from quant import vq
    pts = vq.e8_lattice_points(radius=2)
    assert pts.shape[1] == 8
    fracs = pts.frac()
    # integer: frac near 0 (handle negative: frac(-2.0) = 0, frac(-1.0) = 0)
    is_int = torch.all(fracs.abs() < 0.01, dim=1)
    # half-integer: frac near +/-0.5
    is_half = torch.all((fracs.abs() - 0.5).abs() < 0.01, dim=1)
    assert torch.all(is_int | is_half), "E8 points must be integer or half-integer"
    sums = pts.sum(dim=1)
    assert torch.all(sums % 2 < 0.01), "E8 point sums must be even"


def test_vq_quant_round_trip_shape():
    """VQ quantization must return correct shapes."""
    from quant import vq
    cb = vq.gaussian_kmeans_codebook(4, 16, "cpu", n_samples=1000, iters=5)
    x = torch.randn(8, 4)
    q, idx = vq.quant_vq(x, cb)
    assert q.shape == x.shape
    assert idx.shape == (8,)


def test_vq_rvq_residual_decreases():
    """Each RVQ stage must reduce the residual."""
    from quant import vq
    cbs = [vq.gaussian_kmeans_codebook(4, 4, "cpu", n_samples=1000, iters=5, seed=s)
           for s in range(2)]
    x = torch.randn(16, 4)
    q, indices = vq.quant_rvq(x, cbs)
    residual_after = (x - q).norm()
    # single-stage residual for comparison
    q1, _ = vq.quant_vq(x, cbs[0])
    residual_after_1 = (x - q1).norm()
    assert residual_after < residual_after_1, \
        f"RVQ residual {residual_after:.4f} should be < single-stage {residual_after_1:.4f}"


def test_vq_rvq_dequant_matches():
    """RVQ dequantization must match the forward quantization."""
    from quant import vq
    cbs = [vq.gaussian_kmeans_codebook(4, 4, "cpu", n_samples=1000, iters=5, seed=s)
           for s in range(2)]
    x = torch.randn(8, 4)
    q, indices = vq.quant_rvq(x, cbs)
    q2 = vq.dequant_rvq(indices, cbs)
    assert torch.allclose(q, q2, atol=1e-5), "RVQ dequant mismatch"


def test_quant_dequant_fullrot_vq_runs():
    """fullrot_vq method must run and produce correct shape/dtype."""
    from quant import quantize as Q
    W = (torch.randn(16, 128) * 0.02).to(torch.bfloat16)
    for method in ["fullrot_vq:gaussian:2:1", "fullrot_vq:d4:2:1",
                   "fullrot_vq:gaussian:2:2", "fullrot_vq:d4:2:2"]:
        Wq, err, bpw, G = Q.quant_dequant(W, 2, 128, method)
        assert Wq.shape == W.shape, f"{method}: shape mismatch"
        assert Wq.dtype == W.dtype, f"{method}: dtype mismatch"
        assert err > 0, f"{method}: error should be positive"
        assert bpw > 0, f"{method}: bpw should be positive"


def test_vq_beats_scalar_on_gaussian():
    """VQ at equal bits should have lower reconstruction error than scalar
    Lloyd-Max on Gaussian data (the shaping gain)."""
    from quant import vq
    from quant.codebook import lloyd_max_gaussian
    torch.manual_seed(42)
    x = torch.randn(10000, 4)
    # scalar: quantize each coordinate independently with 2-bit Lloyd-Max
    c_scalar = lloyd_max_gaussian(2, "cpu", n_samples=50000, iters=20)
    bounds = (c_scalar[1:] + c_scalar[:-1]) / 2
    idx_s = torch.bucketize(x, bounds)
    q_scalar = c_scalar[idx_s]
    err_scalar = (x - q_scalar).norm() / x.norm()
    # VQ: 2-bit, d=4, single stage = 2^8 = 256 codewords
    cb = vq.gaussian_kmeans_codebook(4, 256, "cpu", n_samples=50000, iters=20)
    q_vq, _ = vq.quant_vq(x, cb)
    err_vq = (x - q_vq).norm() / x.norm()
    assert err_vq < err_scalar, \
        f"VQ err {err_vq:.4f} should be < scalar err {err_scalar:.4f} (shaping gain)"

