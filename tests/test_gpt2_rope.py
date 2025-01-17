import torch
import pytest
import math
from models.gpt_2_baseline_liger import Attention, GPT2Configuration


def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    return torch.atan2(torch.sin(angle), torch.cos(angle))

def test_freqs_cis_properties():
    dim = 16
    seq_len = 128
    theta = 10000.0

    # Base frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # Verify geometric sequence
    ratios = freqs[1:] / freqs[:-1]
    expected_ratio = theta ** (-2.0/dim)
    assert torch.allclose(ratios, torch.full_like(ratios, expected_ratio), rtol=1e-5)

    # Full rotary embeddings
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs_outer = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)

    # Check basic properties
    assert freqs_cis.shape == (seq_len, dim//2)
    assert torch.allclose(torch.abs(freqs_cis), torch.ones_like(freqs_outer))

    # Position 0 should have no rotation
    assert torch.allclose(freqs_cis[0].real, torch.ones(dim//2))
    assert torch.allclose(freqs_cis[0].imag, torch.zeros(dim//2))

def test_freqs_cis_all_positions():
    dim = 16
    seq_len = 128
    theta = 10000.0

    # Base frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs_outer = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)

    for pos in range(seq_len):
        # Expected angles at this position
        expected_angles = pos * freqs

        # Check real and imaginary parts
        assert torch.allclose(
            freqs_cis[pos].real,
            torch.cos(expected_angles),
            rtol=1e-5
        )
        assert torch.allclose(
            freqs_cis[pos].imag,
            torch.sin(expected_angles),
            rtol=1e-5
        )

        # Check magnitude
        assert torch.allclose(
            torch.abs(freqs_cis[pos]),
            torch.ones_like(freqs),
            rtol=1e-5
        )

def test_freqs_cis_geometric_properties():
    dim = 16
    seq_len = 128
    theta = 10000.0

    # Base frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # Check geometric progression of base frequencies
    for i in range(len(freqs) - 1):
        ratio = freqs[i] / freqs[i + 1]
        expected = theta ** (2.0/dim)
        assert torch.isclose(ratio, torch.tensor(expected), rtol=1e-5), \
            f"Ratio mismatch at index {i}: {ratio} != {expected}"

    # Check rotations
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs_outer = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)

    for dim_idx in range(freqs.shape[0]):
        base_freq = freqs[dim_idx]
        for pos in range(1, seq_len):
            angle1 = torch.angle(freqs_cis[pos, dim_idx])
            angle0 = torch.angle(freqs_cis[pos-1, dim_idx])

            # Calculate difference and normalize
            diff = normalize_angle(angle1 - angle0)

            # Print diagnostic info if failing
            if not torch.isclose(diff, base_freq, rtol=1e-5):
                print(f"\nDiagnostic info for position {pos}, dimension {dim_idx}:")
                print(f"angle1: {angle1}")
                print(f"angle0: {angle0}")
                print(f"diff: {diff}")
                print(f"base_freq: {base_freq}")
                print(f"absolute error: {abs(diff - base_freq)}")
                print(f"relative error: {abs(diff - base_freq) / base_freq}")

            assert torch.isclose(diff, base_freq, rtol=1e-5, atol=1e-7), \
                f"Angle step mismatch at position {pos}, dimension {dim_idx}\n" \
                f"diff={diff}, base_freq={base_freq}, " \
                f"abs_error={abs(diff - base_freq)}, " \
                f"rel_error={abs(diff - base_freq) / base_freq}"

def test_freqs_cis_full_rotation():
    """Test to verify full rotation behavior"""
    dim = 16
    seq_len = 128
    theta = 10000.0

    # Calculate frequencies exactly as in the original implementation
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs_outer = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)

    for dim_idx in range(freqs.shape[0]):
        # Get rotations directly from the original calculation
        expected_rotations = torch.polar(
            torch.ones(seq_len),
            t * freqs[dim_idx]
        )

        # Compare with our generated rotations
        actual_rotations = freqs_cis[:, dim_idx]

        # Compare complex numbers directly instead of angles
        assert torch.allclose(
            actual_rotations,
            expected_rotations,
            rtol=1e-5,
            atol=1e-7
        ), f"""
        Mismatch at dimension {dim_idx}:
        First mismatch at position: {torch.where(~torch.isclose(actual_rotations, expected_rotations))[0][0]}
        """

def print_angles_comparison(actual, expected, dim_idx):
    """Helper function to print detailed angle comparison"""
    print(f"\nDetailed comparison for dimension {dim_idx}:")
    print("pos | actual | expected | diff")
    print("-" * 40)
    for i in range(len(actual)):
        diff = abs(actual[i] - expected[i])
        if diff > 1e-5:
            print(f"{i:3d} | {actual[i]:7.4f} | {expected[i]:7.4f} | {diff:7.4f}")

def test_rotary_position_embeddings():
    """Test that RoPE properly rotates vectors based on position"""
    config = GPT2Configuration(block_size=128, d_model=64, num_heads=4)
    attention = Attention(config)

    # Create simple test vectors
    batch_size = 2
    seq_len = 4
    head_dim = config.d_model // config.num_heads

    # Create simple query/key vectors where we can easily track rotation
    query = torch.zeros(batch_size, config.num_heads, seq_len, head_dim)
    key = torch.zeros(batch_size, config.num_heads, seq_len, head_dim)

    # Set specific values in first head, first batch
    # Use unit vectors in x-y plane for easy rotation tracking
    query[0, 0, :, 0] = 1.0  # x-component
    query[0, 0, :, 1] = 0.0  # y-component
    key[0, 0, :, 0] = 0.0    # x-component
    key[0, 0, :, 1] = 1.0    # y-component

    # Apply RoPE
    rotated_query, rotated_key = attention.apply_rotary_position_embedding(
        query, key, attention.freqs_cis_sp[:seq_len]
    )

    # Test that vectors maintain unit length
    query_magnitudes = torch.sqrt(rotated_query[0, 0, :, 0]**2 + rotated_query[0, 0, :, 1]**2)
    key_magnitudes = torch.sqrt(rotated_key[0, 0, :, 0]**2 + rotated_key[0, 0, :, 1]**2)
    assert torch.allclose(query_magnitudes, torch.ones_like(query_magnitudes), atol=1e-6)
    assert torch.allclose(key_magnitudes, torch.ones_like(key_magnitudes), atol=1e-6)

    def get_rotation_angle(x, y):
        return torch.atan2(y, x)

    def normalize_angle_diff(diff):
        """Normalize angle difference to [-π, π]"""
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    # Get rotation angles at each position
    query_angles = get_rotation_angle(
        rotated_query[0, 0, :, 0],
        rotated_query[0, 0, :, 1]
    )
    key_angles = get_rotation_angle(
        rotated_key[0, 0, :, 0],
        rotated_key[0, 0, :, 1]
    )

    # Check rotation consistency for each consecutive pair
    for i in range(len(query_angles) - 1):
        # Calculate normalized angle differences
        query_diff = normalize_angle_diff(float(query_angles[i+1] - query_angles[i]))
        key_diff = normalize_angle_diff(float(key_angles[i+1] - key_angles[i]))

        # Base frequency for this dimension pair
        base_freq = float(1.0 / (10000.0 ** (0 / head_dim)))  # Using first dimension pair

        # Verify rotations match expected
        assert query_diff > 0, f"Query rotation not positive at position {i}"
        assert key_diff > 0, f"Key rotation not positive at position {i}"
        assert abs(query_diff - base_freq) < 1e-5, f"Query rotation step mismatch at position {i}"
        assert abs(key_diff - base_freq) < 1e-5, f"Key rotation step mismatch at position {i}"

    # Also verify that Q and K maintain relative orientation
    # The angle between Q and K should stay constant (π/2) after rotation
    for pos in range(seq_len):
        q_angle = float(query_angles[pos])
        k_angle = float(key_angles[pos])
        relative_angle = normalize_angle_diff(k_angle - q_angle)
        assert abs(relative_angle - math.pi/2) < 1e-5, f"Q-K relative angle changed at position {pos}"


def test_attention_liger_mode():
    """Test that attention outputs are well-behaved in both modes"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for this test")

    # Test parameters
    batch_size = 2
    seq_len = 4
    d_model = 64
    num_heads = 4

    # Create identical inputs
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Standard config
    config_standard = GPT2Configuration(
        block_size=128,
        d_model=d_model,
        num_heads=num_heads,
        use_liger=False
    )
    attention_standard = Attention(config_standard).to(device)

    # Liger config
    config_liger = GPT2Configuration(
        block_size=128,
        d_model=d_model,
        num_heads=num_heads,
        use_liger=True
    )
    attention_liger = Attention(config_liger).to(device)

    # Get outputs
    with torch.no_grad():
        output_standard = attention_standard(x)
        output_liger = attention_liger(x)

    # Basic tests for both outputs
    for name, output in [("standard", output_standard), ("liger", output_liger)]:
        # Check shape
        assert output.shape == x.shape, f"{name} output shape mismatch"

        # Check finite values
        assert torch.all(torch.isfinite(output)), f"{name} output has non-finite values"

        # Check reasonable magnitudes
        assert torch.all(torch.abs(output) < 100), f"{name} output has unreasonably large values"

    # Both implementations should produce outputs of similar magnitude
    std_mag = torch.norm(output_standard)
    liger_mag = torch.norm(output_liger)
    mag_ratio = std_mag / liger_mag
    assert 0.1 < mag_ratio < 10, "Outputs differ too much in magnitude"

    # The difference between implementations should not be extremely large
    max_diff = torch.max(torch.abs(output_standard - output_liger))
    assert max_diff < 2.0, "Maximum difference between outputs is too large"

    # Outputs should have similar statistics
    std_mean = output_standard.mean()
    liger_mean = output_liger.mean()
    assert torch.abs(std_mean - liger_mean) < 0.5, "Outputs have very different means"

if __name__ == "__main__":
    pytest.main([
        "-v",
        "tests/test_gpt2_rope.py::test_freqs_cis_properties",
        "tests/test_gpt2_rope.py::test_freqs_cis_all_positions",
        "tests/test_gpt2_rope.py::test_freqs_cis_geometric_properties",
        "tests/test_gpt2_rope.py::test_freqs_cis_full_rotation",
        "tests/test_gpt2_rope.py::test_rotary_position_embeddings",
        "tests/test_gpt2_rope.py::test_attention_liger_mode"
    ])