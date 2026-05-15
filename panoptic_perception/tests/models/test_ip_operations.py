from panoptic_perception.models.gdip.ip_operations import (GateModule,
                                                           GammaBalance,
                                                           WhiteBalance,
                                                           Contrast,
                                                           Tone,
                                                           Defog,
                                                           Sharpening,
                                                           tanh_range)

import torch

def test_gate_module(latent_dim, num_gates, latent_embed):
    gate = GateModule(latent_dim, num_gates)
    gate_out = gate(latent_embed)

    assert gate_out.shape == (latent_embed.shape[0], num_gates)
    assert gate_out.min() >= 0.01, f'Gate min {gate_out.min()} below 0.01'
    assert gate_out.max() <= 1.0, f'Gate max {gate_out.max()} above 1.0'

    print(f'GateModule: PASSED  shape={gate_out.shape}  range=[{gate_out.min():.3f}, {gate_out.max():.3f}]')
    return gate_out

def test_white_balance(latent_dim, x, latent_embed, gate):
    wb = WhiteBalance(latent_dim)
    out = wb(x, latent_embed, gate)

    assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
    assert torch.isfinite(out).all(), 'Output contains NaN/Inf'

    print(f'WhiteBalance: PASSED  range=[{out.min():.3f}, {out.max():.3f}]')

def test_gamma_balance(latent_dim, x, latent_embed, gate):
    gamma = GammaBalance(latent_dim)
    out = gamma(x, latent_embed, gate)

    assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
    assert torch.isfinite(out).all(), 'Output contains NaN/Inf'

    print(f'GammaBalance: PASSED  range=[{out.min():.3f}, {out.max():.3f}]')

def test_sharpening(latent_dim, x, latent_embed, gate):
    sharpen = Sharpening(latent_dim, kernel_size=13)
    out = sharpen(x, latent_embed, gate)

    assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
    assert torch.isfinite(out).all(), 'Output contains NaN/Inf'

    print(f'Sharpening: PASSED  range=[{out.min():.3f}, {out.max():.3f}]')

def test_defog(latent_dim, x, latent_embed, gate):
    defog = Defog(latent_dim)
    out = defog(x, latent_embed, gate)

    assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
    assert torch.isfinite(out).all(), 'Output contains NaN/Inf'

    print(f'Defog: PASSED  range=[{out.min():.3f}, {out.max():.3f}]')

def test_contrast(latent_dim, x, latent_embed, gate):
    contrast = Contrast(latent_dim)
    out = contrast(x, latent_embed, gate)

    assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
    assert torch.isfinite(out).all(), 'Output contains NaN/Inf'

    print(f'Contrast: PASSED  range=[{out.min():.3f}, {out.max():.3f}]')

def test_tone(latent_dim, x, latent_embed, gate):
    tone = Tone(latent_dim)
    out = tone(x, latent_embed, gate)

    assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'
    assert torch.isfinite(out).all(), 'Output contains NaN/Inf'

    print(f'Tone: PASSED  range=[{out.min():.3f}, {out.max():.3f}]')

def test_identity(x, gate):
    out = x * gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    assert out.shape == x.shape, f'Shape mismatch: {out.shape} vs {x.shape}'

    print(f'Identity: PASSED  range=[{out.min():.3f}, {out.max():.3f}]')

def test_combined_output(latent_dim, x, latent_embed, gates):
    """Test that all IP operations sum to a valid output (mimics GDIP forward)."""
    wb_out = WhiteBalance(latent_dim)(x, latent_embed, gates[:, 0])
    gamma_out = GammaBalance(latent_dim)(x, latent_embed, gates[:, 1])
    identity_out = x * gates[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    sharpen_out = Sharpening(latent_dim, kernel_size=13)(x, latent_embed, gates[:, 3])
    defog_out = Defog(latent_dim)(x, latent_embed, gates[:, 4])
    contrast_out = Contrast(latent_dim)(x, latent_embed, gates[:, 5])
    tone_out = Tone(latent_dim)(x, latent_embed, gates[:, 6])

    combined = wb_out + gamma_out + identity_out + sharpen_out + defog_out + contrast_out + tone_out
    combined = (combined - combined.min()) / (combined.max() - combined.min())

    assert combined.shape == x.shape, f'Shape mismatch: {combined.shape} vs {x.shape}'
    assert torch.isfinite(combined).all(), 'Combined output contains NaN/Inf'
    assert combined.min() >= 0.0 and combined.max() <= 1.0, 'Combined output not in [0, 1]'

    print(f'Combined GDIP: PASSED  shape={combined.shape}  range=[{combined.min():.3f}, {combined.max():.3f}]')

def test_ip_operations():

    h, w = (768, 1280)
    num_channels = 3
    batch_size = 2

    latent_dim = 256
    num_ip_operations = 7

    x = torch.randint(low=0, high=255, size=(batch_size, num_channels, h, w))
    x = x.float() / 255.0

    latent_embed = torch.rand(batch_size, latent_dim).float()

    print(f'Image: {x.shape}  Latent: {latent_embed.shape}')
    print('=' * 60)

    gates = test_gate_module(latent_dim, num_ip_operations, latent_embed)

    test_white_balance(latent_dim, x, latent_embed, gates[:, 0])
    test_gamma_balance(latent_dim, x, latent_embed, gates[:, 1])
    test_sharpening(latent_dim, x, latent_embed, gates[:, 3])
    test_defog(latent_dim, x, latent_embed, gates[:, 4])
    test_contrast(latent_dim, x, latent_embed, gates[:, 5])
    test_tone(latent_dim, x, latent_embed, gates[:, 6])
    test_identity(x, gates[:, 2])

    print('=' * 60)
    test_combined_output(latent_dim, x, latent_embed, gates)

    print('=' * 60)
    print('ALL TESTS PASSED')

if __name__ == "__main__":
    test_ip_operations()
