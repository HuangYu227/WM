import torch


def test_highpass_removes_constant_and_local_low_frequency():
    from worldttt.sap_binding.value import spatial_highpass

    constant = torch.ones(2, 18, 6) * 7
    result = spatial_highpass(constant, 2, 3, 3)
    torch.testing.assert_close(result, torch.zeros_like(result))
    impulse = constant.clone(); impulse[:, 4] += 1
    result = spatial_highpass(impulse, 2, 3, 3)
    assert result.abs().sum() > 0
    torch.testing.assert_close(result.reshape(2, 2, 9, 6).mean((1, 2)), torch.zeros(2, 6), atol=1e-6, rtol=0)


def test_value_projection_is_fixed_and_whitening_uses_supplied_training_values():
    from worldttt.sap_binding.value import BindingValueEncoder

    encoder = BindingValueEncoder(8, 4, value_dim=8, seed=5)
    assert not any(p.requires_grad for p in encoder.parameters())
    h, z = torch.randn(2, 18, 8), torch.randn(2, 4, 2, 3, 3)
    raw = encoder.raw(h, z, 2, 3, 3)
    encoder.fit_whitening([raw])
    value = encoder(h, z, 2, 3, 3)
    torch.testing.assert_close(value.reshape(-1, 8).mean(0), torch.zeros(8), atol=1e-5, rtol=0)
    torch.testing.assert_close(value.reshape(-1, 8).std(0, unbiased=False), torch.ones(8), atol=2e-4, rtol=0)
    assert encoder.whitening_fitted


def test_latent_projection_supports_required_128_to_256_isometric_expansion():
    from worldttt.sap_binding.value import fixed_orthogonal
    weight = fixed_orthogonal(128, 256, 9)
    assert weight.shape == (256, 128)
    torch.testing.assert_close(weight.T @ weight, torch.eye(128), atol=2e-5, rtol=2e-5)
