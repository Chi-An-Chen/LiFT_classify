from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F

from lift_classify import LiFTClassifier, sample_depth_coordinates
from conftest import activate_modulation


@pytest.mark.parametrize("batch,height,width", [(1, 32, 32), (2, 224, 224), (4, 256, 256), (1, 224, 256)])
@pytest.mark.parametrize("loops", [1, 2, 4, 8])
def test_shapes_and_inference_depth(config, batch, height, width, loops):
    model = LiFTClassifier(replace(config, patch_size=16)).eval()
    with torch.inference_mode():
        logits = model(torch.randn(batch, 3, height, width), num_loops=loops)
    assert logits.shape == (batch, config.num_classes)
    assert torch.isfinite(logits).all()


def test_manual_unroll_boundary_order_conditions_and_sharing(config):
    model = LiFTClassifier(config).eval()
    activate_modulation(model)
    images = torch.randn(2, 3, 16, 16)
    coordinates = torch.tensor([[0.1, 0.4, 1.0], [0.3, 0.9, 1.0]])
    core_calls, prelude_calls, readout_calls = [], [], []
    handles = [model.core.register_forward_pre_hook(
        lambda module, args: core_calls.append((args[0].clone(), args[1].clone(), tuple(id(p) for p in module.parameters())))
    ), model.prelude.register_forward_hook(lambda *args: prelude_calls.append(1)),
        model.readout.register_forward_hook(lambda *args: readout_calls.append(1))]
    output = model(images, num_loops=3, depth_coordinates=coordinates, return_intermediates=True)
    for handle in handles:
        handle.remove()
    condition = model.base_condition.expand(2, -1)
    h0 = model.prelude(model.patch_embedding(images).flatten(2).transpose(1, 2), condition, (4, 4))
    h = h0
    for step in range(3):
        expected_input = h / torch.sqrt(h.square().mean(-1, keepdim=True) + 1e-6) + h0
        expected_condition = condition + model.depth_embedding(coordinates[:, step])
        torch.testing.assert_close(core_calls[step][0], expected_input)
        torch.testing.assert_close(core_calls[step][1], expected_condition)
        h = model.core(expected_input, expected_condition, (4, 4))
        torch.testing.assert_close(output["hidden_states"][step + 1], h)
    torch.testing.assert_close(output["logits"], model.readout(h, condition))
    assert prelude_calls == [1] and readout_calls == [1]
    assert all(call[2] == core_calls[0][2] for call in core_calls)
    assert not torch.allclose(core_calls[0][0], model.boundary_norm(h0 + h0))


def test_num_loops_does_not_change_parameters_or_state_dict(config):
    model = LiFTClassifier(config)
    ids = tuple(id(p) for p in model.parameters())
    keys = tuple(model.state_dict())
    count = sum(p.numel() for p in model.parameters())
    for loops in (1, 2, 4, 8):
        model(torch.randn(1, 3, 16, 16), num_loops=loops)
        assert tuple(id(p) for p in model.parameters()) == ids
        assert tuple(model.state_dict()) == keys
        assert sum(p.numel() for p in model.parameters()) == count
        other = LiFTClassifier(replace(config, num_loops=loops))
        assert sum(p.numel() for p in other.parameters()) == count
        other.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize("states,logits", [(True, False), (False, True), (True, True)])
def test_intermediates_reuse_shared_readout_and_leave_state_unchanged(config, states, logits):
    model = LiFTClassifier(config).eval()
    activate_modulation(model)
    images = torch.randn(2, 3, 16, 16)
    expected = model(images)
    calls = []
    handle = model.readout.register_forward_hook(lambda *args: calls.append(1))
    output = model(images, return_intermediates=states, return_intermediate_logits=logits)
    handle.remove()
    torch.testing.assert_close(output["logits"], expected)
    assert len(calls) == (config.num_loops + 1 if logits else 1)
    assert (output["hidden_states"] is not None) == states
    assert (output["intermediate_logits"] is not None) == logits
    if states:
        assert len(output["hidden_states"]) == config.num_loops + 1
        assert all(h.requires_grad for h in output["hidden_states"])
        for h in output["hidden_states"]:
            before = h.clone()
            model.classify_features(h)
            torch.testing.assert_close(h, before, rtol=0, atol=0)
    if logits:
        assert output["logits"] is output["intermediate_logits"][-1]


def test_readout_normalizes_tokens_before_pooling(config):
    model = LiFTClassifier(config)
    activate_modulation(model.readout)
    h = torch.randn(2, 5, config.hidden_dim) * torch.arange(1, 6)[None, :, None]
    condition = model.base_condition.expand(2, -1)
    shift, scale = model.readout.modulation(condition).chunk(2, -1)
    normalized = h / torch.sqrt(h.square().mean(-1, keepdim=True) + 1e-6)
    expected = model.readout.classifier((normalized * (1 + scale[:, None]) + shift[:, None]).mean(1))
    torch.testing.assert_close(model.classify_features(h), expected)
    wrong = model.readout.classifier(model.readout.norm(h.mean(1)) * (1 + scale) + shift)
    assert not torch.allclose(expected, wrong)


def test_initial_zero_gates_then_ce_unlocks_all_required_groups(config):
    model = LiFTClassifier(config)
    images = torch.randn(5, 3, 16, 16)
    labels = torch.arange(5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    groups = {"patch": model.patch_embedding, "prelude": model.prelude, "core": model.core,
              "depth": model.depth_embedding, "head": model.readout.classifier}
    seen = set()
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels)
        assert torch.isfinite(loss)
        loss.backward()
        for name, module in groups.items():
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert all(torch.isfinite(g).all() for g in grads)
            if grads and sum(g.abs().sum().item() for g in grads) > 0:
                seen.add(name)
        if model.base_condition.grad is not None and model.base_condition.grad.abs().sum() > 0:
            seen.add("condition")
        if step == 0:
            assert model.core.blocks[0].attn.qkv.weight.grad.abs().sum() == 0
            assert model.core.blocks[0].mlp.fc1.weight.grad.abs().sum() == 0
            assert model.depth_embedding.mlp[0].weight.grad.abs().sum() == 0
            assert model.patch_embedding.weight.grad.abs().sum() > 0
            assert model.readout.classifier.weight.grad.abs().sum() > 0
            assert model.core.blocks[0].modulation[-1].bias.grad.abs().sum() > 0
        optimizer.step()
    assert seen == set(groups) | {"condition"}
    assert model.core.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    assert model.core.blocks[0].mlp.fc1.weight.grad.abs().sum() > 0
    assert model.prelude.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    assert model.prelude.blocks[0].mlp.fc1.weight.grad.abs().sum() > 0


def test_intermediate_supervision_and_shared_unroll_gradients(config):
    model = LiFTClassifier(config)
    activate_modulation(model)
    images = torch.randn(2, 3, 16, 16)
    labels = torch.tensor([1, 3])
    output = model(images, return_intermediates=True, return_intermediate_logits=True)
    for h in output["hidden_states"]:
        h.retain_grad()
    sum(F.cross_entropy(z, labels) for z in output["intermediate_logits"]).backward()
    assert all(h.grad is not None and h.grad.abs().sum() > 0 for h in output["hidden_states"])

    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(images), labels).backward()
    expected = {name: p.grad.clone() for name, p in model.core.named_parameters()}
    model.zero_grad(set_to_none=True)
    condition = model.base_condition.expand(2, -1)
    h0 = model.prelude(model.patch_embedding(images).flatten(2).transpose(1, 2), condition, (4, 4))
    h = h0
    for coordinate in (0.5, 1.0):
        h = model.core(model.boundary_norm(h) + h0,
                       condition + model.depth_embedding(torch.full((2,), coordinate)), (4, 4))
    F.cross_entropy(model.readout(h, condition), labels).backward()
    for name, p in model.core.named_parameters():
        torch.testing.assert_close(p.grad, expected[name])


def test_depth_condition_changes_core_behavior_and_random_grids_work(config):
    model = LiFTClassifier(config).eval()
    activate_modulation(model)
    images = torch.randn(2, 3, 16, 16)
    a = model(images, depth_coordinates=torch.tensor([0.1, 1.0]), return_intermediates=True)
    b = model(images, depth_coordinates=torch.tensor([0.9, 1.0]), return_intermediates=True)
    assert not torch.equal(a["hidden_states"][1], b["hidden_states"][1])
    torch.testing.assert_close(a["hidden_states"][0], b["hidden_states"][0], rtol=0, atol=0)
    result = model(images, depth_coordinates=sample_depth_coordinates(2, 2))
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("scale", [0.0, 1e-10, 1.0, 1e4])
@pytest.mark.parametrize("loops", [1, 2, 4, 8])
def test_recurrent_numerical_stability_after_gates_activate(config, scale, loops):
    model = LiFTClassifier(config)
    activate_modulation(model)
    output = model(torch.randn(2, 3, 16, 16) * scale, num_loops=loops, return_intermediates=True)
    assert all(torch.isfinite(h).all() for h in output["hidden_states"])
    assert torch.isfinite(output["logits"]).all()
    F.cross_entropy(output["logits"], torch.tensor([1, 2])).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_bfloat16_autocast_and_native_half(config):
    model = LiFTClassifier(config)
    activate_modulation(model)
    images = torch.randn(2, 3, 16, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(images, num_loops=4, return_intermediates=True)
        loss = F.cross_entropy(output["logits"], torch.tensor([1, 2]))
    loss.backward()
    assert output["logits"].dtype == torch.bfloat16
    assert all(torch.isfinite(h).all() for h in output["hidden_states"])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    half_model = LiFTClassifier(config).half()
    with torch.no_grad():
        assert torch.isfinite(half_model(images.half(), num_loops=4)).all()


def test_no_prelude_and_rectangular_patches(config):
    model = LiFTClassifier(replace(config, image_size=(16, 24), patch_size=(4, 8), prelude_depth=0))
    assert model(torch.randn(1, 3, 20, 32)).shape == (1, config.num_classes)


@pytest.mark.parametrize("images", [torch.randn(2, 3, 15, 16), torch.randn(1, 4, 16, 16),
                                    torch.randn(3, 16, 16), torch.ones(1, 3, 16, 16, dtype=torch.int64),
                                    torch.randn(0, 3, 16, 16)])
def test_invalid_images(config, images):
    with pytest.raises(ValueError, match="images"):
        LiFTClassifier(config)(images)


@pytest.mark.parametrize("loops", [0, -2, 1.5, True])
def test_invalid_forward_loops(config, loops):
    with pytest.raises(ValueError, match="num_loops"):
        LiFTClassifier(config)(torch.randn(1, 3, 16, 16), num_loops=loops)


def test_labels_are_not_an_input(config):
    with pytest.raises(TypeError):
        LiFTClassifier(config)(torch.randn(1, 3, 16, 16), labels=torch.tensor([1]))
