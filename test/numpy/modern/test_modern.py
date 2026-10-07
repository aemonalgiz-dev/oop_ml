"""Check the useful mathematical promises, not merely the fixture outputs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.numpy.modern import (
    AdversarialGaussian,
    CausalLanguageModel,
    DiagonalGaussian,
    GaussianDiffusion,
    LinearAutoencoder,
    LowRankAdapter,
    TermOverlapRetriever,
    TokenCrossEntropy,
    TokenDistribution,
)


@pytest.mark.parametrize("temperature", [0.1, 1, 10])
def test_sampling_is_normalized_and_invariant_to_a_logit_offset(temperature):
    logits = np.array([2.0, 1.0, -3.0])
    first = TokenDistribution.from_logits(logits, temperature)
    second = TokenDistribution.from_logits(logits + 10000, temperature)
    assert_allclose(first.probabilities, second.probabilities)
    assert_allclose(first.probabilities.sum(), 1)
    assert first.greedy() == 0


def test_nucleus_keeps_the_threshold_crossing_token_and_renormalizes():
    result = TokenDistribution.from_logits(np.log([0.5, 0.3, 0.2]), top_p=0.7)
    assert result.retained.tolist() == [True, True, False]
    assert_allclose(result.probabilities, [0.625, 0.375, 0])


def test_large_equal_logits_do_not_overflow_with_low_temperature():
    result = TokenDistribution.from_logits(np.array([1e308, 1e308]), 0.1)
    assert_allclose(result.probabilities, [0.5, 0.5])


@pytest.mark.parametrize("mask", [None, np.array([False, True, True])])
def test_token_loss_gradient_matches_finite_difference(mask):
    logits = np.random.default_rng(2).normal(size=(3, 4))
    targets = np.array([1, 2, 0])
    objective = TokenCrossEntropy()
    analytic = objective.measure(logits, targets, mask)
    numerical = np.empty_like(logits)
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (
            objective.measure(plus, targets, mask).value
            - objective.measure(minus, targets, mask).value
        ) / 2e-6
    assert_allclose(analytic.gradient, numerical, atol=1e-9)
    assert_allclose(analytic.gradient.sum(axis=1), 0, atol=1e-15)


def test_uniform_perplexity_equals_vocabulary_size():
    result = TokenCrossEntropy().measure(np.zeros((3, 7)), np.array([0, 3, 4]))
    assert_allclose(result.perplexity, 7)


def test_causal_decoder_cannot_read_a_changed_future_token():
    model = CausalLanguageModel(7)
    first = model.respond_to(np.array([0, 1, 2]))
    second = model.respond_to(np.array([0, 1, 3]))
    prefix = model.respond_to(np.array([0, 1]))
    assert_allclose(first.logits[:2], second.logits[:2], atol=1e-12)
    assert_allclose(first.logits[:2], prefix.logits, atol=1e-12)
    assert not np.allclose(first.logits[-1], second.logits[-1])
    for head in first.attention:
        assert_allclose(head.sum(axis=1), 1)
        assert_allclose(np.triu(head, 1), 0)


def test_generation_reuses_appended_prefix_and_can_stop_at_end():
    model = CausalLanguageModel(7, max_context=8)
    prefix = np.array([0, 1])
    first = model.generate(prefix, max_new_tokens=20, seed=42)
    second = model.generate(prefix, max_new_tokens=20, seed=42)
    assert [r.token_id for r in first] == [r.token_id for r in second]
    assert len(first) == 6
    for before, after in zip(first, first[1:], strict=False):
        assert after.context == (*before.context, before.token_id)
    stop = model.generate(prefix, seed=42, end_id=first[0].token_id)
    assert len(stop) == 1
    assert prefix.tolist() == [0, 1]


def test_averaging_bottleneck_loses_difference_but_keeps_equal_inputs():
    model = LinearAutoencoder(np.array([[0.5], [0.5]]), np.ones((1, 2)))
    response = model.respond_to(np.array([[3.0, -1.0], [1.0, 1.0]]))
    assert_allclose(response.code, [[1], [1]])
    assert_allclose(response.reconstructed, [[1, 1], [1, 1]])
    assert_allclose(response.mean_squared_error, 2)


def test_gaussian_reparameterization_and_kl_special_cases():
    standard = DiagonalGaussian(np.zeros(2), np.ones(2))
    assert_allclose(standard.kl_to_standard_normal, 0)
    shifted = DiagonalGaussian(np.array([2.0, -1.0]), np.ones(2))
    assert_allclose(shifted.kl_to_standard_normal, 2.5)
    assert_allclose(shifted.sample(np.array([0.5, -0.5])), [2.5, -1.5])


def test_gan_alternation_matches_loss_derivatives():
    noise = np.array([-1.0, -0.5, 0.5, 1.0])
    real = noise + 1.5
    model = AdversarialGaussian()
    rate, epsilon = 0.1, 1e-6
    expected = {}
    for parameter in ["weight", "bias"]:
        original = getattr(model, parameter)
        setattr(model, parameter, original + epsilon)
        plus = model.inspect(real, noise).discriminator_loss
        setattr(model, parameter, original - epsilon)
        minus = model.inspect(real, noise).discriminator_loss
        setattr(model, parameter, original)
        expected[parameter] = original - rate * (plus - minus) / (2 * epsilon)
    model.weight, model.bias = expected["weight"], expected["bias"]
    original = model.mean
    model.mean = original + epsilon
    plus = model.inspect(real, noise).generator_loss
    model.mean = original - epsilon
    minus = model.inspect(real, noise).generator_loss
    expected_mean = original - rate * (plus - minus) / (2 * epsilon)
    actual = AdversarialGaussian().train(real, noise, 1, rate)[-1]
    assert_allclose(actual.discriminator_weight, expected["weight"], atol=1e-10)
    assert_allclose(actual.discriminator_bias, expected["bias"], atol=1e-10)
    assert_allclose(actual.generator_mean, expected_mean, atol=1e-10)


def test_small_gan_reduces_the_shift_error_on_the_fixed_distribution():
    noise = np.random.default_rng(3).normal(size=128)
    history = AdversarialGaussian().train(noise + 1.5, noise, 100)
    assert abs(history[-1].generator_mean - 1.5) < abs(history[0].generator_mean - 1.5)


@pytest.mark.parametrize("step", [0, 1, 10, 20])
def test_diffusion_recovers_clean_signal_when_given_exact_noise(step):
    model = GaussianDiffusion(np.full(20, 0.1))
    clean = np.array([0, 1, 1, 0], dtype=float)
    noise = np.random.default_rng(5).normal(size=4)
    noisy = model.add_noise(clean, noise, step)
    assert_allclose(model.recover_clean(noisy, noise, step), clean, atol=1e-14)
    if step == 0:
        assert_allclose(noisy, clean)


def test_lora_factored_and_merged_outputs_agree_without_mutating_base():
    rng = np.random.default_rng(4)
    base, down, up = (
        rng.normal(size=(6, 5)),
        rng.normal(size=(6, 2)),
        rng.normal(size=(2, 5)),
    )
    saved = base.copy()
    inputs = rng.normal(size=(3, 6))
    adapter = LowRankAdapter(base, down, up, alpha=4)
    assert_allclose(
        adapter.respond_to(inputs), inputs @ base + 2 * (inputs @ down) @ up
    )
    assert_allclose(base, saved)
    assert np.linalg.matrix_rank(adapter.delta) <= 2
    assert adapter.trainable_parameters == 22


def test_retrieval_ranks_exact_matches_and_does_not_invent_unknown_matches():
    model = TermOverlapRetriever(["cat sat", "dog slept", "cat sat"])
    matches = model.retrieve("CAT sat", 3)
    assert [r.index for r in matches] == [0, 2]
    assert_allclose(matches[0].score, 1)
    assert model.retrieve("elephant") == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda: TokenDistribution.from_logits(np.ones(2), temperature=0),
        lambda: TokenDistribution.from_logits(np.ones(2), top_p=0),
        lambda: TokenDistribution.from_logits(np.array([np.nan])),
        lambda: TokenCrossEntropy().measure(np.zeros((2, 2)), np.array([0, 2])),
        lambda: TokenCrossEntropy().measure(
            np.zeros((2, 2)), np.array([0, 1]), np.zeros(2, dtype=bool)
        ),
        lambda: CausalLanguageModel(4, n_heads=3),
        lambda: CausalLanguageModel(4).respond_to(np.array([0.0, 1.0])),
        lambda: CausalLanguageModel(4).respond_to(np.array([4])),
        lambda: DiagonalGaussian(np.zeros(1), np.zeros(1)),
        lambda: GaussianDiffusion(np.array([1.0])),
        lambda: GaussianDiffusion(np.array([0.1])).add_noise(np.ones(2), np.ones(3), 1),
        lambda: LowRankAdapter(np.eye(4), np.ones((4, 2)), np.ones((3, 4))),
        lambda: TermOverlapRetriever([]),
    ],
)
def test_invalid_inputs_fail_at_sdk_boundary(operation):
    with pytest.raises((InvalidValuesError, ShapeMismatchError)):
        operation()
