"""Inspect compression, latent sampling, an adversarial update, and noising."""

from dataclasses import dataclass

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.logistic import stable_logistic
from oop_ml.core.types import FloatArray
from oop_ml.numpy.modern._validation import count, finite, positive


@dataclass(frozen=True)
class Reconstruction:
    code: FloatArray
    reconstructed: FloatArray
    mean_squared_error: float


class LinearAutoencoder:
    """An explicit linear encoder and decoder with caller-supplied weights.

    Useful for exposing information lost at a bottleneck. No fit is implied;
    learned nonlinear encoders use the same encode/decode arrangement.
    """

    def __init__(self, encoder: FloatArray, decoder: FloatArray) -> None:
        self.encoder = finite(encoder, 2, "encoder")
        self.decoder = finite(decoder, 2, "decoder")
        if self.encoder.shape != self.decoder.T.shape:
            raise ShapeMismatchError("decoder must return to the input width")

    def respond_to(self, inputs: FloatArray) -> Reconstruction:
        values = finite(inputs, 2, "inputs")
        if values.shape[1] != self.encoder.shape[0]:
            raise ShapeMismatchError("inputs must match encoder width")
        code = values @ self.encoder
        reconstructed = code @ self.decoder
        return Reconstruction(
            code, reconstructed, float(np.mean((values - reconstructed) ** 2))
        )


class DiagonalGaussian:
    """A VAE's per-example approximate posterior, with explicit sample noise.

    Each code coordinate has a mean and positive standard deviation. The KL is
    against a standard normal prior, summed over code coordinates. The encoder,
    decoder, and reconstruction likelihood are separate operations.
    """

    def __init__(self, mean: FloatArray, deviation: FloatArray) -> None:
        self.mean = finite(mean, 1, "mean")
        self.deviation = finite(deviation, 1, "deviation")
        if self.mean.shape != self.deviation.shape:
            raise ShapeMismatchError("mean and deviation must share a shape")
        if np.any(self.deviation <= 0):
            raise InvalidValuesError("deviations must be strictly positive")

    def sample(self, noise: FloatArray) -> FloatArray:
        """Reparameterize so differentiation can reach mean and deviation."""
        noise = finite(noise, 1, "noise")
        if noise.shape != self.mean.shape:
            raise ShapeMismatchError("noise must match the code width")
        return self.mean + self.deviation * noise

    @property
    def kl_to_standard_normal(self) -> float:
        variance = self.deviation**2
        return float(0.5 * np.sum(self.mean**2 + variance - 1 - np.log(variance)))


@dataclass(frozen=True)
class AdversarialState:
    generator_mean: float
    discriminator_weight: float
    discriminator_bias: float
    real_score: float
    fake_score: float
    discriminator_loss: float
    generator_loss: float


class AdversarialGaussian:
    """Alternating gradients in a deliberately limited one-dimensional GAN.

    G(z) = z + mean learns only a shift. D(x) = sigmoid(weight*x + bias).
    The generator uses non-saturating loss; the discriminator averages its two
    binary cross-entropies. Fixed noise makes before/after comparisons fair.
    This family can match a shifted distribution, but cannot model two modes.
    """

    def __init__(self, generator_mean: float = -1.5) -> None:
        self.mean = float(finite([generator_mean], 1, "generator_mean")[0])
        self.weight = 0.5
        self.bias = 0.0

    def inspect(self, real: FloatArray, noise: FloatArray) -> AdversarialState:
        real = finite(real, 1, "real")
        fake = finite(noise, 1, "noise") + self.mean
        real_logits = self.weight * real + self.bias
        fake_logits = self.weight * fake + self.bias
        return AdversarialState(
            self.mean,
            self.weight,
            self.bias,
            float(stable_logistic(real_logits).mean()),
            float(stable_logistic(fake_logits).mean()),
            float(
                (
                    np.logaddexp(0, -real_logits).mean()
                    + np.logaddexp(0, fake_logits).mean()
                )
                / 2
            ),
            float(np.logaddexp(0, -fake_logits).mean()),
        )

    def train(
        self,
        real: FloatArray,
        noise: FloatArray,
        steps: int = 1,
        learning_rate: float = 0.1,
    ) -> list[AdversarialState]:
        """Update D with G fixed, then update G using the newly updated D."""
        real = finite(real, 1, "real")
        noise = finite(noise, 1, "noise")
        count(steps, "steps", 0)
        rate = positive(learning_rate, "learning_rate")
        history = [self.inspect(real, noise)]
        for _ in range(steps):
            fake = noise + self.mean
            real_error = stable_logistic(self.weight * real + self.bias) - 1
            fake_error = stable_logistic(self.weight * fake + self.bias)
            slope = ((real_error * real).mean() + (fake_error * fake).mean()) / 2
            offset = (real_error.mean() + fake_error.mean()) / 2
            self.weight -= rate * float(slope)
            self.bias -= rate * float(offset)
            generated_error = stable_logistic(self.weight * fake + self.bias) - 1
            self.mean -= rate * float(generated_error.mean()) * self.weight
            history.append(self.inspect(real, noise))
        return history


class GaussianDiffusion:
    """The forward Gaussian schedule and a clean-sample estimate from noise.

    This is the training pair construction, not a trained reverse sampler.
    recover_clean requires a noise estimate supplied by the caller. Passing the
    true added noise is an oracle check, not a way to generate from unknown noise.
    """

    def __init__(self, betas: FloatArray) -> None:
        self.betas = finite(betas, 1, "betas")
        if np.any(self.betas <= 0) or np.any(self.betas >= 1):
            raise InvalidValuesError("each beta must be strictly between 0 and 1")
        self.retention = np.concatenate(([1.0], np.cumprod(1 - self.betas)))
        if self.retention[-1] == 0:
            raise InvalidValuesError("schedule underflows; shorten or reduce it")

    def _at(self, step: int) -> float:
        count(step, "step", 0)
        if step >= len(self.retention):
            raise InvalidValuesError("step lies outside the schedule")
        return float(self.retention[step])

    def add_noise(self, clean: FloatArray, noise: FloatArray, step: int) -> FloatArray:
        clean, noise = finite(clean, 1, "clean"), finite(noise, 1, "noise")
        if clean.shape != noise.shape:
            raise ShapeMismatchError("noise must match the clean sample")
        retained = self._at(step)
        return np.sqrt(retained) * clean + np.sqrt(1 - retained) * noise

    def recover_clean(
        self, noisy: FloatArray, predicted_noise: FloatArray, step: int
    ) -> FloatArray:
        noisy = finite(noisy, 1, "noisy")
        prediction = finite(predicted_noise, 1, "predicted_noise")
        if noisy.shape != prediction.shape:
            raise ShapeMismatchError("prediction must match the noisy sample")
        retained = self._at(step)
        return (noisy - np.sqrt(1 - retained) * prediction) / np.sqrt(retained)
