"""Imitation-learning world-model pretraining with JAX/Flax.

Expected dataset format (.npz):
- observations: [N, T, H, W, C], uint8 or float32
- actions: [N, T, A], float32
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state


@dataclass
class ILPretrainConfig:
    """Configuration for IL world-model pretraining."""

    dataset_path: str = "data/il_sequences.npz"
    output_dir: str = "outputs/il_world_model"
    chunk_size: int = 50
    latent_dim: int = 64
    deter_dim: int = 128
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 3e-4
    kl_scale: float = 1e-3
    seed: int = 0
    save_name: str = "world_model_params.msgpack"


class ImageEncoder(nn.Module):
    """Encode image observation into a compact feature."""

    feature_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.feature_dim)(x)
        return nn.tanh(x)


class ActionEncoder(nn.Module):
    """Encode action vectors."""

    feature_dim: int

    @nn.compact
    def __call__(self, a: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.feature_dim)(a)
        return nn.tanh(x)


class RSSMWorldModel(nn.Module):
    """Action-conditioned latent world model with decoder."""

    latent_dim: int
    deter_dim: int
    action_dim: int
    image_shape: tuple[int, int, int]

    def setup(self) -> None:
        self.encoder = ImageEncoder(feature_dim=self.deter_dim)
        self.action_encoder = ActionEncoder(feature_dim=self.deter_dim)
        self.gru = nn.GRUCell(features=self.deter_dim)
        self.prior_mean = nn.Dense(self.latent_dim)
        self.prior_logvar = nn.Dense(self.latent_dim)
        self.post_mean = nn.Dense(self.latent_dim)
        self.post_logvar = nn.Dense(self.latent_dim)
        self.decoder_fc = nn.Dense(8 * 8 * 64)
        self.deconv1 = nn.ConvTranspose(64, (4, 4), (2, 2))
        self.deconv2 = nn.ConvTranspose(32, (4, 4), (2, 2))
        self.deconv3 = nn.ConvTranspose(self.image_shape[-1], (4, 4), (2, 2))

    def _decode(self, deter: jnp.ndarray, stoch: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([deter, stoch], axis=-1)
        x = nn.relu(self.decoder_fc(x))
        x = x.reshape((x.shape[0], 8, 8, 64))
        x = nn.relu(self.deconv1(x))
        x = nn.relu(self.deconv2(x))
        x = self.deconv3(x)
        return nn.sigmoid(x)

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Posterior rollout for training.

        observations: [B, T, H, W, C]
        actions: [B, T, A]
        """
        bsz, horizon = observations.shape[:2]
        features = observations.reshape((-1,) + observations.shape[2:])
        obs_embed = self.encoder(features).reshape((bsz, horizon, -1))
        act_embed = self.action_encoder(actions.reshape((-1, actions.shape[-1]))).reshape(
            (bsz, horizon, -1)
        )

        deter = jnp.zeros((bsz, self.deter_dim), dtype=jnp.float32)
        stoch = jnp.zeros((bsz, self.latent_dim), dtype=jnp.float32)
        recon_frames = []
        kl_terms = []
        keys = jax.random.split(rng, horizon)

        for t in range(horizon):
            gru_in = jnp.concatenate([stoch, act_embed[:, t]], axis=-1)
            deter, _ = self.gru(deter, gru_in)
            prior_mu = self.prior_mean(deter)
            prior_logvar = self.prior_logvar(deter)

            post_in = jnp.concatenate([deter, obs_embed[:, t]], axis=-1)
            post_mu = self.post_mean(post_in)
            post_logvar = self.post_logvar(post_in)
            eps = jax.random.normal(keys[t], post_mu.shape)
            stoch = post_mu + jnp.exp(0.5 * post_logvar) * eps

            recon_frames.append(self._decode(deter, stoch))
            kl = _gaussian_kl(post_mu, post_logvar, prior_mu, prior_logvar)
            kl_terms.append(jnp.mean(kl))

        recons = jnp.stack(recon_frames, axis=1)
        kl_loss = jnp.mean(jnp.stack(kl_terms, axis=0))
        return recons, kl_loss

    def rollout_predict(
        self,
        init_observation: jnp.ndarray,
        action_seq: jnp.ndarray,
        rng: jax.Array,
    ) -> jnp.ndarray:
        """Predict future frames from one frame + actions.

        init_observation: [B, H, W, C]
        action_seq: [B, T, A]
        """
        bsz, horizon = action_seq.shape[:2]
        init_embed = self.encoder(init_observation)
        deter = init_embed
        stoch = jnp.zeros((bsz, self.latent_dim), dtype=jnp.float32)
        act_embed = self.action_encoder(action_seq.reshape((-1, action_seq.shape[-1]))).reshape(
            (bsz, horizon, -1)
        )
        keys = jax.random.split(rng, horizon)
        preds = []
        for t in range(horizon):
            gru_in = jnp.concatenate([stoch, act_embed[:, t]], axis=-1)
            deter, _ = self.gru(deter, gru_in)
            prior_mu = self.prior_mean(deter)
            prior_logvar = self.prior_logvar(deter)
            eps = jax.random.normal(keys[t], prior_mu.shape)
            stoch = prior_mu + jnp.exp(0.5 * prior_logvar) * eps
            preds.append(self._decode(deter, stoch))
        return jnp.stack(preds, axis=1)


def _gaussian_kl(
    mu_q: jnp.ndarray,
    logvar_q: jnp.ndarray,
    mu_p: jnp.ndarray,
    logvar_p: jnp.ndarray,
) -> jnp.ndarray:
    """KL(q||p) for diagonal Gaussians."""
    var_ratio = jnp.exp(logvar_q - logvar_p)
    sq = (mu_p - mu_q) ** 2 * jnp.exp(-logvar_p)
    return 0.5 * (var_ratio + sq - 1.0 + logvar_p - logvar_q).sum(axis=-1)


class TrainState(train_state.TrainState):
    """Container for model params and optimizer state."""


def _load_dataset(config: ILPretrainConfig) -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(config.dataset_path)
    if "observations" not in raw or "actions" not in raw:
        raise ValueError("Dataset must contain 'observations' and 'actions' arrays.")
    observations = raw["observations"]
    actions = raw["actions"]
    if observations.ndim != 5 or actions.ndim != 3:
        raise ValueError("Expected observations [N,T,H,W,C] and actions [N,T,A].")
    if observations.shape[0] != actions.shape[0] or observations.shape[1] != actions.shape[1]:
        raise ValueError("Observation/action episode length mismatch.")
    observations = observations.astype(np.float32)
    if observations.max() > 1.0:
        observations /= 255.0
    return observations, actions.astype(np.float32)


def _iter_minibatches(
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
):
    n = observations.shape[0]
    order = rng.permutation(n)
    for i in range(0, n, batch_size):
        idx = order[i : i + batch_size]
        yield observations[idx], actions[idx]


def run_il_pretraining(config: ILPretrainConfig) -> None:
    """Train action-conditioned world model for imitation prediction."""
    print("[IL] Starting JAX world-model imitation pretraining:")
    print(config)

    observations, actions = _load_dataset(config)
    _, horizon, h, w, c = observations.shape
    action_dim = actions.shape[-1]
    if horizon < config.chunk_size:
        raise ValueError(
            f"chunk_size={config.chunk_size} > dataset horizon={horizon}. "
            "Use shorter chunk_size or longer sequences."
        )
    observations = observations[:, : config.chunk_size]
    actions = actions[:, : config.chunk_size]

    model = RSSMWorldModel(
        latent_dim=config.latent_dim,
        deter_dim=config.deter_dim,
        action_dim=action_dim,
        image_shape=(h, w, c),
    )

    rng = jax.random.PRNGKey(config.seed)
    init_obs = jnp.asarray(observations[:1])
    init_act = jnp.asarray(actions[:1])
    params = model.init(rng, init_obs, init_act, rng)["params"]
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(config.learning_rate),
    )

    @jax.jit
    def train_step(
        state: TrainState,
        batch_obs: jnp.ndarray,
        batch_actions: jnp.ndarray,
        step_key: jax.Array,
    ) -> tuple[TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def loss_fn(p):
            recons, kl_loss = model.apply({"params": p}, batch_obs, batch_actions, step_key)
            recon_loss = jnp.mean((recons - batch_obs) ** 2)
            loss = recon_loss + config.kl_scale * kl_loss
            return loss, (recon_loss, kl_loss)

        (loss, (recon_loss, kl_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, recon_loss, kl_loss

    np_rng = np.random.default_rng(config.seed)
    step = 0
    for epoch in range(config.epochs):
        epoch_losses = []
        for batch_obs, batch_actions in _iter_minibatches(
            observations, actions, config.batch_size, np_rng
        ):
            rng, step_key = jax.random.split(rng)
            state, loss, recon_loss, kl_loss = train_step(
                state,
                jnp.asarray(batch_obs),
                jnp.asarray(batch_actions),
                step_key,
            )
            epoch_losses.append((float(loss), float(recon_loss), float(kl_loss)))
            step += 1
        mean_loss = np.mean([x[0] for x in epoch_losses])
        mean_recon = np.mean([x[1] for x in epoch_losses])
        mean_kl = np.mean([x[2] for x in epoch_losses])
        print(
            f"[IL][epoch {epoch+1:03d}] "
            f"loss={mean_loss:.6f} recon={mean_recon:.6f} kl={mean_kl:.6f}"
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / config.save_name
    ckpt_bytes = serialization.to_bytes(state.params)
    ckpt_path.write_bytes(ckpt_bytes)
    with (output_dir / "config.txt").open("w", encoding="utf-8") as f:
        f.write(str(asdict(config)))
    print(f"[IL] Saved world model checkpoint to: {ckpt_path}")
    print("[IL] Training complete.")


def load_world_model_checkpoint(
    ckpt_path: str,
    image_shape: tuple[int, int, int],
    action_dim: int,
    latent_dim: int,
    deter_dim: int,
) -> tuple[RSSMWorldModel, dict]:
    """Load world model params from a msgpack checkpoint."""
    model = RSSMWorldModel(
        latent_dim=latent_dim,
        deter_dim=deter_dim,
        action_dim=action_dim,
        image_shape=image_shape,
    )
    dummy_obs = jnp.zeros((1, 1, *image_shape), dtype=jnp.float32)
    dummy_act = jnp.zeros((1, 1, action_dim), dtype=jnp.float32)
    init_key = jax.random.PRNGKey(0)
    params_template = model.init(init_key, dummy_obs, dummy_act, init_key)["params"]
    ckpt_bytes = Path(ckpt_path).read_bytes()
    params = serialization.from_bytes(params_template, ckpt_bytes)
    return model, params


def predict_future_frames(
    model: RSSMWorldModel,
    params: dict,
    init_observation: np.ndarray,
    action_seq: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Predict future frames from one frame + action chunk.

    init_observation: [B, H, W, C], uint8 or float32
    action_seq: [B, T, A], float32
    returns: [B, T, H, W, C], float32 in [0, 1]
    """
    obs = init_observation.astype(np.float32)
    if obs.max() > 1.0:
        obs /= 255.0
    acts = action_seq.astype(np.float32)
    rng = jax.random.PRNGKey(seed)
    preds = model.apply(
        {"params": params},
        jnp.asarray(obs),
        jnp.asarray(acts),
        rng,
        method=RSSMWorldModel.rollout_predict,
    )
    return np.array(preds)
