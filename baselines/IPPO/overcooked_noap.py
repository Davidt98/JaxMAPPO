""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
import wandb
from flax import core, struct
from jax.experimental import io_callback as io_callback

import matplotlib.pyplot as plt


class ShapedRewardCoeffManager(struct.PyTreeNode):
    shaped_reward_coeff: float = 1.0

    @classmethod
    def create(cls, shaped_reward_coeff: float = 1.0):
        return cls(shaped_reward_coeff=shaped_reward_coeff)
    
    def set_new_shaped_reward_coeff(self, new_coeff):
        return self.replace(
            shaped_reward_coeff = new_coeff
        )
    

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        other_action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        other_action = distrax.Categorical(logits=other_action_logits)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, other_action, jnp.squeeze(critic, axis=-1)
    

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    shaped_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    other_action_pred: jnp.ndarray

def get_rollout(train_state, config, layout):
    env = jaxmarl.make(config["ENV_NAME"], **layout)
    # env_params = env.default_params
    # env = LogWrapper(env)

    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x) # initializes the network's weights and biases based on the provided key and observation

    # print("222:", train_state)
    network_params = train_state[0].params # initialize with pre-trained parameters

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        # breakpoint()
        obs = {k: v.flatten() for k, v in obs.items()} # converts multi-dim obs arrays into 1D arrays suitable for feeding into the NN
        # action distributions pi_0, pi_1
        pi_0, noap, _ = network.apply(network_params, obs["agent_0"])
        pi_1, noap, _ = network.apply(network_params, obs["agent_1"])

        # choose one action from action dist with random key
        actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}
        # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
        # env_act = {k: v.flatten() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"] # checks if all agents are done

        state_seq.append(state)

    return state_seq

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config, layout):
    env = jaxmarl.make(config["ENV_NAME"], **layout)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape) # initialization of zero vector with shape of env obs space
        
        init_x = init_x.flatten() # ensures that the input shape matches the expected input shape of the neural network
        
        network_params = network.init(_rng, init_x) # initialize parameters of NN
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        # this object will be used during training to update the network parameters    
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        shapedRewardState = ShapedRewardCoeffManager.create(
            shaped_reward_coeff=1.0
        )
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, shapedRewardState, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                # takes the observations of all agents and stacks them into a single tensor
                # with a shape that matches the expected input shape of the neural network
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                pi, noap, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                # reshapes the batched actions back into individual actions for each agent.
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                
                env_act = {k:v.flatten() for k,v in env_act.items()}
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    info["shaped_reward"], 
                    log_prob,
                    obs_batch,
                    info,
                    noap
                    
                )
                runner_state = (train_state, shapedRewardState, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            # also need last_world_state because TD requires last and current state
            # Temporal Difference (TD) (error): δt​=rt​+γV(st+1​)−V(st​)
            # Generalized Advantage Estimation (GAE): Uses TD to calculate advantage: At​=∑l=0∞​(γλ)lδt+l
            # PPO: LCLIP(θ)=Et​[min(rt​(θ)A^t​,clip(rt​(θ),1−ϵ,1+ϵ)A^t​)]..
            train_state, shapedRewardState, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val, shaped_reward_coeff):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value, shaped_reward_coeff = gae_and_next_value
                    scaled_shaped_reward = shaped_reward_coeff * transition.shaped_reward 

                    # jax.debug.print("shaped {a}", a=shaped_reward_coeff.mean())
                    shaped_reward = transition.reward + scaled_shaped_reward
                    # shaped_reward = transition.reward # only base reward (for comparison)
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        shaped_reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value, shaped_reward_coeff), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, shaped_reward_coeff),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, shapedRewardState.shaped_reward_coeff)
            
            # UPDATE NETWORK
            # Epoch: cycle where all agents have interacted with the environment for a certain number of steps (can include multiple episodes)
            def _update_epoch(update_state, unused):
                # "Minibatches can be formed from the experiences of all agents.
                # Each agent's policy is updated using its own or possibly shared experiences"
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        # applies current network params to obs to get policy (action) dist and value estimates
                        pi, noap, value = network.apply(params, traj_batch.obs)
                        # computes log probs of the actions taken in the traj under current policy
                        log_prob = pi.log_prob(traj_batch.action)

                        other_action_pred_loss_weight = 9.0e-10
                        log_softmax_preds = jax.nn.log_softmax(noap.logits)
                        other_action_prediction_loss = jnp.sum(traj_batch.other_action_pred.logits * log_softmax_preds)

                        # CALCULATE VALUE LOSS
                        # Computes the value loss, which helps to update the value function
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets) # ensures value update change not too drastically
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob) # The probability ratio between the new and old policies
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8) # normalized gae (between 0-1)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        total_loss_actor = loss_actor + other_action_pred_loss_weight * other_action_prediction_loss
                        entropy = pi.entropy().mean()

                        total_loss = (
                            total_loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info

            rng = update_state[-1]


            def callback(metric, sr, traj_batch):
                wandb.log({
                    # # [jnp.array([13, 27, 41, 55,69,83,97,111]), :].mean(),
                    "total_rewards": sr.shaped_reward_coeff*traj_batch.shaped_reward.sum(axis=0).mean(axis=-1) + traj_batch.reward.sum(axis=0).mean(axis=-1),
                    "shaped_coefficient": sr.shaped_reward_coeff,
                    "env_step": metric["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"],
                    "scaled_shaped_reward": sr.shaped_reward_coeff*traj_batch.shaped_reward.sum(axis=0).mean(),
                    "scaled_reward": traj_batch.shaped_reward.sum(axis=0).mean(),
                    "base_reward": traj_batch.reward.sum(axis=0).mean(axis=-1), #[jnp.array([13, 27, 41, 55,69,83,97,111]), :].mean(),
                    # [jnp.array([13, 27, 41, 55,69,83,97,111]), :].mean(),
                    # "final_reward_per_game_estimate": traj_batch.reward.sum(axis=0).mean(axis=-1) / (traj_batch.reward.shape[0]//14),
                })
            metric["update_steps"] = update_steps
            # metric["shaped_reward"] = shapedRewardState.shaped_reward_coeff*traj_batch.shaped_reward.sum(axis=0).mean(axis=-1) + traj_batch.reward.sum(axis=0).mean(axis=-1)
            
            io_callback(callback, None, metric, shapedRewardState, traj_batch)
            # hcb.id_tap(callback, metric)
            # jax.debug.callback(metric)
            update_steps = update_steps + 1
            # new_shaped_reward_coeff_value = 1.0
            new_shaped_reward_coeff_value = jnp.maximum(0.0, 1 - (update_steps * config["NUM_ENVS"] * config["NUM_STEPS"] / config["TOTAL_TIMESTEPS"]))# config["TOTAL_TIMESTEPS"]
            new_shaped_reward_coeff = jnp.full(
                shapedRewardState.shaped_reward_coeff.shape, fill_value=new_shaped_reward_coeff_value
            )

            # jax.debug.print(
            #     "Shaped reward coeff: {a}, real_dsteps: {b}, shaped_reward_steps: {c}",
            #     a=new_shaped_reward_coeff,
            #     b=update_steps * config["NUM_ENVS"] * config["NUM_STEPS"],
            #     c=840000
            # )

            # runner_state[1] is the training state object where the shaped reward coefficient is stored
            shapedRewardState = shapedRewardState.set_new_shaped_reward_coeff(new_coeff=new_shaped_reward_coeff)

            runner_state = (train_state, shapedRewardState, env_state, last_obs, rng)
            return (runner_state, update_steps), metric
            
            # callback_value = callback(metric, train_state)
            # runner_state = callback_value[0]
            
            # runner_state = (train_state, env_state, last_obs, rng)
            # return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, shapedRewardState, env_state, obsv, _rng)
        # run function iteratively (optimized and parallelized by jax)
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        # breakpoint()
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_overcooked")
def main(config):
    config = OmegaConf.to_container(config)
    for layout in config["LAYOUTS"]:
        layout_name = layout
        layout = {'layout': overcooked_layouts[layout]}


        # WandB
        wandb.init(
            name = f"noap_{layout_name}_ippo",
            group="ippo",
            tags=[f"vanilla_{layout_name}", "no_sr", "final_report", "no_other_action_prediction", "vanilla"],
            entity=config["ENTITY"],
            project=config["PROJECT"],
            config=config,
            # Mode: Determines whether wandb will run online or offline
            mode=config["WANDB_MODE"],
            # mode="offline",
        )


        rng = jax.random.PRNGKey(50)
        # num_seeds = 5
        # with jax.disable_jit(False):
            # jax.vmap: enables running multiple env simultaneously by vectorizing make_train fnc
            # jax.jit: compiles vectorized training fnc into optimized machine code
        train_jit = jax.jit(make_train(config, layout))
        # rngs = jax.random.split(rng)#, num_seeds)
        out = train_jit(rng)
        
        print(f'** Saving Results for {layout_name}')
        filename = f'vanilla_{layout_name}'
        rewards = out["metrics"]["returned_episode_returns"].mean(-1)#.reshape((num_seeds, -1))
        reward_mean = rewards.mean(0)  # mean 
        # reward_std = rewards.std(0) / np.sqrt(num_seeds)  # standard error

        plt.plot(reward_mean)
        # plt.fill_between(range(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
        # compute standard error
        plt.xlabel("Update Step")
        plt.ylabel("Return")
        plt.savefig(f'{filename}.png')

        # animate first seed
        # train_state = jax.tree_map(lambda x: x[0], out["runner_state"])
        state_seq = get_rollout(out["runner_state"][0], config, layout)
        viz = OvercookedVisualizer()
        # agent_view_size is hardcoded as it determines the padding around the layout.
        viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

        # for step in range(len(reward_mean)):
        #     wandb.log({"Return_mean": reward_mean[step], "Return_std": reward_std[step]}, step=step)
        # wandb.log({"animation": wandb.Video(f"{filename}.gif", fps=4)}, step=len(reward_mean))
        wandb.finish()

if __name__ == "__main__":
    main()