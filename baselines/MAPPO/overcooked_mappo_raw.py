import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, JaxMARLWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
#from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
import wandb


import matplotlib.pyplot as plt

class OvercookedWorldStateWrapper(JaxMARLWrapper):
    # def __init__(self, env):
    #     self.env = env
    #     self.world_state_size =  # Define the size of the world state based on the environment

    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs, env_state)
        return obs, env_state

    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state(obs, env_state)
        return obs, env_state, reward, done, info

    def world_state(self, obs, env_state):
        """
        Compute the world state based on the observations and environment state.
        """
        # for agent in self._env.agents:
        #     print(f"Observation for {agent}: {obs[agent]}")
        world_state = jnp.concatenate([(obs[agent]) for agent in self._env.agents], axis=-1)
        world_state_inverse = jnp.concatenate([obs[agent] for agent in reversed(self._env.agents)], axis=-1)

        # world_state = {for state in world_state}
        return world_state#, world_state_inverse

    # def world_state_size(self, obs):
    #     # spaces = [self._env.observation_space(agent) for agent in self._env.agents]
    #     # return sum([space.shape[-1] for space in spaces])
    #     world_state = jnp.concatenate([(obs[agent]) for agent in self._env.agents], axis=-1)

    #     return (world_state.shape[0],world_state.shape[1],world_state.shape[2]) 

    # def world_state_size(self):
    #     # spaces = [self._env.observation_space(agent) for agent in self._env.agents]
    #     # return sum([space.shape[-1] for space in spaces])
        
    #     return (4,5,52) # NOTE: hardcoded

# decentralized
class ActorFF(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=action_logits)

        return pi

# centralized
class CriticFF(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, x):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(critic)
            
        return jnp.squeeze(critic, axis=-1) # NOTE nochmal anschauen


class Transition(NamedTuple):
    global_done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray

def get_rollout(train_state, config, layout):
    env = jaxmarl.make(config["ENV_NAME"], **layout)
    # env_params = env.default_params
    # env = LogWrapper(env)

    network_actor = ActorFF(env.action_space().n, config)
    # network_critic = CriticFF(config) # critic not needed, rollout in this case just to simulate trained agents
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network_actor.init(key_a, init_x) # initializes the network's weights and biases based on the provided key and observation
    # network_critic.init(key_a, init_x) # initialize with pre-trained parameters
    network_params = train_state.params

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        obs = {k: v.flatten() for k, v in obs.items()} # converts multi-dim obs arrays into 1D arrays suitable for feeding into the NN

        pi_0 = network_actor.apply(network_params, obs["agent_0"])
        pi_1 = network_actor.apply(network_params, obs["agent_1"])

        # choose one action from action dist with random key
        actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}
        # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
        # env_act = {k: v.flatten() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"] # checks if all agents are done

        state_seq.append(state)
    
    # Evaluate state values using the critic network
    # state_values = []
    # for state in state_seq:
    #     state_value = network_critic.apply(network_params, state)
    #     state_values.append(state_value)

    return state_seq#, state_values

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
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    # 'wraps' the original env with the methods provided in the OvercookedWorldStateWrapper class
    # agents share common world state
    # centralizes e.g. rewards to encourage more cooperative behavior among agents
    env = OvercookedWorldStateWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorFF(env.action_space().n, config)
        critic_network = CriticFF(config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = jnp.zeros(env.observation_space().shape)

        ac_init_x = ac_init_x.flatten()

        actor_network_params = actor_network.init(_rng_actor, ac_init_x)
        
        # world_size = jnp.zeros(env.world_state_size())
        world_size = env.observation_space().shape
        new_world_size = (world_size[0], world_size[1], world_size[2] * env.num_agents)
        world_size = jnp.zeros(new_world_size)
        print(world_size)
        cr_init_x = world_size.flatten()
        # cr_init_x = jnp.zeros((26,)) # NOTE hardcoded >:( nochmal anschauen..
                
        critic_network_params = critic_network.init(_rng_critic, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                pi = actor_network.apply(train_states[0].params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                
                env_act = {k:v.flatten() for k,v in env_act.items()}

                # VALUE
                world_state= last_obs["world_state"]#.swapaxes(0,1)
                print("last_obs", last_obs)
                print("1", world_state)
                # world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                # world_state_batch = batchify(world_state)
                # print(world_state)

                # (stacking world_state, world_state_inv) resulted in worse results
                world_state_batch = jnp.stack([world_state,world_state])#.swapaxes(0,1)

                # print("2", world_state_batch)
                # print(world_state_batch)
                # print(type(world_state_batch))
                # obs space example (cramped room): ShapedArray(uint8[16,4,5,52])
                # 16: Different features such as agent, object positions, order status, etc.
                # 4: height of room
                # 5: width of room
                # 52 (2*26: hardcoded): boolean flags (presence of agents, onions, plates, pots, etc.)
                x_reshape = 2 * world_state.shape[0]
                y_reshape = world_state.shape[1] * world_state.shape[2] * world_state.shape[3]
                world_state_batch = world_state_batch.reshape((x_reshape, y_reshape))
                print("2:", world_state_batch)
                value = critic_network.apply(train_states[1].params, world_state_batch)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, env_state, reward, global_done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                # Assuming done is currently defined as a dictionary
                # done_2 = {key: value for key, value in global_done.items() if key != '__all__'} # NOTE find better solution

                transition = Transition(
                    # jnp.tile(global_done["__all__"], env.num_agents),
                    batchify(global_done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    world_state_batch,
                    info
                )
                runner_state = (train_states, env_state, obsv, rng)
                return runner_state, transition
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, rng = runner_state
            last_world_state = last_obs["world_state"]#.swapaxes(0,1)
            # last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            world_state_batch = jnp.stack([last_world_state,last_world_state])#.swapaxes(0,1)
            x_reshape = 2 * last_world_state.shape[0]
            y_reshape = last_world_state.shape[1] * last_world_state.shape[2] * last_world_state.shape[3]
            world_state_batch = world_state_batch.reshape((x_reshape, y_reshape))
            last_val = critic_network.apply(train_states[1].params, world_state_batch)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, traj_batch, gae):
                        # RERUN NETWORK
                        pi = actor_network.apply(actor_params,traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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
                        entropy = pi.entropy().mean()
                        
                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        
                        actor_loss = (
                            loss_actor
                            - config["ENT_COEF"] * entropy
                        )
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)
                    
                    def _critic_loss_fn(critic_params, traj_batch, targets):
                        # RERUN NETWORK
                        value = critic_network.apply(critic_params, traj_batch.world_state) 
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)
                    
                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, traj_batch, targets
                    )
                    
                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    
                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }
                    
                    return (actor_train_state, critic_train_state), loss_info

                train_states, traj_batch, advantages, targets, rng = update_state
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
                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info
            
            update_state = (
                train_states,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )            
            train_states = update_state[0]
            metric = traj_batch.info
            loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
            metric["loss"] = loss_info
            rng = update_state[-1]

            runner_state = (train_states, env_state, last_obs, rng)
            return runner_state, metric
        
        rng, _rng = jax.random.split(rng)
        runner_state = ((actor_train_state, critic_train_state), env_state, obsv, _rng)
        # runner_state = (actor_train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

@hydra.main(version_base=None, config_path="config", config_name="mappo_ff_overcooked")
def main(config):
    config = OmegaConf.to_container(config)
    for layout in config["LAYOUTS"]:
        layout_dict = {'layout': overcooked_layouts[layout]}
        


        # WandB
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=["IPPO", "FF", config["ENV_NAME"]],
            config=config,
            # Mode: Determines whether wandb will run online or offline
            mode=config["WANDB_MODE"],
            # mode="offline",
        )


        rng = jax.random.PRNGKey(30)
        # num_seeds = 20
        num_seeds = 5
        # with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config, layout_dict)))
        rngs = jax.random.split(rng, num_seeds)
        out = train_jit(rngs)
        print(f'** Saving Results for {layout}**')
        filename = f'mappo_{config["ENV_NAME"]}_{layout}'
        rewards = out["metrics"]["returned_episode_returns"].mean(-1).reshape((num_seeds, -1))
        reward_mean = rewards.mean(0)  # mean 
        reward_std = rewards.std(0) / np.sqrt(num_seeds)  # standard error
        
        plt.figure()  # Create a new figure for each layout
        plt.plot(reward_mean)
        plt.fill_between(range(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
        # compute standard error
        plt.xlabel("Update Step")
        plt.ylabel("Return")
        plt.savefig(f'{filename}.png')

        # animate first seed
        train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0][0])
        state_seq = get_rollout(train_state, config, layout_dict)
        viz = OvercookedVisualizer()
        # agent_view_size is hardcoded as it determines the padding around the layout.
        viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")



        for step in range(len(reward_mean)):
            wandb.log({"Return_mean": reward_mean[step], "Return_std": reward_std[step]}, step=step)
        wandb.log({"animation": wandb.Video(f"{filename}.gif", fps=4)}, step=len(reward_mean))
    wandb.finish()

if __name__ == "__main__":
    main()