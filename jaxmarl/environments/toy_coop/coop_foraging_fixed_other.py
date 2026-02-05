from jaxmarl.environments import MultiAgentEnv, spaces

from typing import Tuple, Dict
import chex
import jax
import jax.numpy as jnp
from functools import partial

from .toy_coop import State
from .coop_foraging import CoopForaging
from enum import IntEnum
from flax import struct


# class OtherAgentType(IntEnum):
#     leader_a = 0  # always goes towards first goal position
#     leader_b = 1  # always goes towards second goal position
#     follower = 2  # always follows the ego agent

@struct.dataclass
class FixedOtherState:
    coop_env_state: State 
    # 0: always goes towards first goal position
    # 1: always goes towards second goal position
    # 2: always follows the ego agent
    other_agent_type: int 

def leader_action(state: FixedOtherState, goal_idx: int, agent_idx: int=1) -> chex.Array:
    """
    A policy where the agent moves towards a goal location chosen at random.

    Args:
        key (chex.PRNGKey): Random key for selecting the goal location.
        state (State): The current state of the environment.

    Returns:
        chex.Array: The action chosen by the fixed policy.
    """
    # Randomly select a goal location (either from `goal_pos` or `other_goal_pos`)
    goal_location = state.goal_pos[goal_idx]

    # Get the current position of the fixed agent (agent_1)
    agent_pos = state.agent_pos[agent_idx]

    # Compute the direction to move towards the goal
    direction = goal_location - agent_pos

    # Map the direction to an action
    action = jnp.where(direction[0] > 0, 0,  # Move right
              jnp.where(direction[0] < 0, 2,  # Move left
              jnp.where(direction[1] > 0, 1,  # Move down
              jnp.where(direction[1] < 0, 3,  # Move up
              4))))  # Stay if already at the goal

    return action 

def follower_action(state: FixedOtherState, agent_idx: int=1) -> chex.Array:
    """
    A policy where the agent follows the ego agent.

    Args:
        state (State): The current state of the environment.

    Returns:
        chex.Array: The action chosen by the fixed policy.
    """
    # Get the positions of both agents
    follower_agent_pos = state.agent_pos[agent_idx]
    leader_agent_pos = state.agent_pos[(agent_idx + 1) % 2]

    # Compute the direction to move towards the ego agent
    direction = leader_agent_pos - follower_agent_pos

    # Map the direction to an action
    action = jnp.where(direction[0] > 0, 0,  # Move right
              jnp.where(direction[0] < 0, 2,  # Move left
              jnp.where(direction[1] > 0, 1,  # Move down
              jnp.where(direction[1] < 0, 3,  # Move up
              4))))  # Stay if already at the ego agent

    return action


class CoopForagingFixedOther(MultiAgentEnv):
    """
    A wrapper for the CoopForaging environment to make it a single-agent environment.
    The second agent operates under a fixed policy, specified to be either leader or follower
    """
    def __init__(self, **kwargs):
        """
        Args:
            env (CoopForaging): The cooperative foraging environment.
            fixed_policy (callable): A function that takes the environment state and returns the action for the second agent.
        """
        super().__init__(num_agents=1)  # Single-agent environment
        self.env = CoopForaging(**kwargs)
        self.agent_id = "agent_0"  # The trainable agent
        self.agents = ["agent_0"]  # List of agents in the environment
        self.fixed_policy_map = (
            lambda state: leader_action(state, goal_idx=0),
            lambda state: leader_action(state, goal_idx=1),
            follower_action
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, FixedOtherState]:
        """
        Reset the environment and return the initial observation for the trainable agent.
        """
        new_key, sub_key = jax.random.split(key)
        # Sample the other agent type, with equal prob of being follower/leader. 
        other_agent_type_idx = jax.random.choice(sub_key, jnp.array([0, 1, 2]), p=jnp.array([0.25, 0.25, 0.5]))
        obs, state = self.env.reset(new_key)
        fixed_other_agent_state = FixedOtherState(
            coop_env_state=state, 
            other_agent_type=other_agent_type_idx,
        )
        return obs[self.agent_id], fixed_other_agent_state
    

    def step(self, key: chex.PRNGKey, state: FixedOtherState, action: chex.Array) -> Tuple[chex.Array, FixedOtherState, float, bool, Dict]:
        """
        Take a step in the environment with the trainable agent's action and the fixed policy for the other agent.
        """
        # Get the action for the fixed agent
        # fixed_action = self.fixed_policy_map[state.other_agent_type](state.coop_env_state)
        fixed_action = jax.lax.switch(
            state.other_agent_type,
            self.fixed_policy_map,
            state.coop_env_state
        )
        # Combine actions for both agents
        actions = {
            "agent_0": action["agent_0"],
            "agent_1": fixed_action
        }

        # Step the environment
        obs, next_state, rewards, dones, infos = self.env.step_env(key, state.coop_env_state, actions)

        next_state = FixedOtherState(coop_env_state=next_state, other_agent_type=state.other_agent_type)
        # Return the observation, reward, and done flag for the trainable agent
        return obs["agent_0"], next_state, rewards["agent_0"], dones["agent_0"], infos

    @partial(jax.jit, static_argnums=[0])
    def is_terminal(self, state: FixedOtherState) -> bool:
        """Check if episode is done."""
        return self.env.is_terminal(state.coop_env_state)

    @property
    def name(self) -> str:
        """Environment name."""
        return "CoopForagingFixedOther"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.env.num_actions

    def action_space(self, agent_id: str = "") -> spaces.Discrete:
        """Action space of the environment."""
        return self.env.action_space(agent_id)

    def observation_space(self, agent_id: str = "") -> spaces.Box:
        """Observation space of the environment."""
        return self.env.observation_space(agent_id)