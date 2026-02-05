from collections import OrderedDict
from enum import IntEnum

import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from flax import struct
from functools import partial
from .toy_coop import ToyCoop, State
import pdb

class CoopForaging(ToyCoop):
    """
    Coorperative Foraging environment with modified reward structure, 
    Agents must reach the same goal in order to maximize reward.
    NOTE: currently only implementing for 1 set of goal locations (state.goal_pos, green goals)
    """
    
    @partial(jax.jit, static_argnums=[0])
    def step_agents(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: chex.Array,
    ) -> Tuple[State, float]:
        """Update agent positions and calculate rewards."""
        # Calculate next positions
        next_pos = state.agent_pos + self.action_to_dir[actions]
        
        # Bound positions to grid
        next_pos = jnp.clip(next_pos, 0, self.width - 1)
        
        # Check if positions would collide
        # would_collide = jnp.all(next_pos[0] == next_pos[1])
        # next_pos = jnp.where(would_collide, state.agent_pos, next_pos)
        
        # Modified reward calculation
        on_goal = lambda x, y: jnp.all(x == y)
        
        # Check which goal each agent is on (if any)
        agent0_green_goal = jax.vmap(on_goal, in_axes=(None, 0))(next_pos[0], state.goal_pos)
        agent1_green_goal = jax.vmap(on_goal, in_axes=(None, 0))(next_pos[1], state.goal_pos)
        
        # Reward agents only if they are on the same goal
        both_on_same_green_goal = jnp.any(jnp.logical_and(agent0_green_goal, agent1_green_goal))
        
        # Reward is 3 if both agents are on the same goal
        reward = 3 * (both_on_same_green_goal).astype(jnp.float32)  
        return state.replace(agent_pos=next_pos), reward - 1  # step cost