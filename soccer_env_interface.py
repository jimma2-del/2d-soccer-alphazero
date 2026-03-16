from soccer_env.game.FootballGame import State as GameState, Action as GameAction, \
    Settings, FootballGame
import chex, functools, jax, dataclasses
from jax import numpy as jnp

from core.types import StepMetadata

PLAYERS_PER_TEAM = 2

N_ACTIONS = 3*3*2 * PLAYERS_PER_TEAM
ACTION_MASK = jnp.ones((N_ACTIONS), dtype=jnp.bool)

def nn_output_to_game_action(action):
    move = jnp.zeros((PLAYERS_PER_TEAM, 2), dtype=jnp.int32)
    kick = jnp.zeros((PLAYERS_PER_TEAM), dtype=jnp.int32)

    for i in range(PLAYERS_PER_TEAM):
        # move axis 0 (3 options: -1, 0, 1)
        # move axis 1 (3 options: -1, 0, 1)
        # kick (2 options: 0, 1)

        move = move.at[i,0].set(action % 3 - 1)
        action //= 3

        move = move.at[i,1].set(action % 3 - 1)
        action //= 3

        kick = kick.at[i].set(action % 2)
        action //= 2

    return GameAction(move=move, kick=kick)

@chex.dataclass(frozen=True)
class State:
    game_state: GameState
    cur_player_id: jnp.number
    prev_action: jnp.number
    step: jnp.number

DT = 0.1
game = FootballGame(dt=DT, settings=Settings(players_per_team=PLAYERS_PER_TEAM))

def init_fn(key):
    game_state = game.reset()

    state = State(
        game_state=game_state,
        cur_player_id=jax.random.randint(key, shape=(), minval=0, maxval=2),
        prev_action=jnp.int32(0),
        step=jnp.int32(1)
    )

    return state, StepMetadata(
        rewards=jnp.zeros((2), dtype=jnp.int32),
        action_mask=ACTION_MASK,
        terminated=False,
        cur_player_id=state.cur_player_id,
        step = jnp.int32(1)
    )

def step_fn(state: State, action):
    actions = jnp.array([action, state.prev_action], dtype=jnp.int32)
    
    new_game_state, goal = game.step(
        state.game_state, 
        nn_output_to_game_action(actions[state.cur_player_id]), 
        nn_output_to_game_action(actions[1 - state.cur_player_id])
    )
    
    new_state = dataclasses.replace(state,
        game_state=new_game_state,
        prev_action=action,
        cur_player_id=1 - state.cur_player_id,
        step=state.step + 1
    )

    rewards=jnp.array((goal, -goal), dtype=jnp.int32)

    return new_state, StepMetadata(
        rewards=rewards,
        action_mask=ACTION_MASK,
        terminated=goal != 0,
        cur_player_id=new_state.cur_player_id,
        step = new_state.step
    )

def state_to_nn_input(state):
    width = game._cached_consts.window_size[1]
    game_state = state.game_state

    return jax.lax.cond(state.cur_player_id==0,
        lambda: jnp.vstack((
            game_state.ball_pos,
            game_state.ball_vel,
            game_state.left_player_pos,
            game_state.left_player_vel,
            game_state.right_player_pos,
            game_state.right_player_vel
        )),
        lambda: jnp.vstack((
            width - game_state.ball_pos,
            -game_state.ball_vel,
            width - game_state.right_player_pos,
            -game_state.right_player_vel,
            width - game_state.left_player_pos,
            -game_state.left_player_vel
        ))
    )