from soccer_env.game.FootballGame import State as GameState, Action as GameAction, \
    Settings, FootballGame
import chex, functools, jax, dataclasses
from jax import numpy as jnp

from core.types import StepMetadata

PLAYERS_PER_TEAM = 2

N_ACTIONS = 3*3*2 * PLAYERS_PER_TEAM
ACTION_MASK = jnp.ones((N_ACTIONS), dtype=jnp.bool)

def nn_output_to_game_action(action):
    move = jnp.zeros((PLAYERS_PER_TEAM, 2), dtype=jnp.int8)
    kick = jnp.zeros((PLAYERS_PER_TEAM), dtype=jnp.float32)

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
    cur_player_id: int
    prev_action: int
    step: int

DT = 0.1
game = FootballGame(dt=DT, settings=Settings(players_per_team=PLAYERS_PER_TEAM))

def init_fn(key):
    game_state = game.reset()

    state = State(
        game_state=game_state,
        cur_player_id=jax.random.randint(key, shape=(), minval=0, maxval=2),
        prev_action=0,
        step=1
    )

    return state, StepMetadata(
        rewards=0,
        action_mask=ACTION_MASK,
        terminated=False,
        cur_player_id=state.current_player,
        step = 1
    )

@functools.partial(jax.jit, static_argnames=('reverse'))
def game_step_reverse_actions(game_state, actions, reverse):
    if reverse:
        return game.step(game_state, actions[1], actions[0])
    else:
        return game.step(game_state, actions[0], actions[1])

def step_fn(state: State, action: GameAction):
    new_game_state, goal = game_step_reverse_actions(
        state.game_state, 
        (nn_output_to_game_action(action), nn_output_to_game_action(state.prev_action)), 
        state.cur_player_id == 1
    )
    
    new_state = dataclasses.replace(state,
        game_state=new_game_state,
        prev_action=action,
        cur_player_id=(state.cur_player_id + 1) % 2,
        step=state.step + 1
    )

    return new_state, StepMetadata(
        rewards=goal * (2*state.cur_player_id - 1),
        action_mask=ACTION_MASK,
        terminated=goal != 0,
        cur_player_id=new_state.cur_player_id,
        step = new_state.step
    )

@functools.partial(jax.jit, static_argnames=('reverse'))
def state_to_nn_input_reverse(game_state: GameState, reverse):
    if reverse:
        width = game._cached_consts.window_size[1]

        return jnp.vstack((
            width - game_state.ball_pos,
            -game_state.ball_vel,
            width - game_state.right_player_pos,
            -game_state.right_player_vel,
            width - game_state.left_player_pos,
            -game_state.left_player_vel
        ))
    
    return jnp.vstack((
        game_state.ball_pos,
        game_state.ball_vel,
        game_state.left_player_pos,
        game_state.left_player_vel,
        game_state.right_player_pos,
        game_state.right_player_vel
    ))

def state_to_nn_input(state):
    return state_to_nn_input_reverse(state.game_state, state.cur_player_id==1)