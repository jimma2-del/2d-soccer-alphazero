from soccer_env.game.FootballGame import State as GameState, Action as GameAction, \
    Settings, FootballGame
import chex, functools, jax, dataclasses
from jax import numpy as jnp

from core.types import StepMetadata

PLAYERS_PER_TEAM = 2
    # keep 1 or 2; 3 is starting to be too much

N_ACTIONS = (3*3*2) ** PLAYERS_PER_TEAM
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
    '''If current player is the right player, flip everything to make it look like left player
    Output: abs pos, abs vel, [ pos relative to player_i for player_i in own team ]
        Shape=(2 * (1 + 2*PLAYERS_PER_TEAM) + (1 + 2*PLAYERS_PER_TEAM - 1) * PLAYERS_PER_TEAM, 2)'''

    game_state = state.game_state

    pos = jax.lax.cond(state.cur_player_id == 0,
        lambda: jnp.vstack((
            game_state.ball_pos,
            game_state.left_player_pos,
            game_state.right_player_pos,
        )),
        lambda: jnp.vstack(( # flip left & right players if right team
            game_state.ball_pos,
            game_state.right_player_pos,
            game_state.left_player_pos,
        ))
    )

    # center pos around center of game area
    pos = pos - jnp.array(game._cached_consts.center, dtype=jnp.float32)

    # normalize so field bounds/goals are +- 1 (players can go slightly beyond)
    pos = pos / jnp.array(game.get_settings().field_size, dtype=jnp.float32) * 2

    vel = jax.lax.cond(state.cur_player_id == 0,
        lambda: jnp.vstack((
            game_state.ball_vel,
            game_state.left_player_vel,
            game_state.right_player_vel
        )),
        lambda: jnp.vstack(( # flip left & right players if right team
            game_state.ball_vel,
            game_state.right_player_vel,
            game_state.left_player_vel
        ))
    )

    # normalize; same normalization factor as position is a good heuristic for this env
    vel = vel / jnp.array(game.get_settings().field_size, dtype=jnp.float32) * 2

    # flip x-coord if on the right team
    pos, vel = jax.lax.cond(state.cur_player_id == 0,
        lambda: (pos, vel), 
        lambda: (pos.at[:, 1].multiply(-1), vel.at[:, 1].multiply(-1))
    )

    ### RELATIVE POSITIONS ### 
    comb = jnp.vstack((pos, vel))

    for i in range(PLAYERS_PER_TEAM):
        other_obj_pos = jnp.delete(pos, 1 + i, axis=0) # objs other than the current player
        comb = jnp.vstack((comb, other_obj_pos - pos[1 + i]))

    # clip; shouldn't be necessary, but good to have in case of extreme values/bugs
    comb = jnp.clip(comb, -2, 2)

    return comb

#static consts for transform fn
flip_y_transform_fn_action_idxs = jnp.arange(N_ACTIONS)
flip_y_transform_fn_action_new_idxs = flip_y_transform_fn_action_idxs

for i in range(PLAYERS_PER_TEAM):
    move_up_mask = (flip_y_transform_fn_action_idxs // 18**i) % 3 == 0
    move_down_mask = (flip_y_transform_fn_action_idxs // 18**i) % 3 == 2

    flip_y_transform_fn_action_new_idxs = flip_y_transform_fn_action_new_idxs \
        .at[move_up_mask].set(flip_y_transform_fn_action_new_idxs[move_down_mask]) \
        .at[move_down_mask].set(flip_y_transform_fn_action_new_idxs[move_up_mask]) # swap    

def flip_y_transform_fn(mask, policy, state):
    # flip observation y-cooridinate 
    height = game._cached_consts.window_size[0]
    game_state = state.game_state

    flip_pos_y = lambda pos: pos.at[0].set(height - pos[0])

    new_game_state = state.game_state.replace(
        left_player_pos = flip_pos_y(game_state.left_player_pos),
        right_player_pos = flip_pos_y(game_state.right_player_pos.at),
        ball_pos = flip_pos_y(game_state.ball_pos),

        left_player_vel = game_state.left_player_vel.at[0].multiply(-1),
        right_player_vel = game_state.right_player_vel.at[0].multiply(-1),
        ball_vel = game_state.ball_vel.at[0].multiply(-1)
    )

    # remap action: swap move up and move down

    # dynamic: swap policy elements to new indices
    policy = jnp.empty_like(policy).at[flip_y_transform_fn_action_new_idxs].set(policy)

    return mask, policy, state.replace(game_state=new_game_state)
    
transforms = [flip_y_transform_fn]    