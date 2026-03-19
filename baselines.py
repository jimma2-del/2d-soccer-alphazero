from core.evaluators.evaluation_fns import make_nn_eval_fn_no_params_callable

import jax.numpy as jnp
from soccer_env_interface import N_ACTIONS, state_to_nn_input, PLAYERS_PER_TEAM

SELF_GOAL_POS = jnp.array((0, -1), dtype=jnp.float32)
OPP_GOAL_POS = jnp.array((0, 1), dtype=jnp.float32)

def ball_dist_to_goal_value(obs):
    '''value = dist to self goal - dist to opp goal'''
    
    # NOTE: obs is "batched" (batch axis, [pos, vel], [ball, *left_players, *right_players], [y, x])
    ball_pos = obs[:, 0, 0]
    
    return jnp.linalg.norm(ball_pos - SELF_GOAL_POS, axis=-1) - jnp.linalg.norm(ball_pos - OPP_GOAL_POS, axis=-1)
    
def ball_dist_to_goal_eval(obs):
    # NOTE: turbozero expects BATCHED return vals
    return jnp.ones((1,N_ACTIONS)), ball_dist_to_goal_value(obs)
        # equal chance for each action; vanilla (ie. no policy net) MCTS

ball_dist_to_goal_eval_fn = make_nn_eval_fn_no_params_callable(ball_dist_to_goal_eval, state_to_nn_input)

def closest_player_dist_to_ball_value(obs):
    '''value = min(opp players dist to ball) - min(self players dist to ball)'''
    
    # NOTE: obs is "batched" (batch axis, [pos, vel], [ball, *left_players, *right_players], [y, x])
    ball_pos = obs[:, 0, 0]
    self_player_pos = obs[:, 1:1 + PLAYERS_PER_TEAM, 0]
    opp_player_pos = obs[:, 1 + PLAYERS_PER_TEAM:, 0]

    value = jnp.min(jnp.linalg.norm(opp_player_pos - ball_pos, axis=-1), axis=-1) \
        - jnp.min(jnp.linalg.norm(self_player_pos - ball_pos, axis=-1), axis=-1)
    
    return value / PLAYERS_PER_TEAM

PLAYER_BALL_DIST_VALUE_WEIGHT = 0.5
    # relative to ball dist to goal, which has weight=1

def player_ball_goal_dist_eval(obs):
    # NOTE: turbozero expects BATCHED return vals
    value = ball_dist_to_goal_value(obs) + PLAYER_BALL_DIST_VALUE_WEIGHT*closest_player_dist_to_ball_value(obs)
    return jnp.ones((1,N_ACTIONS)), value / (1 + PLAYER_BALL_DIST_VALUE_WEIGHT)
        # equal chance for each action; vanilla (ie. no policy net) MCTS

player_ball_goal_dist_eval_fn = make_nn_eval_fn_no_params_callable(player_ball_goal_dist_eval, state_to_nn_input)