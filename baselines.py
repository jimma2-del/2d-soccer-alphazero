from core.evaluators.evaluation_fns import make_nn_eval_fn_no_params_callable

import jax.numpy as jnp
from soccer_env_interface import N_ACTIONS, state_to_nn_input, PLAYERS_PER_TEAM

def make_value_eval_fn(value_fn):
    def eval_fn(obs):
        return jnp.ones((1,N_ACTIONS)), value_fn(obs)
            # equal chance for each action; vanilla (ie. no policy net) MCTS

    return make_nn_eval_fn_no_params_callable(eval_fn, state_to_nn_input)

# NOTE: obs is "batched" (batch axis, [ball, *left_players, *right_players, ...], [y, x])

def get_ball_pos(obs):
    return obs[:, 0]

def get_self_player_pos(obs):
    return obs[:, 1 : 1 + PLAYERS_PER_TEAM]

def get_opp_player_pos(obs):
    return obs[:, 1 + PLAYERS_PER_TEAM : 1 + 2*PLAYERS_PER_TEAM]

### baseline heuristic value functions ###
    # normalized to [-1, 1] -> greatly overestimates value for the most part since it overexplains

SELF_GOAL_POS = jnp.array((0, -1), dtype=jnp.float32)
OPP_GOAL_POS = jnp.array((0, 1), dtype=jnp.float32)

def dist(a, b):
    return jnp.linalg.norm(a - b, axis=-1)

def ball_dist_to_goal_value(obs):
    '''value = dist to self goal - dist to opp goal'''
    
    ball_pos = get_ball_pos(obs)

    value = dist(ball_pos, SELF_GOAL_POS) - dist(ball_pos, OPP_GOAL_POS)
    
    return value / 2 # normalize range from [-2, 2]

MAX_DIST = jnp.linalg.norm(jnp.array((2, 2), dtype=jnp.float32))

def closest_player_dist_to_ball_value(obs):
    '''heuristic for which team is in control of the ball
    value = min(opp players dist to ball) - min(self players dist to ball)'''
    
    ball_pos = get_ball_pos(obs)
    self_player_pos = get_self_player_pos(obs)
    opp_player_pos = get_opp_player_pos(obs)

    value = jnp.min(dist(opp_player_pos, ball_pos), axis=-1) \
        - jnp.min(dist(self_player_pos, ball_pos), axis=-1)
    
    return value / MAX_DIST

def count_defenders_between_ball_and_goal(obs):
    '''Assigns one team as attackers and other team as defenders depending on
    who has control of the ball, then finds number of defenders between ball and goal.
    Additionally returns True if self team attacking and False if opp team attacking.'''

    self_team_attacking = closest_player_dist_to_ball_value(obs) >= 0

    ball_pos = get_ball_pos(obs)
    self_player_pos = get_self_player_pos(obs)
    opp_player_pos = get_opp_player_pos(obs)

    defenders = jnp.where(self_team_attacking, opp_player_pos, self_player_pos)
    goal = jnp.where(self_team_attacking, OPP_GOAL_POS, SELF_GOAL_POS)

    num_defenders_btw_goal = jnp.sum(dist(defenders, goal) < dist(ball_pos, goal), axis=1)

    return num_defenders_btw_goal, self_team_attacking

def defenders_between_ball_and_goal_value(obs):
    '''Value function based on count_defenders_between_ball_and_goal() with modiciations
    Increasing Returns: going from 1 to 0 defenders is much better than from 2 to 1.
    Signed: positive if self attacking and negative if opp attacking; gives bonus for controlling ball
    '''

    num_defenders_btw_goal, self_team_attacking = count_defenders_between_ball_and_goal(obs)
    
    DEGREE = 0.2 # adjustable parameter; lower -> lower values before spiking at x=0
    return (1 - (num_defenders_btw_goal/PLAYERS_PER_TEAM) ** DEGREE) * (2*self_team_attacking - 1)