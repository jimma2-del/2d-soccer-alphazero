from networks import SimpleResNetMLP

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS

from core.memory.replay_memory import EpisodeReplayBuffer

from functools import partial
from core.testing.two_player_baseline import TwoPlayerBaseline
from core.testing.two_player_tester import TwoPlayerTester
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
import optax

from os import path
import jax

from soccer_env_interface import state_to_nn_input, step_fn, init_fn, N_ACTIONS, transforms
from baselines import make_value_eval_fn, \
    ball_dist_to_goal_value, closest_player_dist_to_ball_value, defenders_between_ball_and_goal_value

resnet = SimpleResNetMLP(
    policy_head_out_size=N_ACTIONS,
    num_blocks=2,
    hidden_dim=128,
)

# alphazero can take an arbirary search `backend`
# here we use classic MCTS
def make_az_evaluator(eval_fn, testing=False):
    NUM_ITERATIONS = 64,#256,#32,
    MAX_NODES = 65,#257,#40,

    NUM_ITERATIONS_TESTING = 128,#512,#64,
    MAX_NODES_TESTING = 129,#513,#80,
    
    return AlphaZero(MCTS)(
        eval_fn = eval_fn,
        num_iterations = NUM_ITERATIONS_TESTING if testing else NUM_ITERATIONS,
        max_nodes = MAX_NODES_TESTING if testing else MAX_NODES,
        branching_factor = N_ACTIONS,
        action_selector = PUCTSelector(),
        temperature = 0.0 if testing else 1.0
    )
    
az_evaluator = make_az_evaluator(make_nn_eval_fn(resnet, state_to_nn_input), testing=False)
az_evaluator_test = make_az_evaluator(make_nn_eval_fn(resnet, state_to_nn_input), testing=True)

# baselines
def heuristic_value(obs):
    ball_goal = ball_dist_to_goal_value(obs)
    player_ball = closest_player_dist_to_ball_value(obs)
    defenders = defenders_between_ball_and_goal_value(obs)

    return 0.5 * defenders + 0.1 * ball_goal**3 + 0.02 * player_ball**3
        # ** n -> increasing returns

player_ball_goal_dist_evaluator = make_az_evaluator(make_value_eval_fn(heuristic_value), testing=True)

replay_memory = EpisodeReplayBuffer(capacity=2000)#1000)

trainer = Trainer(
    batch_size = 1024,
    train_batch_size = 4096,
    warmup_steps = 0,
    collection_steps_per_epoch = 512,#256,
    train_steps_per_epoch = 128,#64,
    nn = resnet,
    loss_fn = partial(az_default_loss_fn, l2_reg_lambda = 0.0),
    optimizer = optax.adam(1e-3),
    evaluator = az_evaluator,
    memory_buffer = replay_memory,
    max_episode_steps = 512,#600,#80, # avoid possible infinite loops?
    env_step_fn = step_fn,
    env_init_fn = init_fn,
    state_to_nn_input_fn=state_to_nn_input,
    testers = [
        TwoPlayerBaseline(num_episodes=64, baseline_evaluator=player_ball_goal_dist_evaluator, 
                          #render_fn=render_fn, render_dir='.', 
                          name='player_ball_goal_dist'),
    ],
    #testers = [],
    #testers=[TwoPlayerTester(num_episodes=64)],
    evaluator_test = az_evaluator_test,
    data_transform_fns=transforms,
    #wandb_project_name = 'turbozero-soccer' ,
    ckpt_dir = path.join(path.dirname(path.realpath(__file__)), "tmp", "checkpoints")
)

output = trainer.train_loop(seed=0, num_epochs=80, eval_every=5)