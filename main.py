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

resnet = SimpleResNetMLP(
    policy_head_out_size=N_ACTIONS,
    num_blocks=2,
    hidden_dim=128,
)

# alphazero can take an arbirary search `backend`
# here we use classic MCTS
az_evaluator = AlphaZero(MCTS)(
    eval_fn = make_nn_eval_fn(resnet, state_to_nn_input),
    num_iterations = 64,#256,#32,
    max_nodes = 65,#257,#40,
    branching_factor = N_ACTIONS,
    action_selector = PUCTSelector(),
    temperature = 1.0
)

az_evaluator_test = AlphaZero(MCTS)(
    eval_fn = make_nn_eval_fn(resnet, state_to_nn_input),
    num_iterations = 128,#512,#64,
    max_nodes = 129,#513,#80,
    branching_factor = N_ACTIONS,
    action_selector = PUCTSelector(),
    temperature = 0.0
)

replay_memory = EpisodeReplayBuffer(capacity=1000)

trainer = Trainer(
    batch_size = 1024,
    train_batch_size = 4096,
    warmup_steps = 0,
    collection_steps_per_epoch = 512,#256,
    train_steps_per_epoch = 64,
    nn = resnet,
    loss_fn = partial(az_default_loss_fn, l2_reg_lambda = 0.0),
    optimizer = optax.adam(1e-3),
    evaluator = az_evaluator,
    memory_buffer = replay_memory,
    max_episode_steps = 512,#600,#80, # avoid possible infinite loops?
    env_step_fn = step_fn,
    env_init_fn = init_fn,
    state_to_nn_input_fn=state_to_nn_input,
    # testers = [
    #     TwoPlayerBaseline(num_episodes=128, baseline_evaluator=baseline_az, render_fn=render_fn, render_dir='.', name='pretrained'),
    #     TwoPlayerBaseline(num_episodes=128, baseline_evaluator=greedy_az, render_fn=render_fn, render_dir='.', name='greedy'),
    # ],
    testers = [],
    #testers=[TwoPlayerTester(num_episodes=64)],
    evaluator_test = az_evaluator_test,
    data_transform_fns=transforms,
    wandb_project_name = 'turbozero-soccer' ,
    ckpt_dir = path.join(path.dirname(path.realpath(__file__)), "tmp", "checkpoints")
)

output = trainer.train_loop(seed=0, num_epochs=80, eval_every=5)