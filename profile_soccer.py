import jax
import jax.numpy as jnp
import time
from typing import Callable

from soccer_env_interface import state_to_nn_input, step_fn, init_fn, N_ACTIONS

def profile(
    num_steps: int = 1000, 
    batch_size: int = 1024
):
    def rollout_step(carry, unused_input):
        state, key = carry
        key, action_key = jax.random.split(key)
        action = jax.random.randint(action_key, (), 0, N_ACTIONS)
        next_state, metadata = step_fn(state, action)
        key, reset_key = jax.random.split(key)
        return (jax.lax.cond(metadata.terminated, lambda: init_fn(reset_key)[0], lambda: next_state), key), None

    @jax.vmap
    def run_trajectory(key):
        init_key, loop_key = jax.random.split(key)
        initial_state, metadata = init_fn(init_key)
        # Scan over num_steps
        (final_state, _), _ = jax.lax.scan(
            rollout_step, 
            (initial_state, loop_key), 
            None, 
            length=num_steps
        )
        return final_state

    key = jax.random.PRNGKey(42)
    
    # 4. Warm-up
    print(f"Compiling for batch size {batch_size}...")
    key, warm_up_key = jax.random.split(key)
    keys = jax.random.split(warm_up_key, batch_size)
    _ = run_trajectory(keys)
    jax.block_until_ready(_)

    # 5. Timing
    print(f"Profiling {num_steps} steps...")
    keys = jax.random.split(key, batch_size)
    start = time.perf_counter()
    _ = run_trajectory(keys)
    jax.block_until_ready(_)
    end = time.perf_counter()

    # 6. Results
    total_time = end - start
    sps = (num_steps * batch_size) / total_time
    print(f"\nResults for Batch {batch_size}:")
    print(f"Total SPS: {sps:,.2f}")
    print(f"Time per trajectory: {total_time:.4f}s")

profile(1000, 1024)