import jax
import jax.numpy as jnp
import time
from typing import Callable

from soccer_env_interface import state_to_nn_input, step_fn, init_fn, N_ACTIONS

def profile(
    num_steps: int = 1000, 
    batch_size: int = 1024
):
    # 1. Setup Vectorized functions
    v_reset = jax.vmap(init_fn)
    v_step = jax.vmap(step_fn)

    # 2. Define the Scan Body
    def rollout_step(carry, unused_input):
        state, key = carry
        key, action_key = jax.random.split(key)
        action = random.randint(action_key, (batch_size), 0, N_ACTIONS)
        next_state, metadata = v_step(state, action)
        return (next_state, key), None

    # 3. JIT Compile the entire sequence
    @jax.jit
    def run_trajectory(key):
        init_key, loop_key = jax.random.split(key)
        # Reset all envs in the batch
        initial_state, metadata = v_reset(jax.random.split(init_key, batch_size))
        # Scan over num_steps
        (final_state, _), _ = jax.lax.scan(
            rollout_step, 
            (initial_state, loop_key), 
            None, 
            length=num_steps
        )
        return final_state

    # 4. Warm-up
    print(f"Compiling for batch size {batch_size}...")
    main_key = jax.random.PRNGKey(42)
    _ = run_trajectory(main_key)
    jax.block_until_ready(_)

    # 5. Timing
    print(f"Profiling {num_steps} steps...")
    start = time.perf_counter()
    _ = run_trajectory(main_key)
    jax.block_until_ready(_)
    end = time.perf_counter()

    # 6. Results
    total_time = end - start
    sps = (num_steps * batch_size) / total_time
    print(f"\nResults for Batch {batch_size}:")
    print(f"Total SPS: {sps:,.2f}")
    print(f"Time per trajectory: {total_time:.4f}s")

profile(1000, 1024)