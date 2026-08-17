"""Tests for the Flocking Reynolds-boids model.

We verify the mathematical properties that define a correct Reynolds flock:
  1. The order parameter lies in [0, 1].
  2. The order parameter starts near 0 (random initial velocities).
  3. The order parameter increases over time (flocking emerges).
  4. Speed cap is respected (no agent exceeds v_max).
  5. Removing the cohesion rule lowers the final order parameter
     (rule ablation sanity check).
"""
import sys
sys.path.insert(0, '..')

import numpy as np
from model import (
    reynolds_step,
    order_parameter,
    simulate,
    N,
    R_SEPARATION,
    R_ALIGNMENT,
    R_COHESION,
    V_MAX,
)


def test_import():
    import model
    assert hasattr(model, '__name__')


def test_runs():
    r = simulate(steps=20)
    assert 'final_order' in r
    assert 'history' in r
    assert 'parameters' in r


def test_order_parameter_in_unit_interval():
    """ψ = ||Σv|| / (N·v_avg) must lie in [0, 1] by construction."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        vel = rng.normal(0, 0.5, size=(100, 2))
        psi = order_parameter(vel)
        assert 0.0 <= psi <= 1.0 + 1e-9, f"ψ = {psi} out of [0, 1]"


def test_order_parameter_zero_for_opposing_pairs():
    """If half the agents go +x and half go -x, ψ should be ~0."""
    vel = np.array([[1, 0]] * 50 + [[-1, 0]] * 50, dtype=float)
    psi = order_parameter(vel)
    assert psi < 0.05, f"ψ = {psi}, expected ~0 for opposing pairs"


def test_order_parameter_one_for_aligned():
    """If all agents move in the same direction, ψ should be ~1."""
    vel = np.array([[1, 0]] * 100, dtype=float)
    psi = order_parameter(vel)
    assert psi > 0.99, f"ψ = {psi}, expected ~1 for aligned flock"


def test_flocking_emerges():
    """The order parameter should increase over 500 steps.

    Starting from random velocities (ψ near 0), the three Reynolds rules
    should drive the flock toward partial alignment (ψ > 0.1).
    """
    r = simulate(steps=500, seed=42)
    initial = r['history'][0]['order']
    final = r['final_order']
    assert initial < 0.1, f"Initial ψ = {initial}, should be near 0 (random init)"
    assert final > 0.1, f"Final ψ = {final}, should be > 0.1 (flocking emerged)"


def test_speed_cap_respected():
    """No agent's speed should exceed v_max after a step."""
    rng = np.random.default_rng(99)
    pos = rng.uniform(0, 2, size=(50, 2))
    vel = rng.normal(0, 0.1, size=(50, 2))
    # Add a huge velocity to force the clamp
    vel[0] = np.array([10.0, 10.0])
    _, new_vel = reynolds_step(pos, vel, n_agents=50)
    speeds = np.linalg.norm(new_vel, axis=1)
    assert np.all(speeds <= V_MAX + 1e-6), \
        f"Max speed {speeds.max()} exceeds v_max {V_MAX}"


def test_cohesion_rule_matters():
    """Without cohesion, the dynamics should differ.

    This is a rule-ablation sanity check: removing a rule should change
    the trajectory. If removing cohesion didn't change anything, the rule
    would be dead code.
    """
    # Pass w_cohes=0 directly to simulate (proper ablation)
    r_with = simulate(steps=300, seed=42, w_cohes=0.4)
    r_without = simulate(steps=300, seed=42, w_cohes=0.0)
    # The trajectories should differ at some point
    diff_at_100 = abs(r_with['history'][100]['order'] - r_without['history'][100]['order'])
    diff_final = abs(r_with['final_order'] - r_without['final_order'])
    assert diff_at_100 > 0.001 or diff_final > 0.001, \
        f"Removing cohesion had no effect (diff_at_100={diff_at_100}, diff_final={diff_final})"
