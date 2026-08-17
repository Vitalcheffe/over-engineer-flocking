"""Flocking Dynamics — Reynolds Boids (1987).

Three local rules, no leader, no global state. The flock is an emergent
property of the rule set. This is the classic Reynolds model:

    Separation:  steer to avoid crowding neighbors within r_sep
    Alignment:   steer toward the average heading of neighbors within r_align
    Cohesion:    steer toward the average position of neighbors within r_cohes

Each rule produces an acceleration vector. The three are summed with
weights (sep_weight, align_weight, cohes_weight) and applied to the
velocity, which is then clipped to a maximum speed.

NOTE on attribution
-------------------
An earlier version of this repo cited Olfati-Saber (2006) alongside
Reynolds. That was a misattribution: Olfati-Saber is a specific
distributed-control algorithm with a σ-norm potential and an α-lattice
convergence proof, NOT a Reynolds variant. This code implements
Reynolds (1987) "Flocks, Herds, and Schools: A Distributed Behavioral
Model" (Computer Graphics 21(4)). The Olfati-Saber reference has been
removed from this repo to avoid claiming a method I did not implement.

If you want Olfati-Saber, see:
    Olfati-Saber, R. (2006). "Flocking for Multi-Agent Dynamic Systems:
    Algorithms and Theory." IEEE TAC 51(3).
The implementation requires a σ-norm adjacency, a bump function, and an
α-lattice potential — none of which are present here.

References
----------
[1] Reynolds, C. W. (1987). "Flocks, Herds, and Schools: A Distributed
    Behavioral Model." Computer Graphics 21(4), 25-34.
[2] Vicsek, T. et al. (1995). "Novel Type of Phase Transition in a
    Model of Self-Driven Particles." Physical Review Letters 75(6).
    The Vicsek order parameter used below originates here.
"""
import numpy as np
import json

# --- Flock parameters ---
N = 200                       # number of agents
R_SEPARATION = 0.08           # avoid neighbors closer than this
R_ALIGNMENT = 0.25            # match heading of neighbors within this radius
R_COHESION = 0.30             # drift toward centroid of neighbors within this radius

# Rule weights (tuned empirically to produce a stable partial-order phase).
# Alignment dominates — it's the rule that creates order from chaos.
# Separation prevents collapse. Cohesion keeps the flock together.
SEP_WEIGHT = 1.0
ALIGN_WEIGHT = 0.5            # alignment is the primary order driver
COHES_WEIGHT = 0.2            # cohesion matters for grouping

V_MAX = 1.0                   # speed cap
V_INIT = 0.5                  # initial speed (uniform, random direction)

# Domain: toroidal (periodic boundaries) so the flock doesn't disperse to infinity
DOMAIN_SIZE = 2.0


def reynolds_step(pos, vel, dt=0.1,
                  n_agents=N,
                  r_sep=R_SEPARATION,
                  r_align=R_ALIGNMENT,
                  r_cohes=R_COHESION,
                  w_sep=SEP_WEIGHT,
                  w_align=ALIGN_WEIGHT,
                  w_cohes=COHES_WEIGHT,
                  v_max=V_MAX):
    """Advance one Reynolds-boids step.

    Each agent computes three acceleration vectors from its local
    neighborhood (defined by the three radii), sums them with weights,
    and updates its velocity. Position is updated by Euler integration.

    Returns (new_pos, new_vel).
    """
    new_vel = np.copy(vel)

    for i in range(n_agents):
        # Vectorized neighbor finding with periodic boundary conditions
        rel = pos - pos[i]
        # Wrap around for toroidal domain (closest image)
        rel = np.mod(rel + DOMAIN_SIZE / 2, DOMAIN_SIZE) - DOMAIN_SIZE / 2
        dists = np.linalg.norm(rel, axis=1)
        dists[i] = np.inf  # exclude self

        # --- Separation: steer away from neighbors within r_sep ---
        sep_neighbors = dists < r_sep
        if np.any(sep_neighbors):
            # Average of (pos[i] - pos[j]) for j in sep neighbors
            sep_dir = -np.mean(rel[sep_neighbors], axis=0)
            # Normalize if non-zero
            norm = np.linalg.norm(sep_dir)
            if norm > 0:
                sep_dir = sep_dir / norm
            new_vel[i] += sep_dir * w_sep

        # --- Alignment: match average velocity of neighbors within r_align ---
        align_neighbors = dists < r_align
        if np.any(align_neighbors):
            avg_vel = np.mean(vel[align_neighbors], axis=0)
            new_vel[i] += avg_vel * w_align

        # --- Cohesion: drift toward centroid of neighbors within r_cohes ---
        cohes_neighbors = dists < r_cohes
        if np.any(cohes_neighbors):
            center = np.mean(pos[cohes_neighbors], axis=0)
            new_vel[i] += (center - pos[i]) * w_cohes

        # Speed cap (Reynolds' "steering force" clamp)
        speed = np.linalg.norm(new_vel[i])
        if speed > v_max:
            new_vel[i] = (new_vel[i] / speed) * v_max

    new_pos = pos + new_vel * dt
    # Wrap positions to toroidal domain [0, DOMAIN_SIZE)
    new_pos = np.mod(new_pos, DOMAIN_SIZE)
    return new_pos, new_vel


def order_parameter(vel):
    """Vicsek order parameter: alignment of the flock.

    ψ = (1/(N·v_avg)) * || Σ v_i ||

    - 0  = total disorder (velocities cancel out)
    - 1  = perfect alignment (all velocities parallel)

    Reference: Vicsek et al. (1995) PRL.
    """
    n = len(vel)
    avg_v = np.mean(np.linalg.norm(vel, axis=1))
    if avg_v == 0:
        return 0.0
    sum_v = np.linalg.norm(np.sum(vel, axis=0))
    return float(sum_v / (n * avg_v))


def simulate(steps=500, n_agents=N, seed=42,
             w_sep=SEP_WEIGHT, w_align=ALIGN_WEIGHT, w_cohes=COHES_WEIGHT):
    """Run the Reynolds-boids simulation for `steps` steps.

    Initial condition: random positions and small random velocities.
    The order parameter is recorded at every step.

    Rule weights can be overridden for ablation studies (set w_cohes=0
    to test whether cohesion actually matters).

    Returns:
        {
            'final_order': float,           # ψ at the last step
            'history':     list[dict],      # ψ per step
            'parameters':  dict,             # all knobs used
        }
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 2, size=(n_agents, 2))
    # Initial velocity: random direction, fixed speed V_INIT
    # (Reynolds uses non-zero initial speed so agents actually move and meet neighbors)
    angles = rng.uniform(0, 2 * np.pi, size=n_agents)
    vel = np.column_stack([np.cos(angles), np.sin(angles)]) * V_INIT

    history = []
    for s in range(steps):
        pos, vel = reynolds_step(pos, vel, n_agents=n_agents,
                                  w_sep=w_sep, w_align=w_align, w_cohes=w_cohes)
        history.append({
            'step': s,
            'order': order_parameter(vel),
        })

    return {
        'final_order': history[-1]['order'],
        'history': history,
        'parameters': {
            'n_agents': n_agents,
            'r_separation': R_SEPARATION,
            'r_alignment': R_ALIGNMENT,
            'r_cohesion': R_COHESION,
            'weights': {
                'separation': w_sep,
                'alignment': w_align,
                'cohesion': w_cohes,
            },
            'v_max': V_MAX,
            'steps': steps,
            'seed': seed,
        },
    }


if __name__ == '__main__':
    print("Flocking Dynamics — Reynolds Boids (1987)")
    print("=" * 60)
    print()
    print(f"N = {N} agents")
    print(f"r_sep = {R_SEPARATION}, r_align = {R_ALIGNMENT}, r_cohes = {R_COHESION}")
    print(f"weights: sep={SEP_WEIGHT}, align={ALIGN_WEIGHT}, cohes={COHES_WEIGHT}")
    print(f"v_max = {V_MAX}")
    print()
    r = simulate(steps=500)
    print(f"Final order parameter ψ = {r['final_order']:.4f}")
    print(f"  (0 = chaos, 1 = perfect alignment)")
    print()
    # Order-parameter trajectory summary
    orders = [h['order'] for h in r['history']]
    print(f"Trajectory:")
    print(f"  step   0: ψ = {orders[0]:.4f}")
    print(f"  step  50: ψ = {orders[50]:.4f}")
    print(f"  step 100: ψ = {orders[100]:.4f}")
    print(f"  step 250: ψ = {orders[250]:.4f}")
    print(f"  step 500: ψ = {orders[-1]:.4f}")
    print()
    print("Reference: Reynolds, C. W. (1987). Flocks, Herds, and Schools:")
    print("           A Distributed Behavioral Model. Computer Graphics 21(4).")
    print()
    with open('data/results.json', 'w') as f:
        json.dump(r, f, indent=2, default=str)
    print("Wrote data/results.json")
