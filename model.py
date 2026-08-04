"""Flocking Dynamics — Reynolds Boids + Olfati-Saber"""
import numpy as np, json
N = 200; R_SEPARATION = 0.05; R_ALIGNMENT = 0.15; R_COHESION = 0.20
def step(pos, vel, dt=0.1):
    new_vel = np.copy(vel)
    for i in range(N):
        dists = np.linalg.norm(pos - pos[i], axis=1)
        mask_sep = (dists < R_SEPARATION) & (dists > 0)
        mask_align = (dists < R_ALIGNMENT) & (dists > 0)
        mask_cohes = (dists < R_COHESION) & (dists > 0)
        if np.any(mask_sep):
            sep = -np.mean(pos[mask_sep] - pos[i], axis=0)
            new_vel[i] += sep * 2.0
        if np.any(mask_align):
            new_vel[i] += np.mean(vel[mask_align], axis=0) * 0.5
        if np.any(mask_cohes):
            center = np.mean(pos[mask_cohes], axis=0)
            new_vel[i] += (center - pos[i]) * 0.3
        new_vel[i] = np.clip(new_vel[i], -1, 1)
    new_pos = pos + new_vel * dt
    return new_pos, new_vel
def order_parameter(vel):
    """Measure alignment: 0=chaos, 1=ordered flock."""
    avg_v = np.mean(np.linalg.norm(vel, axis=1))
    sum_v = np.linalg.norm(np.sum(vel, axis=0))
    return sum_v / (N * avg_v) if avg_v > 0 else 0
def simulate(steps=500):
    pos = np.random.rand(N, 2) * 2
    vel = np.random.randn(N, 2) * 0.1
    history = []
    for s in range(steps):
        pos, vel = step(pos, vel)
        history.append({'step': s, 'order': float(order_parameter(vel))})
    return {'final_order': history[-1]['order'], 'history': history}
if __name__ == '__main__':
    r = simulate()
    print(f"Flocking: final order parameter = {r['final_order']:.3f} (1.0 = perfect flock)")
    with open('data/results.json', 'w') as f: json.dump(r, f, indent=2, default=str)
