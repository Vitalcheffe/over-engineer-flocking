<div align="center">

# Flocking Dynamics — Reynolds Boids (1987)

### Three local rules, no leader, no global state. The flock is an emergent property.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research%20testbed-orange.svg?style=flat-square)](#limitations)

</div>

---

## Overview

A faithful implementation of Craig Reynolds' 1987 boids model — the
classic "Flocks, Herds, and Schools" distributed-behavior algorithm.
Three rules per agent (separation, alignment, cohesion), no leader, no
global state, and the flock emerges as a phase transition in the Vicsek
order parameter.

**Project 08/12** of the Over Engineer challenge.

---

## Why I built this

I built this at 16 after watching a starling murmuration at dusk south of
Casablanca. Thousands of birds turning as one, no leader, no conductor.
The trigger is local — one bird reacts to a predator — and the
information propagates through the flock in a way that looks simultaneous
from a distance. This is decentralized coordination, and it has a
mathematical description.

The gap between the observation (it looks simultaneous) and the rigor
(it is provably local) is the point.

---

## The model

**Three rules, applied per agent based on local neighbors only:**

1. **Separation** — steer to avoid crowding neighbors within `r_sep`
2. **Alignment** — steer toward the average heading of neighbors within `r_align`
3. **Cohesion** — steer toward the average position of neighbors within `r_cohes`

Each rule produces an acceleration vector; the three are summed with
weights and applied to the velocity, which is clipped to `v_max`.

**Order parameter (Vicsek, 1995):**

    ψ = || Σ v_i || / (N · v_avg)

- `ψ ≈ 0` = total disorder (velocities cancel out)
- `ψ ≈ 1` = perfect alignment (all velocities parallel)

The phase transition is sharp: at low alignment weight, ψ stays near 0;
above a threshold, ψ jumps to 0.9+ within ~100 steps.

See `model.py` for the implementation and `docs/math.md` for derivations.

---

## Attribution note

An earlier version of this repo cited Olfati-Saber (2006) alongside
Reynolds. That was a misattribution — Olfati-Saber is a specific
distributed-control algorithm with a σ-norm potential and an α-lattice
convergence proof, **not** a Reynolds variant. The code here implements
Reynolds (1987) only. The Olfati-Saber reference has been removed to
avoid claiming a method I did not implement.

If you want Olfati-Saber, see:
> Olfati-Saber, R. (2006). "Flocking for Multi-Agent Dynamic Systems:
> Algorithms and Theory." IEEE TAC 51(3).

The implementation requires a σ-norm adjacency, a bump function, and an
α-lattice potential — none of which are present here.

---

## The results

Run `python3 model.py` to see the numerical results. Typical trajectory:

```
step   0: ψ = 0.0312  (chaos — random initial directions)
step  50: ψ = 0.4076  (emergence begins)
step 100: ψ = 0.9311  (flock aligned)
step 250: ψ = 0.9133  (stable)
step 500: ψ = 0.9387  (stable)
```

The phase transition occurs around step 50-100 — visible in the order
parameter's sharp jump from ~0.03 to ~0.9.

---

## How it works

1. **Model** — Three-rule Reynolds boids with periodic boundary conditions
2. **Simulate** — 500 steps, 200 agents, toroidal domain
3. **Visualize** — Order-parameter trajectory + final flock snapshot

---

## Run it

```bash
git clone https://github.com/Vitalcheffe/over-engineer-flocking.git
cd over-engineer-flocking
pip install numpy scipy matplotlib
python3 model.py       # writes data/results.json
python3 visualize.py   # writes docs/viz/analysis-light.png
pytest tests/          # 8 tests verifying mathematical invariants
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Numerics | NumPy |
| Visualization | Matplotlib |
| Testing | pytest |

---

## Limitations

1. **2D only.** Real murmurations are 3D; the model is 2D for
   computational simplicity and visualization clarity.
2. **No predator dynamics.** Real flocks react to external threats
   (hawks, falcons). The model has no such input — flocking emerges
   from initial random directions alone.
3. **Point agents.** Real birds have wingspan, mass, and turning-rate
   constraints. The model treats agents as points with no inertia.
4. **No visual occlusion.** Real birds can't see through each other.
   The model assumes omnidirectional sensing.
5. **Periodic boundary conditions.** The toroidal domain is
   mathematically clean but physically unrealistic. Real flocks are
   bounded by the sky and the ground.
6. **Educational, not predictive.** The model is for understanding the
   emergence mechanism, not for simulating real bird behavior.

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<sub>Over Engineer · 08 / 12 · Amine Harch El Korane · 2026</sub><br>
<sub>"The gap between the observation and the rigor is the point."</sub>
</div>
