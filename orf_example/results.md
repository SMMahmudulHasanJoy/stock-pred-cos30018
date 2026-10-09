# Approximating S1 and S2 with small KANs

Setup: `S(t) = sum_j A_j * max(0, t - theta_j)` on `t = 0:0.01:10`, slopes
`A = [2, -1, 4, -3, -1]` for both; knots `theta = [0,2,4,6,8]` for S1 (evenly
spaced) and `[0,3,5,6,7]` for S2. Both are piecewise linear with five knots.
Fits use PyKAN 0.2.8, L-BFGS, 200 steps, grid range [0, 10], seed 0.
Error = max |KAN(t) - S(t)| over the 1001 points; RMSE in brackets.

## A. One spline edge (KAN width [1,1]), linear splines (k = 1)

| grid intervals | S1 max err | S2 max err | params |
|---|---|---|---|
| 5  | **0.0000** | **1.4761** (0.448) | 12 |
| 10 | 0.0000 | 0.0000 | 17 |
| 20 | 0.0241 (0.003) | 0.0199 (0.003) | 27 |
| 50 | 0.0039 | 0.0033 | 57 |
| 100 | 0.0010 | 0.0008 | 107 |

With 5 intervals the KAN's knots sit at 0, 2, 4, 6, 8, 10: S1 is represented
exactly, S2 is not (its knots at 3, 5, 7 fall between grid points). With 10
intervals the grid knots are the integers, so S2 is also exact, because its
knots happen to be integers. From 20 intervals up, the small residual is
optimisation (more parameters, same 200 steps), not representability.

## B. Same, cubic splines (k = 3, PyKAN's default and the paper's setting)

| grid | S1 | S2 |
|---|---|---|
| 5 | 0.535 | 0.882 |
| 10 | 0.344 | 0.315 |
| 20 | 0.163 | 0.170 |
| 50 | 0.064 | 0.063 |
| 100 | 0.032 | 0.032 |

A cubic spline cannot reproduce a kink, so neither function is ever exact;
the error falls roughly like 1/grid for both. Knot placement matters much
less than the kink itself here.

## C. One hidden layer (k = 1)

| width | grid | S1 | S2 |
|---|---|---|---|
| [1,2,1] | 5 | 0.066 | 1.064 |
| [1,2,1] | 10 | 0.020 | 0.025 |
| [1,3,1] | 5 | 0.016 | **0.030** |
| [1,3,1] | 10 | 0.047 | 0.073 |

Composing splines lets a coarse grid produce effective knots off the grid:
[1,3,1] on 5 intervals brings S2 from 1.48 (single edge) to 0.03. The price
is six times the parameters and a harder optimisation (grid 10 is worse than
grid 5 for [1,3,1], a local-minimum effect).

## D. Reference: free-knot linear spline (knots learned, 11 parameters)

| | max err | knots recovered |
|---|---|---|
| S1 | 0.0000 | 0, 2, 4, 6, 8 |
| S2 | 0.0000 | 0, 3, 5, 6, 7 |

Exact for both with the minimal number of parameters, but only with 14
random restarts: with 6 restarts S2 stuck at knots (3, 4.27, 4.97, 6.25).
That non-convexity is the free-knot difficulty referee 1 alludes to.

## What the example shows

1. A KAN edge is a fixed-grid spline. It is exact when the target's knots
   lie on its grid (S1 at grid 5) and not otherwise (S2 at grid 5: max
   error 1.48, about 7% of the function's range).
2. Refining the uniform grid recovers accuracy, at the cost of parameters:
   S2 needs 10 intervals for what S1 gets with 5. For a target whose knots
   are not on any coarse uniform grid, the cost grows further.
3. Depth can substitute for grid resolution (C), free knots do it with the
   fewest parameters (D), but free-knot fitting is non-convex.
4. With cubic splines, the paper's setting, no fixed grid captures a kink
   exactly; accuracy is bought with grid size for both functions alike.

Suggestion: S2's knots are integers, so a uniform grid of 10 catches them
by luck. Moving them to non-integer positions (e.g. 0, 3.3, 5.1, 6.4, 7.2)
would make the point hold for every uniform grid, which is stronger.

Files: `fit_kan.py` (regenerates everything, ~3 min), `results.json`,
`figure_fits.png`, `figure_error_vs_grid.png`, `S1.txt`, `S2.txt`,
`ExampleForORF.m` (Nadia's generator).
