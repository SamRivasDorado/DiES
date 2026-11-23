# Crack coordinates, angles and rotation in DiES

This note documents how `displacements()` and `stresses()` (in `DiES_core.py`) evaluate the
analytical solution for a straight 2D crack of **any orientation**: how each point is located
relative to the crack, how the angles are measured, and how the results are rotated back to the
global X–Y axes. It also records the sign conventions and the verification of the implementation.

## 1. Why the solution is evaluated in a local frame

The solution (Pollard and Segall, 1987; Pollard and Martel, 2020) is the plane-strain solution for
a straight crack of half-length `a` whose walls are loaded by uniform driving stresses: `dpx`
(normal to the walls, mode I), `dpy` (parallel to the walls, in the plane, mode II) and `dpz`
(parallel to the walls, out of the plane, mode III). Its equations are written for a crack lying
along one axis of its own reference frame, with the angles measured from that axis. They are exact
in that frame and only there. Evaluating them for an inclined crack therefore needs three steps:

1. Build the local frame of the crack (`crack_frame`).
2. Express every point in that frame and compute its **signed** tripolar coordinates
   (`local_coordinates`, `tripolar_coordinates`).
3. Evaluate the equations in the local frame and rotate the displacement vector and the stress
   tensor to the global axes (`rotate_vector_to_global`, `rotate_stress_to_global`).

Remote stresses (`rxx`, `ryy`, `rxy`, `rxz`, `ryz`) and the lithostatic stresses are given in the
global axes, so they are added **after** the rotation.

## 2. The local frame of the crack — `crack_frame(upper_tip, lower_tip)`

```
                 upper tip  (xi = -a)
                     o
                      \        n  (t rotated +90°, points into the hanging wall
        footwall       \     ↗       when the crack dips towards +x)
                        \  /
                         \/   center (xi = 0, eta = 0)
                         /\
                        /  \  hanging wall
                       v    \
                       t     \
               (down-dip)     o
                          lower tip (xi = +a)
```

- **Tips.** The upper tip is the shallower one. If the tips are given the other way round, they are
  swapped, so the frame does not depend on the order of the inputs. For a horizontal crack, the
  tip with the smaller x is taken as the upper one.
- **Center and half-length.** `center = (upper_tip + lower_tip)/2`, `a = |lower_tip - upper_tip|/2`.
  `displacements()` and `stresses()` check that the `crack_center` (and `a`) they receive agree with
  the tips, and raise `InputError` otherwise.
- **Unit vector along the crack**, pointing down-dip, from the upper to the lower tip:

  `t = (lower_tip - upper_tip) / |lower_tip - upper_tip| = (tx, ty)`, with `ty ≤ 0`.

- **Unit normal**, `t` rotated +90° (counterclockwise):

  `n = (-ty, tx)`.

  `(t, n)` is a right-handed frame, like `(X, Y)`. Because `ty ≤ 0`, `nx = -ty ≥ 0`: `n` always
  points towards +x. For a vertical crack `t = (0, -1)` and `n = (1, 0)`.
- **Orientation factor `h`** for the shear driving stresses:
  - `h = +1` when `n` points into the hanging wall, i.e. `ny = tx > 0` (crack dipping towards +x),
    and for vertical cracks (`tx = 0`), whose block at x > 0 plays the role of the hanging wall.
  - `h = -1` when `n` points into the footwall (`tx < 0`, crack dipping towards -x).

  The solution is evaluated with `s = h·dpy` and `s3 = h·dpz` (Section 6). This makes the sign of
  `dpy` and `dpz` mean the same physical sense whatever the dip direction.

## 3. Local coordinates — `local_coordinates(x, y, center, t, n)`

The position of a point relative to the center, `d = (x - xc, y - yc)`, is projected onto the two
unit vectors:

```
xi  = d · t = (x - xc)·tx + (y - yc)·ty      along the crack (upper tip at -a, lower tip at +a)
eta = d · n = (x - xc)·nx + (y - yc)·ny      normal to the crack (> 0 on the n side)
```

This is the inverse rotation of Section 5. `x` and `y` may be scalars or arrays, so a whole grid
can be evaluated in one call.

**Points on the crack plane.** If `eta` is exactly 0 (including `-0.0`) it is set to `+0.0`. Points
lying on the crack then take the values of its n-side wall. This matters because the displacements
and some stresses are discontinuous across the crack. The previous code shifted such points 0.1 m
in x instead.

## 4. Signed tripolar coordinates — `tripolar_coordinates(xi, eta, a)`

Each point is located by its distance and angle from the center and from the two tips:

| quantity | from | definition |
|---|---|---|
| `r`, `theta` | center (xi = 0) | `r = hypot(xi, eta)`, `theta = arctan2(eta, xi)` |
| `r1`, `theta1` | lower tip (xi = +a) | `r1 = hypot(xi - a, eta)`, `theta1 = arctan2(eta, xi - a)` |
| `r2`, `theta2` | upper tip (xi = -a) | `r2 = hypot(xi + a, eta)`, `theta2 = arctan2(eta, xi + a)` |
| `R`, `THETA` | — | `R = (r1·r2)^0.5`, `THETA = (theta1 + theta2)/2` |

All angles are measured from the +xi direction (along `t`, down-dip) and are **positive towards
+eta** (towards `n`). They range from -π to π. **The sign is what tells the two sides of the
crack apart**: `theta`, `theta1` and `theta2` are positive on the n side and negative on the other.

With `z = xi + i·eta`, these coordinates are the polar forms of the complex quantities in the
solution:

```
z                       = r · exp(i·theta)
z - a                   = r1 · exp(i·theta1)
z + a                   = r2 · exp(i·theta2)
(z² - a²)^0.5           = R · exp(i·THETA)          (principal branch: cut along the crack)
z / (z² - a²)^0.5       = (r/R) · exp(i·(theta - THETA))
```

Taking the square roots of `z - a` and `z + a` separately, each with its principal argument in
(-π, π], gives the branch of `(z² - a²)^0.5` that behaves like `z` far from the crack and is
discontinuous only across the crack itself. That is the physically correct branch.

**What was wrong before.** The angles were computed with `angle_between_lines`, i.e. with
`arccos` of a dot product. That is always between 0 and 180°, so it cannot tell one side of the
crack from the other. The equations were then right on one side only. The code tried to fix the
other side by flipping the sign of `dpx` (in `displacements`) or `dpy` (in `stresses`) depending on
the side, but a single flip cannot correct all the components. In addition, the results were never
rotated, so for an inclined crack the "normal" and "parallel" components were returned as if they
were x and y. See Section 8 for the consequences.

## 5. Equations in the local frame

Shorthand used in the code (all with the signed angles of Section 4):

```
im_Z = R·sin(THETA) - r·sin(theta)            = Im[(z²-a²)^0.5 - z]
re_Z = R·cos(THETA) - r·cos(theta)            = Re[(z²-a²)^0.5 - z]
eta_re = r·sin(theta)·((r/R)·cos(theta-THETA) - 1) = eta·Re[z/(z²-a²)^0.5 - 1]
eta_im = r·sin(theta)·((r/R)·sin(theta-THETA))     = eta·Im[z/(z²-a²)^0.5]
C = (r/R)·cos(theta-THETA) - 1                = Re[z/(z²-a²)^0.5] - 1
S = (r/R)·sin(theta-THETA)                    = Im[z/(z²-a²)^0.5]
q = a²·r·sin(theta)/R³                        = a²·eta/R³
```

with `G` the shear modulus, `ν` Poisson's ratio, `p = dpx`, `s = h·dpy`, `s3 = h·dpz`.

**Displacements** (`displacements()`):

```
u_n = [ p·(2(1-ν)·im_Z - eta_re) - s·((1-2ν)·re_Z + eta_im) ] / (2G)     normal to the crack (along n)
u_t = [ s·(2(1-ν)·im_Z + eta_re) + p·((1-2ν)·re_Z - eta_im) ] / (2G)     along the crack (along t)
u_z = 2·s3·im_Z / (2G)                                                   out of plane
```

**Stresses** (`stresses()`), tension positive, crack-induced part:

```
s_nn = p·(C + q·sin3Θ) + s·(q·cos3Θ)              normal stress across the crack (planes parallel to it)
s_tt = p·(C - q·sin3Θ) + s·(2S - q·cos3Θ)         normal stress along the crack (planes normal to it)
s_tn = s·(C - q·sin3Θ) + p·(q·cos3Θ)              in-plane shear stress
s_nz = s3·C                                       out-of-plane shear on planes parallel to the crack
s_tz = s3·S                                       out-of-plane shear on planes normal to the crack
```

These are the same equations DiES has always used (`ux` → `u_n`, `uy` → `u_t`, `sxx` → `s_nn`,
`syy` → `s_tt`, `sxy` → `s_tn`), written for a crack along the local `t` axis. They are the
Westergaard solution for a crack with uniformly loaded walls, `Z = z/(z²-a²)^0.5 - 1`, expanded in
tripolar form. On the walls (`eta → 0`, `|xi| < a`) they give `s_nn = -p`, `s_tn = -s` and
`s_nz = -s3`: the crack-induced stresses cancel the driving stresses. The only change to the
equations themselves is in the out-of-plane shear on planes normal to the crack: it was written
`s3·(S - 1)`, but the `-1` made it tend to `-s3` far from the crack instead of 0. It is now `s3·S`,
which is consistent with `u_z` through Hooke's law.

## 6. Sign conventions

- **Tension positive** for all stresses, as in the rest of DiES.
- **`dpx` > 0** opens the crack (mode I). The opening is always normal to the crack.
- **`dpy` > 0** makes the **hanging wall slide down-dip** relative to the footwall (normal sense);
  **`dpy` < 0**, up-dip (reverse sense). For a vertical crack, `dpy` > 0 makes the block at x > 0
  move down relative to the block at x < 0. This is the sense the previous code already produced
  for vertical cracks, so their displacements do not change. With `s = h·dpy`, the n-side block
  moves along +t (down-dip) relative to the other one when `s` > 0, and `h` makes the n side the
  hanging wall.
- **`dpz` > 0** makes the hanging wall (the block at x > 0 for a vertical crack) move towards +z
  relative to the footwall.
- **`dpx` in `dynamic_model`** keeps its meaning for faults (aperture 0): > 0 dilatant fault,
  < 0 compaction band, 0 pure shear.

## 7. Rotation to the global axes

The local unit vectors are the columns of the rotation matrix `Q = [t n]`:

```
Q = | tx  nx |
    | ty  ny |
```

**Vectors** (`rotate_vector_to_global`): `u = Q·(u_t, u_n)`, i.e.

```
ux = u_t·tx + u_n·nx
uy = u_t·ty + u_n·ny
```

The out-of-plane shear stresses `(s_tz, s_nz)` are the components of a vector (the traction on
planes normal to z) and are rotated the same way to `(sxz, syz)`.

**In-plane stress tensor** (`rotate_stress_to_global`): `σ = Q·σ'·Qᵀ`, with
`σ' = [[s_tt, s_tn], [s_tn, s_nn]]`, i.e. `σ = s_tt·(t⊗t) + s_nn·(n⊗n) + s_tn·(t⊗n + n⊗t)`:

```
sxx = s_tt·tx² + s_nn·nx² + 2·s_tn·tx·nx
syy = s_tt·ty² + s_nn·ny² + 2·s_tn·ty·ny
sxy = s_tt·tx·ty + s_nn·nx·ny + s_tn·(tx·ny + nx·ty)
```

For a vertical crack (`t = (0, -1)`, `n = (1, 0)`) this reduces to `sxx = s_nn`, `syy = s_tt`,
`sxy = -s_tn`, `ux = u_n`, `uy = -u_t`. Afterwards the remote stresses are added, and then
`szz = ν(sxx + syy)` (plane strain) and the maximum in-plane shear stress
`t = 0.5·((syy - sxx)² + 4·sxy²)^0.5` are computed, all in the global axes.

## 8. Verification

`tests/reference_westergaard.py` is an independent implementation of the same problem written
with complex potentials (Westergaard, 1939) instead of tripolar coordinates.
`tests/test_core_against_reference.py` checks, with `pytest` or `python tests/test_core_against_reference.py`:

1. **Agreement with the reference** to machine precision (relative error ~1e-15) for
   `sxx, syy, sxy, ux, uy` at 2000 random points. Six geometries (vertical, dipping 60° towards +x
   and towards -x, dipping 30°, horizontal, tips given in reverse order) and three loads
   (mode I, mode II, mixed).
2. **Physics, independent of the reference:**
   - the displacement jump across the crack is normal to it in mode I and parallel to it in
     mode II, with the theoretical maximum `2(1-ν)·p·a/G` (or `s` instead of `p`);
   - the tractions on the walls are `-p` (normal) and `-s` (shear);
   - the stresses obtained from the displacements by Hooke's law (plane strain, numerical
     derivatives) match `stresses()`, including mode III (`sxz = G·∂uz/∂x`, `syz = G·∂uz/∂y`);
   - the perturbation vanishes far from the crack.
3. **Shear sense:** with `dpy` > 0 the hanging wall moves down-dip for cracks dipping towards
   +x and towards -x, and the block at x > 0 moves down for a vertical crack.
4. **Continuity with the previous code** where it was right: for a vertical crack the
   displacements (modes I and II), and `sxx`, `syy` in mode I, are unchanged to 1e-12.

`tests/visual_check.py` complements this with figures (a fixed vertical dike and a 60° pure-shear fault, plotted after every change).

**Impact on previous results.**

- *Vertical cracks in mode I* (the dikes of Rivas-Dorado et al., 2023): displacements, `sxx` and
  `syy` were already correct. `sxy` had the wrong sign at x > 0, which does not change σ1, σ3 or the
  failure modes (they depend on `sxy²`). This was checked on a dike with lithostatic stresses:
  σ1, σ3 and the failure modes of `dynamic_model` are identical. The principal orientations
  (`s1a`, `s3a`) at x > 0 were mirrored, and so were the stress trajectories and failure-plane
  angles derived from them on that side.
- *Vertical cracks in mode II*: the displacements were right; `sxx` and `syy` had the wrong sign
  and `sxy` was wrong on one side.
- *Inclined cracks*: displacements and stresses were wrong (errors of 20–200 %). For example, a
  60° crack opened horizontally instead of normal to its walls.
- *Mode III*: `uz` had no jump across the crack (same value on both sides), and `syz` had the
  extra `-1`.

## 9. Singularities and numerical notes

- The solution is singular at the tips (`R = 0`): a grid node exactly on a tip gives `inf`/`NaN`.
  See item 12 of Phase 0 in the roadmap.
- `displacements()` and `stresses()` accept arrays for `point = (x, y)`, so the whole grid can be
  evaluated in one call. The result is identical to evaluating point by point (differences
  ~1e-16, from rounding).

## References

- Pollard, D. D., and Segall, P. (1987). Theoretical displacements and stresses near fractures in
  rock: with applications to faults, joints, veins, dikes, and solution surfaces. In Atkinson,
  B. K. (ed.), *Fracture Mechanics of Rock*, Academic Press, 277–349.
- Pollard, D. D., and Martel, S. J. (2020). *Structural Geology: A Quantitative Introduction*.
  Cambridge University Press.
- Westergaard, H. M. (1939). Bearing pressures and cracks. *Journal of Applied Mechanics*, 6, A49–A53.
- Rivas-Dorado, S., Ruíz, J., and Romeo, I. (2023). Modeling of dike-induced graben nucleation in the
  Elysium region, Mars: The role of planetary gravity. *Journal of Structural Geology*, 167, 104778.
