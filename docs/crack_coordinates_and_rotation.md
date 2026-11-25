# Crack coordinates, angles and rotation in DiES

This note documents how `displacements()` and `stresses()` (in `DiES_core.py`) evaluate the
analytical solution for a straight 2D crack of **any orientation**: how each point is located
relative to the crack, how the angles are measured, and how the results are rotated back to the
global X–Y axes. It also records the sign conventions and the verification of the implementation.

## Change log

Every change to the calculation of coordinates, angles and orientations, with the date it was
found and fixed and the commit on branch `phase0-update-bools-crack-naming` (named
`update-bools-crack-naming` until 2026-11-25) of `SamRivasDorado/DiES`.
Dates are local time in Madrid (CET).

| Date | Commit | Change | Section |
|---|---|---|---|
| 2026-11-23 | `1915dc4` | Crack solution evaluated in the local frame of the crack (`crack_frame`, `local_coordinates`), with signed tripolar angles (`arctan2`) instead of the unsigned `arccos` of `angle_between_lines`, and the results rotated to the global axes. The side-dependent sign flips of `dpx`/`dpy` were removed. | 2–5, 7 |
| 2026-11-23 | `1915dc4` | Shear sense made independent of the dip direction (factor `h`): `dpy > 0` = hanging wall slides down-dip. | 2, 6 |
| 2026-11-23 | `1915dc4` | Mode III: the spurious `-1` in the out-of-plane shear on planes normal to the crack removed; `uz` now jumps across the crack. | 5, 8 |
| 2026-11-23 | `1915dc4` | Points exactly on the crack take the values of its n-side wall (previously shifted 0.1 m in x). | 3 |
| 2026-11-24 | `ace6c94` | Orientation of σ1/σ3: `½·atan2(2σxy, σxx−σyy)`, counterclockwise from +x, in (−90°, 90°]. Removed the 0.001 replacement of a zero denominator (0° instead of 90° where σ1 is vertical with σxy = 0) and the 90° forced over the whole grid when `dpx == 0 and s11 == 0`. | 10.2 |
| 2026-11-24 | `ace6c94` | Orientations drawn as `(cos a, sin a)` instead of the mirrored `(cos a, −sin a)`, in `plot_stress_trajectories` and in the failure planes of `dynamic_model`. | 10.1, 10.3, 10.4 |
| 2026-11-24 | `ace6c94` | Failure planes of `dynamic_model` computed by `failure_plane_orientation()` (conjugate plane dipping towards the crack) instead of four hand-tuned branches depending on the sign of `s1a`. | 10.4 |
| 2026-11-24 | `ace6c94` | Sign of T in `evaluate_stress_state`: `T = T/1e6` instead of `T = (−T)/1e6`; T must be negative. | 11 |

Related changes to the figures, outside the angle calculations: on 2026-11-23 (`1915dc4`) and 2026-11-24 (`2a4e5f3`), the colour and arrow scale of `plot_displacement` changed from `(ux+uy)/2` to the magnitude of the plotted vector, and the `uxy` contours stopped crashing.

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

**Points on the crack plane** (since 2026-11-23, `1915dc4`). If `eta` is exactly 0 (including `-0.0`) it is set to `+0.0`. Points
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

**What was wrong before** (found and fixed on 2026-11-23, `1915dc4`). The angles were computed with `angle_between_lines`, i.e. with
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
equations themselves (2026-11-23, `1915dc4`) is in the out-of-plane shear on planes normal to the crack: it was written
`s3·(S - 1)`, but the `-1` made it tend to `-s3` far from the crack instead of 0. It is now `s3·S`,
which is consistent with `u_z` through Hooke's law.

## 6. Sign conventions

- **Tension positive** for all stresses, as in the rest of DiES.
- **`dpx` > 0** opens the crack (mode I). The opening is always normal to the crack.
- **`dpy` > 0** (convention set on 2026-11-23, `1915dc4`) makes the **hanging wall slide down-dip** relative to the footwall (normal sense);
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

Tests added on 2026-11-23 (`1915dc4`).


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
  σ1, σ3 and the failure modes of `dynamic_model` are identical. The value of `s1a`, `s3a`
  changed sign at x > 0; how the orientations were computed and drawn, and their own errors, are
  described in Section 10.
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

## 10. Principal stress orientations and failure planes

The stresses of Sections 5–7 feed the principal stresses, their orientations and, through them,
the orientations of the failure planes that `dynamic_model` predicts. This section documents how
those orientations are computed and drawn since 2026-11-24 (`ace6c94`), and what was wrong before.
The errors were found on 2026-11-24, while reviewing every figure DiES produces (`tests/plot_gallery.py`).

### 10.1 Convention

All orientations in the X–Y section are angles **counterclockwise from the +x axis**, given modulo
180° (a line at `a` and at `a + 180°` is the same line) and reported in (−90°, 90°]. A line with
orientation `a` has the direction `(cos a, sin a)`; that is how every orientation is drawn
(`quiver(x, y, cos a, sin a)`). 90° is vertical. The principal stresses use the
compression-positive convention: `σ1` is the maximum compressive stress.

### 10.2 Orientation of σ1 and σ3 — `model.principal_stresses`

With `σxx`, `σyy`, `σxy` compression positive, the direction of `σ1` satisfies
`tan 2a = 2σxy / (σxx − σyy)` (Jaeger et al., 2007). `tan 2a` alone does not tell `σ1` from `σ3`,
because it has two solutions 90° apart. The two-argument arctangent picks the right one from the
signs of `σxy` and `σxx − σyy`:

```
s1a = ½ · atan2(2σxy, σxx − σyy)        (degrees, in (−90°, 90°])
s3a = s1a − 90°  if s1a > 0,  s1a + 90°  otherwise
```

Checks: with `σxy = 0` and `σxx > σyy` (horizontal σ1), `atan2(0, +) = 0` gives 0°; with `σxy = 0`
and `σxx < σyy` (vertical σ1, e.g. the lithostatic state) `atan2(0, −) = 180°` gives 90°. There is
no division, so no special case is needed.

**What was wrong before** (found and fixed on 2026-11-24, `ace6c94`).

- **(a)** The angle was `arctan(2σxy / ((σxx − σyy) + R))`, with `R = ((σxx − σyy)² + 4σxy²)^0.5`.
  Where `σxy = 0` and `σxx < σyy` the denominator is 0. It was replaced by 0.001, which gave
  `arctan(0) = 0` (horizontal) instead of 90° (vertical): a 90° error, for example along the whole
  row through the center of a vertical dike.
- **(b)** The whole grid was forced to `s1a = 90°` when `dpx == 0 and s11 == 0`. This was meant for
  "no crack and no remote stress", but it also caught faults loaded only in shear (`dpx = 0`,
  `dpy ≠ 0`), whose effect on the orientations was then ignored (errors up to ~87° next to the fault).
- **(c)** The orientations were drawn as `(cos a, −sin a)`, i.e. mirrored about the horizontal.
- **(d)** Combined with the old sign error of `σxy` at x > 0 (Section 4), (c) cancelled out on the
  right of a vertical crack, so x > 0 was drawn correctly and x < 0 was mirrored (up to ~18° near the
  dike, 95th percentile). After the core was corrected on 2026-11-23 (Section 4), both sides were mirrored until
  (a)–(c) were fixed on 2026-11-24.

The new orientations match the eigenvectors of the stress tensor to ~1e−6° everywhere.

### 10.3 Optimal shear planes — `plot_stress_trajectories(planes=True)`

Drawn as `(cos, sin)` since 2026-11-24 (`ace6c94`); the formulas of the planes did not change.

The two conjugate optimal planes lie at `±coula` from σ1, where `coula = ½·arctan(1/μ)` is the angle
between σ1 and the planes (`θ_opt = 90° − coula`). Their orientations are `s1a − coula` and
`s1a + coula`, drawn as `(cos, sin)`.

### 10.4 Failure planes in `dynamic_model` — `failure_plane_orientation`

Since 2026-11-24 (`ace6c94`). At each node, `evaluate_stress_state` returns the failure mode and `2θ`, the angle on the Mohr
circle from the σ1 point to the point of failure. `θ` is the angle between σ1 and the **normal** to
the failure plane, so the plane itself lies at `δ = 90° − θ` from σ1.
`failure_plane_orientation(s1a, θ, mode, side)` in `Failure_Determination.py` returns its orientation:

- **Mode I** (`2θ = 180°`, `θ = 90°`): the plane contains σ1, `b = s1a + θ − 90° = s1a`.
- **Shear modes** (`'Mixed-mode I-II'`, `'Mode II - tension-compression'`,
  `'Mode II - compression'`): there are two conjugate planes, `s1a + δ` and `s1a − δ`. As in the
  original code, the one that **dips towards the crack** is kept. On the left of the crack
  (x ≤ x of its center) that is the plane dipping towards +x; on the right, the one dipping towards
  −x. Above a dike these are the graben-forming normal faults.
  - A trace with orientation `b` (modulo 180°) dips towards +x when `90° < b < 180°`, i.e. when
    `sin 2b < 0`, and towards −x when `sin 2b > 0`. The candidate with the most negative `sin 2b`
    (left) or the most positive (right) is selected.
  - If σ1 is so tilted that both conjugates dip the same way (possible next to an inclined fault),
    this picks the one closest to dipping towards the crack.
- `'Below'`, `'Above'` or `θ = NaN`: no plane (`NaN`).

The traces are drawn as `(cos b, sin b)`.

**What was wrong before** (found and fixed on 2026-11-24, `ace6c94`). The original code used four hand-tuned branches, depending on the side
of the crack and on the sign of `s1a`:

```
left,  s1a ≥ 0:  b = s1a + θ − 90                 right, s1a ≥ 0:  b = s1a − θ + 90
left,  s1a < 0:  b = −(|s1a| − θ + 90)            right, s1a < 0:  b = −(|s1a| + θ − 90)
```

With `s1a = −|s1a|` these reduce to `b = s1a − δ` (left) and `b = s1a + δ` (right), whatever the
sign of `s1a`. Drawn as `(cos b, −sin b)`, they gave the right plane only where σ1 was exactly
vertical: the mirror leaves a vertical σ1 unchanged and swaps the two conjugates. Elsewhere the
error was twice the tilt of σ1, which explains the comments `#Why? This does not match with my
drawings` in the original code.

### 10.5 Verification

Tests added on 2026-11-24 (`ace6c94`).


`tests/test_orientations_and_failure.py`:

1. `s1a` and `s3a` match the eigenvectors of the stress tensor (largest and smallest eigenvalue)
   to 1e−4° for a vertical dike and for faults dipping towards +x and towards −x, and lie in
   (−90°, 90°].
2. In the lithostatic state alone, σ1 is vertical (`s1a = 90°`) everywhere.
3. Rules of `failure_plane_orientation`: with σ1 vertical and `θ = 60°` it gives 120° on the left
   and 60° on the right; in Mode I it gives `s1a`; for `'Below'`/`'Above'` it gives `NaN`.
4. **Independent check of the failure planes:** for every node in shear failure, the full stress
   tensor is resolved on the selected plane (normal `(−sin b, cos b)`). The normal and shear
   stresses obtained match the tangent point `(σn, τ)` returned by `evaluate_stress_state` to
   0.02 MPa. Of the two conjugates, the selected one is the closest to dipping towards the crack.
   A mirrored or wrongly selected plane fails this test. In Mode I the difference can reach a few
   % of T, which is the 5 % tolerance with which `evaluate_stress_state` detects tensile tangency.
5. The sign of T (Section 11).

## 11. Sign of the tensile strength T

The Mohr diagram uses the compression-positive convention, so tension plots on the negative side
of the normal-stress axis and the tensile strength **T is always negative** (e.g. −1.5 MPa). In
`evaluate_stress_state`, the Griffith parabola `τ² = 4|T|(σn − T)` has its vertex at `σn = T`, and
the Coulomb line `τ = μσn − 2T` intersects the shear-stress axis at `−2T = 2|T|`, the cohesion
(C = 2|T|, Rivas-Dorado et al., 2023, eq. 12b).

**What was wrong before** (found on 2026-11-24 in `2a4e5f3`, fixed on 2026-11-24 in `ace6c94`; the convention, T always negative, was confirmed by Sam the same day). The function converted the inputs with `T = (−T)/1e6`, which turned
T positive. The rest of the function (`x < T` for points beyond the tensile strength, the parabola
vertex at `T`, the intercept `−2T`) is written for a negative T, so the envelope ended up in
compression. Most stable states were classified as `'Above'`: for example, the lithostatic state at
300 m (σ1 = 3.2, σ3 = 1.1 MPa) with T = −1.5 MPa, or σ3 = −1.4 MPa with |T| = 1.5 MPa. This was
independent of the size of the pressure steps of `dynamic_model`, and it was already present in
`main`. The conversion is now `T = T/1e6`, and a positive T raises `InputError`. With T negative,
the results are identical to those the old code gave with the same |T| passed as positive.

## References

- Pollard, D. D., and Segall, P. (1987). Theoretical displacements and stresses near fractures in
  rock: with applications to faults, joints, veins, dikes, and solution surfaces. In Atkinson,
  B. K. (ed.), *Fracture Mechanics of Rock*, Academic Press, 277–349.
- Pollard, D. D., and Martel, S. J. (2020). *Structural Geology: A Quantitative Introduction*.
  Cambridge University Press.
- Jaeger, J. C., Cook, N. G. W., and Zimmerman, R. W. (2007). *Fundamentals of Rock Mechanics*,
  4th ed. Blackwell.
- Westergaard, H. M. (1939). Bearing pressures and cracks. *Journal of Applied Mechanics*, 6, A49–A53.
- Rivas-Dorado, S., Ruíz, J., and Romeo, I. (2023). Modeling of dike-induced graben nucleation in the
  Elysium region, Mars: The role of planetary gravity. *Journal of Structural Geology*, 167, 104778.
