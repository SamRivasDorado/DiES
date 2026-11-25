"""
Explanatory figure of the analytical Griffith-Coulomb criterion: the critical radius
(docs/failure_criterion.md). Self-contained: the formulas are written here again, as
in failure_contact (Failure_Determination.py), so that the figure explains them.

    python docs/figures/critical_radius.py      # writes docs/figures/critical_radius.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MU, K = 0.57, 1.5          # coefficient of internal friction, |T| in MPa
ROOT = np.sqrt(1 + MU**2)


def envelope(sigma):
    """Shear stress of the Griffith-Coulomb envelope (MPa): parabola for sigma < 0, line for sigma >= 0."""
    sigma = np.asarray(sigma, dtype=float)
    return np.where(sigma < 0, np.sqrt(np.clip(4*K*(sigma + K), 0, None)), 2*K + MU*sigma)


def critical(c):
    """Critical radius r_c (distance from (c, 0) to the envelope), tangent point and branch."""
    sg = c - 2*K
    if sg < -K:
        dG, ptG, bG = abs(c + K), (-K, 0.0), 'vertex'
    elif sg > 0:
        dG, ptG, bG = np.hypot(c, 2*K), (0.0, 2*K), 'kink'
    else:
        dG, ptG, bG = np.sqrt(4*K*c), (sg, np.sqrt(4*K*(sg + K))), 'parabola'
    dC = (MU*c + 2*K)/ROOT
    sc = c - dC*MU/ROOT
    if sc >= 0:
        dCc, ptC, bC = dC, (sc, 2*K + MU*sc), 'line'
    else:
        dCc, ptC, bC = np.hypot(c, 2*K), (0.0, 2*K), 'kink'
    return (dG, ptG, bG) if dG <= dCc else (dCc, ptC, bC)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1.15, 1]})

# ---- Panel 1: one centre, three radii (below, at and above failure)
sig = np.linspace(-K, 14, 600)
ax1.plot(sig, envelope(sig), c='k', lw=2, label='Griffith-Coulomb envelope')
ax1.axvline(0, c='0.7', lw=0.8, ls=':')
c = 7.0
rc, (sn, ss), _ = critical(c)
t = np.linspace(0, np.pi, 300)
for f, col, lab in ((0.8, 'tab:blue', 'r < r_c: below failure'),
                    (1.0, 'tab:green', 'r = r_c: at failure (tangent)'),
                    (1.2, 'tab:red', 'r > r_c: above failure')):
    r = f*rc
    ax1.plot(c + r*np.cos(t), r*np.sin(t), c=col, lw=2 if f == 1 else 1.4, label=lab)
ax1.plot([c, sn], [0, ss], c='tab:green', ls='--', lw=1.2)
ax1.annotate('r$_c$ = distance from the\ncentre to the envelope', xy=((c + sn)/2, ss/2), xytext=(9.2, 1.2),
             fontsize=10, arrowprops=dict(arrowstyle='->', lw=0.8))
ax1.plot([c], [0], 'ko', ms=4)
ax1.annotate('centre c = (σ$_1$+σ$_3$)/2', xy=(c, 0), xytext=(c - 1.6, -1.2), fontsize=10)
ax1.plot([sn], [ss], 'o', c='tab:green', ms=7)
ax1.annotate('tangent point:\n(σ$_n$, τ) on the\nfailure plane', xy=(sn, ss), xytext=(-2.3, 4.6), fontsize=9,
             arrowprops=dict(arrowstyle='->', lw=0.8))
ax1.set_xlim(-2.5, 14) ; ax1.set_ylim(-2, 10.5) ; ax1.set_aspect('equal')
ax1.set_xlabel('σ$_n$ (MPa, compression positive)') ; ax1.set_ylabel('τ (MPa)')
ax1.set_title('A Mohr circle of centre c touches the envelope\nwhen its radius r = (σ$_1$−σ$_3$)/2 equals r$_c$(c)', fontsize=11)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax1.text(-2.3, -1.75, 'T = −1.5 MPa, μ′ = 0.57', fontsize=9)

# ---- Panel 2: r_c as a function of the centre, coloured by the part of the envelope it touches
cs = np.linspace(-1.4, 14, 800)
rcs = np.array([critical(v)[0] for v in cs])
branch = np.array([critical(v)[2] for v in cs])
zones = (('vertex', 'tab:purple', 'Mode I: touches the vertex (σ$_3$ = T)\nr$_c$ = c + |T|'),
         ('parabola', 'tab:orange', 'Mixed-mode I-II: tangent to the parabola\nr$_c$ = √(4|T|c)'),
         ('line', 'tab:brown', 'Mode II: tangent to the line\nr$_c$ = (2|T| + μ′c)/√(1+μ′²)'))
for name, col, lab in zones:
    m = branch == name
    ax2.plot(cs[m], rcs[m], c=col, lw=3, label=lab)
m = branch == 'kink'
if m.any():
    ax2.plot(cs[m], rcs[m], c='0.4', lw=3, label='touches the kink at σ$_n$ = 0')
ax2.fill_between(cs, 0, rcs, color='tab:blue', alpha=0.08)
ax2.fill_between(cs, rcs, 16, color='tab:red', alpha=0.08)
ax2.text(9.5, 3.0, 'below failure\n(r < r$_c$)', color='tab:blue', fontsize=10, ha='center')
ax2.text(11.5, 12.0, 'above failure\n(r > r$_c$)', color='tab:red', fontsize=10, ha='center')
ax2.axvline(K, c='0.6', lw=0.8, ls=':') ; ax2.axvline(2*K, c='0.6', lw=0.8, ls=':')
ax2.text(K - 0.1, 0.3, 'c = |T|', fontsize=8, ha='right') ; ax2.text(2*K + 0.1, 0.3, 'c = 2|T|', fontsize=8, ha='left')
ax2.set_xlim(-1.5, 14) ; ax2.set_ylim(0, 16)
ax2.set_xlabel('centre of the circle c (MPa)') ; ax2.set_ylabel('critical radius r$_c$ (MPa)')
ax2.set_title('Each node: compute c and r, compare r with r$_c$(c)', fontsize=11)
ax2.legend(loc='upper left', fontsize=8.5, framealpha=0.95)

fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'critical_radius.png'), dpi=130)
print('saved')
