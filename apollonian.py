"""Apollonian gasket: circle geometry, Descartes' theorem, brightness-driven DFS."""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Circle:
    z: complex  # center as complex number
    r: float    # radius
    k: float    # signed curvature: +1/r for interior, -1/r for outer bounding

    @property
    def cx(self) -> float:
        return self.z.real

    @property
    def cy(self) -> float:
        return self.z.imag

    @property
    def b(self) -> complex:
        # "bend times center", used in complex Descartes theorem
        return self.k * self.z


def _tangent_ok(c_new: Circle, c_ref: Circle, tol: float = 1e-3) -> bool:
    d = abs(c_new.z - c_ref.z)
    return (math.isclose(d, c_new.r + c_ref.r, abs_tol=tol) or
            math.isclose(d, abs(c_new.r - c_ref.r), abs_tol=tol))


def _find_two_descartes(c1: Circle, c2: Circle, c3: Circle) -> list[Circle]:
    """Given 3 mutually tangent circles, return the (up to 2) Descartes 4th circles."""
    k1, k2, k3 = c1.k, c2.k, c3.k
    disc_k = k1 * k2 + k2 * k3 + k3 * k1
    root_k = 2 * math.sqrt(max(0.0, disc_k))
    k4_pair = (k1 + k2 + k3 + root_k, k1 + k2 + k3 - root_k)

    b1, b2, b3 = c1.b, c2.b, c3.b
    root_b = 2 * cmath.sqrt(b1 * b2 + b2 * b3 + b3 * b1)
    b4_pair = (b1 + b2 + b3 + root_b, b1 + b2 + b3 - root_b)

    seen = set()
    results = []
    for k4 in k4_pair:
        if abs(k4) < 1e-12:
            continue
        for b4 in b4_pair:
            z4 = b4 / k4
            r4 = 1.0 / abs(k4)
            cand = Circle(z=z4, r=r4, k=k4)
            if not (_tangent_ok(cand, c1) and _tangent_ok(cand, c2) and _tangent_ok(cand, c3)):
                continue
            key = (round(z4.real, 3), round(z4.imag, 3), round(r4, 3))
            if key in seen:
                continue
            seen.add(key)
            results.append(cand)
    return results


def _reflect_new(triple: tuple[Circle, Circle, Circle], parent: Circle) -> Circle:
    """Given a triple + the known 4th (parent), return the OTHER Descartes 4th."""
    a, b, c = triple
    k_new = 2 * (a.k + b.k + c.k) - parent.k
    b_new = 2 * (a.b + b.b + c.b) - parent.b
    z_new = b_new / k_new
    r_new = 1.0 / abs(k_new)
    return Circle(z=z_new, r=r_new, k=k_new)


def make_initial(canvas_size: int, R: float, n: int, rng) -> tuple[Circle, list[Circle]]:
    """Build the outer big circle and 2 or 3 inner mutually-tangent small circles."""
    cx = cy = canvas_size / 2.0
    big = Circle(z=complex(cx, cy), r=R, k=-1.0 / R)

    if n == 2:
        for _ in range(30):
            r1 = rng.uniform(0.25 * R, 0.7 * R)
            r2 = rng.uniform(0.15 * R, R - r1 - 1e-3)
            if r2 <= 0.05 * R:
                continue
            cos_t = ((R - r1) ** 2 + (R - r2) ** 2 - (r1 + r2) ** 2) / (2 * (R - r1) * (R - r2))
            if abs(cos_t) > 1.0:
                continue
            theta = math.acos(cos_t)
            alpha1 = rng.uniform(0, 2 * math.pi)
            alpha2 = alpha1 + theta
            z1 = complex(cx + (R - r1) * math.cos(alpha1), cy + (R - r1) * math.sin(alpha1))
            z2 = complex(cx + (R - r2) * math.cos(alpha2), cy + (R - r2) * math.sin(alpha2))
            return big, [Circle(z=z1, r=r1, k=1.0 / r1),
                          Circle(z=z2, r=r2, k=1.0 / r2)]
        # fallback: two equal circles side by side
        r = R / 2.0
        return big, [Circle(z=complex(cx - r, cy), r=r, k=1.0 / r),
                      Circle(z=complex(cx + r, cy), r=r, k=1.0 / r)]

    # n == 3
    for _ in range(80):
        r1 = rng.uniform(0.2 * R, 0.5 * R)
        r2 = rng.uniform(0.2 * R, 0.5 * R)
        k_big, k1, k2 = -1.0 / R, 1.0 / r1, 1.0 / r2
        disc = k_big * k1 + k1 * k2 + k2 * k_big
        if disc <= 0:
            continue
        candidates = [k_big + k1 + k2 + 2 * math.sqrt(disc),
                      k_big + k1 + k2 - 2 * math.sqrt(disc)]
        candidates = [k for k in candidates if k > 1.0 / (0.55 * R)]
        if not candidates:
            continue
        k3 = min(candidates)  # smaller curvature = larger radius
        r3 = 1.0 / k3
        if r3 > 0.55 * R or r3 < 0.08 * R:
            continue

        cos_t = ((R - r1) ** 2 + (R - r2) ** 2 - (r1 + r2) ** 2) / (2 * (R - r1) * (R - r2))
        if abs(cos_t) > 1.0:
            continue
        theta = math.acos(cos_t)
        alpha1 = rng.uniform(0, 2 * math.pi)
        alpha2 = alpha1 + rng.choice([-1, 1]) * theta
        z1 = complex(cx + (R - r1) * math.cos(alpha1), cy + (R - r1) * math.sin(alpha1))
        z2 = complex(cx + (R - r2) * math.cos(alpha2), cy + (R - r2) * math.sin(alpha2))
        s1 = Circle(z=z1, r=r1, k=k1)
        s2 = Circle(z=z2, r=r2, k=k2)

        # Solve for s3 position via complex Descartes centers, filter to matching k3
        candidates3 = _find_two_descartes(big, s1, s2)
        s3 = None
        for cand in candidates3:
            if abs(cand.k - k3) / max(k3, 1e-9) < 0.02:
                s3 = cand
                break
        if s3 is None:
            continue
        return big, [s1, s2, s3]

    # symmetric fallback: 3 equal circles at 120°
    r = R * math.sqrt(3) / (2 + math.sqrt(3))
    d = R - r
    alpha0 = rng.uniform(0, 2 * math.pi)
    smalls = []
    for i in range(3):
        a = alpha0 + i * 2 * math.pi / 3
        z = complex(cx + d * math.cos(a), cy + d * math.sin(a))
        smalls.append(Circle(z=z, r=r, k=1.0 / r))
    return big, smalls


def _budget_from_brightness(b: float) -> int:
    if b > 0.8:
        return 4
    if b > 0.5:
        return 2
    if b > 0.2:
        return 1
    return 0


def _valid_new(c: Circle, canvas_size: int, min_r: float) -> bool:
    if not math.isfinite(c.r) or c.r < min_r:
        return False
    if c.k <= 0:  # only interior positive-curvature circles
        return False
    # circle must lie within the outer big circle roughly (bounds check on canvas)
    if not (0 <= c.cx <= canvas_size and 0 <= c.cy <= canvas_size):
        return False
    return True


def _recurse(triple, parent, budget, depth, out, brightness_fn, canvas_size, max_depth, min_r):
    if budget <= 0 or depth > max_depth:
        return
    new_c = _reflect_new(triple, parent)
    if not _valid_new(new_c, canvas_size, min_r):
        return
    b_val = brightness_fn(new_c.cx, new_c.cy, new_c.r)
    local = _budget_from_brightness(b_val)
    # Decouple from parent budget: bright regions refresh to their own local budget,
    # so recursion continues as long as brightness allows (bounded by max_depth).
    new_budget = local
    out.append((new_c, depth))
    if new_budget <= 0:
        return
    for i in range(3):
        sub_triple = tuple(x for j, x in enumerate(triple) if j != i) + (new_c,)
        sub_parent = triple[i]
        _recurse(sub_triple, sub_parent, new_budget, depth + 1,
                 out, brightness_fn, canvas_size, max_depth, min_r)


def build_gasket(big, smalls, brightness_fn, canvas_size,
                 max_depth=10, min_r=0.5):
    """Return list of (Circle, depth) covering the outer, the initial smalls, and all recursed."""
    out = [(big, 0)]
    for s in smalls:
        out.append((s, 0))

    if len(smalls) == 2:
        s1, s2 = smalls
        roots = _find_two_descartes(big, s1, s2)
        base = (big, s1, s2)
        for new_c in roots:
            if not _valid_new(new_c, canvas_size, min_r):
                continue
            b_val = brightness_fn(new_c.cx, new_c.cy, new_c.r)
            budget = _budget_from_brightness(b_val)
            out.append((new_c, 1))
            if budget <= 0:
                continue
            for excl in range(3):
                sub_triple = tuple(x for j, x in enumerate(base) if j != excl) + (new_c,)
                sub_parent = base[excl]
                _recurse(sub_triple, sub_parent, budget, 2, out,
                         brightness_fn, canvas_size, max_depth, min_r)
    else:
        quad = [big] + list(smalls)
        for parent_idx in range(4):
            triple = tuple(x for j, x in enumerate(quad) if j != parent_idx)
            parent = quad[parent_idx]
            new_c = _reflect_new(triple, parent)
            if not _valid_new(new_c, canvas_size, min_r):
                continue
            b_val = brightness_fn(new_c.cx, new_c.cy, new_c.r)
            budget = _budget_from_brightness(b_val)
            out.append((new_c, 1))
            if budget <= 0:
                continue
            for excl in range(3):
                sub_triple = tuple(x for j, x in enumerate(triple) if j != excl) + (new_c,)
                sub_parent = triple[excl]
                _recurse(sub_triple, sub_parent, budget, 2, out,
                         brightness_fn, canvas_size, max_depth, min_r)

    return out
