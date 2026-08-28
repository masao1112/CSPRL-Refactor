"""
Verification for the M/M/c/N station queue in helpers.avg_waiting.

Two claims are being checked, and they are independent of each other:

  1. Multi-server is a strict generalisation. With one charger the queue must
     reproduce the single-server M/M/1/N formulae the model used before, to
     floating-point precision, at every traffic intensity -- including rho = 1
     and rho > 1, where the old code needed separate algebraic branches.

  2. The system capacity is structural. N = (1 + kappa) * c depends on the site,
     never on the demand routed to it. The demand-sized buffer it replaces made
     the blocking probability non-monotone in D, so monotonicity of P_N in D is
     the property that pins the fix down.

The remaining tests cover the behaviour that motivated the change (charging time
must follow the power of ONE charger, not the pooled station power), the two
limits of kappa, numerical stability where the closed form overflows, and the
saturation identity D * P_N -> D - mu that makes unserved demand a
buffer-independent congestion signal.

Usage:
    python tests/test_queue_mmcn.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import custom_environment.helpers as H

POWER_INDEX = {int(p): i for i, p in enumerate(H.CHARGING_POWER)}

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print("  [{}] {}{}".format(status, name, "  --  " + detail if detail else ""))
    if not ok:
        FAILURES.append(name)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def make_station(n_chargers, power_kw, demand):
    """A bare station of n identical chargers with D(s) = demand routed to it."""
    s_x = np.zeros(len(H.CHARGING_POWER), dtype=int)
    s_x[POWER_INDEX[power_kw]] = n_chargers
    station = [(0, {"x": 0.0, "y": 0.0, "land_price": 0.0}), s_x, {}]
    H.charging_capability(station)
    H.service_rate(station)
    station[2]["D_s"] = demand
    return station


def solve(n_chargers, power_kw, demand, kappa=None):
    station = make_station(n_chargers, power_kw, demand)
    H.avg_waiting(station, kappa=kappa)
    return station[2]


def mm1n_reference(rho, N, D, eps=1e-9):
    """Closed-form M/M/1/N, written out independently of the implementation."""
    if math.isclose(rho, 1.0, rel_tol=1e-7):
        PN = 1.0 / (N + 1)
        L = N / 2.0
    elif rho > 1.0:
        u = 1.0 / rho
        PN = (1.0 - u) / (1.0 - u ** (N + 1))
        L = (N - (N + 1) * u + u ** (N + 1)) / ((1.0 - u) * (1.0 - u ** (N + 1)))
    else:
        PN = (rho ** N * (1.0 - rho)) / (1.0 - rho ** (N + 1))
        L = rho / (1.0 - rho) - ((N + 1) * rho ** (N + 1)) / (1.0 - rho ** (N + 1))
    return PN, L / (D * (1.0 - PN) + eps)


def erlang_b(a, c):
    """Erlang loss formula by the standard recursion."""
    b = 1.0
    for n in range(1, c + 1):
        b = a * b / (n + a * b)
    return b


def aggregate_rate(power_kw, n):
    return n * power_kw / H.BATTERY


def test_single_server_equivalence():
    print("\n1. c = 1 reproduces M/M/1/N exactly")
    # kappa fixes N = (1 + kappa) * 1, so the reference gets the same N.
    for kappa in (0.0, 1.0, 4.0):
        N = int(1 + math.ceil(kappa))
        for power_kw, demand in [(250, 1), (250, 3), (120, 5), (30, 12), (7, 40)]:
            d = solve(1, power_kw, demand, kappa=kappa)
            rho = demand / aggregate_rate(power_kw, 1)
            PN_ref, W_ref = mm1n_reference(rho, N, demand)
            check("kappa={} {}kW D={}  P_N".format(kappa, power_kw, demand),
                  close(d["P_N"], PN_ref, 1e-9),
                  "impl={:.12f} ref={:.12f}".format(d["P_N"], PN_ref))
            check("kappa={} {}kW D={}  W_s".format(kappa, power_kw, demand),
                  close(d["W_s"], W_ref, 1e-9),
                  "impl={:.9f}h ref={:.9f}h".format(d["W_s"], W_ref))


def test_capacity_is_structural():
    print("\n2. N is structural: P_N and unserved are monotone in D")
    # The demand-sized buffer broke exactly here: at 20x150 kW it gave
    # P_N(D=20) = 0.00605 > P_N(D=21) = 0.00541.
    prev_pn, prev_unserved = -1.0, -1.0
    monotone = True
    for D in range(1, 61):
        d = solve(20, 150, D)
        if d["P_N"] < prev_pn - 1e-15 or d["unserved"] < prev_unserved - 1e-12:
            monotone = False
            print("      break at D={}: P_N {:.6f} -> {:.6f}".format(D, prev_pn, d["P_N"]))
        prev_pn, prev_unserved = d["P_N"], d["unserved"]
    check("P_N and D*P_N non-decreasing in D over D=1..60 (20x150 kW)", monotone)

    # ...and the buffer itself must not move with demand.
    caps = set()
    for D in (1, 7, 30, 200):
        caps.add(solve(8, 30, D)["c_servers"])
    check("c(s) independent of D", caps == {8}, "c_servers seen: {}".format(sorted(caps)))


def test_charging_time_follows_one_charger():
    print("\n3. Charging time follows the power of ONE charger")
    # Same C(s) = 240 kW, same rho, different granularity. The single-server
    # model returned ~30 min for all three; the sojourn time must instead be at
    # least 1/mu_1 = E_avg / (C/n), the time one charger needs for one session.
    for n, kw in [(2, 120), (4, 60), (8, 30)]:
        d = solve(n, kw, 2)
        one_charger_h = H.BATTERY / kw
        check("{}x{}kW light load -> W_s >= 1/mu_1".format(n, kw),
              d["W_s"] >= one_charger_h - 1e-9,
              "W_s={:.1f} min, 1/mu_1={:.1f} min".format(d["W_s"] * 60, one_charger_h * 60))

    # Splitting the same capacity more finely must never look cheaper.
    ws = [solve(n, kw, 8)["W_s"] for n, kw in [(2, 120), (4, 60), (8, 30)]]
    check("W_s increases as the same 240 kW is split more finely",
          ws[0] < ws[1] < ws[2],
          " < ".join("{:.1f}min".format(w * 60) for w in ws))


def test_kappa_limits():
    print("\n4. kappa endpoints")
    # kappa = 0 is the Erlang loss system: no waiting room, so the sojourn time
    # is exactly one charging session and P_N is the Erlang-B formula.
    for n, kw, D in [(8, 30, 12), (20, 150, 30), (4, 60, 3)]:
        d = solve(n, kw, D, kappa=0.0)
        a = D / (kw / H.BATTERY)
        check("kappa=0 {}x{}kW D={}  P_N == Erlang-B".format(n, kw, D),
              close(d["P_N"], erlang_b(a, n), 1e-9),
              "impl={:.9f} erlangB={:.9f}".format(d["P_N"], erlang_b(a, n)))
        check("kappa=0 {}x{}kW D={}  W_s == 1/mu_1".format(n, kw, D),
              close(d["W_s"], H.BATTERY / kw, 1e-9),
              "W_s={:.4f}h 1/mu_1={:.4f}h".format(d["W_s"], H.BATTERY / kw))

    # Growing kappa lengthens the wait but leaves blocking essentially fixed:
    # that is why unserved demand, not W_s, is the robust congestion signal.
    ws = [solve(8, 30, 12, kappa=k)["W_s"] for k in (0.5, 1.0, 2.0, 4.0)]
    pns = [solve(8, 30, 12, kappa=k)["P_N"] for k in (0.5, 1.0, 2.0, 4.0)]
    check("W_s grows with kappa", all(x < y for x, y in zip(ws, ws[1:])),
          " < ".join("{:.0f}min".format(w * 60) for w in ws))
    check("P_N nearly invariant to kappa under saturation",
          max(pns) - min(pns) < 1e-2,
          "spread={:.2e}".format(max(pns) - min(pns)))


def test_numerical_stability():
    print("\n5. Numerical stability where the closed form overflows")
    # A large site of slow chargers drives the offered load into the thousands of
    # Erlang, and a ** N then leaves float64 range. The log-space recursion is
    # what keeps this reachable configuration computable at all.
    for n, kw, D in [(60, 3, 200), (100, 7, 400), (30, 3, 100)]:
        d = solve(n, kw, D)
        a = D / (kw / H.BATTERY)
        N = d["N_sys"]
        try:
            a ** N
            overflows = False
        except OverflowError:
            overflows = True
        print("      {}x{}kW D={}: a={:.0f} Erlang, N={}, a**N overflows: {}".format(
            n, kw, D, a, N, overflows))
        check("{}x{}kW D={}  W_s finite".format(n, kw, D),
              math.isfinite(d["W_s"]), "W_s={:.3f} h".format(d["W_s"]))
        check("{}x{}kW D={}  P_N in [0,1]".format(n, kw, D),
              0.0 <= d["P_N"] <= 1.0, "P_N={:.6f}".format(d["P_N"]))

    # At least one of the above must actually be out of reach for the closed
    # form, otherwise this test proves nothing.
    a = 200 / (3 / H.BATTERY)
    N = solve(60, 3, 200)["N_sys"]
    try:
        a ** N
        overflowed = False
    except OverflowError:
        overflowed = True
    check("60x3kW D=200 really does overflow a**N (guard is not vacuous)", overflowed,
          "a={:.0f}, N={}, a**N ~ 1e{:.0f}".format(a, N, N * math.log10(a)))


def test_saturation_identity():
    print("\n6. Saturation: D * P_N -> D - mu, independent of kappa")
    for n, kw, D in [(8, 30, 12), (20, 7, 60), (2, 250, 20)]:
        mu = aggregate_rate(kw, n)
        shortfall = D - mu
        for kappa in (0.5, 1.0, 2.0, 4.0):
            d = solve(n, kw, D, kappa=kappa)
            check("{}x{}kW D={} kappa={}  unserved ~ D-mu".format(n, kw, D, kappa),
                  abs(d["unserved"] - shortfall) < 0.05 * max(1.0, abs(shortfall)),
                  "unserved={:.4f} D-mu={:.4f}".format(d["unserved"], shortfall))


def test_aggregate_rate_preserved():
    print("\n7. Aggregate service rate unchanged (rho keeps its old meaning)")
    for n, kw in [(1, 250), (8, 30), (20, 150)]:
        st = make_station(n, kw, 5)
        mu, mu1, c = st[2]["service rate"], st[2]["service rate 1"], st[2]["c_servers"]
        check("{}x{}kW  c * mu_1 == mu".format(n, kw), close(c * mu1, mu),
              "c={} mu_1={:.4f} mu={:.4f}".format(c, mu1, mu))


def test_edge_cases():
    print("\n8. Edge cases")
    d = solve(4, 60, 0)
    check("no demand -> no wait, no blocking", d["W_s"] == 0.0 and d["P_N"] == 0.0)

    s_x = np.zeros(len(H.CHARGING_POWER), dtype=int)
    empty = [(0, {"x": 0.0, "y": 0.0, "land_price": 0.0}), s_x, {}]
    H.charging_capability(empty)
    H.service_rate(empty)
    empty[2]["D_s"] = 5
    H.avg_waiting(empty)
    check("no charger -> everything blocked",
          empty[2]["P_N"] == 1.0 and empty[2]["unserved"] == 5,
          "W_s={}".format(empty[2]["W_s"]))

    plan = [make_station(8, 30, 12), make_station(20, 150, 30)]
    for st in plan:
        H.avg_waiting(st)
    total = sum(st[2]["unserved"] for st in plan)
    check("unserved_demand sums the plan", close(H.unserved_demand(plan), total / H.time_unit),
          "{:.4f} veh/h".format(H.unserved_demand(plan)))

    # unserved_ratio is normalised by the plan's own total demand, not by p_0,
    # so it is an absolute fraction and needs no baseline threaded into it.
    demand = sum(st[2]["D_s"] for st in plan)
    ratio = H.unserved_ratio(plan)
    check("unserved_ratio == unserved_demand / sum D(s)", close(ratio, total / demand),
          "{:.6f}".format(ratio))
    check("unserved_ratio in [0,1]", 0.0 <= ratio <= 1.0)
    check("unserved_ratio of an empty plan is 0", H.unserved_ratio([]) == 0.0)


if __name__ == "__main__":
    print("M/M/c/N queue verification (kappa default = {})".format(H.WAIT_BAY_RATIO))
    test_single_server_equivalence()
    test_capacity_is_structural()
    test_charging_time_follows_one_charger()
    test_kappa_limits()
    test_numerical_stability()
    test_saturation_identity()
    test_aggregate_rate_preserved()
    test_edge_cases()

    print("\n" + "=" * 60)
    if FAILURES:
        print("{} FAILED:".format(len(FAILURES)))
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print("all checks passed")
