"""服务点接待容量分配。

把“可达”与“可接待”分开：路径可行只产生候选项，实际服务人数受
services.capacity（同一开放窗口）约束。主解用 MILP（先最大化接待总量，
再在不减少总量的前提下向弱势/高可靠度倾斜），另给透明贪心做交叉验证；
两者差异本身也是结果，不隐藏。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from .config import VULNERABILITY_WEIGHT


@dataclass
class Option:
    request: str
    residence: str
    profile: str
    service: str
    persons: float
    reliability: float          # Monte Carlo 可靠度
    p50_total: float            # 秒
    p90_total: float
    eligible: bool
    reason: str = ""            # 不合格原因
    path: tuple = ()


def _option_matrix(options: list[Option]) -> dict:
    groups = sorted({o.request for o in options})
    services = sorted({o.service for o in options})
    gidx = {g: i for i, g in enumerate(groups)}
    sidx = {s: j for j, s in enumerate(services)}
    return {"groups": groups, "services": services, "gidx": gidx, "sidx": sidx}


def milp_allocate(options: list[Option], capacities: dict[str, float],
                  persons_total: dict[str, float]) -> dict:
    """两级 MILP：最大接待量 → 弱势与可靠度加权的最优分配。"""
    elig = [o for o in options if o.eligible]
    if not elig:
        return {"allocation": {}, "served": 0.0, "stage": "empty"}
    mat = _option_matrix(elig)
    n = len(elig)
    c = np.zeros(n)

    # 第一级：最大化总接待（最小化 -1）
    res1 = milp(c=-np.ones(n),
                bounds=Bounds(np.zeros(n), np.array([o.persons for o in elig])),
                constraints=_constraints(elig, mat, capacities, persons_total),
                options={"disp": False})
    if not res1.success:
        return {"allocation": {}, "served": 0.0, "stage": f"milp失败:{res1.message}"}
    max_served = -float(res1.fun)

    # 第二级：固定总接待量，最大化弱势权重 × 可靠度
    w = np.array([VULNERABILITY_WEIGHT.get(o.profile, 1) * (0.5 + o.reliability)
                  for o in elig])
    extra = _constraints(elig, mat, capacities, persons_total)
    total_a = np.ones((1, n))
    extra.append(LinearConstraint(total_a, max_served - 1e-6, max_served + 1e-6))
    res2 = milp(c=-w, bounds=Bounds(np.zeros(n), np.array([o.persons for o in elig])),
                constraints=extra, options={"disp": False})
    x = res2.x if res2.success else res1.x
    alloc = {}
    for o, val in zip(elig, x):
        if val > 1e-6:
            alloc[(o.request, o.service)] = float(min(val, o.persons))
    return {"allocation": alloc, "served": float(sum(alloc.values())),
            "stage": "optimal", "max_served": max_served}


def _constraints(options: list[Option], mat: dict, capacities: dict,
                 persons_total: dict[str, float]) -> list[LinearConstraint]:
    n = len(options)
    cons = []
    # 服务点容量
    for s, j in mat["sidx"].items():
        a = np.zeros(n)
        for k, o in enumerate(options):
            if o.service == s:
                a[k] = 1.0
        cons.append(LinearConstraint(a.reshape(1, -1), -np.inf, capacities[s]))
    # 每组总人数
    for g, i in mat["gidx"].items():
        a = np.zeros(n)
        for k, o in enumerate(options):
            if o.request == g:
                a[k] = 1.0
        cons.append(LinearConstraint(a.reshape(1, -1), -np.inf, persons_total[g]))
    return cons


def greedy_allocate(options: list[Option], capacities: dict[str, float],
                    persons_total: dict[str, float]) -> dict:
    """透明贪心：弱势优先、可选服务少者优先，再按可靠度从高到低。"""
    by_group = {}
    for o in options:
        if o.eligible:
            by_group.setdefault(o.request, []).append(o)
    order = sorted(by_group, key=lambda g: (
        -VULNERABILITY_WEIGHT.get(by_group[g][0].profile, 1),
        len(by_group[g]), -max(o.reliability for o in by_group[g])))
    cap = dict(capacities)
    alloc = {}
    for g in order:
        remaining = persons_total[g]
        for o in sorted(by_group[g], key=lambda x: -x.reliability):
            take = min(remaining, cap.get(o.service, 0.0))
            if take > 1e-9:
                alloc[(g, o.service)] = take
                cap[o.service] -= take
                remaining -= take
            if remaining <= 1e-9:
                break
    return {"allocation": alloc, "served": float(sum(alloc.values())),
            "stage": "greedy"}
