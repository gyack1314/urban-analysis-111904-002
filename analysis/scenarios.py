"""项目情景构造与可达性评估引擎。

情景轴：
  网络状态（基线/恢复未兑现/电梯完全失效/各项目施工与完工/组合/敏感性）
  × 需求情景（人口三档 × R1 出口两个），两者交叉但参数不互相挑选。

评估流程（每个网络 × 需求情景）：
  1. 静态物理过滤得到各群体可行边与阻断原因；
  2. 全部请求组、全部出发时段联合做“路由 ↔ 电梯排队”不动点迭代，
     候梯时间随全体居民的共同路径选择变化，而不是各组单独排队；
  3. 固定路径上对每个出发时段做 Monte Carlo，按时段人数份额加权得到
     时限/开放窗口内到达的可靠度与用时分位（早到计入等待，不判失败）；
  4. 合格选项交给 MILP/贪心分配服务容量，产出服务量与未满足需求。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .allocate import Option, greedy_allocate, milp_allocate
from .demand import departure_bins
from .feasibility import (build_usable_edges, next_availability_vectorized,
                          profiles_from_df)
from .travel import EdgeTimeModel, TimeDependentRouter, simulate_weighted_lift_queue


# --------------------------------------------------------------- 网络情景
@dataclass
class NetworkState:
    name: str
    kind: str                     # baseline / construction / completion / sensitivity
    description: str
    cost_yuan: float = 0.0
    construction_days: int = 0
    capacity_overrides: dict = field(default_factory=dict)   # service -> capacity
    weak_flags: list = field(default_factory=list)


def _project_map(projects: pd.DataFrame) -> dict[str, dict]:
    return {p["project"]: p for p in projects.to_dict("records")}


def build_scenario_registry(projects: pd.DataFrame, budget: float) -> list[NetworkState]:
    pm = _project_map(projects)
    reg = [
        NetworkState("baseline", "baseline",
                     "基线：actions.json 的 L1 午间恢复生效，停运收缩为 660–720"),
        NetworkState("baseline_no_restore", "sensitivity",
                     "反事实：L1 恢复未兑现，停运维持 660–780",
                     weak_flags=["L1恢复为已宣布动作，未兑现时本情景展示后果"]),
        NetworkState("lift_L1_failed", "sensitivity",
                     "极端：L1 整个开放窗口不可用（地图最短路径仍在但梯不可乘）",
                     weak_flags=["梯边 e4/e4r 全天移除，用于停运风险压力测试"]),
    ]
    for pid in ("P1", "P2", "P3"):
        p = pm[pid]
        reg.append(NetworkState(
            f"{pid}_construction", "construction",
            f"{pid}施工期：封闭 {p['closed_edges']}，新增边尚未可用",
            cost_yuan=float(p["cost_yuan"]), construction_days=int(p["construction_days"]),
            weak_flags=[f"封闭边按契约字面单向执行：{p['closed_edges']}"]))
        caps = {}
        if p["service_capacity_target"] and p["new_capacity"] is not None:
            caps[p["service_capacity_target"]] = float(p["new_capacity"])
        reg.append(NetworkState(
            f"{pid}_completion", "completion",
            f"{pid}完工：封闭边恢复，新边/改善生效"
            + (f"，{p['service_capacity_target']} 窗口容量改为 {p['new_capacity']}" if caps else ""),
            cost_yuan=float(p["cost_yuan"]), construction_days=int(p["construction_days"]),
            capacity_overrides=caps))
    p1, p2 = pm["P1"], pm["P2"]
    reg.append(NetworkState(
        "P1P2_construction", "construction",
        "P1+P2 同期施工：e4、e7 封闭，新边均未可用",
        cost_yuan=float(p1["cost_yuan"] + p2["cost_yuan"]),
        construction_days=max(int(p1["construction_days"]), int(p2["construction_days"])),
        weak_flags=["同期施工假设，实际排期可能错开"]))
    reg.append(NetworkState(
        "P1P2_completion", "completion",
        "P1+P2 完工：N1→N4→S1 贯通（两项目单独都不成立），S1 容量 45",
        cost_yuan=float(p1["cost_yuan"] + p2["cost_yuan"]),
        construction_days=max(int(p1["construction_days"]), int(p2["construction_days"])),
        capacity_overrides={"S1": float(p2["new_capacity"])}))
    reg.append(NetworkState(
        "P1P2_construction_closeboth", "construction",
        "敏感性：施工封闭按双向理解（e4 同时封 e4r、e7 同时封 e7r）",
        cost_yuan=float(p1["cost_yuan"] + p2["cost_yuan"]),
        construction_days=max(int(p1["construction_days"]), int(p2["construction_days"])),
        weak_flags=["契约 closed_edges 为单向列表，本情景仅作敏感性对照"]))
    reg.append(NetworkState(
        "P3_completion_bothdir", "sensitivity",
        "敏感性：P3 坡度改善同时作用于 e5 与 e5r（契约字面只作用 e5）",
        cost_yuan=float(pm["P3"]["cost_yuan"]),
        construction_days=int(pm["P3"]["construction_days"]),
        weak_flags=["宽度 1.1m 仍不满足轮椅 1.2m，故该敏感性不改变轮椅可达性"]))
    for s in reg:
        s.within_budget = s.cost_yuan <= budget  # type: ignore[attr-defined]
    return reg


def apply_network_state(state: NetworkState, inputs) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """返回 (调整后的 network 表, 实际停运表, 容量覆盖)。"""
    net = inputs.network.copy()
    outages = inputs._outages
    pm = _project_map(inputs.projects)

    if state.name == "baseline_no_restore":
        outages = inputs.outages.copy()
    elif state.name == "lift_L1_failed":
        w = inputs.lift_windows.loc[inputs.lift_windows["lift"] == "L1"].iloc[0]
        outages = pd.concat([inputs.outages, pd.DataFrame([{
            "lift": "L1", "start": float(w["open"]), "end": float(w["close"]),
            "reason": "压力测试全天失效"}])], ignore_index=True)

    closed, added, improved = [], [], []
    # 注意：必须按状态名判断施工/完工，不能用 kind——部分敏感性状态
    # （如双向封闭）kind="sensitivity" 但语义上仍是施工期。
    if state.name.startswith("P1_"):
        is_construction = state.name.endswith("construction")
        closed, added, improved = _project_edges(pm["P1"],
                                                 "construction" if is_construction else "completion")
    elif state.name.startswith("P2_"):
        is_construction = state.name.endswith("construction")
        closed, added, improved = _project_edges(pm["P2"],
                                                 "construction" if is_construction else "completion")
    elif state.name.startswith("P3_"):
        p = pm["P3"]
        if state.name.endswith("construction"):
            closed = list(p["closed_edges"])
        elif state.name == "P3_completion_bothdir":
            improved = [("e5", float(p["new_slope"])), ("e5r", -float(p["new_slope"]))]
        else:
            improved = [("e5", float(p["new_slope"]))]
    elif state.name.startswith("P1P2_"):
        is_construction = "construction" in state.name
        kind = "construction" if is_construction else "completion"
        c1, a1, _ = _project_edges(pm["P1"], kind)
        c2, a2, _ = _project_edges(pm["P2"], kind)
        closed, added = c1 + c2, a1 + a2
        if state.name == "P1P2_construction_closeboth":
            closed = closed + ["e4r", "e7r"]

    if closed:
        net = net[~net["edge"].isin(closed)].copy()
    rows = net.to_dict("records")
    existing = {r["edge"] for r in rows}
    for ne in added:
        if ne["edge"] not in existing:
            rows.append(dict(ne))
    for eid, slope in improved:
        for r in rows:
            if r["edge"] == eid:
                r["slope"] = slope
    return pd.DataFrame(rows), outages, state.capacity_overrides


def _project_edges(p: dict, kind: str):
    if kind == "construction":
        return list(p["closed_edges"]), [], []
    added = [dict(p["new_edge"])] if p["new_edge"] else []
    improved = [(p["improve_edge"], float(p["new_slope"]))] if p["improve_edge"] else []
    return [], added, improved


# --------------------------------------------------------------- 评估引擎
class Engine:
    def __init__(self, inputs, outages_effective):
        self.inputs = inputs
        self.profiles = profiles_from_df(inputs.profiles)
        inputs._outages = outages_effective
        self.rng = np.random.default_rng(config.RNG_SEED)
        self._sample_cache: dict[tuple[str, str], np.ndarray] = {}

    def _samples(self, model: EdgeTimeModel) -> np.ndarray:
        """每条（边,群体）预抽一组用时样本，跨情景复用，保证可复现。"""
        key = (model.edge, model.profile)
        if key not in self._sample_cache:
            self._sample_cache[key] = self.rng.lognormal(
                model.mu, model.sigma, config.MC_DRAWS)
        return self._sample_cache[key]

    def evaluate(self, state: NetworkState, demand_scenario) -> dict:
        inputs = self.inputs
        net, outages, cap_override = apply_network_state(state, inputs)
        models = inputs._base_models
        usable, blocked = build_usable_edges(net, self.profiles)
        edge_rows = {r["edge"]: r for r in net.to_dict("records")}
        lift_edges = {r["edge"]: r["lift_id"] for r in net.to_dict("records")
                      if r["lift_id"]}
        win_map = {w["lift"]: w for w in inputs.lift_windows.to_dict("records")}
        svc_info = {r["service"]: r for r in inputs.services.to_dict("records")}
        caps = {r["service"]: float(r["capacity"])
                for r in inputs.services.to_dict("records")}
        caps.update(cap_override)

        reqs = inputs.requests.to_dict("records")
        bins_by_req = {r["request"]: departure_bins(r) for r in reqs}
        persons_by_req = {r["request"]: float(r["persons"]) * demand_scenario.scale
                          for r in reqs}
        targets_by_req = {r["request"]: {s for s, info in svc_info.items()
                                         if info["category"] == r["category"]}
                          for r in reqs}

        routes, wait_tables = self._fixed_point(
            reqs, bins_by_req, demand_scenario, usable, models, edge_rows,
            lift_edges, win_map, outages, persons_by_req)

        options, route_evidence = [], []
        for req in reqs:
            rid, prof = req["request"], req["profile"]
            for svc in sorted(targets_by_req[rid]):
                rel, p50, p90, fails, paths_seen, weak = self._reliability(
                    req, svc, self._routes_for(routes, rid), bins_by_req[rid],
                    models, prof, outages, lift_edges, win_map, edge_rows,
                    svc_info[svc], wait_tables)
                persons = persons_by_req[rid]
                reachable_persons = persons * rel
                eligible = rel >= config.RELIABILITY_THRESHOLD
                reason = "" if eligible else \
                    f"加权可靠度{rel:.2f}<{config.RELIABILITY_THRESHOLD}；" \
                    + ",".join(sorted(set(fails))[:3])
                options.append(Option(rid, req["residence"], prof, svc,
                                      reachable_persons, rel, p50, p90,
                                      eligible, reason, tuple(paths_seen[0][0]
                                                              if paths_seen else ())))
                route_evidence.append({
                    "state": state.name, "demand_scenario": demand_scenario.name,
                    "request": rid, "service": svc,
                    "paths": [list(p) for p, _ in paths_seen],
                    "path_share": [round(w, 3) for _, w in paths_seen],
                    "reliability": round(rel, 3),
                    "reachable_persons": round(reachable_persons, 2),
                    "p50_seconds": None if math.isinf(p50) else round(p50, 1),
                    "p90_seconds": None if math.isinf(p90) else round(p90, 1),
                    "weak_edges": sorted(set(weak)),
                    "fail_reasons": sorted(set(fails)),
                })
        result = self._allocate(options, caps, persons_by_req)
        result["routes"] = route_evidence
        result["capacity"] = caps
        result["blocked"] = blocked
        result["edges"] = sorted(edge_rows)
        result["nominal_persons"] = dict(persons_by_req)
        return result

    # -------------------------------------------------- 路由↔排队不动点
    def _fixed_point(self, reqs, bins_by_req, demand_scenario, usable, models,
                     edge_rows, lift_edges, win_map, outages, persons_by_req):
        wait_tables: dict[str, dict[float, float]] = {}
        routes: dict[tuple[str, float], dict] = {}
        prev_signature = None
        for _ in range(config.ASSIGNMENT_MAX_ITERS):
            router = TimeDependentRouter(edge_rows, usable, models,
                                         self.inputs.lift_windows, outages,
                                         wait_funcs=wait_tables, quantile=0.5)
            routes = {}
            for req in reqs:
                origin = demand_scenario.exit_map[req["residence"]]
                for t, _share in bins_by_req[req["request"]]:
                    routes[(req["request"], t)] = router.shortest(
                        req["profile"], origin, t,
                        targets_by_req_set(req, self.inputs))
            # 联合收集全部请求组在各电梯前的到达；
            # 同一请求组到不同服务点的备选路径若共用一部梯，只计一次（取最早到达），
            # 因为一个居民不会因备选目的地而重复乘梯。
            arrivals: dict[str, list[tuple[float, float]]] = {}
            for req in reqs:
                rid, prof = req["request"], req["profile"]
                bin_n = len(bins_by_req[rid])
                for t, _share in bins_by_req[rid]:
                    batch = persons_by_req[rid] / bin_n
                    first_arrival: dict[str, float] = {}
                    for sol in routes[(rid, t)].values():
                        cur = t
                        for e in sol["path"]:
                            if lift_edges.get(e):
                                lift = lift_edges[e]
                                if lift not in first_arrival or cur < first_arrival[lift]:
                                    first_arrival[lift] = cur
                            cur += models[(e, prof)].quantile(0.5) / 60.0
                    for lift, at in first_arrival.items():
                        arrivals.setdefault(lift, []).append((at, batch))
            wait_tables = {
                lift: simulate_weighted_lift_queue(evs, win_map[lift], outages)
                for lift, evs in arrivals.items()}
            signature = tuple(sorted(
                (k, t, tuple(sorted((s, tuple(v["path"]))
                                    for s, v in sols.items())))
                for (k, t), sols in routes.items()))
            if signature == prev_signature:
                break
            prev_signature = signature
        return routes, wait_tables

    def _routes_for(self, routes, rid):
        return {t: sols for (k, t), sols in routes.items() if k == rid}

    # -------------------------------------------------- Monte Carlo 可靠度
    def _reliability(self, req, svc, per_bin, bins, models, prof, outages,
                     lift_edges, win_map, edge_rows, service, wait_tables):
        n = config.MC_DRAWS
        rels, p50s, p90s, fails = [], [], [], []
        path_weight: dict[tuple, float] = {}
        weak = []
        for (t, share) in bins:
            sol = per_bin.get(t, {}).get(svc)
            if sol is None:
                rels.append(0.0)
                p50s.append(math.inf)
                p90s.append(math.inf)
                fails.append("无物理可行路径")
                continue
            path = tuple(sol["path"])
            path_weight[path] = path_weight.get(path, 0.0) + share
            totals = np.full(n, t * 60.0)  # 当前时钟（秒）
            ok = np.ones(n, dtype=bool)
            bin_fails = set()
            for e in path:
                m = models[(e, prof)]
                if m.weak:
                    weak.append(e)
                d = self._samples(m).copy()
                floor = m.evidence.get("conservative_floor_seconds")
                if floor:
                    d = np.maximum(d, float(floor))
                if lift_edges.get(e):
                    lift = lift_edges[e]
                    cur_min = totals / 60.0
                    avail_min = next_availability_vectorized(
                        lift, cur_min, self.inputs.lift_windows, outages)
                    no_lift = np.isinf(avail_min)
                    table = wait_tables.get(lift, {})
                    keys = sorted(table)
                    if keys:
                        idx = np.searchsorted(keys, cur_min, side="right") - 1
                        idx = np.clip(idx, 0, len(keys) - 1)
                        queue_wait = np.array([table[keys[i]] for i in idx])
                    else:
                        queue_wait = np.full(n, np.inf)
                    wait_sec = np.maximum(avail_min * 60.0 - totals,
                                          queue_wait * 60.0)
                    ok &= ~no_lift & ~np.isinf(queue_wait)
                    totals = np.where(ok, totals + wait_sec + d, totals)
                else:
                    totals = totals + d
            # 服务窗口：早到等待开门；关门后到达或超时限即失败
            open_sec, close_sec = service["open"] * 60.0, service["close"] * 60.0
            waited_open = np.maximum(totals, open_sec)
            total_time = waited_open - t * 60.0
            success = ok & (waited_open < close_sec) & \
                (total_time <= float(req["max_total_seconds"]))
            rel = float(success.mean())
            rels.append(rel)
            good = total_time[success]
            if len(good):
                p50s.append(float(np.median(good)))
                p90s.append(float(np.quantile(good, 0.9)))
            else:
                p50s.append(math.inf)
                p90s.append(math.inf)
                bin_fails.add("该出发时段无成功抽样")
            if rel < 1.0:
                if (waited_open >= close_sec).any():
                    bin_fails.add("到达不早于关门")
                if (total_time > float(req["max_total_seconds"])).any():
                    bin_fails.add("超过可接受总时间")
                if (~ok).any():
                    bin_fails.add("电梯闭门前无法乘梯")
            fails.extend(bin_fails)
        weights = np.array([w for _t, w in bins])
        rel_w = float(np.dot(rels, weights))
        # p50/p90 只在“有成功抽样”的时段间、按份额重归一化后加权；
        # 失败时段不贡献伪零值，避免把条件用时人为拉低。
        def conditional_weighted(values):
            mask = np.array([math.isfinite(v) for v in values], dtype=bool)
            if not mask.any():
                return math.inf
            w = weights[mask] / weights[mask].sum()
            return float(np.dot(np.array(values)[mask], w))
        p50_w = conditional_weighted(p50s)
        p90_w = conditional_weighted(p90s)
        paths_seen = sorted(path_weight.items(), key=lambda x: -x[1])
        return rel_w, p50_w, p90_w, fails, paths_seen, weak

    def _allocate(self, options, caps, persons_total) -> dict:
        milp_res = milp_allocate(options, caps, persons_total)
        greedy_res = greedy_allocate(options, caps, persons_total)
        return {"milp": milp_res, "greedy": greedy_res, "options": options}


def targets_by_req_set(req, inputs):
    return {r["service"] for r in inputs.services.to_dict("records")
            if r["category"] == req["category"]}
