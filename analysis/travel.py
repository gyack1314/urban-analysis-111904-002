"""路段用时模型、电梯排队与时间依赖最短路。

证据处理原则（对应任务要求）：
- completed 观测用于拟合对数正态用时；
- observation_end 是右删失：只作为“真实用时不低于该值”的下界，不并入均值；
- turned_back 不产生用时分布，只作为该边对该群体通行困难的证据；
- 没有可支持观测的必经边绝不按零分钟：依次尝试反向同段转移、速度模型外推、
  保守先验带，并在 evidence 中登记 weak/assumption 标记。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .feasibility import next_lift_availability


# ---------------------------------------------------------------- 用时模型
@dataclass
class EdgeTimeModel:
    """单条边对单个群体的用时模型（秒）。"""
    edge: str
    profile: str
    is_lift: bool
    mu: float = math.log(60.0)       # 对数正态参数
    sigma: float = 0.3
    source: str = "fallback"         # observed / reverse-analog / speed-model / lift-analog / fallback
    evidence: dict = field(default_factory=dict)

    def quantile(self, q: float) -> float:
        return float(stats.lognorm.ppf(q, s=self.sigma, scale=math.exp(self.mu)))

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.lognormal(self.mu, self.sigma))

    @property
    def weak(self) -> bool:
        return self.source != "observed"


def _fit_lognormal(values: list[float]) -> tuple[float, float]:
    mu = float(np.mean(np.log(values)))
    sigma = float(np.std(np.log(values), ddof=1)) if len(values) > 1 else 0.15
    # sigma 下限避免单点/极少样本给出虚假的确定性
    return mu, max(sigma, 0.08)


def build_edge_time_models(traversals: pd.DataFrame, network: pd.DataFrame,
                           projects: pd.DataFrame, config) -> dict[tuple[str, str], EdgeTimeModel]:
    lift_edges = {r["edge"] for r in network.to_dict("records") if r["lift_id"]}
    models: dict[tuple[str, str], EdgeTimeModel] = {}
    notes: dict[tuple[str, str], dict] = {}

    profiled_edges = sorted(set(traversals["edge"]))
    profiles = sorted(set(traversals["profile"]))

    # 1) 有观测的边：completed 拟合；删失值约束保守端；折返登记为通行困难证据
    for edge in profiled_edges:
        for prof in profiles:
            sub = traversals[(traversals["edge"] == edge) & (traversals["profile"] == prof)]
            if sub.empty:
                continue
            completed = sub.loc[sub["status"] == "completed", "duration_seconds"].tolist()
            censored = sub.loc[sub["status"] == "observation_end", "duration_seconds"].tolist()
            turned = sub.loc[sub["status"] == "turned_back", "duration_seconds"].tolist()
            ev: dict = {"n_completed": len(completed), "n_censored": len(censored),
                        "n_turned_back": len(turned)}
            if completed:
                mu, sigma = _fit_lognormal(completed)
                m = EdgeTimeModel(edge, prof, edge in lift_edges, mu, sigma,
                                  "observed", ev)
                if censored:
                    p90 = m.quantile(0.90)
                    ev["censored_max"] = float(max(censored))
                    if max(censored) > p90:
                        # 右删失观测比拟合 p90 还大：把保守端拉到删失值，不假装分布更窄
                        ev["conservative_floor_seconds"] = float(max(censored))
            else:
                # 只有折返/删失、没有完成记录：不给零，也不给均值，落保守先验
                m = EdgeTimeModel(edge, prof, edge in lift_edges,
                                  source="fallback", evidence=ev)
                m.mu, m.sigma = math.log(config.FALLBACK_SPEED_MPS[1] ** -1 * 60), 0.5
                ev["reason"] = "无完成观测（全部折返或删失）"
            notes[(edge, prof)] = ev
            models[(edge, prof)] = m

    # 2) 每个群体的非梯边速度（秒/米）分布，供无观测边外推
    speed_ratio: dict[str, tuple[float, float]] = {}
    for prof in profiles:
        ratios = []
        for edge in profiled_edges:
            if edge in lift_edges:
                continue
            completed = traversals.loc[
                (traversals["edge"] == edge)
                & (traversals["profile"] == prof)
                & (traversals["status"] == "completed"),
                "duration_seconds"].tolist()
            if not completed:
                continue
            length = float(network.loc[network["edge"] == edge, "length_m"].iloc[0])
            ratios.extend(d / length for d in completed)
        if ratios:
            speed_ratio[prof] = _fit_lognormal(ratios)

    edge_rows = {r["edge"]: r for r in network.to_dict("records")}

    def speed_model(edge: str, prof: str, inflate: float, source: str,
                    reason: str, length_override: float | None = None) -> EdgeTimeModel:
        length = (float(length_override) if length_override is not None
                  else float(edge_rows[edge]["length_m"]))
        if prof in speed_ratio:
            r_mu, r_sigma = speed_ratio[prof]
            m = EdgeTimeModel(edge, prof, edge in lift_edges,
                              mu=r_mu + math.log(length),
                              sigma=r_sigma * inflate, source=source,
                              evidence={"reason": reason, "length_m": length})
        else:
            lo, hi = config.FALLBACK_SPEED_MPS
            m = EdgeTimeModel(edge, prof, edge in lift_edges,
                              mu=math.log(length / ((lo + hi) / 2)), sigma=0.5,
                              source="fallback",
                              evidence={"reason": reason + "；群体无速度样本，启用先验带"})
        return m

    # 3) 网络中存在但缺少某群体观测的边：反向同段转移 → 速度模型 → 先验。
    # 反向命名规则：以 r 结尾则去掉末尾 r，否则追加 r。
    # 不能用 [:-1]+'r'——多位数边名（e11）会被错算成 e1r 而非 e11r。
    reverse = {r["edge"]: (r["edge"][:-1] if r["edge"].endswith("r")
                           else r["edge"] + "r")
               for r in network.to_dict("records")}
    for edge in network["edge"]:
        for prof in profiles:
            if (edge, prof) in models:
                continue
            rev = reverse.get(edge)
            if rev and (rev, prof) in models and models[(rev, prof)].source == "observed":
                base = models[(rev, prof)]
                m = EdgeTimeModel(edge, prof, edge in lift_edges, base.mu,
                                  min(base.sigma * 1.15, 0.8),
                                  "reverse-analog",
                                  {"reason": f"由反向同段 {rev} 转移，已加宽不确定性"})
            else:
                m = speed_model(edge, prof, inflate=1.25, source="speed-model",
                                reason="该群体对该边无观测，按同群体秒/米分布外推")
            models[(edge, prof)] = m

    # 4) 项目新建边：明确为假设值
    analog = config.ANALOG_LIFT_EDGE
    for p in projects.to_dict("records"):
        ne = p["new_edge"]
        if not ne:
            continue
        for prof in profiles:
            if ne["lift_id"]:
                base = models.get((analog, prof))
                if base is not None:
                    m = EdgeTimeModel(ne["edge"], prof, True, base.mu,
                                      min(base.sigma * 1.2, 0.8), "lift-analog",
                                      {"reason": f"新建电梯边，类比 {analog} 轿厢用时（无实测）",
                                       "analog": analog})
                else:
                    m = speed_model(ne["edge"], prof, 1.5, "fallback",
                                    "新建电梯边且无类比来源")
            else:
                m = speed_model(ne["edge"], prof, inflate=1.5, source="speed-model",
                                reason="新建普通边无实测，按同群体速度外推并加宽预测带",
                                length_override=float(ne["length_m"]))
            models[(ne["edge"], prof)] = m

    return models


# ---------------------------------------------------------------- 电梯排队
def simulate_weighted_lift_queue(arrivals: list[tuple[float, float]], window: dict,
                                 outages: pd.DataFrame) -> dict[float, float]:
    """先到先服务的加权批量排队（跨全部请求组联合模拟）。

    arrivals: (到达分钟, 到达人数，可为分数)。
    返回 {到达分钟: 该批人员的加权平均候梯分钟}。停运区间内的计划梯次顺延到
    恢复之后；闭门前无法乘梯的批次候梯记为 inf（路由层据此排除该梯边）。
    """
    import collections

    cycle_min = window["cycle_seconds"] / 60.0
    cap = float(window["capacity"])
    open_t, close_t = float(window["open"]), float(window["close"])
    outage = sorted((float(o["start"]), float(o["end"]))
                    for o in outages.to_dict("records") if o["lift"] == window["lift"])

    def push_past_outage(t: float) -> float:
        for s, e in outage:
            if s <= t < e:
                t = e
        return t

    batches: dict[float, float] = {}
    for t, n in arrivals:
        key = round(t, 4)
        batches[key] = batches.get(key, 0.0) + n

    queue: collections.deque = collections.deque()  # [到达分钟, 剩余人数]
    acc: dict[float, list[float]] = {}              # 到达分钟 -> [Σwait·人数, Σ人数]
    dep = push_past_outage(open_t)

    def run_elevator(until: float) -> None:
        nonlocal dep
        while queue and dep <= until and dep < close_t:
            cap_left = cap
            while cap_left > 1e-12 and queue and queue[0][0] <= dep:
                at, n = queue[0]
                seated = min(n, cap_left)
                a = acc.setdefault(at, [0.0, 0.0])
                a[0] += (dep - at) * seated
                a[1] += seated
                cap_left -= seated
                if n - seated <= 1e-12:
                    queue.popleft()
                else:
                    queue[0][1] = n - seated
            dep = push_past_outage(dep + cycle_min)

    for at in sorted(batches):
        run_elevator(at)
        queue.append([at, batches[at]])
    run_elevator(close_t)
    for at, _ in queue:  # 闭门前仍未运走：该到达时刻无法乘梯
        acc.setdefault(at, [math.inf, 1.0])
    return {at: (math.inf if math.isinf(w_sum) else w_sum / w)
            for at, (w_sum, w) in acc.items()}



# ---------------------------------------------------------------- 时变最短路
class TimeDependentRouter:
    """给定边集合、用时模型与当前候梯函数，做 FIFO 时间依赖 Dijkstra。"""

    def __init__(self, edge_rows: dict[str, dict], usable: dict[str, dict],
                 models: dict[tuple[str, str], EdgeTimeModel],
                 lift_windows: pd.DataFrame, outages: pd.DataFrame,
                 wait_funcs: dict[str, dict[float, float]] | None = None,
                 quantile: float = 0.5):
        self.edge_rows = edge_rows
        self.usable = usable
        self.models = models
        self.windows = {r["lift"]: r for r in lift_windows.to_dict("records")}
        self.outages = outages
        # wait_funcs[lift] = {到达分钟: 候梯分钟}，按到达分钟最近向下取键
        self.wait_funcs = wait_funcs or {}
        self.q = quantile
        self.adj: dict[str, dict[str, list[dict]]] = {}
        for prof, edges in usable.items():
            self.adj[prof] = {}
            for eid, e in edges.items():
                self.adj[prof].setdefault(e["source"], []).append(e)

    def _queue_wait(self, lift: str, t: float) -> float | None:
        """返回 t 时刻到梯后的候梯分钟；闭门前无可用梯次返回 None。"""
        avail = next_lift_availability(lift, t,
                                       pd.DataFrame(list(self.windows.values())),
                                       self.outages)
        if avail is None:
            return None
        base = max(0.0, avail - t)
        table = self.wait_funcs.get(lift)
        if not table:
            return base
        keys = sorted(table)
        import bisect
        i = bisect.bisect_right(keys, t) - 1
        queued = table[keys[i]] if i >= 0 else table[keys[0]]
        if math.isinf(queued):
            return None  # 联合排队下该时刻已无梯次可乘
        return max(base, queued)

    def shortest(self, profile: str, origin: str, depart_minute: float,
                 targets: set[str]) -> dict[str, dict]:
        import heapq
        dist = {origin: depart_minute}
        prev: dict[str, tuple[str, str]] = {}
        pq = [(depart_minute, origin)]
        while pq:
            t, u = heapq.heappop(pq)
            if t > dist.get(u, math.inf):
                continue
            for e in self.adj.get(profile, {}).get(u, ()):
                model = self.models.get((e["edge"], profile))
                if model is None:
                    continue
                if e["lift_id"]:
                    wait = self._queue_wait(e["lift_id"], t)
                    if wait is None:
                        continue
                    onboard = model.quantile(self.q) / 60.0
                    nd = t + wait + onboard
                else:
                    nd = t + model.quantile(self.q) / 60.0
                if nd < dist.get(e["target"], math.inf):
                    dist[e["target"]] = nd
                    prev[e["target"]] = (u, e["edge"])
                    heapq.heappush(pq, (nd, e["target"]))
        out = {}
        for tgt in targets:
            if tgt not in dist:
                continue
            path_edges = []
            cur = tgt
            while cur != origin:
                if cur not in prev:
                    path_edges = []
                    break
                prev_node, eid = prev[cur]
                path_edges.append(eid)
                cur = prev_node
            out[tgt] = {"arrival": dist[tgt], "path": list(reversed(path_edges))}
        return out
