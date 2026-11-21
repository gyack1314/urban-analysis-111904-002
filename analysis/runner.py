"""编排：读原件 → 全情景评估 → 汇总指标与证据登记 → 写 outputs/。

输出（均为分析结果，不回写 inputs/）：
  scenario_metrics.csv  每个网络状态 × 需求情景的服务量、未满足需求、可靠度
  route_evidence.csv    每个请求组到每个服务点的路径、分位用时与弱证据边
  evidence_gaps.csv     证据薄弱/纯假设条目登记
  comparison_summary.json 面向项目组的情景比较与互补/挤出判断
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .allocate import Option
from .demand import build_demand_scenarios
from .load import effective_outages, load_inputs
from .scenarios import Engine, build_scenario_registry
from .travel import build_edge_time_models


def run(inputs_root: str | Path, outputs_root: str | Path) -> dict:
    inputs = load_inputs(inputs_root)
    out_dir = Path(outputs_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    outages_eff = effective_outages(inputs.outages, inputs.actions, apply_restore=True)
    engine = Engine(inputs, outages_eff)
    # 一次性建立全部边（含项目新边）的用时模型
    inputs._base_models = build_edge_time_models(
        inputs.traversals, inputs.network, inputs.projects, config)

    registry = build_scenario_registry(inputs.projects, float(inputs.planning["budget_yuan"]))
    demand_scenarios = build_demand_scenarios(inputs.origins, inputs.requests)

    metrics_rows, route_rows, gap_rows = [], [], []
    raw = {}
    for state in registry:
        raw[state.name] = {}
        for ds in demand_scenarios:
            res = engine.evaluate(state, ds)
            raw[state.name][ds.name] = res
            metrics_rows.extend(_metric_rows(state, ds, res))
            route_rows.extend(res["routes"])
        # 证据登记按状态实际出现的边生成，避免跨状态无意义重复
        gap_rows.extend(_evidence_gaps(state, inputs, engine, res))

    metrics = pd.DataFrame(metrics_rows)
    routes = pd.DataFrame(route_rows).drop_duplicates(
        subset=["state", "demand_scenario", "request", "service"])
    gaps = pd.DataFrame(gap_rows).drop_duplicates(subset=["state", "item"])

    metrics.to_csv(out_dir / "scenario_metrics.csv", index=False, encoding="utf-8-sig")
    routes.to_csv(out_dir / "route_evidence.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(out_dir / "evidence_gaps.csv", index=False, encoding="utf-8-sig")

    summary = _comparison_summary(raw, registry, demand_scenarios, inputs)
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metrics": metrics, "routes": routes, "gaps": gaps, "summary": summary,
            "raw": raw, "registry": registry, "demand": demand_scenarios}


def _metric_rows(state, ds, res) -> list[dict]:
    milp = res["milp"]
    greedy = res["greedy"]
    options = res["options"]
    nominal = res["nominal_persons"]
    # 每个请求组只计一次名义需求；可靠可达人数按组内最高加权可靠度折算
    per_request: dict[str, list[Option]] = {}
    for o in options:
        per_request.setdefault(o.request, []).append(o)
    demand = sum(nominal[rid] for rid in per_request)
    reachable = sum(max(o.reliability for o in g) * nominal[rid]
                    for rid, g in per_request.items())
    served = milp.get("served", 0.0)
    by_profile: dict[str, dict[str, float]] = {}
    for (req, svc), persons in milp.get("allocation", {}).items():
        prof = next(o.profile for o in options if o.request == req)
        by_profile.setdefault(prof, {"served": 0.0, "demand": 0.0})
        by_profile[prof]["served"] += persons
    for req, group in per_request.items():
        prof = group[0].profile
        by_profile.setdefault(prof, {"served": 0.0, "demand": 0.0})
        by_profile[prof]["demand"] += nominal[req]
    rels = [max(o.reliability for o in g) for g in per_request.values()]
    return [{
        "state": state.name, "kind": state.kind, "within_budget":
        getattr(state, "within_budget", True), "cost_yuan": state.cost_yuan,
        "construction_days": state.construction_days,
        "demand_scenario": ds.name,
        "demand_persons": round(demand, 2),
        "reachable_persons": round(reachable, 2),
        "served_persons_milp": round(served, 2),
        "served_persons_greedy": round(greedy.get("served", 0.0), 2),
        "unmet_persons": round(demand - served, 2),
        "service_rate": round(served / demand, 4) if demand else 0.0,
        "mean_reliability": round(float(np.mean(rels)), 4) if rels else 0.0,
        "served_wheelchair": round(by_profile.get("wheelchair", {}).get("served", 0), 2),
        "served_slowwalk": round(by_profile.get("slow-walk", {}).get("served", 0), 2),
        "served_walking": round(by_profile.get("walking", {}).get("served", 0), 2),
        "milp_stage": milp.get("stage", ""),
    }]


def _evidence_gaps(state, inputs, engine: Engine, res: dict | None = None) -> list[dict]:
    rows = []
    models = inputs._base_models
    present_edges = set(res["edges"]) if res and "edges" in res else None
    for (edge, prof), m in models.items():
        if present_edges is not None and edge not in present_edges:
            continue
        if m.weak:
            rows.append({"state": state.name, "item": f"{edge}|{prof}",
                         "type": m.source,
                         "detail": m.evidence.get("reason", "无完成观测的保守用时带"),
                         "n_completed": m.evidence.get("n_completed", 0)})
    # 项目状态登记其自身的假设标记
    for flag in state.weak_flags:
        rows.append({"state": state.name, "item": "scenario-assumption",
                     "type": "assumption", "detail": flag, "n_completed": 0})
    # 折返证据：本状态中该边仍存在、对该群体无完成记录
    tb = (inputs.traversals[inputs.traversals["status"] == "turned_back"]
          [["edge", "profile"]].drop_duplicates())
    for _, r in tb.iterrows():
        if present_edges is not None and r["edge"] not in present_edges:
            continue
        rows.append({"state": state.name,
                     "item": f"{r['edge']}|{r['profile']}",
                     "type": "turned-back-evidence",
                     "detail": "存在折返记录且无完成观测，按不可独立通行处理",
                     "n_completed": 0})
    return rows


def _comparison_summary(raw, registry, demand_scenarios, inputs) -> dict:
    nominal = [d.name for d in demand_scenarios if d.name.startswith("nominal")]

    def agg(state_name, key):
        vals = [raw[state_name][d]["milp"].get("served", 0.0) for d in nominal]
        return float(np.mean(vals)) if vals else 0.0

    base_served = agg("baseline", None)
    states = []
    for st in registry:
        served = agg(st.name, None)
        row = {
            "state": st.name, "kind": st.kind,
            "cost_yuan": st.cost_yuan,
            "within_budget": getattr(st, "within_budget", True),
            "construction_days": st.construction_days,
            "nominal_served_persons": round(served, 2),
            "delta_vs_baseline": round(served - base_served, 2),
            "description": st.description,
        }
        if st.kind == "construction":
            # 双向封闭敏感性状态没有独立的完工对照，回落到常规组合完工
            completion_name = st.name.replace("construction", "completion")
            if completion_name not in raw:
                completion_name = "P1P2_completion"
            completed = agg(completion_name, None)
            row["construction_window_loss_persons"] = round(base_served - served, 2)
            row["construction_day_loss_estimate"] = round(
                max(0.0, base_served - served) * st.construction_days, 2)
            row["post_completion_gain_persons"] = round(completed - base_served, 2)
            # 服务量不变不代表无损失：登记施工期可靠可达人数与平均可靠度的退化
            row.update(_construction_reliability_delta(raw, st.name, nominal))
        states.append(row)

    # 互补性：P1、P2 单独完工 vs 组合完工。
    # 注意区分两类收益来源：P2 单独只带来 S1 容量扩容（30→45），
    # 新链路 new-link 的起点 N4 在没有 P1 时是孤岛，不产生任何新路径。
    def served(name):
        return agg(name, None)
    complement = {
        "P1_only_served": round(served("P1_completion"), 2),
        "P2_only_served": round(served("P2_completion"), 2),
        "P2_only_gain_source": "仅来自 S1 窗口容量 30→45；new-link 起点 N4 无可达路径，新链路不生效",
        "P1P2_combined_served": round(served("P1P2_completion"), 2),
        "combined_linkage_reliability_note": "组合后 N1→N4(new-lift,L2)→N4→S1(new-link) 贯通，"
                                              "轮椅群体可靠度才达到 1.0（见 route_evidence）",
        "note": "P1 只到 N4、P2 只从 N4 出发；单独实施均不产生到 S1 的新路径，组合才贯通",
    }
    # 挤出：比较组合完工下各 profile 的服务量与基线
    crowd = _crowding_out(raw, nominal)
    # 容量结构缺口
    meal_caps = {r["service"]: int(r["capacity"])
                 for r in inputs.services.to_dict("records")
                 if r["category"] == "meal"}
    total_nominal_demand = sum(r["persons"] for r in inputs.requests.to_dict("records"))
    budget = float(inputs.planning["budget_yuan"])
    portfolio_view = []
    for st in registry:
        if st.kind not in ("completion", "baseline", "sensitivity"):
            continue
        if st.name in ("baseline_no_restore", "lift_L1_failed",
                       "P3_completion_bothdir"):
            continue
        portfolio_view.append({
            "state": st.name,
            "cost_yuan": st.cost_yuan,
            "within_budget": st.cost_yuan <= budget,
            "nominal_served": round(agg(st.name, None), 2),
            "delta_vs_baseline": round(agg(st.name, None) - base_served, 2),
            "cost_per_added_served_yuan": (
                round(st.cost_yuan / max(1e-9, agg(st.name, None) - base_served), 1)
                if agg(st.name, None) - base_served > 1e-9 else None),
        })
    return {
        "budget_yuan": inputs.planning["budget_yuan"],
        "baseline_served_nominal": round(base_served, 2),
        "total_nominal_meal_demand": total_nominal_demand,
        "meal_capacity_total_baseline": sum(meal_caps.values()),
        "structural_capacity_gap": total_nominal_demand - sum(meal_caps.values()),
        "complementarity": complement,
        "crowding_out": crowd,
        "portfolio_view": portfolio_view,
        "states": states,
        "assumptions": config.ASSUMPTIONS,
        "re_run_note": "现场补测或 L1 恢复确认后，更新 inputs 原件并以同一脚本重跑即可重新比较；"
                       "本分析不派单、不给出唯一必选项目。",
    }


def _construction_reliability_delta(raw, state_name, nominal_scenarios) -> dict:
    """施工期相对基线的可靠可达人数与平均可靠度退化（服务量口径之外的损失）。"""
    def agg_rel(name):
        reach, relsum, count = 0.0, 0.0, 0
        for ds in nominal_scenarios:
            res = raw[name][ds]
            nominal = res["nominal_persons"]
            per_request: dict[str, list] = {}
            for o in res["options"]:
                per_request.setdefault(o.request, []).append(o)
            for rid, group in per_request.items():
                best = max(o.reliability for o in group)
                reach += best * nominal[rid]
                relsum += best
                count += 1
        return reach / len(nominal_scenarios), relsum / count
    base_reach, base_rel = agg_rel("baseline")
    cur_reach, cur_rel = agg_rel(state_name)
    return {
        "construction_reachable_delta": round(cur_reach - base_reach, 2),
        "construction_mean_reliability_baseline": round(base_rel, 4),
        "construction_mean_reliability_state": round(cur_rel, 4),
    }


def _crowding_out(raw, nominal_scenarios) -> list[dict]:
    out = []
    for ds in nominal_scenarios:
        base = _profile_served(raw["baseline"][ds])
        combo = _profile_served(raw["P1P2_completion"][ds])
        for prof in sorted(set(base) | set(combo)):
            b, c = base.get(prof, 0.0), combo.get(prof, 0.0)
            if c < b - 1e-9:
                out.append({"demand_scenario": ds, "profile": prof,
                            "baseline_served": round(b, 2),
                            "combo_served": round(c, 2),
                            "displaced": round(b - c, 2)})
    return out


def _profile_served(res) -> dict[str, float]:
    out: dict[str, float] = {}
    for (req, _svc), persons in res["milp"].get("allocation", {}).items():
        prof = next(o.profile for o in res["options"] if o.request == req)
        out[prof] = out.get(prof, 0.0) + persons
    return out
