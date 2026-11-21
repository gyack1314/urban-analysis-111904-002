"""读取并校验原始资料。

原件（inputs/）只读；actions.json 的 L1 午间恢复在这里以“情景开关”方式应用，
不改写任何输入文件。所有读取走 pyarrow/duckdb，与 materials 的物理校验保持一致。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Inputs:
    network: pd.DataFrame
    profiles: pd.DataFrame
    origins: pd.DataFrame
    traversals: pd.DataFrame
    lift_windows: pd.DataFrame
    outages: pd.DataFrame
    services: pd.DataFrame
    requests: pd.DataFrame
    projects: pd.DataFrame
    actions: list
    planning: dict
    root: Path

    def edge_index(self) -> dict[str, dict]:
        return {r["edge"]: r for r in self.network.to_dict("records")}


def _read(root: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(root / f"{name}.parquet")


def load_inputs(inputs_root: str | Path) -> Inputs:
    root = Path(inputs_root).resolve()
    data = {
        name: _read(root, name)
        for name in (
            "network", "profiles", "origins", "traversals", "lift_windows",
            "outages", "services", "requests", "projects",
        )
    }
    actions = json.loads((root / "actions.json").read_text(encoding="utf-8"))
    planning = json.loads((root / "planning-context.json").read_text(encoding="utf-8"))
    _check_references(data, actions)
    return Inputs(**data, actions=actions, planning=planning, root=root)


def _check_references(d: dict, actions: list) -> None:
    edges = set(d["network"]["edge"])
    lifts = set(d["lift_windows"]["lift"])
    assert set(d["traversals"]["edge"]) <= edges, "观测引用了不存在的边"
    assert d["network"]["lift_id"].dropna().isin(lifts).all(), "边引用了未登记电梯"
    assert d["outages"]["lift"].isin(lifts).all(), "停运记录引用了未登记电梯"
    for act in actions:
        if act["action"] == "restore_lift":
            assert act["lift"] in lifts, "恢复动作引用了未登记电梯"
    # 项目引用的封闭边与改善边必须存在；new_edge 不得与现有边重名。
    for p in d["projects"].to_dict("records"):
        for e in p["closed_edges"]:
            assert e in edges, f"项目 {p['project']} 封闭了不存在的边 {e}"
        if p["improve_edge"]:
            assert p["improve_edge"] in edges
        if p["new_edge"]:
            assert p["new_edge"]["edge"] not in edges
            assert p["new_edge"]["lift_id"] is None or p["new_edge"]["lift_id"] in lifts


def effective_outages(outages: pd.DataFrame, actions: list,
                      apply_restore: bool) -> pd.DataFrame:
    """返回实际生效的停运区间。

    actions.json 宣布 L1 在 [720,780) 恢复，原停运 [660,780) 收缩为 [660,720)。
    apply_restore=False 用于评估“恢复未兑现/未发生”的反事实。
    """
    rows = outages.to_dict("records")
    if not apply_restore:
        return pd.DataFrame(rows)
    result = []
    restore = [a for a in actions if a["action"] == "restore_lift"]
    for row in rows:
        cuts = [a for a in restore if a["lift"] == row["lift"]
                and a["start"] >= row["start"] and a["end"] <= row["end"]]
        cursor = row["start"]
        for a in sorted(cuts, key=lambda x: x["start"]):
            if a["start"] > cursor:
                result.append({**row, "start": cursor, "end": a["start"],
                               "reason": row["reason"] + "(恢复前)"})
            cursor = a["end"]
        if cursor < row["end"]:
            result.append({**row, "start": cursor, "end": row["end"],
                           "reason": row["reason"] + "(恢复后残余)"})
    return pd.DataFrame(result)
