"""需求情景：人口区间、住宅出口与出发窗口。

契约要求：origins 的人数区间不是确定出口分配；同住宅各 profile 互斥；
出口与人数选择必须在共同情景中保持一致。这里把不确定性显式展开成一组
“情景坐标轴”，所有项目情景在同一组轴上比较，不逐项目挑参数。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DEPARTURE_BIN_MINUTES


@dataclass(frozen=True)
class DemandScenario:
    name: str
    scale: float                 # 相对名义（requests.persons）的缩放系数
    exit_map: dict[str, str]     # residence -> 实际使用出口节点
    label: str


def build_demand_scenarios(origins: pd.DataFrame, requests: pd.DataFrame
                           ) -> list[DemandScenario]:
    """人口三档 × R1 两个可能出口 = 6 个共同需求情景。

    名义总量取 requests 中同住宅各 profile 之和（30 人）；
    origins 区间为 25–40 人，因此缩放下界 25/30、上界 40/30。
    """
    nominal = requests.groupby("residence")["persons"].sum().to_dict()
    scales = {"low": 1.0, "nominal": 1.0, "high": 1.0}
    lo_scale = next(iter(origins[origins["residence"] == "R1"]["persons_low"])) / nominal["R1"]
    hi_scale = next(iter(origins[origins["residence"] == "R1"]["persons_high"])) / nominal["R1"]
    scales = {"low": float(lo_scale), "nominal": 1.0, "high": float(hi_scale)}

    exit_variants: list[dict[str, dict[str, str]]] = []
    multi = [r for r in origins.to_dict("records") if len(r["possible_exits"]) > 1]
    if not multi:
        exit_variants.append({r["residence"]: r["possible_exits"][0]
                              for r in origins.to_dict("records")})
    else:
        # 当前只有 R1 有两个出口；分别固定，不允许为单个项目任选。
        r1 = multi[0]
        for chosen in r1["possible_exits"]:
            mapping = {r["residence"]: r["possible_exits"][0]
                       for r in origins.to_dict("records")}
            mapping[r1["residence"]] = chosen
            exit_variants.append(mapping)

    out = []
    for size_name, scale in scales.items():
        for i, mapping in enumerate(exit_variants):
            tag = f"exit-{mapping['R1']}" if "R1" in mapping else f"exit-{i}"
            out.append(DemandScenario(
                name=f"{size_name}|{tag}",
                scale=scale,
                exit_map=dict(mapping),
                label=f"人口{size_name}(×{scale:.3f})；R1出口={mapping.get('R1')}",
            ))
    return out


def departure_bins(req: dict) -> list[tuple[float, float]]:
    """出发窗口切成固定宽度时段，返回 (分钟, 人数份额)。"""
    start, end = float(req["depart_from"]), float(req["depart_to"])
    n = max(1, round((end - start) / DEPARTURE_BIN_MINUTES))
    step = (end - start) / n
    return [(start + i * step, 1.0 / n) for i in range(n)]
