"""物理可行性：按人群画像过滤边，并处理电梯开放/停运的时间维度。

规则来源 docs/input-contract.md：
- profiles 给最大坡度、最小宽度、台阶许可；
- 坡度限制按上坡正值适用；下坡（负值）数据契约未规定，本分析采用同一绝对值
  阈值的保守安全假设（见 config.ASSUMPTIONS["slope_downhill"]）；
- 梯边还要满足电梯在到达时刻处于开放且非停运状态。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Profile:
    name: str
    max_slope: float
    min_width_m: float
    steps_allowed: bool


def profiles_from_df(df: pd.DataFrame) -> dict[str, Profile]:
    return {
        r["profile"]: Profile(r["profile"], float(r["max_slope"]),
                              float(r["min_width_m"]), bool(r["steps_allowed"]))
        for r in df.to_dict("records")
    }


def edge_physically_usable(edge: dict, prof: Profile) -> tuple[bool, str | None]:
    """静态物理判断（不含电梯时刻）。返回 (是否可行, 阻断原因)。"""
    if not prof.steps_allowed and int(edge["steps"]) > 0:
        return False, f"台阶{edge['steps']}级"
    if float(edge["width_m"]) < prof.min_width_m:
        return False, f"宽度{edge['width_m']}m<{prof.min_width_m}m"
    if abs(float(edge["slope"])) > prof.max_slope:
        direction = "上坡" if float(edge["slope"]) >= 0 else "下坡安全假设"
        return False, f"{direction}坡度{edge['slope']}>{prof.max_slope}"
    return True, None


def build_usable_edges(network: pd.DataFrame, profiles: dict[str, Profile]
                       ) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, str]]]:
    """返回 (各群体静态可行边集合, 各群体被阻断边及原因)。"""
    usable: dict[str, dict[str, dict]] = {}
    blocked: dict[str, dict[str, str]] = {}
    for name, prof in profiles.items():
        ok, no = {}, {}
        for edge in network.to_dict("records"):
            good, reason = edge_physically_usable(edge, prof)
            if good:
                ok[edge["edge"]] = edge
            else:
                no[edge["edge"]] = reason
        usable[name] = ok
        blocked[name] = no
    return usable, blocked


def lift_open_at(lift: str, at_minute: float, windows: pd.DataFrame,
                 outages: pd.DataFrame) -> bool:
    """电梯在某本地分钟是否可用：须在开放窗口 [open, close) 且不在停运区间。"""
    win = windows.loc[windows["lift"] == lift]
    if win.empty:
        return False
    w = win.iloc[0]
    if not (float(w["open"]) <= at_minute < float(w["close"])):
        return False
    for o in outages.to_dict("records"):
        if o["lift"] == lift and float(o["start"]) <= at_minute < float(o["end"]):
            return False
    return True


def next_lift_availability(lift: str, at_minute: float, windows: pd.DataFrame,
                           outages: pd.DataFrame) -> float | None:
    """返回不早于 at_minute 的最早可用分钟；当天窗口已过则 None。"""
    win = windows.loc[windows["lift"] == lift]
    if win.empty:
        return None
    w = win.iloc[0]
    close = float(w["close"])
    if at_minute >= close:
        return None
    t = max(at_minute, float(w["open"]))
    for o in sorted([o for o in outages.to_dict("records") if o["lift"] == lift],
                    key=lambda x: x["start"]):
        if t < float(o["start"]):
            return t
        if t < float(o["end"]):
            t = float(o["end"])
    return t if t < close else None


def next_availability_vectorized(lift: str, minutes, windows: pd.DataFrame,
                                 outages: pd.DataFrame):
    """向量化版本：输入分钟数组，返回最早可用分钟数组（窗口已过为 inf）。"""
    import numpy as np
    minutes = np.asarray(minutes, dtype=float)
    win = windows.loc[windows["lift"] == lift]
    if win.empty:
        return np.full_like(minutes, np.inf)
    w = win.iloc[0]
    open_t, close_t = float(w["open"]), float(w["close"])
    t = np.maximum(minutes, open_t)
    for o in sorted([o for o in outages.to_dict("records") if o["lift"] == lift],
                    key=lambda x: x["start"]):
        s, e = float(o["start"]), float(o["end"])
        t = np.where((s <= t) & (t < e), e, t)
    return np.where(t < close_t, t, np.inf)
