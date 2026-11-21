"""命令行入口：python -I -m analysis [inputs_dir] [outputs_dir]。

隔离模式（-I）下解释器不把当前目录加入 sys.path，因此先把仓库根目录显式加入，
保证在容器或任意工作目录中都能定位到 analysis 包，且不依赖被污染的环境路径。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.runner import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="适老出行可达性与项目组合比较")
    p.add_argument("inputs", nargs="?", default=str(ROOT / "inputs"))
    p.add_argument("outputs", nargs="?", default=str(ROOT / "outputs"))
    a = p.parse_args()
    res = run(a.inputs, a.outputs)
    s = res["summary"]
    print(json.dumps({
        "baseline_served_nominal": s["baseline_served_nominal"],
        "structural_capacity_gap": s["structural_capacity_gap"],
        "complementarity": s["complementarity"],
        "states": len(s["states"]),
        "outputs": ["scenario_metrics.csv", "route_evidence.csv",
                    "evidence_gaps.csv", "comparison_summary.json"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
