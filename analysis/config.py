"""集中管理分析假设：所有没有直接观测支撑的判断都在这里显式声明。"""
from __future__ import annotations

# 固定随机种子，保证 Monte Carlo 与抽样结果可复现（现场补测后可换种子重跑）。
RNG_SEED = 20250521

# Monte Carlo 次数：用于路径在时限内到达的可靠度。
MC_DRAWS = 400

# 需求出发窗口切片（分钟），同一请求组的人数均匀分到各切片。
DEPARTURE_BIN_MINUTES = 5

# 电梯排队模拟的时间分辨率（秒）。
QUEUE_TICK_SECONDS = 10

# 路径选择与候梯时间不动点迭代上限。
ASSIGNMENT_MAX_ITERS = 8

# 可靠度门槛：Monte Carlo 中能在时限并在服务开放内到达的比例达到该值，
# 才把该（请求组, 服务点）对计入“可靠可达”。这是分析约定，不是业务承诺。
RELIABILITY_THRESHOLD = 0.80

# 人群弱势权重（容量分配第二目标用）：台阶/坡度受限越严重越优先。
VULNERABILITY_WEIGHT = {"wheelchair": 3, "slow-walk": 2, "walking": 1}

# 无任何可支持观测的边，用时先验带（秒/米）的保守系数区间。
# 仅在转移与速度模型都无法使用时兜底；结果带会被标记为 weak-evidence。
FALLBACK_SPEED_MPS = (0.4, 1.4)  # 对应很慢到一般适老人群步行速度

# 新建边的类比来源（契约允许“按明确假设范围比较”，但不得冒充实测）。
ANALOG_LIFT_EDGE = "e4"        # new-lift(L2) 类比 L1 梯边 e4 的轿厢内用时
# new-link 无同类边，使用非梯边速度模型外推并加宽预测带。

# 施工封闭边按 projects.closed_edges 字面执行；另出一组双向封闭敏感性结果。
SENSITIVITY_CLOSE_BOTH_DIRECTIONS = True

# P3 契约只给 e5（单向）新坡度；另出 e5/e5r 双向改善的敏感性结果。
SENSITIVITY_P3_BOTH_DIRECTIONS = True

# 施工损失换算：假设每个工作日的需求窗口形态相同，窗口损失 × construction_days。
# 该假设在报告中显式声明，数据本身不包含多日需求波动。
CONSTRUCTION_DAY_PATTERN_ASSUMED = True

# 文档化的分析假设清单（写入报告）。
ASSUMPTIONS = {
    "slope_downhill": "契约只规定上坡正值按 max_slope 限制；下坡按同一绝对值阈值作安全假设处理。",
    "censored": "observation_end 为右删失下界，不并入完成用时均值；若其值超过拟合 p90 则加宽保守端。",
    "turned_back": "turned_back 记录作为该边对该群体不可独立完成的证据，用时不参与完成分布拟合。",
    "unobserved_edge": "无观测边不按零分钟处理：优先反向同段转移，其次速度模型外推，均失败时给保守先验带并标记。",
    "new_edge": "新建边无实测：梯边类比 e4 轿厢用时，普通边用速度模型外推，全部标记为假设值。",
    "lift_queue": "电梯先到先服务，按 capacity/cycle_seconds 排队；停运时段到达需等待恢复，超过窗口或时限即失败。",
    "exit": "住宅出口在情景层固定（R1 的两个可能出口分别成情景），不为单栋楼挑选最有利出口。",
    "population": "origins 人数区间按共同情景缩放各 profile 请求人数，所有项目情景使用同一套人口与出口参数。",
    "construction_window_field": "planning-context 的 construction_departure_window=2 契约未定义语义，分析不使用其数值，仅登记。",
    "return_trip": "requests 只描述出行窗口的单向需求；new_edge 为单向时，返程可用性在证据登记中提示，不虚构反向边。",
}
