"""适老出行可达性统计分析层。

只读取 inputs/ 中的合成原始资料，结果写入独立 outputs/，不修改任何原件。
模块划分：load 读取校验，feasibility 物理可行，travel 用时与排队路径，
demand 人口与出口情景，allocate 服务容量分配，scenarios 项目情景比较，report 输出。
"""
