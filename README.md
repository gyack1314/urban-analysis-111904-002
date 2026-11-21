# 电梯停运下的适老出行可达性评估

本项目保留城市更新研究场景的本地合成原始资料和物理读取工具，并在 `analysis/`
实现统计分析应用：物理可行过滤、删失感知的用时模型、联合电梯排队的时变路径、
Monte Carlo 可靠度与服务容量分配，以及项目组合/施工期/完工情景比较。所有结论
均为带假设与不确定性的分析输出，不自动派单、不给出唯一必选项目。

Python 3.11 与 requirements.txt 中固定的分析依赖配合使用。安装命令为 python -m pip install -r requirements.txt。运行 python -m materials inputs 校验摘要、大小、Parquet物理行数并经DuckDB实际读取。python -m unittest discover -s tests -v 核对JSON/Parquet值一致、损坏拒绝、领域原件引用及分析层方法测试。

分析层运行（隔离模式，不污染原件，结果写入 outputs/）：

```bash
python -I analysis/__main__.py inputs outputs
python -I tests/test_analysis.py
```

方法与输出解释见 docs/methodology.md。

容器用法为 docker build -t urban-materials .，再运行 docker run --rm urban-materials；输入测试使用 docker run --rm urban-materials python -m unittest discover -s tests -v。原件检查不连接真实业务系统。

inputs/保存原始观测及研究窗口，docs/input-contract.md说明语义。materials.build_inputs只重建同值Parquet与物理清单，不产生分析结果。actions.json内的新资料尚未处理；原件应保留，后续产品结果另存。实际分析接口、方法适用性和业务验证须在功能实现后补充，现有输入测试不替代未来产品验收。
