# 城市更新资料契约

全部楼宇、机构、居民群体、设备、交易及事件均为本地合成，不对应新闻所涉真实主体，不声称新闻中发生这些业务问题。seed.json及各同名Parquet保留完全相同的原始值；actions.json是尚未应用的后续资料。inventory.json仅记录物理大小、摘要及行数，不是业务结论。原始量纲、空值和时区按下文解释。

network为有向边，反向须单列；length_m/slope/steps/width_m/lift_id描述通行。profiles给最大坡度/最小宽度/台阶许可，origins给住宅可能出口和人数区间，非确定出口分配。traversals.duration_seconds在status=turned_back/observation_end时仅为已观察时间。lift_windows为本地分钟[open,close)，outages优先，capacity/cycle_seconds约束梯次。services给类别/开放窗口/接待量，requests给同工作日出行人数/时间窗/可接受总时间。projects列新增或改善边、施工封闭边与成本。actions为L1午间恢复。资源与路径须在相同需求情景解释，无真实居民身份。

requests是名义需求，同住宅各profile互斥不重叠；origins人数区间表示名义总量的不确定性，出口分配和人数变化要在共同情景保持一致。项目new_edge为单向；P2同时把S1同一窗口接待上限改为new_capacity，不是新增另一家食堂。坡度限制按上坡正值适用，下坡亦需说明安全假设。未实测新边用时不能冒充已观察值，可按明确假设范围比较。
