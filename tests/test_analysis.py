"""分析层单元测试：物理过滤、用时证据处理、电梯排队与容量分配。

测试只用合成 inputs/ 中的真实结构，不引入外部数据；运行：
  python -I /workspace/tests/test_analysis.py
"""
import json
import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.allocate import Option, greedy_allocate, milp_allocate
from analysis.feasibility import build_usable_edges, edge_physically_usable, profiles_from_df
from analysis.load import effective_outages, load_inputs
from analysis.scenarios import apply_network_state, build_scenario_registry
from analysis.travel import (EdgeTimeModel, build_edge_time_models,
                             simulate_weighted_lift_queue)
from analysis import config


class FeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_inputs(ROOT / "inputs")
        cls.profiles = profiles_from_df(cls.inputs.profiles)

    def test_steps_block_wheelchair_and_slow_walk(self):
        e3 = next(e for e in self.inputs.network.to_dict("records") if e["edge"] == "e3")
        for name in ("wheelchair", "slow-walk"):
            ok, reason = edge_physically_usable(e3, self.profiles[name])
            self.assertFalse(ok)
            self.assertIn("台阶", reason)
        ok, _ = edge_physically_usable(e3, self.profiles["walking"])
        self.assertTrue(ok)

    def test_width_blocks_wheelchair_on_e5(self):
        e5 = next(e for e in self.inputs.network.to_dict("records") if e["edge"] == "e5")
        ok, reason = edge_physically_usable(e5, self.profiles["wheelchair"])
        self.assertFalse(ok)
        self.assertIn("宽度", reason)

    def test_e5_binding_constraint_is_width_not_slope(self):
        # 数据事实：e5 坡度 0.07 对缓坡群体（上限 0.08）本就可行；
        # 轮椅被宽度 1.1m<1.2m 卡住，P3 只改坡度，放松坡度后仍不可行。
        e5 = next(e for e in self.inputs.network.to_dict("records") if e["edge"] == "e5")
        ok_slow, _ = edge_physically_usable(e5, self.profiles["slow-walk"])
        self.assertTrue(ok_slow)
        ok_w, _ = edge_physically_usable(e5, self.profiles["wheelchair"])
        self.assertFalse(ok_w)
        relaxed = dict(e5, slope=0.04)
        ok_w2, reason = edge_physically_usable(relaxed, self.profiles["wheelchair"])
        self.assertFalse(ok_w2)
        self.assertIn("宽度", reason)

    def test_blocked_table_returned(self):
        usable, blocked = build_usable_edges(self.inputs.network, self.profiles)
        self.assertIn("e3", blocked["wheelchair"])
        self.assertIn("e4", usable["wheelchair"])


class TravelModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_inputs(ROOT / "inputs")
        cls.models = build_edge_time_models(
            cls.inputs.traversals, cls.inputs.network, cls.inputs.projects, config)

    def test_observed_model_not_weak_and_positive(self):
        m = self.models[("e1", "walking")]
        self.assertEqual(m.source, "observed")
        self.assertFalse(m.weak)
        self.assertGreater(m.quantile(0.5), 0)

    def test_turned_back_edge_never_zero_time(self):
        # e3 对轮椅只有折返：模型不得给零，且应落保守来源
        m = self.models[("e3", "wheelchair")]
        self.assertGreater(m.quantile(0.5), 0)
        self.assertNotEqual(m.source, "observed")
        self.assertEqual(m.evidence["n_completed"], 0)

    def test_unobserved_wheelchair_e11_gets_nonzero_analog(self):
        m = self.models[("e11", "wheelchair")]
        self.assertGreater(m.quantile(0.5), 0)
        self.assertTrue(m.weak)
        self.assertIn(m.source, ("reverse-analog", "speed-model", "fallback"))

    def test_reverse_analog_uses_same_segment_not_prefix_edge(self):
        # 回归：e11 的反向必须是 e11r，不能因多位数边名错算成 e1r
        m = self.models[("e11", "wheelchair")]
        self.assertEqual(m.source, "reverse-analog")
        self.assertIn("e11r", m.evidence["reason"])
        # e11 长 180m，转移用时应与 e11r 同量级，远高于 80m 的 e1r
        self.assertGreater(m.quantile(0.5),
                           self.models[("e1r", "wheelchair")].quantile(0.5))

    def test_new_edges_are_flagged_assumptions(self):
        self.assertEqual(self.models[("new-lift", "wheelchair")].source, "lift-analog")
        self.assertTrue(self.models[("new-link", "walking")].weak)

    def test_censored_value_never_drops_distribution_floor(self):
        # 每条有删失观测的边，p50 为正且有限
        for (edge, prof), m in self.models.items():
            if m.evidence.get("n_censored"):
                self.assertTrue(math.isfinite(m.quantile(0.5)))
                self.assertGreater(m.quantile(0.5), 0)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.window = {"lift": "L1", "open": 480, "close": 1080,
                       "capacity": 4, "cycle_seconds": 100}
        self.no_outage = pd.DataFrame(columns=["lift", "start", "end", "reason"])

    def test_first_batch_departs_immediately(self):
        waits = simulate_weighted_lift_queue([(500, 3)], self.window, self.no_outage)
        self.assertAlmostEqual(waits[500], 0.0)

    def test_overflow_waits_one_cycle(self):
        waits = simulate_weighted_lift_queue(
            [(500, 6)], self.window, self.no_outage)
        # 4 人即走、2 人等一个周期（100 秒=1.6667 分钟），
        # 加权平均候梯 = 2/6 × 1.6667 = 0.5556 分钟，且严格为正
        self.assertGreater(waits[500], 0.0)
        self.assertAlmostEqual(waits[500], (2 / 6) * (100 / 60), places=4)

    def test_outage_postpones_departure(self):
        outage = pd.DataFrame([{"lift": "L1", "start": 500, "end": 520, "reason": "x"}])
        waits = simulate_weighted_lift_queue([(500, 2)], self.window, outage)
        self.assertAlmostEqual(waits[500], 20.0)

    def test_arrival_after_close_is_infinite(self):
        waits = simulate_weighted_lift_queue([(1090, 1)], self.window, self.no_outage)
        self.assertTrue(math.isinf(waits[1090]))

    def test_joint_arrivals_share_capacity(self):
        # 两批各 3 人同时到：共 6 人，4 人即走 2 人等周期
        waits = simulate_weighted_lift_queue(
            [(500, 3), (500, 3)], self.window, self.no_outage)
        self.assertGreater(waits[500], 0)


class OutageActionTests(unittest.TestCase):
    def test_restore_shrinks_outage(self):
        inputs = load_inputs(ROOT / "inputs")
        actions = json.loads((ROOT / "inputs/actions.json").read_text("utf-8"))
        eff = effective_outages(inputs.outages, actions, apply_restore=True)
        self.assertEqual(len(eff), 1)
        self.assertEqual(eff.iloc[0]["start"], 660)
        self.assertEqual(eff.iloc[0]["end"], 720)
        raw = effective_outages(inputs.outages, actions, apply_restore=False)
        self.assertEqual(raw.iloc[0]["end"], 780)


class AllocationTests(unittest.TestCase):
    def _opt(self, rid, svc, persons, rel, prof="wheelchair"):
        return Option(rid, "R", prof, svc, persons, rel, 100.0, 120.0, rel >= 0.8)

    def test_capacity_constraint_binds(self):
        opts = [self._opt("g1", "S1", 10, 0.9), self._opt("g2", "S1", 10, 0.9)]
        res = milp_allocate(opts, {"S1": 12}, {"g1": 10, "g2": 10})
        self.assertAlmostEqual(res["served"], 12.0)

    def test_vulnerable_group_prioritized_at_tight_capacity(self):
        opts = [self._opt("gw", "S1", 10, 0.9, "wheelchair"),
                self._opt("gk", "S1", 10, 0.9, "walking")]
        res = milp_allocate(opts, {"S1": 10}, {"gw": 10, "gk": 10})
        alloc = res["allocation"]
        self.assertAlmostEqual(alloc.get(("gw", "S1"), 0), 10.0)
        self.assertNotIn(("gk", "S1"), alloc)

    def test_greedy_respects_capacity_and_matches_total(self):
        opts = [self._opt("g1", "S1", 10, 0.95),
                self._opt("g1", "S2", 10, 0.9),
                self._opt("g2", "S1", 10, 0.85)]
        caps = {"S1": 12, "S2": 8}
        milp_res = milp_allocate(opts, caps, {"g1": 10, "g2": 10})
        greedy_res = greedy_allocate(opts, caps, {"g1": 10, "g2": 10})
        self.assertAlmostEqual(milp_res["served"], greedy_res["served"], places=6)
        self.assertLessEqual(milp_res["served"], 20.0)

    def test_ineligible_options_never_allocated(self):
        opts = [self._opt("g1", "S1", 10, 0.2)]
        res = milp_allocate(opts, {"S1": 30}, {"g1": 10})
        self.assertEqual(res["served"], 0.0)


class ScenarioNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_inputs(ROOT / "inputs")
        cls.inputs._outages = effective_outages(
            cls.inputs.outages, cls.inputs.actions, apply_restore=True)
        cls.registry = {s.name: s for s in build_scenario_registry(
            cls.inputs.projects, float(cls.inputs.planning["budget_yuan"]))}

    def _edges(self, name):
        net, _, _ = apply_network_state(self.registry[name], self.inputs)
        return set(net["edge"])

    def test_construction_states_have_no_new_edge(self):
        for name in ("P1_construction", "P2_construction",
                     "P1P2_construction", "P1P2_construction_closeboth"):
            edges = self._edges(name)
            self.assertNotIn("new-lift", edges, name)
            self.assertNotIn("new-link", edges, name)

    def test_completion_states_include_their_new_edge(self):
        self.assertIn("new-lift", self._edges("P1_completion"))
        self.assertIn("new-link", self._edges("P2_completion"))
        combo = self._edges("P1P2_completion")
        self.assertIn("new-lift", combo)
        self.assertIn("new-link", combo)

    def test_closeboth_removes_reverse_edges_during_construction(self):
        edges = self._edges("P1P2_construction_closeboth")
        for e in ("e4", "e4r", "e7", "e7r"):
            self.assertNotIn(e, edges)
        # 常规单向施工封闭保留反向边
        normal = self._edges("P1P2_construction")
        self.assertIn("e4r", normal)
        self.assertIn("e7r", normal)

    def test_p2_alone_leaves_n4_disconnected(self):
        # P2 单独完工时 new-link 存在，但 N4 的唯一入边 new-lift 属于 P1、未建成
        net = self._edges("P2_completion")
        self.assertIn("new-link", net)
        self.assertNotIn("new-lift", net)

    def test_p3_does_not_relax_width_for_wheelchair(self):
        net, _, _ = apply_network_state(self.registry["P3_completion"], self.inputs)
        profiles = profiles_from_df(self.inputs.profiles)
        e5 = next(r for r in net.to_dict("records") if r["edge"] == "e5")
        ok, reason = edge_physically_usable(e5, profiles["wheelchair"])
        self.assertFalse(ok)
        self.assertIn("宽度", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
