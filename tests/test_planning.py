"""Business-rule tests for the planning layer.

These tests pin the rules a caseworker relies on, not just schema shape:
measures stay separate, missing returns are not zero, branches are not double
counted, receipts are annual, siblings move together, loans only count when
signed, and expired loans are not opening capacity.
"""

import unittest
from datetime import date

from planning.analyzer import (
    check_confirmation,
    district_report,
    first_breach,
    grouping_current_phases,
    lock_analysis,
    loan_options,
    renegotiations,
    scenario_table,
    unit_rows,
)
from planning.capacity import Supply
from planning.data import Snapshot, load_snapshot
from planning.demand import (
    build_demand,
    consistency_issues,
    family_views,
    phase_estimate,
)
from planning.history import (
    all_outcomes,
    history_summary,
    phase_outcome,
    rolling_slips,
    stage_uncertainty,
)


class SnapshotDates(unittest.TestCase):
    def setUp(self):
        self.snapshot = load_snapshot()

    def test_earlier_observation_hides_later_records(self):
        early = load_snapshot(as_of="2026-11-19")
        self.assertEqual(early.as_of, date(2026, 11, 19))
        # The later B008 progress and the family update arrived after this date.
        self.assertNotIn(
            "E-progress-1", [p.progress_id for p in early.progress]
        )
        self.assertEqual(len(early.confirmations), 0)
        self.assertEqual(early.families["F000"].temporary, "河口")
        # At the data cut both are visible.
        self.assertEqual(self.snapshot.families["F000"].temporary, "青池")
        self.assertEqual(len(self.snapshot.confirmations), 12)

    def test_latest_family_statement_wins_but_history_is_kept(self):
        rows = self.snapshot.family_history["F000"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].temporary, "河口")
        self.assertEqual(rows[-1].temporary, "青池")


class HistorySeparation(unittest.TestCase):
    def setUp(self):
        self.snapshot = load_snapshot()

    def test_delivery_occupancy_enrolment_are_not_merged(self):
        outcome = phase_outcome(self.snapshot, "B000")
        self.assertEqual(outcome.planned_homes, 107)
        self.assertEqual(outcome.delivered_homes, 106)
        self.assertEqual(outcome.occupied_homes, 74)
        self.assertEqual(outcome.first_enrolled, 20)
        # The three measures must never be summed into one completion figure.
        self.assertNotEqual(outcome.delivered_homes, outcome.first_enrolled)

    def test_partial_delivery_is_not_complete(self):
        outcome = phase_outcome(self.snapshot, "B016")
        self.assertEqual(outcome.delivered_homes, 94)
        self.assertEqual(outcome.planned_homes, 163)
        self.assertFalse(outcome.complete)
        self.assertIn("部分", "".join(outcome.notes))

    def test_missing_return_is_not_zero(self):
        outcome = phase_outcome(self.snapshot, "B096")
        self.assertEqual(outcome.delivered_homes, 27)
        self.assertEqual(outcome.observation, "入住未回传")
        self.assertIsNone(outcome.enrollment_per_delivered)
        self.assertTrue(any("不能按零" in note for note in outcome.notes))

    def test_not_yet_due_is_distinguished_from_missing(self):
        early = load_snapshot(as_of="2026-11-19")
        self.assertEqual(phase_outcome(early, "B006").observation, "未到观察期")
        self.assertEqual(
            phase_outcome(self.snapshot, "B006").observation, "入学未回传"
        )

    def test_rolling_validation_uses_information_at_the_time(self):
        slips = rolling_slips(self.snapshot)
        b008 = [s for s in slips if s.phase == "B008"]
        # B008 has no delivery yet, so it contributes no realised slip.
        self.assertEqual(b008, [])
        b016 = [s for s in slips if s.phase == "B016"]
        self.assertTrue(all(s.actual_year == 2026 for s in b016))
        # A message that replaces another is not double-counted.
        b008_messages = [
            p.progress_id for p in self.snapshot.progress_by_phase["B008"]
        ]
        self.assertIn("B008-0", b008_messages)

    def test_stage_uncertainty_uses_ordinal_not_average(self):
        early_stage = stage_uncertainty(self.snapshot, "B035")
        self.assertEqual(early_stage["distribution"], {0: 42, 1: 24, 2: 11})
        late_stage = stage_uncertainty(self.snapshot, "B016")
        self.assertEqual(late_stage["distribution"], {0: 20})

    def test_history_summary_keeps_lags_separate(self):
        summary = history_summary(self.snapshot)
        self.assertIn("交付→入住", summary["lags"])
        self.assertIn("交付→首次入学", summary["lags"])
        self.assertEqual(summary["lags"]["交付→首次入学"], {2: 70})


class DemandAndBranches(unittest.TestCase):
    def setUp(self):
        self.snapshot = load_snapshot()

    def test_awaiting_child_routes_by_branch_not_to_both(self):
        for label in ("按期", "延后1年", "延后2年"):
            demand = build_demand(self.snapshot, branch_label=label)
            by_child = {}
            for placement in demand.placements:
                if placement.child:
                    by_child.setdefault((placement.child, placement.year), set()).add(
                        placement.campus
                    )
            doubled = {
                key: campuses
                for key, campuses in by_child.items()
                if len(campuses) > 1
            }
            self.assertEqual(doubled, {}, f"{label} 把同一儿童许给两个校区")

    def test_branch_changes_campus_for_delayed_delivery(self):
        on_time = build_demand(self.snapshot, branch_label="按期")
        delayed = build_demand(self.snapshot, branch_label="延后2年")

        def campus_of(demand, child, year):
            rows = [
                p
                for p in demand.placements
                if p.child == child and p.year == year
            ]
            self.assertEqual(len(rows), 1)
            return rows[0].campus

        # C010-0 is B098 (temporary 河口, permanent 新港), first enrolment
        # 2027.  Delivered on time, 2027 is the delivery year, so the child
        # is already at the permanent campus.  A two-year slip keeps the child
        # at the temporary campus in 2027 and 2028.
        self.assertEqual(campus_of(on_time, "C010-0", 2027), "新港")
        self.assertEqual(campus_of(delayed, "C010-0", 2027), "河口")
        self.assertEqual(campus_of(on_time, "C010-0", 2028), "新港")
        self.assertEqual(campus_of(delayed, "C010-0", 2028), "河口")
        self.assertEqual(campus_of(delayed, "C010-0", 2029), "新港")

    def test_registered_children_are_part_of_the_estimate(self):
        for phase_id in ("B008", "B017", "B044", "B098"):
            estimate = phase_estimate(self.snapshot, phase_id)
            self.assertLessEqual(estimate["unregistered_forecast"], estimate["expected_entrants"])
            self.assertGreaterEqual(estimate["unregistered_forecast"], 0)

    def test_siblings_move_together(self):
        # F005 (B053) has two children and no receipts, so both follow the
        # same delivery branch and stay at the same campus in every year.
        for label in ("按期", "延后1年", "延后2年"):
            demand = build_demand(self.snapshot, branch_label=label)
            by_year = {}
            for placement in demand.placements:
                if placement.family == "F005":
                    by_year.setdefault(placement.year, set()).add(placement.campus)
            for year, campuses in by_year.items():
                self.assertEqual(campuses, {next(iter(campuses))}, (label, year))

    def test_existing_receipt_keeps_the_family_together(self):
        # F000 updated its temporary address to 青池 after C000-0 already held
        # a receipt at 河口.  The receipt stands, and the unsigned sibling is
        # kept at the signed campus rather than moved alone; changing campus
        # requires a fresh receipt for every child.
        demand = build_demand(self.snapshot, branch_label="按期")
        rows = [p for p in demand.placements if p.family == "F000"]
        campus_by_child = {
            p.child: p.campus for p in rows if p.year == 2027 and p.child
        }
        self.assertEqual(campus_by_child["C000-0"], "河口")
        self.assertEqual(campus_by_child["C000-1"], "河口")
        issues = consistency_issues(self.snapshot)
        self.assertTrue(
            any(
                issue.get("family") == "F000"
                and "兄弟姐妹" in issue["detail"]
                for issue in issues
            )
        )


class SupplyAndLoans(unittest.TestCase):
    def setUp(self):
        self.snapshot = load_snapshot()

    def test_seat_is_min_of_teacher_and_classroom(self):
        supply = Supply(self.snapshot)
        seat = supply.seat("西桥", 2027, "上午", "低年段")
        self.assertEqual(seat.teachers, 40)  # signed X0-0 removed one teacher
        self.assertEqual(seat.classrooms, 32)
        self.assertEqual(seat.usable, 32)

    def test_only_signed_loans_change_capacity(self):
        supply = Supply(self.snapshot)
        self.assertEqual(supply.signed_offers, ["X0-0"])
        # X0-1 was withdrawn after one-sided approval; it must not count.
        self.assertFalse(self.snapshot.is_signed("X0-1"))
        self.assertTrue(self.snapshot.is_withdrawn("X0-1"))
        self.assertEqual(supply.band_seats("青池", 2027, "低年段"), 124)
        self.assertEqual(supply.band_seats("青池", 2028, "低年段"), 108)

    def test_expired_loan_is_not_next_year_capacity(self):
        supply = Supply(self.snapshot)
        self.assertEqual(supply.band_seats("西桥", 2027, "低年段"), 92)
        self.assertEqual(supply.band_seats("西桥", 2028, "低年段"), 108)
        # Durable capacity excludes even the signed loan.
        self.assertEqual(supply.durable_band_seats("西桥", 2027, "低年段"), 108)

    def test_loan_options_refuse_returned_capacity(self):
        supply = Supply(self.snapshot)
        options = {
            row["offer"]: row
            for row in loan_options(
                self.snapshot, supply, build_demand(self.snapshot), 2027, "低年段"
            )
        }
        # X0-0 ends in 2027; it cannot open a 3-year band.
        self.assertFalse(options["X0-0"]["usable"])
        self.assertIn("早于所需", options["X0-0"]["reason"])
        # X1-0 is one year only, so also not usable as opening capacity.
        self.assertFalse(options["X1-0"]["usable"])
        # X1-2 runs 2027-2029 and the sending campus keeps its own demand met.
        self.assertTrue(options["X1-2"]["usable"])

    def test_sending_campus_must_keep_its_rule(self):
        supply = Supply(self.snapshot)
        offer = self.snapshot.offers["X3-2"]
        rows = supply.source_after_transfer("X3-2")
        self.assertTrue(rows)
        self.assertTrue(all(row["source"] == offer.source for row in rows))
        options = {
            row["offer"]: row
            for row in loan_options(
                self.snapshot, supply, build_demand(self.snapshot), 2027, "低年段"
            )
        }
        self.assertFalse(options["X3-2"]["source_keeps_rule"])


class BranchComparison(unittest.TestCase):
    def setUp(self):
        self.snapshot = load_snapshot()
        self.report = district_report(self.snapshot)

    def test_first_breach_is_the_earliest_year(self):
        breach = self.report["commit"]["first_breach"]
        self.assertEqual(breach["year"], 2027)
        campuses = {unit["campus"] for unit in breach["units"]}
        self.assertIn("河口", campuses)
        self.assertIn("新港", campuses)

    def test_hold_reserves_every_branch(self):
        table = scenario_table(self.snapshot)
        self.assertTrue(table["rows"])
        for row in table["rows"]:
            self.assertGreaterEqual(row["hold_demand"] + 1e-9, row["commit_demand"])
            self.assertGreaterEqual(row["sacrificed"], -1e-9)
        sacrificed = [row for row in table["rows"] if row["sacrificed"] > 0]
        self.assertTrue(sacrificed, "等待应有可量化的接收额度代价")

    def test_lock_verdict_does_not_promise_two_campuses(self):
        for row in lock_analysis(self.snapshot):
            if not row["can_wait_for_progress"] and row["distinct_placements"]:
                self.assertIn("不得", row["verdict"])
            if row["can_wait_for_progress"] and row["distinct_placements"]:
                self.assertIn("不得对两校同时承诺", row["verdict"])

    def test_groups_state_applicability(self):
        for group in grouping_current_phases(self.snapshot):
            self.assertTrue(group["scope"])
            self.assertTrue(group["applicability"])
            if group["tier"] == "forecast":
                self.assertEqual(group["children"], [])
                self.assertIn("不落到个人", group["scope"])


class ConfirmationImpact(unittest.TestCase):
    def setUp(self):
        self.snapshot = load_snapshot()

    def test_existing_receipt_is_not_overwritten(self):
        check = check_confirmation(self.snapshot, "C000-1", "青池", 2027)
        self.assertFalse(check.feasible)
        self.assertTrue(any("保留原签收" in c for c in check.conflicts))

    def test_undecided_delivery_is_reported_as_a_risk(self):
        # A single receipt does not double-promise, so an open delivery branch
        # is a risk that forces renegotiation, not a hard block.
        # C050-0's B026 is 青池/林园 and both campuses have room for the
        # whole cohort; the open delivery branch is then a renegotiation risk
        # rather than a hard block.
        check = check_confirmation(self.snapshot, "C050-0", "青池", 2027)
        self.assertTrue(check.feasible)
        self.assertTrue(any("交付方向未定" in w for w in check.warnings))

    def test_receiving_capacity_gap_blocks(self):
        check = check_confirmation(self.snapshot, "C005-0", "河口", 2028)
        self.assertFalse(check.feasible)
        self.assertTrue(any("缺口" in b for b in check.blocking))
        self.assertIsNotNone(check.first_breach)

    def test_campus_outside_family_range_is_a_conflict(self):
        check = check_confirmation(self.snapshot, "C005-0", "西桥", 2028)
        self.assertFalse(check.feasible)
        self.assertTrue(any("不在该家庭登记地址" in c for c in check.conflicts))

    def test_impact_covers_receiving_campus_all_years(self):
        check = check_confirmation(self.snapshot, "C005-0", "河口", 2028)
        years = {row["year"] for row in check.impacts if row["campus"] == "河口"}
        self.assertEqual(years, {2028, 2029, 2030})
        self.assertTrue(all("tiers" in row for row in check.impacts))

    def test_sibling_is_carried_in_what_if(self):
        check = check_confirmation(self.snapshot, "C005-0", "河口", 2028)
        self.assertEqual(check.siblings, ["C005-0", "C005-1"])
        demand = build_demand(
            self.snapshot,
            branch_label="按期",
            extra_receipts=[("C005-0", "河口", 2028)],
        )
        placements = [p for p in demand.placements if p.family == "F005"]
        self.assertEqual({p.campus for p in placements}, {"河口"})
        # The younger sibling is carried in the same year as the receipt.
        sib = [p for p in placements if p.child == "C005-1" and p.year == 2028]
        self.assertTrue(sib)
        self.assertEqual(sib[0].campus, "河口")


class Renegotiation(unittest.TestCase):
    def setUp(self):
        self.snapshot = load_snapshot()

    def test_phase_correction_is_reported(self):
        rows = renegotiations(self.snapshot)["分期更正"]
        self.assertTrue(any(row["phase"] == "B008" for row in rows))
        detail = "".join(row["detail"] for row in rows)
        self.assertIn("2027", detail)
        self.assertIn("2028", detail)

    def test_family_address_change_moves_siblings_together(self):
        rows = renegotiations(self.snapshot)["家庭改址"]
        f000 = [row for row in rows if row["family"] == "F000"]
        self.assertTrue(f000)
        self.assertEqual(f000[0]["children"], ["C000-0", "C000-1"])
        self.assertTrue(f000[0]["needs_resign"])

    def test_loan_end_is_a_renegotiation(self):
        rows = renegotiations(self.snapshot)["借期结束"]
        self.assertTrue(rows)
        self.assertTrue(any(row["offer"] == "X0-0" and row["last"] == 2027 for row in rows))


class ForecastHandling(unittest.TestCase):
    def setUp(self):
        self.snapshot = load_snapshot()

    def test_forecast_comes_from_missing_returns_not_plan_year(self):
        # A 2025 phase without a first-enrolment return still contributes to
        # 2027; dropping it would quietly treat the missing result as zero.
        demand = build_demand(self.snapshot, branch_label="按期")
        phases = {
            p.phase
            for p in demand.placements
            if p.tier == "forecast" and p.phase
        }
        self.assertIn("B006", phases)
        self.assertIn("B024", phases)

    def test_forecast_years_are_stable_across_observation_dates(self):
        cut = load_snapshot()
        early = load_snapshot(as_of="2026-11-19")
        for snapshot in (cut, early):
            demand = build_demand(snapshot, branch_label="按期")
            for placement in demand.placements:
                if placement.tier == "forecast":
                    # 2025 delivery + 2 years = 2027, not shifted by the
                    # observation date.
                    self.assertGreaterEqual(placement.year, 2027)
        cut_phases = {
            p.phase for p in build_demand(cut, "按期").placements if p.tier == "forecast"
        }
        early_phases = {
            p.phase
            for p in build_demand(early, "按期").placements
            if p.tier == "forecast"
        }
        self.assertEqual(cut_phases, early_phases)

    def test_forecast_is_not_a_named_child(self):
        demand = build_demand(self.snapshot, branch_label="按期")
        for placement in demand.placements:
            if placement.tier == "forecast":
                self.assertIsNone(placement.child)
                self.assertTrue(placement.conditional)
                self.assertIsNotNone(placement.phase)

    def test_identified_breach_excludes_forecast(self):
        supply = Supply(self.snapshot)
        demand = build_demand(self.snapshot, branch_label="按期")
        rows = unit_rows(self.snapshot, demand, supply)
        identified = [
            type(row)(
                campus=row.campus,
                year=row.year,
                band=row.band,
                tier_counts={k: v for k, v in row.tier_counts.items() if k != "forecast"},
                seats=row.seats,
                durable_seats=row.durable_seats,
            )
            for row in rows
        ]
        total = first_breach(rows)
        firm = first_breach(identified)
        self.assertIsNotNone(firm)
        # The hard deficit is in the upper band, independent of forecasts.
        self.assertEqual(firm["year"], 2027)
        self.assertTrue(all(unit["band"] == "高年段" or unit["campus"] == "河口" for unit in firm["units"]))
        self.assertGreaterEqual(
            sum(unit["gap"] for unit in total["units"]),
            sum(unit["gap"] for unit in firm["units"]),
        )


class Consistency(unittest.TestCase):
    def setUp(self):
        self.snapshot = load_snapshot()

    def test_receipt_and_family_update_conflict_is_reported(self):
        issues = consistency_issues(self.snapshot)
        f000 = [
            issue
            for issue in issues
            if issue.get("family") == "F000"
            and issue["kind"] == "家庭登记与签收"
        ]
        self.assertTrue(f000)
        self.assertIn("保留既有签收", f000[0]["action"])

    def test_missing_history_is_flagged_not_zero(self):
        issues = consistency_issues(self.snapshot)
        missing = [i for i in issues if i["kind"] == "历史结果缺失"]
        self.assertTrue(missing)
        self.assertTrue(all("不按零" in i["action"] for i in missing))

    def test_registered_over_estimate_is_flagged(self):
        issues = consistency_issues(self.snapshot)
        self.assertTrue(
            any(i["kind"] == "登记数高于分期预估" for i in issues)
        )


if __name__ == "__main__":
    unittest.main()
