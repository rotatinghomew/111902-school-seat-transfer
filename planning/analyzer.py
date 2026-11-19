"""Projection, branch comparison, confirmation impact and renegotiation.

The functions here answer the operational questions, not just arithmetic:

* where (campus, grade, year) demand exceeds real seats, and which is the
  first school year that cannot be honoured;
* what is safe to promise now and what can only be an alternative;
* what a wait costs in receiving quota per school, and whether the district
  can even afford to wait until the next progress report;
* what a new confirmation does to *both* schools over the coming years;
* which arrangements must be renegotiated because a phase moved, a family
  changed address, or a loan expired.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

from .capacity import Supply
from .data import BANDS, Snapshot, band_of
from .demand import (
    COMMITTED_TIERS,
    TIERS,
    TIER_LABELS,
    DemandSet,
    branch_map,
    build_demand,
    family_views,
)
from .history import all_outcomes, campus_history, observation_status, stage_uncertainty

# A phase's delivery outcome is treated as distinguishable from its plan only
# when the probability of the alternative branch exceeds this share.
DISTINGUISHING_SHARE = 0.0


@dataclass
class UnitRow:
    campus: str
    year: int
    band: str
    tier_counts: dict[str, float]
    seats: int
    durable_seats: int
    forecast_unreported: float = 0.0
    forecast_unbuilt: float = 0.0

    @property
    def demand(self) -> float:
        return sum(self.tier_counts.values())

    @property
    def confirmed(self) -> float:
        """Demand that may be treated as signed capacity this year.

        ``choice`` is registered demand but not a signed quota; ``intent``,
        ``awaiting`` and ``forecast`` are excluded for the same reason.
        """

        return sum(self.tier_counts.get(t, 0) for t in COMMITTED_TIERS)

    @property
    def waiting(self) -> float:
        return self.tier_counts.get("awaiting", 0)

    @property
    def forecast(self) -> float:
        return self.tier_counts.get("forecast", 0)

    @property
    def identified(self) -> float:
        """Demand that is not a group forecast."""

        return self.demand - self.forecast

    @property
    def slack(self) -> float:
        return self.seats - self.demand

    @property
    def free_after_confirmed(self) -> float:
        return self.seats - self.confirmed

    @property
    def planned(self) -> float:
        return self.demand

    @property
    def slack_without_loans(self) -> int:
        return self.durable_seats - self.demand


def firm_demand(demand: DemandSet) -> DemandSet:
    """The same demand without unregistered group forecasts.

    Forecasts have no identity and may not materialise, so they must not on
    their own block a named child's confirmation; they are reported as an
    additional risk.
    """

    return DemandSet(
        branch=demand.branch,
        placements=[p for p in demand.placements if p.tier != "forecast"],
    )


def unit_rows(snapshot: Snapshot, demand: DemandSet, supply: Supply) -> list[UnitRow]:
    years = sorted({p.year for p in demand.placements})
    tiers = demand.by_tier()
    out: list[UnitRow] = []
    for campus in snapshot.campuses:
        for year in years:
            for band in BANDS:
                counts = {
                    t: tiers.get((campus, year, band, t), 0.0) for t in TIERS
                }
                unreported = sum(
                    p.count
                    for p in demand.placements
                    if p.campus == campus
                    and p.year == year
                    and p.band == band
                    and p.tier == "forecast"
                    and p.source_kind == "unreported"
                )
                unbuilt = counts.get("forecast", 0.0) - unreported
                if not any(counts.values()) and not supply.band_seats(campus, year, band):
                    continue
                out.append(
                    UnitRow(
                        campus=campus,
                        year=year,
                        band=band,
                        tier_counts=counts,
                        forecast_unreported=unreported,
                        forecast_unbuilt=unbuilt,
                        seats=supply.band_seats(campus, year, band),
                        durable_seats=supply.durable_band_seats(campus, year, band),
                    )
                )
    return out


def grade_rows(snapshot: Snapshot, demand: DemandSet) -> list[dict]:
    """Campus/year/grade detail, with identified and forecast split."""

    agg: dict[tuple[str, int, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for placement in demand.placements:
        agg[(placement.campus, placement.year, placement.grade)][placement.tier] += placement.count
    out = []
    for (campus, year, grade), counts in sorted(agg.items()):
        forecast = counts.get("forecast", 0.0)
        out.append(
            {
                "campus": campus,
                "year": year,
                "grade": grade,
                "band": band_of(grade),
                "tiers": {t: round(counts.get(t, 0), 2) for t in TIERS if counts.get(t)},
                "total": round(sum(counts.values()), 2),
                "identified": round(sum(counts.values()) - forecast, 2),
                "forecast": round(forecast, 2),
            }
        )
    return out


def first_breach(rows: Iterable[UnitRow]) -> dict | None:
    """Earliest year with demand above seats, with the binding units.

    Seats are band-level, so a breach is reported at campus/year/band.  An
    over-capacity band is a scheduling gap even if another band has space,
    because teachers and classrooms are not interchangeable across bands in
    the same block.
    """

    breaches = [r for r in rows if r.demand > r.seats + 1e-9]
    if not breaches:
        return None
    year = min(r.year for r in breaches)
    units = [r for r in breaches if r.year == year]
    return {
        "year": year,
        "units": [
            {
                "campus": r.campus,
                "band": r.band,
                "demand": round(r.demand, 1),
                "seats": r.seats,
                "gap": round(r.demand - r.seats, 1),
                "tiers": {k: round(v, 1) for k, v in r.tier_counts.items() if v},
            }
            for r in sorted(units, key=lambda x: (x.campus, x.band))
        ],
        "first_year": year,
    }


@dataclass
class Scenario:
    name: str
    branch: str | None
    rows: list[UnitRow]
    demand: DemandSet


def scenario_rows(
    snapshot: Snapshot,
    branch: str | None,
    supply: Supply,
    first_year: int | None = None,
) -> Scenario:
    first_year = first_year or snapshot.as_of.year
    years = range(first_year, snapshot.resource_years()[1] + 1)
    demand = build_demand(snapshot, branch, years=years)
    return Scenario(
        name=branch or "最可能（各分期按期）",
        branch=branch,
        rows=unit_rows(snapshot, demand, supply),
        demand=demand,
    )


def scenario_table(snapshot: Snapshot, supply: Supply | None = None) -> dict:
    """Commit-now versus hold-to-lock comparison.

    ``commit`` chooses the on-time branch for every open phase.  ``hold`` keeps
    enough seats for *every* branch that is still possible, so no child or
    classroom is double-promised.  The difference per campus/year/band is the
    receiving quota sacrificed by waiting.
    """

    supply = supply or Supply(snapshot)
    first_supply_year = snapshot.resource_years()[0]
    branches = branch_map(snapshot)
    labels = sorted({b.label for options in branches.values() for b in options})
    commit_label = "按期" if "按期" in labels else (labels[0] if labels else None)
    commit = scenario_rows(snapshot, commit_label, supply, first_supply_year)

    # Envelope: per unit, the maximum demand over all branches, plus confirmed
    # demand that does not move.  This is what a hold must reserve.
    branch_rows = {}
    for label in labels:
        branch_rows[label] = {
            (r.campus, r.year, r.band): r
            for r in scenario_rows(snapshot, label, supply, first_supply_year).rows
        }
    envelope: dict[tuple[str, int, str], dict] = {}
    for key in {(r.campus, r.year, r.band) for r in commit.rows}:
        samples = [branch_rows[label][key] for label in labels if key in branch_rows[label]]
        if not samples:
            continue
        seats = samples[0].seats
        # Confirmed tiers are identical across branches; waiting and forecast
        # can differ.  Reserve the worst case for each tier.
        counts = {}
        for tier in TIERS:
            counts[tier] = max(s.tier_counts.get(tier, 0) for s in samples)
        envelope[key] = {"tiers": counts, "seats": seats}

    comparison = []
    for key, env in sorted(envelope.items()):
        campus, year, band = key
        commit_row = next(
            (r for r in commit.rows if (r.campus, r.year, r.band) == key), None
        )
        if commit_row is None:
            continue
        commit_free = commit_row.seats - commit_row.demand
        hold_free = env["seats"] - sum(env["tiers"].values())
        comparison.append(
            {
                "campus": campus,
                "year": year,
                "band": band,
                "seats": env["seats"],
                "commit_demand": round(commit_row.demand, 1),
                "hold_demand": round(sum(env["tiers"].values()), 1),
                "commit_free": round(commit_free, 1),
                "hold_free": round(hold_free, 1),
                "sacrificed": round(commit_free - hold_free, 1),
                "hold_gap": round(max(0.0, sum(env["tiers"].values()) - env["seats"]), 1),
            }
        )

    return {
        "commit_branch": commit_label,
        "branches": labels,
        "rows": comparison,
        "commit_breach": first_breach(commit.rows),
        "note": (
            "保留代价按全部仍可能走向的上包络计算；awaiting 与 forecast 在保留方案中"
            "占位但不落到个人。"
        ),
    }


def lock_analysis(snapshot: Snapshot, campus: str | None = None) -> list[dict]:
    """Decide whether waiting for the next progress return is affordable.

    The relevant lock date is the *binding* offer for a destination campus: the
    one whose expiry would strand the cohort.  That is the longest-lasting
    offer that can serve the campus in the phase's delivery years, because a
    one-year offer is not a solution for a cohort that needs the band later.

    Two branches are only treated as distinguishable if they imply different
    *placements*, not merely different delivery dates.  A family's temporary
    address and delivery address decide whether a delayed delivery changes the
    campus at all.
    """

    branches = branch_map(snapshot)
    out = []
    for pid, options in branches.items():
        phase = snapshot.phase(pid)
        members = [v for v in family_views(snapshot).values() if v.phase == pid]
        if not members:
            continue
        next_progress = snapshot.next_scheduled_progress(pid)
        for destination in phase.campuses:
            if campus and destination != campus:
                continue
            offers = [
                offer
                for offer in snapshot.offers.values()
                if offer.destination == destination
                and not snapshot.is_withdrawn(offer.offer)
                and offer.last >= phase.plan_year
            ]
            binding = max(offers, key=lambda o: (o.last, o.lock_by)) if offers else None
            earliest = min(offers, key=lambda o: o.lock_by) if offers else None
            # The earliest lock decides whether waiting is affordable; the
            # binding (longest) offer sets the latest effective date for a plan
            # that actually relies on a transfer.
            lock = binding.lock_by if binding else None
            decision_date = earliest.lock_by if earliest else None
            # Distinct placements: temporary vs permanent campus for children
            # whose entry year straddles the delivery year in at least one
            # branch.
            # Only families without a chosen campus are branch-dependent; a
            # family that already chose a campus is placed there regardless of
            # the delivery year.
            branch_dependent = [v for v in members if not v.choice]
            placement_paths = set()
            for view in branch_dependent:
                for child in view.children:
                    for branch in options:
                        campus_for_child = (
                            view.temporary
                            if child.first_year < branch.delivery_year
                            else view.permanent
                        )
                        placement_paths.add((child.child, campus_for_child))
            distinct = len(placement_paths) > 1
            can_wait = bool(next_progress) and (
                decision_date is None or next_progress <= decision_date
            )
            if can_wait and not distinct:
                verdict = (
                    f"可等下一次进度（{next_progress.isoformat()}）；"
                    "方向已不影响校区，等待期不占两校资源"
                )
            elif can_wait and distinct:
                verdict = (
                    f"可等下一次进度（{next_progress.isoformat()}）到 {decision_date.isoformat()}；"
                    "但方向仍影响校区，等待期间不得对两校同时承诺同一儿童/教室"
                )
            elif distinct:
                verdict = (
                    f"下一次进度 {next_progress.isoformat() if next_progress else '无'}"
                    f" 晚于最早锁定日 {decision_date.isoformat() if decision_date else '无'}；"
                    "两种走向到锁定日仍无法区分，不得把同一儿童/教室许给两校"
                )
            else:
                verdict = (
                    f"须在最早锁定日 {decision_date.isoformat() if decision_date else '无'} "
                    "前确认；方向已定，可先按该方向排资源"
                )
            out.append(
                {
                    "phase": pid,
                    "destination": destination,
                    "branches": [
                        {
                            "label": b.label,
                            "delivery_year": b.delivery_year,
                            "weight": round(b.weight, 2),
                            "basis": b.basis,
                        }
                        for b in options
                    ],
                    "distinct_placements": distinct,
                    "next_progress": next_progress.isoformat() if next_progress else None,
                    "next_lock": lock.isoformat() if lock else None,
                    "earliest_lock": decision_date.isoformat() if decision_date else None,
                    "decision_date": decision_date.isoformat() if decision_date else None,
                    "earliest_offer": earliest.offer if earliest else None,
                    "binding_offer": binding.offer if binding else None,
                    "binding_last": binding.last if binding else None,
                    "notification": (
                        f"{pid} 下一次进度回报，预计 {next_progress.isoformat()}"
                        if next_progress
                        else "该分期已过全部常规回报节点，没有可依赖的下一次通知"
                    ),
                    "latest_effective": lock.isoformat() if lock else None,
                    "unsigned_intent_void": (
                        f"{binding.offer} 若到 {lock.isoformat()} 未双方签认，另一方不得把该意向当作容量"
                        if binding and lock
                        else None
                    ),
                    "can_wait_for_progress": can_wait,
                    "verdict": verdict,
                }
            )
    return out


def _claim_by_other(snapshot: Snapshot, offer_id: str) -> str | None:
    """A loan already promised to another district cannot be offered again."""

    offer = snapshot.offers[offer_id]
    for other in snapshot.offers.values():
        if other.offer == offer_id:
            continue
        if other.source != offer.source or other.destination != offer.destination:
            continue
        if not (offer.first <= other.last and other.first <= offer.last):
            continue
        # Only a signed transfer actually consumes the named resources.  A
        # bare intent (even with one-sided approval) is not a claim.
        if snapshot.is_signed(other.offer):
            return f"同批资源已由 {other.offer} 向 {other.destination} 签认"
    return None


def loan_options(
    snapshot: Snapshot,
    supply: Supply,
    demand: DemandSet,
    year: int,
    band: str | None = None,
) -> list[dict]:
    """Remaining transfer options that add seats in ``year`` without breaking
    the sending campus's own teacher-and-classroom rule, and without relying
    on capacity that must be returned in a later year.
    """

    rows = unit_rows(snapshot, demand, supply)
    current = {
        (r.campus, r.year, r.band): r for r in rows if r.year == year
    }
    out = []
    for offer in snapshot.offers.values():
        if snapshot.is_withdrawn(offer.offer):
            out.append(
                {
                    "offer": offer.offer,
                    "status": "已撤回",
                    "usable": False,
                    "reason": "意向已撤回，不能作为容量",
                }
            )
            continue
        if claim := _claim_by_other(snapshot, offer.offer):
            out.append(
                {
                    "offer": offer.offer,
                    "status": "已被占用",
                    "usable": False,
                    "reason": claim,
                }
            )
            continue
        if not (offer.first <= year <= offer.last):
            out.append(
                {
                    "offer": offer.offer,
                    "status": "不在借期",
                    "usable": False,
                    "reason": f"借期 {offer.first}-{offer.last}，{year} 年无此资源",
                }
            )
            continue
        if band is not None and offer.band != band:
            continue
        state = snapshot.loan_state(offer.offer)
        signed = state == "双方签认"
        destination = offer.destination
        source = offer.source
        dest_row = current.get((destination, year, offer.band))
        src_row = current.get((source, year, offer.band))
        if dest_row is None:
            continue
        # A loan adds min(teacher, classroom) capacity, not the teacher figure.
        caps: dict[str, int] = {}
        for name in offer.resources:
            kind = name.split("-")[2]
            caps[kind] = max(
                (
                    row.capacity
                    for row in snapshot.resource_rows(name)
                    if row.year == year and row.enabled
                ),
                default=0,
            )
        added = min(caps.values()) if caps else 0
        if added == 0:
            out.append(
                {
                    "offer": offer.offer,
                    "status": "资源不可用",
                    "usable": False,
                    "reason": "该学年资源未启用或不存在",
                }
            )
            continue
        # Sending campus must still meet the identified demand under the same
        # teacher-and-classroom rule.  The unregistered group forecast is not
        # an identified child, so it does not veto a transfer on its own; it
        # is reported separately as a risk.
        source_planned = src_row.demand - src_row.forecast if src_row else 0.0
        source_after = src_row.seats - added if src_row else None
        source_ok = src_row is None or source_after >= source_planned
        source_forecast_risk = bool(
            src_row is not None and source_after is not None and source_after < src_row.demand
        )
        # A loan cannot be the only reason a cohort is placed when it expires
        # before the cohort finishes the band.  A 3-year band starting at
        # grade 1 needs the resource through year+2.  Report the shortfall
        # instead of calling the option "usable".
        needed_until = min(year + 2, snapshot.resource_years()[1])
        covers_band_end = offer.last >= needed_until
        expires_early = offer.last < needed_until
        # A signed transfer is already reflected in the supply; re-checking it
        # as a new option would double count.  Only unsigned offers are
        # candidate options here.
        usable = (
            not signed
            and (state in ("意向", "调出认可"))
            and source_ok
            and not expires_early
        )
        reasons = []
        if not source_ok and src_row is not None:
            reasons.append(
                f"调出后 {source} 自身缺口 {abs(round(source_after - source_planned, 1))}"
            )
        if expires_early:
            reasons.append(
                f"借期到 {offer.last} 年，早于所需 {needed_until} 年；"
                "不能把开办容量当次年必还资源"
            )
        if signed:
            reasons.append("该意向已双方签认，容量已计入，不再作为新增选项")
        reason = "；".join(reasons) if reasons else "可调拨"
        out.append(
            {
                "offer": offer.offer,
                "status": state,
                "usable": usable,
                "already_applied": signed,
                "signed": signed,
                "source": source,
                "destination": destination,
                "band": offer.band,
                "first": offer.first,
                "last": offer.last,
                "lock_by": offer.lock_by.isoformat(),
                "seats_added": added,
                "destination_free_before": round(dest_row.free_after_confirmed, 1),
                "destination_free_after": round(dest_row.free_after_confirmed + added, 1),
                "source_slack_after": (
                    round(source_after - source_planned, 1)
                    if src_row is not None and source_after is not None
                    else None
                ),
                "source_slack_including_forecast": (
                    round(source_after - src_row.demand, 1)
                    if src_row is not None and source_after is not None
                    else None
                ),
                "source_keeps_rule": source_ok,
                "source_forecast_risk": source_forecast_risk,
                "source_identified_demand": round(source_planned, 1) if src_row else 0.0,
                "source_forecast": round(src_row.forecast, 1) if src_row else 0.0,
                "expires_after": offer.last,
                "covers_declared_years": covers_band_end,
                "needed_until": needed_until,
                "reason": reason,
            }
        )
    return out


@dataclass
class ConfirmationCheck:
    child: str
    campus: str
    year: int
    feasible: bool
    family: str | None
    siblings: list[str]
    sibling_status: list[dict]
    grade_impacts: list[dict]
    impacts: list[dict]
    conflicts: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    blocking_detail: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    no_option_reason: str | None = None
    first_breach: dict | None = None
    alternatives: list[dict] = field(default_factory=list)


def check_confirmation(
    snapshot: Snapshot,
    child: str,
    campus: str,
    year: int,
    at: date | None = None,
) -> ConfirmationCheck:
    """Evaluate a proposed receiving confirmation at signing time.

    The check is rerun against the evidence and resources that exist at
    ``at`` (default: the latest data cut).  It never edits an existing
    receipt.  If the child or a sibling already holds a receipt at another
    campus, the proposal is a conflict and the original stands.  Siblings are
    carried together, and the sending campus is checked for the seats it
    retains.
    """

    at = at or snapshot.as_of
    if at != snapshot.as_of:
        # Re-run against the evidence actually available on the signing date.
        snapshot = Snapshot(snapshot.tables, as_of=at)
    view = family_views(snapshot).get(
        snapshot.child_applicant.get(child).family
    ) if child in snapshot.child_applicant else None
    if view is None:
        return ConfirmationCheck(
            child=child,
            campus=campus,
            year=year,
            feasible=False,
            family=None,
            siblings=[],
            sibling_status=[],
            grade_impacts=[],
            impacts=[],
            conflicts=["儿童不在登记名册"],
            warnings=[],
        )

    conflicts: list[str] = []
    for other, receipt in view.receipts.items():
        if receipt.campus != campus:
            conflicts.append(
                f"{other} 已有 {receipt.receipt} 签收在 {receipt.campus}（{receipt.year}），"
                "保留原签收，不擅自改校"
            )
    allowed_campuses = {view.temporary, view.permanent}
    if view.choice:
        allowed_campuses.add(view.choice)
    if campus not in allowed_campuses:
        conflicts.append(
            f"拟签校区 {campus} 不在该家庭登记地址/选校范围内"
            f"（{view.temporary}/{view.permanent}"
            + (f"，选校 {view.choice}" if view.choice else "")
            + "）"
        )

    siblings = [c.child for c in view.children]
    sibling_status = []
    for sibling in view.children:
        receipt = view.receipts.get(sibling.child)
        sibling_status.append(
            {
                "child": sibling.child,
                "receipt": receipt.receipt if receipt else None,
                "receipt_campus": receipt.campus if receipt else None,
                "action": (
                    "已有签收且与新签收同校，原签收继续有效"
                    if receipt and receipt.campus == campus
                    else (
                        f"原签收在 {receipt.campus}，与本签收不同校；保留原签收，须先重新签认"
                        if receipt
                        else "随家庭一起切换，需随本签收一并确认"
                    )
                ),
            }
        )
    supply = Supply(snapshot)
    demand = build_demand(
        snapshot,
        branch_label=None,
        extra_receipts=[(child, campus, year)],
    )
    rows = unit_rows(snapshot, demand, supply)
    firm_rows = unit_rows(snapshot, firm_demand(demand), supply)
    # Forecast children are not named, but they are part of the phase's
    # expected intake and still occupy seats.  They therefore count in the
    # capacity check; the report separates them so a caseworker can see how
    # much of the gap is a group estimate.
    breach = first_breach(rows)

    # Grade detail for the receiving campus and the campuses the family would
    # otherwise use, so a school can see which year group is hit.
    grade_demand = build_demand(
        snapshot,
        branch_label=None,
        extra_receipts=[(child, campus, year)],
    )
    grade_agg: dict[tuple[str, int, int, str], float] = defaultdict(float)
    for placement in grade_demand.placements:
        if placement.campus not in {campus, view.temporary, view.permanent}:
            continue
        if placement.year < year:
            continue
        grade_agg[
            (placement.campus, placement.year, placement.grade, placement.tier)
        ] += placement.count
    grade_detail = [
        {
            "campus": campus_name,
            "year": grade_year,
            "grade": grade,
            "tier": tier,
            "count": round(count, 2),
        }
        for (campus_name, grade_year, grade, tier), count in sorted(grade_agg.items())
    ]

    impacts = []
    counterpart = (
        view.permanent if campus == view.temporary else view.temporary
    )
    if view.choice and campus == view.choice:
        counterpart = view.permanent
    for r in rows:
        # Both schools' accounts: the receiving campus and the family's other
        # address campus, for every future year, plus every campus in the
        # signing year.  The full district table lives in the projection report.
        if r.year >= year and (
            r.campus in {campus, counterpart} or r.year == year
        ):
            impacts.append(
                {
                    "campus": r.campus,
                    "year": r.year,
                    "band": r.band,
                    "demand": round(r.demand, 1),
                    "seats": r.seats,
                    "slack": round(r.slack, 1),
                    "tiers": {k: round(v, 1) for k, v in r.tier_counts.items() if v},
                }
            )

    # A shortfall elsewhere in the district does not by itself make this
    # receipt infeasible; it must be reported, but the block is specific to
    # the receiving campus/year.  Deliveries whose direction is still open
    # are the reason a receipt cannot be finalised for an unbuilt phase.
    applicant = snapshot.child_applicant[child]
    cohort: list[dict] = []
    for unit in rows:
        if unit.campus != campus or unit.year < year:
            continue
        grade = applicant.first_grade + (unit.year - applicant.first_year)
        if grade < 1 or grade > 6 or unit.band != band_of(grade):
            continue
        if unit.demand > unit.seats + 1e-9:
            cohort.append(
                {
                    "year": unit.year,
                    "grade": grade,
                    "band": unit.band,
                    "demand": round(unit.demand, 1),
                    "seats": unit.seats,
                    "gap": round(unit.demand - unit.seats, 1),
                }
            )
    receiving_gap = bool(cohort)
    cohort_breach = (
        {"first_year": min(c["year"] for c in cohort), "units": cohort}
        if cohort
        else None
    )
    pending_branch = any(
        pid == view.phase and len(options) > 1
        for pid, options in branch_map(snapshot).items()
    )
    feasible = not conflicts and not receiving_gap
    blocking = []
    blocking_detail: list[dict] = []
    warnings: list[str] = []
    if conflicts:
        for conflict in conflicts:
            blocking.append(conflict)
            blocking_detail.append({"kind": "已确认去向", "detail": conflict})
        warnings.append(
            "存在与原签收冲突的更新：保留原签收，不自动改校，也不默认顺延使用下一份调拨意向"
        )
    for unit in cohort:
        blocking.insert(
            0,
            f"{campus} {unit['band']} {unit['year']} 学年缺口 {unit['gap']}"
            f"（该儿童读 {unit['grade']} 年级）",
        )
        firm = next(
            (
                r
                for r in firm_rows
                if r.campus == campus and r.year == unit["year"] and r.band == unit["band"]
            ),
            None,
        )
        blocking_detail.insert(
            0,
            {
                "kind": "受约束资源",
                "detail": (
                    f"{campus} {unit['band']} {unit['year']} 年需求 {unit['demand']}，"
                    f"座位 {unit['seats']}"
                    + (
                        f"（其中未登记预估 {unit['demand'] - firm.demand:.1f}）"
                        if firm and unit["demand"] > firm.demand
                        else ""
                    )
                ),
            },
        )
    if breach and not cohort:
        district_note = (
            f"区内存量缺口在 {breach['year']} 年："
            + "、".join(f"{u['campus']}{u['band']}{u['gap']}" for u in breach["units"])
            + "；不直接卡住该儿童，但需同批处理"
        )
        (blocking if not feasible else warnings).append(district_note)
        blocking_detail.append(
            {"kind": "同批受约束资源", "detail": f"区内最早缺口在 {breach['year']} 年"}
        )
    branch = next(
        (options for pid, options in branch_map(snapshot).items() if pid == view.phase),
        None,
    )
    if branch and len(branch) > 1 and year >= min(b.delivery_year for b in branch):
        # A receipt is a single promise, so it does not double-promise; but if
        # the family's actual campus depends on the branch, the seat may sit
        # unused.  Report it as a risk, not as a hard block.
        warnings.append(
            f"{view.phase} 交付方向未定（"
            + "；".join(f"{b.label} {b.delivery_year} 年" for b in branch)
            + "）；若按此签认，交付走向变化时须重议"
        )

    # Remaining transfer options must actually serve the receiving campus in
    # the child's band at the signing year.  A one-year or expired offer is
    # not counted as an option.
    child_band = band_of(applicant.first_grade)
    relevant_options = loan_options(
        snapshot,
        supply,
        build_demand(
            snapshot,
            branch_label=None,
            extra_receipts=[(child, campus, year)],
        ),
        year,
        child_band,
    )
    alternatives = [
        option
        for option in relevant_options
        if option.get("usable") and option.get("destination") == campus
    ]
    if not alternatives:
        nearby = [
            option
            for option in relevant_options
            if option.get("usable")
        ]
        alternative_note = (
            "可用的调拨在别的校区："
            + "、".join(
                f"{o['offer']} {o['source']}→{o['destination']}" for o in nearby
            )
            if nearby
            else "同年级同借期内没有可用的调拨选项"
        )
    else:
        alternative_note = None

    return ConfirmationCheck(
        child=child,
        campus=campus,
        year=year,
        feasible=feasible,
        family=view.family,
        siblings=siblings,
        sibling_status=sibling_status,
        grade_impacts=grade_detail,
        impacts=impacts,
        conflicts=conflicts,
        blocking=blocking,
        blocking_detail=blocking_detail,
        no_option_reason=(
            None
            if alternatives
            else (
                alternative_note
                + "；卡住新增确认的是："
                + "、".join(sorted({item["kind"] for item in blocking_detail}))
                if blocking_detail
                else alternative_note
            )
        ),
        warnings=warnings,
        first_breach=cohort_breach,
        alternatives=alternatives,
    )


def renegotiations(snapshot: Snapshot) -> dict[str, list[dict]]:
    """Arrangements that must be revisited, by cause.

    * phase correction: a later progress message changed the expected year;
    * family address change: a newer family statement moves the temporary
      address, which moves siblings together;
    * loan expiry: a signed transfer ends before the cohort finishes.
    """

    out: dict[str, list[dict]] = {"分期更正": [], "家庭改址": [], "借期结束": []}

    for pid in snapshot.phase_order:
        rows = snapshot.progress_by_phase.get(pid, [])
        if len(rows) < 2:
            continue
        replaced = {r.replaces for r in rows if r.replaces}
        superseded = [r for r in rows if r.progress_id in replaced]
        latest = snapshot.current_progress(pid)
        for old in superseded:
            if latest and old.expected_year != latest.expected_year:
                out["分期更正"].append(
                    {
                        "phase": pid,
                        "detail": (
                            f"{old.progress_id}（{old.received.isoformat()}）预计 {old.expected_year}，"
                            f"被 {latest.progress_id}（{latest.received.isoformat()}）改为 {latest.expected_year}"
                        ),
                        "effect": "按最新预计重排；被替换的回报仍留在滚动校验中",
                    }
                )

    for family, rows in snapshot.family_history.items():
        if len(rows) < 2:
            continue
        for previous, current in zip(rows, rows[1:]):
            if previous.temporary != current.temporary or previous.choice != current.choice:
                view = family_views(snapshot).get(family)
                kids = [c.child for c in view.children] if view else []
                out["家庭改址"].append(
                    {
                        "family": family,
                        "phase": current.phase,
                        "children": kids,
                        "detail": (
                            f"{previous.received.isoformat()} 暂住 {previous.temporary}/"
                            f"{previous.permanent} 选校 {previous.choice} → "
                            f"{current.received.isoformat()} 暂住 {current.temporary}/"
                            f"{current.permanent} 选校 {current.choice}"
                        ),
                        "effect": "同家庭兄弟姐妹一起切换；原签收保留，需重新签认后才能改校",
                        "needs_resign": any(view.receipts) if view else False,
                    }
                )

    for offer in snapshot.offers.values():
        if not snapshot.is_signed(offer.offer):
            continue
        # Does the destination rely on this loan beyond its last year?
        out["借期结束"].append(
            {
                "offer": offer.offer,
                "source": offer.source,
                "destination": offer.destination,
                "band": offer.band,
                "last": offer.last,
                "detail": f"{offer.offer} 借期到 {offer.last} 年；{offer.last + 1} 年起该班仍需按自有资源重排",
                "effect": "不能用借入的开办容量替代次年必还资源，下一届缺口需另找方案",
            }
        )
    return out


def grouping_current_phases(snapshot: Snapshot, year: int | None = None) -> list[dict]:
    """Schedulable groups for current phases, with explicit scope.

    A group is campus + entry grade + entry year + provenance + count.  It is
    what a school needs to plan classes; the report never assigns a named child
    to a class.  Each group states what it can be used for: a signed receipt is
    this year's quota, a receipt continuation or a chosen campus is demand but
    not signed quota, an awaiting group is conditional on the delivery branch,
    and a forecast group has no identity at all and is only for class counts.
    """

    year = year or snapshot.as_of.year
    check_years = [year, year + 1]
    demand = build_demand(
        snapshot,
        branch_label=None,
        years=range(min(check_years), snapshot.resource_years()[1] + 1),
    )
    views = family_views(snapshot)
    groups: dict[tuple, dict] = {}
    for placement in demand.placements:
        if placement.tier == "continuing":
            continue
        if placement.year not in check_years:
            continue
        if placement.phase is None:
            continue
        if placement.child and placement.tier != "forecast":
            applicant = snapshot.child_applicant.get(placement.child)
            # Class-planning groups are entry cohorts; later years are the same
            # children and are counted in the projection, not as new groups.
            if applicant is None or placement.year != applicant.first_year:
                continue
        phase = snapshot.phase(placement.phase)
        if phase.plan_year not in (year - 1, year, year + 1):
            continue
        if placement.tier == "forecast":
            key = (
                placement.campus,
                placement.year,
                placement.grade,
                "forecast",
                placement.phase,
            )
            entry = groups.setdefault(
                key,
                {
                    "campus": placement.campus,
                    "entry_year": placement.year,
                    "grade": placement.grade,
                    "tier": "forecast",
                    "tier_label": TIER_LABELS["forecast"],
                    "phase": placement.phase,
                    "count": 0.0,
                    "children": [],
                    "scope": (
                        f"{placement.phase} 未登记预估，按校区历史份额拆分；"
                        "只用于成组排班，不落到个人，也不能作为签认额度"
                    ),
                    "applicability": "仅用于班级数测算；不得按此对家庭作承诺",
                    "alternatives": [],
                },
            )
            entry["count"] = round(entry["count"] + placement.count, 2)
            continue
        view = views.get(placement.family or "")
        key = (
            placement.campus,
            placement.year,
            placement.grade,
            placement.tier,
            placement.phase,
        )
        entry = groups.setdefault(
            key,
            {
                "campus": placement.campus,
                "entry_year": placement.year,
                "grade": placement.grade,
                "tier": placement.tier,
                "tier_label": TIER_LABELS[placement.tier],
                "phase": placement.phase,
                "count": 0,
                "children": [],
                "scope": (
                    f"{placement.phase}（{phase.plan_year} 年，{phase.market}，"
                    f"{'/'.join(phase.campuses)}）"
                ),
                "applicability": {
                    "receipt": "本学年签认额度，可用于锁定教室与教师",
                    "intent": "签收校续读意向；本学年未签认，不能当作已确认额度",
                    "choice": "家庭已选校但未签收；可预留，不可对外承诺",
                    "awaiting": "依赖交付方向；两个走向都未排除时不能同时许给两校",
                }.get(placement.tier, ""),
                "alternatives": [],
            },
        )
        entry["count"] += 1
        if placement.child:
            entry["children"].append(placement.child)
        if placement.tier == "awaiting" and view is not None and not entry["alternatives"]:
            for branch in branch_map(snapshot).get(placement.phase, []):
                entry["alternatives"].append(
                    f"{branch.label}（{branch.delivery_year} 年交付）→ "
                    f"{view.permanent if placement.year >= branch.delivery_year else view.temporary}"
                )
    return sorted(
        groups.values(),
        key=lambda g: (g["entry_year"], g["campus"], g["grade"], g["tier"]),
    )


def district_report(snapshot: Snapshot) -> dict:
    """One-call assembly used by the CLI and tests."""

    supply = Supply(snapshot)
    # Supply only exists from the first calendar year with a resource row;
    # earlier demand is historical and must not be compared to absent rows.
    first_supply_year = snapshot.resource_years()[0]
    labels = sorted({b.label for options in branch_map(snapshot).values() for b in options})
    commit_label = "按期" if "按期" in labels else (labels[0] if labels else None)
    scenarios = {
        label: scenario_rows(snapshot, label, supply, first_supply_year)
        for label in labels
    }
    commit = scenarios.get(commit_label) if commit_label else scenario_rows(
        snapshot, None, supply, first_supply_year
    )

    breaches = {}
    for label, scenario in scenarios.items():
        breach = first_breach(scenario.rows)
        if breach:
            breaches[label] = breach

    comparison = scenario_table(snapshot, supply)
    history = {
        "outcomes": [
            {
                "phase": o.phase,
                "plan_year": o.plan_year,
                "market": o.market,
                "planned": o.planned_homes,
                "delivered": o.delivered_homes,
                "occupied": o.occupied_homes,
                "first_enrolled": o.first_enrolled,
                "observation": o.observation,
                "notes": o.notes,
            }
            for o in all_outcomes(snapshot)
            if o.first_delivery_year is not None or o.notes
        ],
        "stage": {
            pid: stage_uncertainty(snapshot, pid) for pid in snapshot.phase_order
        },
        "by_plan_year": observation_status(snapshot),
    }
    return {
        "as_of": snapshot.as_of.isoformat(),
        "data_cut": snapshot.data_cut.isoformat(),
        "commit": {
            "branch": commit_label,
            "rows": [
                {
                    "campus": r.campus,
                    "year": r.year,
                    "band": r.band,
                    "seats": r.seats,
                    "durable_seats": r.durable_seats,
                    "demand": round(r.demand, 1),
                    "tiers": {k: round(v, 1) for k, v in r.tier_counts.items() if v},
                    "identified": round(r.identified, 1),
                    "forecast_unreported": round(r.forecast_unreported, 1),
                    "forecast_unbuilt": round(r.forecast_unbuilt, 1),
                    "forecast": round(r.forecast, 1),
                    "slack": round(r.slack, 1),
                    "slack_on_identified": round(r.seats - r.identified, 1),
                }
                for r in commit.rows
            ],
            "first_breach": first_breach(commit.rows),
            "first_breach_identified": first_breach(
                [
                    UnitRow(
                        campus=r.campus,
                        year=r.year,
                        band=r.band,
                        tier_counts={
                            k: v for k, v in r.tier_counts.items() if k != "forecast"
                        },
                        seats=r.seats,
                        durable_seats=r.durable_seats,
                    )
                    for r in commit.rows
                ]
            ),
            "forecast_phases": sorted(
                {
                    p.phase
                    for p in commit.demand.placements
                    if p.tier == "forecast" and p.phase
                }
            ),
        },
        "branches": labels,
        "breaches": breaches,
        "scenario_table": comparison,
        "lock": lock_analysis(snapshot),
        "loans": supply.loan_summary(),
        "loan_warnings": supply.loan_warnings,
        "loan_options": loan_options(
            snapshot, supply, commit.demand, max(snapshot.as_of.year, first_supply_year)
        ),
        "groups": grouping_current_phases(snapshot),
        "grade_rows": grade_rows(snapshot, commit.demand),
        "history_sufficiency": [
            {
                "campus": row.campus,
                "market": row.market,
                "observed_phases": row.observed_phases,
                "delivered_homes": row.delivered_homes,
                "first_enrolled": row.first_enrolled,
                "ratio": round(row.ratio, 4) if row.ratio is not None else None,
                "dispersion": round(row.dispersion, 4) if row.dispersion is not None else None,
                "sufficient": row.sufficient,
                "reasons": row.reasons,
            }
            for row in campus_history(snapshot)
        ],
        "renegotiations": renegotiations(snapshot),
        "history": history,
    }
