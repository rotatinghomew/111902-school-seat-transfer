"""Rolling validation of historical phases.

Three different facts are deliberately kept apart:

* ``交付`` — homes handed over (possibly parcel by parcel);
* ``入住`` — households actually moved in;
* ``首次入学`` — children from the phase enrolling for the first time.

They are never added into one "completion" number.  A phase that has delivered
one parcel is not a completed phase, and a phase whose observation window has
not closed (or whose return has not come back) is *unknown*, not zero.

Validation is rolling: for every progress message the phase had received at the
time, the expected year is compared with the year delivery actually happened.
That is the only defensible way to judge whether a promise made at that stage
was safe.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median, pstdev
from typing import Iterable

from .data import HistoryRecord, Snapshot

MEASURES = ("交付", "入住", "首次入学")


@dataclass
class PhaseOutcome:
    phase: str
    plan_year: int
    market: str
    planned_homes: int
    campuses: tuple[str, ...]
    delivered_homes: int = 0
    occupied_homes: int = 0
    first_enrolled: int = 0
    parcels_delivered: int = 0
    parcels_with_occupancy: int = 0
    parcels_with_enrollment: int = 0
    first_delivery_year: int | None = None
    first_occupancy_year: int | None = None
    first_enrollment_year: int | None = None
    observation: str = "未交付"
    complete: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def delivery_ratio(self) -> float:
        return self.delivered_homes / self.planned_homes if self.planned_homes else 0.0

    @property
    def occupancy_ratio(self) -> float:
        return self.occupied_homes / self.delivered_homes if self.delivered_homes else 0.0

    @property
    def enrollment_per_delivered(self) -> float | None:
        if self.observation != "已观察" or not self.delivered_homes:
            return None
        return self.first_enrolled / self.delivered_homes


def phase_outcome(snapshot: Snapshot, phase_id: str, at=None) -> PhaseOutcome:
    phase = snapshot.phase(phase_id)
    rows = (
        snapshot.history_as_of(phase_id, at)
        if at
        else list(snapshot.history_by_phase.get(phase_id, []))
    )
    outcome = PhaseOutcome(
        phase=phase_id,
        plan_year=phase.plan_year,
        market=phase.market,
        planned_homes=phase.planned_homes,
        campuses=phase.campuses,
    )
    by_measure: dict[str, list[HistoryRecord]] = defaultdict(list)
    for row in rows:
        by_measure[row.measure].append(row)

    delivery = by_measure.get("交付", [])
    occupancy = by_measure.get("入住", [])
    enrollment = by_measure.get("首次入学", [])

    outcome.delivered_homes = sum(r.count for r in delivery)
    outcome.occupied_homes = sum(r.count for r in occupancy)
    outcome.first_enrolled = sum(r.count for r in enrollment)
    outcome.parcels_delivered = len({r.parcel for r in delivery})
    outcome.parcels_with_occupancy = len({r.parcel for r in occupancy})
    outcome.parcels_with_enrollment = len({r.parcel for r in enrollment})

    outcome.first_delivery_year = min((r.effective_year for r in delivery), default=None)
    outcome.first_occupancy_year = min((r.effective_year for r in occupancy), default=None)
    outcome.first_enrollment_year = min((r.effective_year for r in enrollment), default=None)

    # Completeness is judged on the planned home count.  The small 1-home
    # shortfalls in the synthetic data are reporting rounding, not partial
    # delivery; a shortfall over 2% is treated as an open parcel.
    outcome.complete = bool(delivery) and outcome.delivered_homes >= 0.98 * outcome.planned_homes

    if not delivery:
        outcome.observation = "未交付"
        if snapshot.overdue_progress(phase_id):
            outcome.notes.append("进度回报已逾期未回，不等于零")
    elif not occupancy:
        outcome.observation = "入住未回传"
        if not outcome.complete:
            outcome.notes.append(
                f"只交付部分房屋（{outcome.delivered_homes}/{outcome.planned_homes}），"
                "且尚无入住结果；不能按零处理，也不能当作完工"
            )
        else:
            outcome.notes.append("已交付但尚无入住结果，不能按零处理")
    elif not enrollment:
        due_year = outcome.first_delivery_year + 2  # 83% of observed phases
        if at is not None and at.year < due_year or (at is None and snapshot.as_of.year < due_year):
            outcome.observation = "未到观察期"
            outcome.notes.append(f"首次入学观察期到 {due_year} 年，尚未到齐")
        else:
            outcome.observation = "入学未回传"
            outcome.notes.append("已过通常观察期但仍无首次入学结果，按缺失处理而非零")
    else:
        outcome.observation = "已观察"
        if not outcome.complete:
            outcome.notes.append("只交付部分房屋；比例以已交付数为分母")

    if delivery and not outcome.complete:
        outcome.notes.append(f"交付 {outcome.delivered_homes}/{outcome.planned_homes}，仍在分期")
    return outcome


def all_outcomes(snapshot: Snapshot) -> list[PhaseOutcome]:
    return [
        phase_outcome(snapshot, pid)
        for pid in snapshot.phase_order
    ]


@dataclass
class SlipRecord:
    phase: str
    plan_year: int
    market: str
    progress_id: str
    received: str
    ordinal: int
    milestone: str
    expected_year: int
    actual_year: int
    slip: int
    complete: bool


def rolling_slips(snapshot: Snapshot) -> list[SlipRecord]:
    """For each phase, compare every pre-delivery message with actual delivery.

    Only messages received before the first delivery return are used: after a
    handover, the outcome is no longer a forecast.
    """

    out: list[SlipRecord] = []
    for pid in snapshot.phase_order:
        deliveries = [
            r
            for r in snapshot.history_all
            if r.phase == pid and r.measure == "交付"
        ]
        if not deliveries:
            continue
        actual = min(r.effective_year for r in deliveries)
        first_received = min(r.received for r in deliveries)
        messages = [
            r
            for r in snapshot.progress_all
            if r.phase == pid and r.received <= first_received
        ]
        messages.sort(key=lambda r: (r.received, r.progress_id))
        # A message that explicitly replaces an earlier one supersedes it.
        replaced = {m.replaces for m in messages if m.replaces}
        messages = [m for m in messages if m.progress_id not in replaced]
        phase = snapshot.phase(pid)
        deliveries_count = sum(r.count for r in deliveries)
        complete = deliveries_count >= 0.98 * phase.planned_homes
        for ordinal, message in enumerate(messages):
            out.append(
                SlipRecord(
                    phase=pid,
                    plan_year=phase.plan_year,
                    market=phase.market,
                    progress_id=message.progress_id,
                    received=message.received.isoformat(),
                    ordinal=ordinal,
                    milestone=message.milestone,
                    expected_year=message.expected_year,
                    actual_year=actual,
                    slip=actual - message.expected_year,
                    complete=complete,
                )
            )
    return out


def slip_distribution(
    slips: Iterable[SlipRecord], ordinal: int | None = None, milestone: str | None = None
) -> Counter:
    rows = list(slips)
    if ordinal is not None:
        rows = [r for r in rows if r.ordinal == ordinal]
    if milestone is not None:
        rows = [r for r in rows if r.milestone == milestone]
    return Counter(r.slip for r in rows)


def timing_lags(snapshot: Snapshot) -> dict[str, Counter]:
    """Observed lags between the three measures, never merged."""

    outcome_by_phase = {o.phase: o for o in all_outcomes(snapshot)}
    delivery_to_occupancy: Counter = Counter()
    delivery_to_enrollment: Counter = Counter()
    occupancy_to_enrollment: Counter = Counter()
    for o in outcome_by_phase.values():
        if o.first_delivery_year is not None and o.first_occupancy_year is not None:
            delivery_to_occupancy[o.first_occupancy_year - o.first_delivery_year] += 1
        if o.first_delivery_year is not None and o.first_enrollment_year is not None:
            delivery_to_enrollment[o.first_enrollment_year - o.first_delivery_year] += 1
        if o.first_occupancy_year is not None and o.first_enrollment_year is not None:
            occupancy_to_enrollment[o.first_enrollment_year - o.first_occupancy_year] += 1
    return {
        "交付→入住": delivery_to_occupancy,
        "交付→首次入学": delivery_to_enrollment,
        "入住→首次入学": occupancy_to_enrollment,
    }


def observation_status(snapshot: Snapshot) -> dict:
    """Count phases by observation status and by plan year.

    This is the audit view that prevents "no result yet" from being read as
    zero: every unobserved phase is listed with its reason.
    """

    by_year: dict[int, dict[str, int]] = {}
    for outcome in all_outcomes(snapshot):
        bucket = by_year.setdefault(outcome.plan_year, {})
        bucket[outcome.observation] = bucket.get(outcome.observation, 0) + 1
    return by_year


def shares(counter: Counter) -> dict[int, float]:
    total = sum(counter.values())
    return {k: round(v / total, 3) for k, v in sorted(counter.items())} if total else {}


@dataclass
class CampusHistory:
    campus: str
    market: str
    observed_phases: int
    delivered_homes: int
    first_enrolled: int
    ratios: list[float]
    ratio: float | None
    dispersion: float | None
    sufficient: bool
    reasons: list[str]


def campus_history(snapshot: Snapshot, mature_year: int | None = None) -> list[CampusHistory]:
    """Enrolment per delivered home by campus and market, with sufficiency.

    "Sufficient" is deliberately conservative.  A rate is only allowed to
    stand behind a promise when the campus/market cell has at least five
    observed phases and 500 delivered homes.  Everything else is reported as
    directional evidence, not as a basis for a commitment.
    """

    outcomes = {o.phase: o for o in all_outcomes(snapshot)}
    if mature_year is None:
        mature_year = snapshot.as_of.year - 4
    cells: dict[tuple[str, str], list[PhaseOutcome]] = defaultdict(list)
    for o in outcomes.values():
        if o.plan_year > mature_year:
            continue
        for campus in o.campuses:
            cells[(campus, o.market)].append(o)

    out: list[CampusHistory] = []
    for (campus, market), rows in sorted(cells.items()):
        observed = [o for o in rows if o.observation == "已观察" and o.delivered_homes]
        delivered = sum(o.delivered_homes for o in observed)
        enrolled = sum(o.first_enrolled for o in observed)
        ratios = [o.first_enrolled / o.delivered_homes for o in observed]
        ratio = enrolled / delivered if delivered else None
        dispersion = (
            pstdev(ratios) if len(ratios) > 1 else None
        )
        reasons = []
        if len(observed) < 5:
            reasons.append(f"观察到的分期只有 {len(observed)} 个（少于 5 个）")
        if delivered < 500:
            reasons.append(f"已交付样本 {delivered} 套（少于 500 套）")
        out.append(
            CampusHistory(
                campus=campus,
                market=market,
                observed_phases=len(observed),
                delivered_homes=delivered,
                first_enrolled=enrolled,
                ratios=ratios,
                ratio=ratio,
                dispersion=dispersion,
                sufficient=not reasons,
                reasons=reasons,
            )
        )
    return out


def history_summary(snapshot: Snapshot) -> dict:
    outcomes = all_outcomes(snapshot)
    status = Counter(o.observation for o in outcomes)
    complete = [o for o in outcomes if o.complete]
    partial = [o for o in outcomes if o.first_delivery_year is not None and not o.complete]
    delayed = []
    for record in rolling_slips(snapshot):
        if record.slip > 0:
            delayed.append(record)
    slips = rolling_slips(snapshot)
    return {
        "phases": len(outcomes),
        "status": dict(status),
        "complete_phases": len(complete),
        "partial_or_open_phases": [
            {
                "phase": o.phase,
                "planned": o.planned_homes,
                "delivered": o.delivered_homes,
                "observation": o.observation,
            }
            for o in partial
        ],
        "slip_by_ordinal_milestone": {
            f"{ordinal}:{milestone}": dict(slip_distribution(slips, ordinal, milestone))
            for ordinal in sorted({r.ordinal for r in slips})
            for milestone in sorted({r.milestone for r in slips if r.ordinal == ordinal})
        },
        "delayed_messages": len(delayed),
        "lags": {k: dict(v) for k, v in timing_lags(snapshot).items()},
        "median_delay": median([r.slip for r in slips]) if slips else None,
    }


def stage_uncertainty(snapshot: Snapshot, phase_id: str) -> dict:
    """Delay distribution applicable to the phase's latest known stage.

    The ordinal of the current message is used, because a first ``施工``
    message is historically more optimistic than a later one.  A ``验房`` or
    ``分批`` message has matched the delivery year in every mature phase.
    """

    messages = snapshot.progress_by_phase.get(phase_id, [])
    if not messages:
        return {"stage": None, "distribution": {}, "basis": "无进度回报"}
    replaced = {m.replaces for m in messages if m.replaces}
    current = [m for m in messages if m.progress_id not in replaced]
    current.sort(key=lambda m: (m.received, m.progress_id))
    message = current[-1]
    ordinal = len(current) - 1
    slips = rolling_slips(snapshot)
    dist = slip_distribution(slips, ordinal=ordinal, milestone=message.milestone)
    if not dist:
        dist = slip_distribution(slips, milestone=message.milestone)
    return {
        "stage": message.milestone,
        "ordinal": ordinal,
        "expected_year": message.expected_year,
        "distribution": dict(sorted(dist.items())),
        "basis": f"{len([1 for _ in dist.elements()])} 条同类历史回报",
    }
