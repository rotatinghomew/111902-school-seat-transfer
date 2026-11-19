"""Turn registered families and school history into year-by-year placements.

Demand has several provenance tiers that must never be silently merged:

``continuing``
    Children already enrolled, advanced one grade per year, less confirmed
    leavers.  This is independent of housing delivery.

``receipt``
    A child with a signed receiving confirmation.  The confirmation is an
    annual document; later years are intent, not yet a signed quota.

``choice``
    A registered family that has chosen a campus but not signed a receipt.

``awaiting``
    A registered family whose campus depends on when the phase is delivered.
    The child is routed to the temporary address before delivery and to the
    delivery address afterwards; siblings move in the same year.  A child in
    this tier must never be counted as a promise at both campuses.

``forecast``
    The part of a phase's expected first enrolment not covered by registered
    children.  It has no individual identity or grade, so it is reported as a
    group forecast with an explicit scope, never as a child.

Registered children are *part of* the phase estimate, not an addition to it:
``forecast = max(0, expected - registered)``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .data import (
    GRADES,
    Snapshot,
    band_of,
)
from .history import all_outcomes, campus_history, phase_outcome, stage_uncertainty


TIERS = ("continuing", "receipt", "intent", "choice", "awaiting", "forecast")
TIER_LABELS = {
    "continuing": "已有学籍续读",
    "receipt": "已签接收确认（本学年）",
    "intent": "签收校续读意向（未签认）",
    "choice": "登记已选校",
    "awaiting": "待交付方向",
    "forecast": "分期未登记预估",
}
# Tiers a school may treat as signed capacity.  A registered choice is demand
# the district must plan for, but it is not yet a signed quota.  ``intent``,
# ``awaiting`` and ``forecast`` are demand signals but never promises.
COMMITTED_TIERS = ("continuing", "receipt")
# Tiers that need a seat but are not signed: choices, receipt continuation,
# branch-dependent families and group forecasts.
PLANNED_TIERS = ("choice", "intent", "awaiting", "forecast")


@dataclass(frozen=True)
class Placement:
    """One headcount unit.

    Individual children and unregistered group forecasts are both represented;
    ``count`` is 1 for an identified child and may be fractional for a group
    forecast split by historical share.  A group forecast always has
    ``child=None`` and ``conditional=True``.
    """

    campus: str
    year: int
    grade: int
    tier: str
    basis: str
    count: float = 1.0
    family: str | None = None
    child: str | None = None
    phase: str | None = None
    branch: str | None = None
    conditional: bool = False
    note: str | None = None
    # For forecast rows: "unbuilt" (phase not delivered) or "unreported"
    # (delivered but the first-enrolment return has not arrived).
    source_kind: str | None = None

    @property
    def band(self) -> str:
        return band_of(self.grade)


@dataclass
class Branch:
    phase: str
    label: str
    delivery_year: int
    weight: float
    basis: str


@dataclass
class FamilyView:
    family: str
    phase: str
    temporary: str
    permanent: str
    choice: str | None
    children: list
    receipts: dict[str, object]
    status: str
    issues: list[str] = field(default_factory=list)
    # A family split between an existing receipt and a newer choice must not
    # have its unsigned siblings follow the new choice until they re-sign.
    anchor_campus: str | None = None


def family_views(snapshot: Snapshot) -> dict[str, FamilyView]:
    cached = getattr(snapshot, "_family_views_cache", None)
    if cached is not None:
        return cached
    views: dict[str, FamilyView] = {}
    for family, record in snapshot.families.items():
        children = snapshot.children_of(family)
        receipts = {c.child: snapshot.receipt_for(c.child) for c in children if snapshot.receipt_for(c.child)}
        issues: list[str] = []
        if receipts and not record.choice:
            issues.append("有签收但最新家庭登记没有选校")
        for child, receipt in receipts.items():
            if record.choice and receipt.campus != record.choice:
                issues.append(
                    f"{child} 签收在 {receipt.campus}，更新后的家庭选校为 {record.choice}；保留原签收，不自行改校"
                )
        if receipts and len(receipts) < len(children):
            issues.append("同一家庭有兄弟姐妹未随签收，需一并确认")
        if record.choice and record.choice not in (record.temporary, record.permanent):
            issues.append("选校不在家庭登记的两个地址校区内")
        # Choose "receipt" only when every child carries a receipt for the
        # current entry year; partial receipts still need siblings added.
        if receipts and len(receipts) == len(children):
            status = "receipt"
        elif record.choice:
            status = "choice"
        else:
            status = "awaiting"
        anchor_campus = None
        if receipts and record.choice and any(
            receipt.campus != record.choice for receipt in receipts.values()
        ):
            # Keep the family on the signed campus; a change needs a new
            # receipt for every child, in order.
            signed = [receipt.campus for receipt in receipts.values()]
            anchor_campus = max(set(signed), key=signed.count)
        views[family] = FamilyView(
            family=family,
            phase=record.phase,
            temporary=record.temporary,
            permanent=record.permanent,
            choice=record.choice,
            children=children,
            receipts=receipts,
            status=status,
            issues=issues,
            anchor_campus=anchor_campus,
        )
    snapshot._family_views_cache = views
    return views


def phase_rate(snapshot: Snapshot, phase_id: str) -> tuple[float, list[str]]:
    """First enrolments per delivered home for the phase's campus pair.

    Campus and market cells are used when they are sufficient.  Otherwise the
    pooled campus figure is used and flagged, never a precise promise.
    """

    cache = getattr(snapshot, "_phase_rate_cache", None)
    if cache is None:
        cache = {}
        snapshot._phase_rate_cache = cache
    if phase_id in cache:
        return cache[phase_id]
    phase = snapshot.phase(phase_id)
    cells = campus_history(snapshot)
    values = []
    reasons = []
    for campus in phase.campuses:
        match = [c for c in cells if c.campus == campus and c.market == phase.market and c.ratio]
        if match and match[0].sufficient:
            values.append(match[0].ratio)
        else:
            pooled = [c for c in cells if c.campus == campus and c.ratio]
            if pooled:
                values.append(sum(c.first_enrolled for c in pooled) / sum(c.delivered_homes for c in pooled))
                reasons.append(f"{campus}/{phase.market} 样本不足，用校区合并比例")
    if not values:
        result = (0.0, ["无可用历史比例"])
    else:
        result = (sum(values) / len(values), reasons)
    cache[phase_id] = result
    return result


def phase_estimate(snapshot: Snapshot, phase_id: str) -> dict:
    phase = snapshot.phase(phase_id)
    rate, reasons = phase_rate(snapshot, phase_id)
    expected = phase.planned_homes * rate
    registered = [
        child
        for view in family_views(snapshot).values()
        if view.phase == phase_id
        for child in view.children
    ]
    return {
        "phase": phase_id,
        "planned_homes": phase.planned_homes,
        "rate": round(rate, 4),
        "expected_entrants": round(expected, 1),
        "registered_children": len(registered),
        "unregistered_forecast": round(max(0.0, expected - len(registered)), 1),
        "rate_basis": reasons,
    }


def branch_scenarios(snapshot: Snapshot, phase_id: str) -> list[Branch]:
    """Delivery branches for a phase's not-yet-observed delivery.

    Slip probabilities come from the phase's current progress stage.  A
    message that is already at ``分批`` or ``验房`` has historically matched the
    delivered year; a first ``施工`` message has slipped by one or two years.
    """

    uncertainty = stage_uncertainty(snapshot, phase_id)
    dist = uncertainty.get("distribution") or {}
    if not dist:
        # No progress at all: keep the plan year as one unresolved branch.
        return [
            Branch(
                phase=phase_id,
                label="无进度回报",
                delivery_year=snapshot.phase(phase_id).plan_year,
                weight=1.0,
                basis="没有任何进度消息，方向无法区分",
            )
        ]
    total = sum(dist.values())
    branches = []
    for slip, count in sorted(dist.items()):
        label = "按期" if slip == 0 else f"延后{slip}年"
        branches.append(
            Branch(
                phase=phase_id,
                label=label,
                delivery_year=uncertainty["expected_year"] + slip,
                weight=count / total,
                basis=f"{uncertainty['stage']} 阶段第 {uncertainty['ordinal'] + 1} 次回报，"
                f"历史 {count}/{total} 次滑移 {slip} 年",
            )
        )
    return branches


def branch_map(snapshot: Snapshot) -> dict[str, list[Branch]]:
    """Branch scenarios only for phases whose delivery is still open.

    A delivered phase uses its actual year and contributes no branches.
    Results are cached per snapshot because the branch set is immutable.
    """

    cached = getattr(snapshot, "_branch_map_cache", None)
    if cached is not None:
        return cached
    out: dict[str, list[Branch]] = {}
    for pid in snapshot.phase_order:
        outcome = phase_outcome(snapshot, pid)
        if outcome.first_delivery_year is not None:
            continue
        out[pid] = branch_scenarios(snapshot, pid)
    snapshot._branch_map_cache = out
    return out


def _branch_for_phase(branches: dict[str, list[Branch]], phase_id: str, label: str | None) -> list[Branch]:
    options = branches.get(phase_id)
    if not options:
        return [Branch(phase_id, "已交付", 0, 1.0, "已交付")]
    if label is None:
        return options
    return [b for b in options if b.label == label] or options


def _child_placements(
    snapshot: Snapshot,
    view: FamilyView,
    delivery_year: int,
    branch_label: str,
    years: Iterable[int],
) -> list[Placement]:
    out: list[Placement] = []
    for child in view.children:
        first_year = child.first_year
        for year in years:
            if year < first_year:
                continue
            grade = child.first_grade + (year - first_year)
            if grade > 6:
                continue
            receipt = view.receipts.get(child.child)
            if receipt and year >= receipt.year:
                # A receipt is annual.  Only its own school year is signed
                # quota; later years follow the same campus as intent until
                # that year's receipt is signed.
                signed = year == receipt.year
                tier = "receipt" if signed else "intent"
                basis = (
                    f"签收 {receipt.receipt}（{receipt.at.isoformat()}，{receipt.year} 学年）"
                    if signed
                    else f"{receipt.receipt} 签收校延续（{year} 学年未单独签认）"
                )
                out.append(
                    Placement(
                        campus=receipt.campus,
                        year=year,
                        grade=grade,
                        tier=tier,
                        basis=basis,
                        family=view.family,
                        child=child.child,
                        phase=view.phase,
                        branch=branch_label,
                        note=None if signed else "续读意向，尚非签认额度",
                    )
                )
                continue
            if view.anchor_campus:
                out.append(
                    Placement(
                        campus=view.anchor_campus,
                        year=year,
                        grade=grade,
                        tier="choice",
                        basis="同家庭已有签收校；未重签前兄弟姐妹不单独改校",
                        family=view.family,
                        child=child.child,
                        phase=view.phase,
                        branch=branch_label,
                        conditional=True,
                        note="等待同家庭重新签认",
                    )
                )
                continue
            if view.choice:
                out.append(
                    Placement(
                        campus=view.choice,
                        year=year,
                        grade=grade,
                        tier="choice",
                        basis="家庭已选校",
                        family=view.family,
                        child=child.child,
                        phase=view.phase,
                        branch=branch_label,
                    )
                )
                continue
            campus = view.temporary if year < delivery_year else view.permanent
            out.append(
                Placement(
                    campus=campus,
                    year=year,
                    grade=grade,
                    tier="awaiting",
                    basis=(
                        f"交付前暂住 {view.temporary}"
                        if year < delivery_year
                        else f"交付后迁入 {view.permanent}（{delivery_year} 年交付）"
                    ),
                    family=view.family,
                    child=child.child,
                    phase=view.phase,
                    branch=branch_label,
                    note=f"{branch_label}走向",
                )
            )
    return out


def _phase_forecasts(
    snapshot: Snapshot,
    branch_label: str | None,
    years: Iterable[int],
) -> list[Placement]:
    """Unregistered phase entrants as one group forecast per phase.

    The forecast is the part of the phase's expected first enrolment not
    covered by registered children.  Its first enrolment follows delivery by
    the observed lag (two years in 83% of observed phases, three in the rest).
    It has no child identity, so it cannot be promised to one campus per child;
    the campus split is a historical share and is flagged conditional.
    """

    out: list[Placement] = []
    views = family_views(snapshot)
    cells = campus_history(snapshot)
    branches = branch_map(snapshot)
    observed = {o.phase: o for o in all_outcomes(snapshot)}
    year_set = set(years)
    for phase_id in snapshot.phase_order:
        outcome = observed[phase_id]
        # Drive forecasts by observation status, not by plan-year recency.  A
        # 2025 phase whose first-enrolment return has not arrived still has
        # children expected in 2027; dropping it would silently treat a
        # missing result as zero, and would make the same phase appear or
        # disappear depending on the observation date.
        if outcome.observation == "已观察":
            continue
        phase = snapshot.phase(phase_id)
        estimate = phase_estimate(snapshot, phase_id)
        remaining = estimate["unregistered_forecast"]
        if remaining <= 0:
            continue
        options = branches.get(phase_id)
        if options:
            chosen = [b for b in options if b.label == branch_label] or [
                b for b in options if b.label == options[0].label
            ]
            if branch_label is not None and not [b for b in options if b.label == branch_label]:
                continue
            delivery_year = chosen[0].delivery_year
        elif outcome.first_delivery_year is not None:
            delivery_year = outcome.first_delivery_year
        else:
            continue
        enrol_year = delivery_year + 2
        if enrol_year not in year_set:
            continue
        rates = []
        for campus in phase.campuses:
            pooled = [c for c in cells if c.campus == campus and c.ratio]
            rates.append(
                sum(c.first_enrolled for c in pooled) / sum(c.delivered_homes for c in pooled)
                if pooled
                else 1.0
            )
        total_rate = sum(rates) or 1.0
        kids = [
            child
            for view in views.values()
            if view.phase == phase_id
            for child in view.children
        ]
        grade_mix = Counter(c.first_grade for c in kids) or Counter({1: 1})
        mix_total = sum(grade_mix.values())
        for campus, rate in zip(phase.campuses, rates):
            share = remaining * rate / total_rate
            for grade, count in grade_mix.items():
                portion = share * count / mix_total
                if portion < 0.05:
                    continue
                kind = (
                    "unreported"
                    if outcome.first_delivery_year is not None
                    else "unbuilt"
                )
        reason = (
            "已交付但首次入学结果未回传，不能按零处理"
            if kind == "unreported"
            else "分期尚未交付"
        )
        out.append(
                    Placement(
                        campus=campus,
                        year=enrol_year,
                        grade=grade,
                        tier="forecast",
                        count=round(portion, 2),
                        basis=(
                            f"{phase_id}（{reason}）预计首次入学 "
                            f"{estimate['expected_entrants']} 人，"
                            f"登记儿童已占 {estimate['registered_children']} 人，余 {remaining:.1f} 人；"
                            f"按校区历史份额 {rate / total_rate:.0%} 拆分，预计 {enrol_year} 年入学"
                        ),
                        phase=phase_id,
                        branch=branch_label,
                        conditional=True,
                        source_kind=kind,
                    )
                )
    return out


def continuing_placements(snapshot: Snapshot, years: Iterable[int]) -> list[Placement]:
    """Existing pupils advanced by grade, less confirmed leavers.

    A confirmed leaver leaves after the recorded school year, so the reduction
    applies on the transition out of that grade, not to the recorded year.
    """

    base = {(row.campus, row.grade): row for row in snapshot.continuing}
    out: list[Placement] = []
    for campus in snapshot.campuses:
        for year in years:
            for grade in GRADES:
                source_grade = grade - (year - 2026)
                if source_grade < 1 or source_grade > 6:
                    continue
                row = base.get((campus, source_grade))
                if not row:
                    continue
                count = row.count - row.confirmed_leavers
                if year == 2026:
                    count = row.count
                if count <= 0:
                    continue
                out.append(
                    Placement(
                        campus=campus,
                        year=year,
                        grade=grade,
                        tier="continuing",
                        count=count,
                        basis=(
                            f"2026 学年 {campus} {source_grade} 年级 {row.count} 人"
                            + (f"，确认离校 {row.confirmed_leavers} 人" if row.confirmed_leavers else "")
                        ),
                    )
                )
    return out


@dataclass
class DemandSet:
    branch: str
    placements: list[Placement]

    def by_unit(self) -> dict[tuple[str, int, str], list[Placement]]:
        out: dict[tuple[str, int, str], list[Placement]] = defaultdict(list)
        for p in self.placements:
            out[(p.campus, p.year, p.band)].append(p)
        return out

    def by_grade(self) -> dict[tuple[str, int, int], list[Placement]]:
        out: dict[tuple[str, int, int], list[Placement]] = defaultdict(list)
        for p in self.placements:
            out[(p.campus, p.year, p.grade)].append(p)
        return out

    def by_tier(self) -> dict[tuple[str, int, str, str], float]:
        out: dict[tuple[str, int, str, str], float] = defaultdict(float)
        for p in self.placements:
            out[(p.campus, p.year, p.band, p.tier)] += p.count
        return out

    def total(self, **filters) -> float:
        rows = self.placements
        for name, value in filters.items():
            rows = [p for p in rows if getattr(p, name) == value]
        return sum(p.count for p in rows)


def build_demand(
    snapshot: Snapshot,
    branch_label: str | None = None,
    years: Iterable[int] | None = None,
    extra_receipts: Iterable[tuple[str, str, int]] = (),
) -> DemandSet:
    """Assemble every placement for one delivery branch.

    ``extra_receipts`` lets a what-if confirmation be tested without changing
    the recorded data.  Each tuple is ``(child, campus, year)``.
    """

    years = list(years or range(snapshot.as_of.year, snapshot.resource_years()[1] + 1))
    views = family_views(snapshot)
    branches = branch_map(snapshot)
    placements: list[Placement] = list(continuing_placements(snapshot, years))

    extra = {}
    for child, campus, year in extra_receipts:
        extra.setdefault(child, []).append((campus, year))

    # Every phase routes under every branch once, so per-campus demand can be
    # compared across branches.  Families in delivered phases route to their
    # recorded delivery year.
    for phase_id in snapshot.phase_order:
        options = _branch_for_phase(branches, phase_id, branch_label)
        if not options:
            continue
        if not options or options[0].delivery_year == 0:
            outcome = phase_outcome(snapshot, phase_id)
            branch = Branch(
                phase_id,
                "已交付",
                outcome.first_delivery_year or 0,
                1.0,
                "已交付，不再分走向",
            )
        else:
            branch = options[0]
        members = [v for v in views.values() if v.phase == phase_id]
        if branch.delivery_year:
            for view in members:
                placements.extend(
                    _child_placements(snapshot, view, branch.delivery_year, branch.label, years)
                )


    # Unregistered phase entrants are one group forecast per phase; they are
    # not built per phase inside the family loop above.  Only phases whose
    # delivery in this branch has a known year contribute.
    placements.extend(_phase_forecasts(snapshot, branch_label, years))

    # What-if receipts: replace the child's family routing with the receipt
    # campus, and carry siblings on the same campus for the family to stay
    # together.  The original recorded receipt is never overridden.
    if extra:
        by_child = {p.child: p for p in placements if p.child}
        for child, rows in extra.items():
            applicant = snapshot.child_applicant.get(child)
            if not applicant:
                continue
            family = views[applicant.family]
            siblings = {c.child for c in family.children}
            campus, start = max(rows, key=lambda x: x[1])
            for placement in list(placements):
                if placement.child in siblings and placement.year >= start:
                    placements.remove(placement)
            for sib in family.children:
                for year in years:
                    if year < max(start, sib.first_year):
                        continue
                    grade = sib.first_grade + (year - sib.first_year)
                    if grade > 6:
                        continue
                    placements.append(
                        Placement(
                            campus=campus,
                            year=year,
                            grade=grade,
                            tier="receipt" if year == start else "intent",
                            basis=(
                                f"拟签 {child} 带动同家庭兄弟姐妹"
                                if year == start
                                else f"拟签校 {campus} 续读意向"
                            ),
                            family=family.family,
                            child=sib.child,
                            phase=family.phase,
                            branch=branch_label,
                            note=None if year == start else "后续学年待单独签认",
                        )
                    )

    return DemandSet(branch=branch_label or "合并", placements=placements)


def demand_summary(demand: DemandSet) -> dict:
    """Compact campus/year/band/tier totals for exports."""

    tier = demand.by_tier()
    out: dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    for (campus, year, band, t), count in tier.items():
        out[str(year)][campus][f"{band}:{t}"] = round(count, 2)
    return {year: dict(campuses) for year, campuses in out.items()}


def consistency_issues(snapshot: Snapshot) -> list[dict]:
    """Data and business-rule conflicts a caseworker must resolve."""

    issues: list[dict] = []
    views = family_views(snapshot)
    for view in views.values():
        if view.issues:
            issues.append(
                {
                    "kind": "家庭登记与签收",
                    "family": view.family,
                    "phase": view.phase,
                    "detail": "；".join(view.issues),
                    "action": "保留既有签收；兄弟姐妹一起改址前需重新签认",
                }
            )
        phase = snapshot.phase(view.phase)
        if {view.temporary, view.permanent} != set(phase.campuses):
            issues.append(
                {
                    "kind": "地址校区与分期不符",
                    "family": view.family,
                    "phase": view.phase,
                    "detail": f"家庭地址 {view.temporary}/{view.permanent}，分期校区 {phase.campuses}",
                    "action": "按分期校区核对，不按家庭自报跨区承诺",
                }
            )

    # Family record updates are retained; flag cases where a later update
    # would move a sibling away from an existing receipt.
    for family, rows in snapshot.family_history.items():
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]
        if (first.temporary, first.choice) != (last.temporary, last.choice):
            issues.append(
                {
                    "kind": "分期更正/家庭改址",
                    "family": family,
                    "phase": last.phase,
                    "detail": (
                        f"{first.received.isoformat()} 暂住 {first.temporary} 选校 {first.choice} → "
                        f"{last.received.isoformat()} 暂住 {last.temporary} 选校 {last.choice}"
                    ),
                    "action": "既有签收保留；同家庭兄弟姐妹需一起切换并重新签认",
                }
            )

    # Registered children already occupy the phase estimate rather than being
    # added on top.  If registrations exceed the estimate, that is a signal,
    # not a reason to inflate demand.
    for phase_id in snapshot.phase_order:
        estimate = phase_estimate(snapshot, phase_id)
        if estimate["registered_children"] > estimate["expected_entrants"] and estimate["registered_children"]:
            issues.append(
                {
                    "kind": "登记数高于分期预估",
                    "phase": phase_id,
                    "detail": (
                        f"已登记 {estimate['registered_children']} 人，历史预估 "
                        f"{estimate['expected_entrants']} 人；登记儿童已占分期预计人数"
                    ),
                    "action": "按登记数作为下限，不叠加分期预估",
                }
            )

    # Parcels delivered but missing the later measure are unknown, not zero.
    for pid in snapshot.phase_order:
        outcome = phase_outcome(snapshot, pid)
        if outcome.observation in ("入住未回传", "入学未回传", "未到观察期"):
            issues.append(
                {
                    "kind": "历史结果缺失",
                    "phase": pid,
                    "detail": f"交付 {outcome.delivered_homes}/{outcome.planned_homes}，状态 {outcome.observation}",
                    "action": "不按零处理；按已交付分母或等待结果回传",
                }
            )
    return issues
