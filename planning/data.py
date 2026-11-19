"""Load the synthetic planning tables and build date-aware snapshots.

The repository is a record of what different offices reported and when.  A
planning answer must therefore be reproducible for a chosen observation date:
a confirmation signed in February cannot rely on a progress message received in
March, and a family update received in March cannot rewrite a January decision.

Only the standard library is used.  All dates are ISO strings in the source and
``datetime.date`` objects in this module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "supplies.json"

CAMPUSES = ("西桥", "青池", "林园", "河口", "新港")
BANDS = ("低年段", "高年段")
BLOCKS = ("上午", "下午")
GRADES = tuple(range(1, 7))


def band_of(grade: int) -> str:
    """Return the resource band used by the calendar for a primary grade."""

    return "低年段" if grade <= 3 else "高年段"


def d(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _max_date(values: Iterable[str | date | None]) -> date | None:
    dates = [d(v) for v in values if v]
    return max(dates) if dates else None


@dataclass(frozen=True)
class Phase:
    phase: str
    plan_year: int
    planned_homes: int
    market: str
    campuses: tuple[str, ...]

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Phase":
        return cls(
            phase=row["phase"],
            plan_year=row["plan_year"],
            planned_homes=row["planned_homes"],
            market=row["market"],
            campuses=tuple(row["campuses"]),
        )


@dataclass(frozen=True)
class ProgressRecord:
    phase: str
    progress_id: str
    received: date
    expected_year: int
    milestone: str
    replaces: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ProgressRecord":
        return cls(
            phase=row["phase"],
            progress_id=row["progress_id"],
            received=d(row["received"]),  # type: ignore[arg-type]
            expected_year=row["expected_year"],
            milestone=row["milestone"],
            replaces=row["replaces"],
        )


@dataclass(frozen=True)
class HistoryRecord:
    phase: str
    parcel: int
    measure: str
    effective_year: int
    received: date
    count: int

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "HistoryRecord":
        return cls(
            phase=row["phase"],
            parcel=row["parcel"],
            measure=row["measure"],
            effective_year=row["effective_year"],
            received=d(row["received"]),  # type: ignore[arg-type]
            count=row["count"],
        )


@dataclass(frozen=True)
class FamilyRecord:
    family: str
    phase: str
    received: date
    temporary: str
    permanent: str
    choice: str | None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "FamilyRecord":
        return cls(
            family=row["family"],
            phase=row["phase"],
            received=d(row["received"]),  # type: ignore[arg-type]
            temporary=row["temporary"],
            permanent=row["permanent"],
            choice=row["choice"],
        )


@dataclass(frozen=True)
class Applicant:
    family: str
    child: str
    first_grade: int
    first_year: int

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Applicant":
        return cls(
            family=row["family"],
            child=row["child"],
            first_grade=row["first_grade"],
            first_year=row["first_year"],
        )


@dataclass(frozen=True)
class Continuing:
    campus: str
    grade: int
    year: int
    count: int
    confirmed_leavers: int

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Continuing":
        return cls(
            campus=row["campus"],
            grade=row["grade"],
            year=row["year"],
            count=row["count"],
            confirmed_leavers=row["confirmed_leavers"],
        )

    @property
    def continuing_count(self) -> int:
        return max(0, self.count - self.confirmed_leavers)


@dataclass(frozen=True)
class Resource:
    resource: str
    home: str
    year: int
    block: str
    grade_band: str
    kind: str
    capacity: int
    enabled: bool

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Resource":
        return cls(
            resource=row["resource"],
            home=row["home"],
            year=row["year"],
            block=row["block"],
            grade_band=row["grade_band"],
            kind=row["kind"],
            capacity=row["capacity"],
            enabled=row["enabled"],
        )


@dataclass(frozen=True)
class LoanOffer:
    offer: str
    resources: tuple[str, ...]
    destination: str
    first: int
    last: int
    lock_by: date

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "LoanOffer":
        return cls(
            offer=row["offer"],
            resources=tuple(row["resources"]),
            destination=row["destination"],
            first=row["first"],
            last=row["last"],
            lock_by=d(row["lock_by"]),  # type: ignore[arg-type]
        )

    @property
    def source(self) -> str:
        """The source campus encoded in the resource identifiers."""

        return self.resources[0].split("-")[0]

    @property
    def band(self) -> str:
        return self.resources[0].split("-")[1]

    @property
    def block(self) -> str:
        # The numbered resource pair is threaded through one block.  The
        # calendar is the authority; this fallback is only used in reports.
        return "上午"


@dataclass(frozen=True)
class LoanEvent:
    offer: str
    state: str
    received: date

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "LoanEvent":
        return cls(
            offer=row["offer"],
            state=row["state"],
            received=d(row["received"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class Confirmation:
    child: str
    campus: str
    year: int
    at: date
    receipt: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Confirmation":
        return cls(
            child=row["child"],
            campus=row["campus"],
            year=row["year"],
            at=d(row["at"]),  # type: ignore[arg-type]
            receipt=row["receipt"],
        )


class Snapshot:
    """Immutable view of all tables at one observation date.

    ``as_of=None`` means the data cut: the latest received date across event
    tables.  Passing an explicit date is how a caseworker reconstructs what was
    knowable when a receipt was signed.
    """

    def __init__(self, tables: Mapping[str, list[dict[str, Any]]], as_of: str | date | None = None):
        self.tables = tables
        self.data_cut = _max_date(
            row.get(field)
            for table, fields in (
                ("progress", ("received",)),
                ("history", ("received",)),
                ("families", ("received",)),
                ("loan_events", ("received",)),
                ("confirmations", ("at",)),
            )
            for row in tables.get(table, [])
            for field in fields
        )
        self.as_of = d(as_of) if as_of else self.data_cut
        assert self.as_of is not None

        self.phases: dict[str, Phase] = {
            row["phase"]: Phase.from_row(row) for row in tables.get("phases", [])
        }
        self.phase_order = [row["phase"] for row in tables.get("phases", [])]
        self.campuses = tuple(
            dict.fromkeys(
                campus for row in tables.get("phases", []) for campus in row["campuses"]
            )
        ) or CAMPUSES

        self.progress_all = [ProgressRecord.from_row(r) for r in tables.get("progress", [])]
        self.history_all = [HistoryRecord.from_row(r) for r in tables.get("history", [])]
        self.families_all = [FamilyRecord.from_row(r) for r in tables.get("families", [])]
        self.applicants = [Applicant.from_row(r) for r in tables.get("applicants", [])]
        self.continuing = [Continuing.from_row(r) for r in tables.get("continuing", [])]
        self.resources = [Resource.from_row(r) for r in tables.get("resource_calendar", [])]
        self.offers = {
            row["offer"]: LoanOffer.from_row(row) for row in tables.get("loan_offers", [])
        }
        self.loan_events_all = [LoanEvent.from_row(r) for r in tables.get("loan_events", [])]
        self.confirmations_all = [
            Confirmation.from_row(r) for r in tables.get("confirmations", [])
        ]

        self.progress = [r for r in self.progress_all if r.received <= self.as_of]
        self.history = [r for r in self.history_all if r.received <= self.as_of]
        self.family_rows = [r for r in self.families_all if r.received <= self.as_of]
        self.loan_events = [r for r in self.loan_events_all if r.received <= self.as_of]
        self.confirmations = [r for r in self.confirmations_all if r.at <= self.as_of]

        # Latest family statement per family.  A later statement supersedes an
        # earlier one, but both remain available for renegotiation reports.
        self.family_history: dict[str, list[FamilyRecord]] = defaultdict(list)
        for row in sorted(self.family_rows, key=lambda r: (r.received, self.families_all.index(r))):
            self.family_history[row.family].append(row)
        self.families: dict[str, FamilyRecord] = {
            family: rows[-1] for family, rows in self.family_history.items()
        }

        self.applicants_by_family: dict[str, list[Applicant]] = defaultdict(list)
        self.child_applicant: dict[str, Applicant] = {}
        for applicant in self.applicants:
            self.applicants_by_family[applicant.family].append(applicant)
            self.child_applicant[applicant.child] = applicant
        for children in self.applicants_by_family.values():
            children.sort(key=lambda a: (a.first_year, a.first_grade, a.child))

        # Progress is retained per phase in received order.  ``replaces`` is
        # honoured when selecting the current message, but the replaced record
        # stays in the rolling history.
        self.progress_by_phase: dict[str, list[ProgressRecord]] = defaultdict(list)
        for row in sorted(self.progress, key=lambda r: (r.received, r.progress_id)):
            self.progress_by_phase[row.phase].append(row)

        self.history_by_phase: dict[str, list[HistoryRecord]] = defaultdict(list)
        for row in sorted(self.history, key=lambda r: (r.effective_year, r.parcel, r.measure)):
            self.history_by_phase[row.phase].append(row)

        self.replaced_ids = {
            r.replaces for r in self.progress if r.replaces is not None
        }

        self.confirmations_by_child: dict[str, list[Confirmation]] = defaultdict(list)
        for row in sorted(self.confirmations, key=lambda r: (r.at, r.receipt)):
            self.confirmations_by_child[row.child].append(row)

        self.loan_events_by_offer: dict[str, list[LoanEvent]] = defaultdict(list)
        for row in sorted(self.loan_events, key=lambda r: (r.received, r.state)):
            self.loan_events_by_offer[row.offer].append(row)

        # Lazily filled by the planning layer.  These depend only on this
        # snapshot's data and observation date, so they are safe to keep here
        # instead of in module-level caches keyed by object identity.
        self._family_views_cache: dict | None = None
        self._phase_rate_cache: dict | None = None
        self._branch_map_cache: dict | None = None

    # ------------------------------------------------------------------
    # Convenience lookups
    # ------------------------------------------------------------------
    def phase(self, phase_id: str) -> Phase:
        return self.phases[phase_id]

    def children_of(self, family: str) -> list[Applicant]:
        return list(self.applicants_by_family.get(family, []))

    def family_for_child(self, child: str) -> FamilyRecord | None:
        applicant = self.child_applicant.get(child)
        return self.families.get(applicant.family) if applicant else None


    def current_progress(self, phase_id: str) -> ProgressRecord | None:
        rows = self.progress_by_phase.get(phase_id, [])
        if not rows:
            return None
        # A message with ``replaces`` supersedes the older one even if both
        # were received on the same day; otherwise the latest received wins.
        replaced = {r.replaces for r in rows if r.replaces}
        candidates = [r for r in rows if r.progress_id not in replaced]
        return max(candidates or rows, key=lambda r: (r.received, r.progress_id))

    def history_as_of(self, phase_id: str, at: date) -> list[HistoryRecord]:
        return [
            r
            for r in self.history_all
            if r.phase == phase_id and r.received <= at
        ]

    def loan_state(self, offer_id: str) -> str:
        """Return one of ``双方签认``, ``调出认可``, ``撤回意向``, ``意向``.

        The latest event is authoritative.  A withdrawal is terminal for this
        snapshot; a fresh offer would need a new identifier.
        """

        events = self.loan_events_by_offer.get(offer_id, [])
        if not events:
            return "意向"
        return events[-1].state

    def is_signed(self, offer_id: str) -> bool:
        return self.loan_state(offer_id) == "双方签认"

    def is_withdrawn(self, offer_id: str) -> bool:
        return self.loan_state(offer_id) == "撤回意向"

    def receipt_for(self, child: str) -> Confirmation | None:
        rows = self.confirmations_by_child.get(child, [])
        return rows[-1] if rows else None

    def cadence_template(self) -> list[tuple[int, int, int]]:
        """Modal (year offset, month, day) of the first three progress reports.

        Derived from the data rather than hard-coded: each phase's first three
        messages are counted and the most common calendar slot wins.  If the
        data were to change cadence, the promise dates would follow it.
        """

        from collections import Counter

        slots: Counter = Counter()
        for phase_id, rows in self.progress_by_phase.items():
            year = self.phase(phase_id).plan_year
            for ordinal, row in enumerate(sorted(rows, key=lambda r: r.received)[:3]):
                slots[(ordinal, row.received.month, row.received.day)] += 1
        template = []
        for ordinal in range(3):
            candidates = [k for k in slots if k[0] == ordinal]
            if not candidates:
                continue
            _, month, day = max(candidates, key=lambda k: slots[k])
            template.append((ordinal, month, day))
        return template or [(0, 10, 10), (1, 2, 9), (2, 6, 20)]

    def progress_cadence(self, phase_id: str) -> list[date]:
        """Scheduled progress reporting dates for the phase.

        A historical phase uses the calendar offsets actually observed.  A
        future phase uses the same offsets projected from its plan year, which
        is the only reporting promise the district can rely on.
        """

        year = self.phase(phase_id).plan_year
        offsets = {0: -1, 1: 0, 2: 0}
        return [
            date(year + offsets.get(ordinal, 0), month, day)
            for ordinal, month, day in self.cadence_template()
        ]

    def next_scheduled_progress(self, phase_id: str) -> date | None:
        cadence = self.progress_cadence(phase_id)
        future = [x for x in cadence if x > self.as_of]
        return min(future) if future else None

    def overdue_progress(self, phase_id: str) -> date | None:
        """Latest scheduled reporting date that passed without a message.

        A missing return is not a zero result; it is reported as overdue and
        the phase stays unresolved.
        """

        if any(
            r.measure == "交付" for r in self.history_by_phase.get(phase_id, [])
        ):
            return None
        received_dates = sorted(
            r.received for r in self.progress_by_phase.get(phase_id, [])
        )
        # Each scheduled slot is satisfied by a message received in the window
        # from that slot until the next one.  A slot with no message is
        # overdue: the return is missing, which is not the same as zero.
        cadence = self.progress_cadence(phase_id)
        for index, slot in enumerate(cadence):
            if slot >= self.as_of:
                continue
            window_end = cadence[index + 1] if index + 1 < len(cadence) else self.as_of
            if not any(slot <= received < window_end for received in received_dates):
                return slot
        return None

    def resource_years(self) -> tuple[int, int]:
        years = [r.year for r in self.resources]
        return min(years), max(years)

    def resource_rows(self, name: str) -> list[Resource]:
        return [r for r in self.resources if r.resource == name]


def load_snapshot(
    path: str | Path = DEFAULT_DATA, as_of: str | date | None = None
) -> Snapshot:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Snapshot(payload["tables"], as_of=as_of)


