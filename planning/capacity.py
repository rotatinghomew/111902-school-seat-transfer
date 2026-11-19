"""Seat supply and loan application.

A seat is real only when a teacher and a classroom are available in the same
campus, school year, grade band and time block.  The calendar stores the two
kinds separately, so ``min(teachers, classrooms)`` is the usable figure for a
block; band supply is the sum of its blocks because a class occupies one block.

Loans move a *named* resource pair.  A loan counts only after both sides sign
(``双方签认``).  A one-sided approval or a bare intent is never capacity.  A
loan also stops being capacity after its ``last`` year; it cannot be used as
opening capacity for a cohort that needs the following year.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .data import BANDS, BLOCKS, LoanOffer, Snapshot


@dataclass(frozen=True)
class Seat:
    campus: str
    year: int
    block: str
    band: str
    teachers: int
    classrooms: int

    @property
    def usable(self) -> int:
        return max(0, min(self.teachers, self.classrooms))


@dataclass(frozen=True)
class LoanUse:
    """A loan treated as already applied to the supply picture."""

    offer: LoanOffer
    years: tuple[int, ...]
    signed: bool
    state: str

    @property
    def expires_after(self) -> int:
        return self.offer.last

    @property
    def usable_years(self) -> tuple[int, ...]:
        return self.years if self.signed else ()


class Supply:
    """Supply model for one snapshot, optionally with proposed loans applied."""

    def __init__(
        self,
        snapshot: Snapshot,
        extra_signed: tuple[str, ...] = (),
        include_signed: bool = True,
    ):
        self.snapshot = snapshot
        self._seat_cache: dict[tuple[str, int, str, str], Seat] = {}
        self.base: dict[tuple[str, int, str, str], dict[str, int]] = defaultdict(
            lambda: {"教师": 0, "教室": 0}
        )
        for row in snapshot.resources:
            if not row.enabled:
                continue
            self.base[(row.home, row.year, row.block, row.grade_band)][row.kind] += row.capacity

        self.signed_offers: list[str] = []
        if include_signed:
            self.signed_offers = [
                offer_id
                for offer_id in snapshot.offers
                if snapshot.is_signed(offer_id)
            ]
        for offer_id in extra_signed:
            if offer_id not in self.signed_offers:
                self.signed_offers.append(offer_id)

        # Validate that offered resources actually exist for the promised span.
        self.loan_warnings: list[dict] = []
        self._loans: list[LoanUse] = []
        for offer_id in self.signed_offers:
            offer = snapshot.offers[offer_id]
            years = tuple(range(offer.first, offer.last + 1))
            missing = [
                name
                for name in offer.resources
                if not any(
                    row.year in years and row.enabled for row in snapshot.resource_rows(name)
                )
            ]
            if missing:
                self.loan_warnings.append(
                    {
                        "offer": offer_id,
                        "kind": "资源缺失或停用",
                        "resources": missing,
                    }
                )
            self._loans.append(
                LoanUse(
                    offer=offer,
                    years=years,
                    signed=True,
                    state=snapshot.loan_state(offer_id),
                )
            )

    # ------------------------------------------------------------------
    def _resource_seat(self, name: str, year: int, block: str, band: str) -> int:
        total = 0
        for row in self.snapshot.resource_rows(name):
            if row.year == year and row.block == block and row.grade_band == band and row.enabled:
                total += row.capacity
        return total

    def seat(self, campus: str, year: int, block: str, band: str) -> Seat:
        key = (campus, year, block, band)
        cached = self._seat_cache.get(key)
        if cached is not None:
            return cached
        teachers = self.base[(campus, year, block, band)]["教师"]
        classrooms = self.base[(campus, year, block, band)]["教室"]
        for loan in self._loans:
            if year not in loan.years:
                continue
            for name in loan.offer.resources:
                moved = self._resource_seat(name, year, block, band)
                if moved == 0:
                    continue
                if loan.offer.source == campus:
                    if name.split("-")[2] == "教师":
                        teachers -= moved
                    else:
                        classrooms -= moved
                elif loan.offer.destination == campus:
                    if name.split("-")[2] == "教师":
                        teachers += moved
                    else:
                        classrooms += moved
        seat = Seat(
            campus=campus,
            year=year,
            block=block,
            band=band,
            teachers=max(0, teachers),
            classrooms=max(0, classrooms),
        )
        self._seat_cache[key] = seat
        return seat

    def band_seats(self, campus: str, year: int, band: str) -> int:
        return sum(self.seat(campus, year, block, band).usable for block in BLOCKS)

    def block_seats(self, campus: str, year: int, band: str) -> dict[str, int]:
        return {block: self.seat(campus, year, block, band).usable for block in BLOCKS}

    def durable_band_seats(self, campus: str, year: int, band: str) -> int:
        """Seats available without any loan that begins this year.

        Used to tell durable capacity from opening capacity that must be
        returned.  A signed loan still counts for its own years, but not for
        the next year once it has expired.
        """

        durable = Supply(self.snapshot, include_signed=False)
        return durable.band_seats(campus, year, band)

    def source_after_transfer(self, offer_id: str) -> list[dict]:
        """Check the sending campus keeps the same teacher/classroom rule.

        The caller compares the returned residual seats with that campus's
        demand; this function only reports the figures, not a verdict.
        """

        offer = self.snapshot.offers[offer_id]
        rows = []
        for year in range(offer.first, offer.last + 1):
            for band in BANDS:
                rows.append(
                    {
                        "offer": offer_id,
                        "source": offer.source,
                        "year": year,
                        "band": band,
                        "seats_after": self.band_seats(offer.source, year, band),
                        "durable_after": self.durable_band_seats(offer.source, year, band),
                    }
                )
        return rows

    def loan_summary(self) -> list[dict]:
        out = []
        for loan in self._loans:
            offer = loan.offer
            out.append(
                {
                    "offer": offer.offer,
                    "source": offer.source,
                    "destination": offer.destination,
                    "band": offer.band,
                    "block": self._offer_block(offer),
                    "first": offer.first,
                    "last": offer.last,
                    "lock_by": offer.lock_by.isoformat(),
                    "state": loan.state,
                    "seats": self._offer_seats(offer),
                    "expires_after": loan.expires_after,
                }
            )
        return sorted(out, key=lambda row: row["offer"])

    def _offer_block(self, offer: LoanOffer) -> str:
        for name in offer.resources:
            for row in self.snapshot.resource_rows(name):
                if row.year == offer.first:
                    return row.block
        return offer.block

    def _offer_seats(self, offer: LoanOffer) -> int:
        # The pair is one teacher plus one classroom; the smaller cap binds.
        caps: dict[str, int] = {}
        for name in offer.resources:
            kind = name.split("-")[2]
            caps[kind] = max(
                (r.capacity for r in self.snapshot.resource_rows(name) if r.enabled),
                default=0,
            )
        return min(caps.values()) if caps else 0


