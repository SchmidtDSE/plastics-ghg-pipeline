"""Utilities for normalizing subtype to overall net trade ratios.

License: BSD
"""
import functools
import typing

import const
import data_struct

OBSERVATION_MAYBE = typing.Optional[data_struct.Observation]
OBSERVATIONS = typing.Iterable[data_struct.Observation]
OBSERVATIONS_MAYBE = typing.Iterable[typing.Optional[data_struct.Observation]]


class RatioReduceRecord:
    """Internal data structure for reducing across potentiall incomplete subtype ratios."""

    def __init__(self, invalid: float, valid: float):
        """Create a new record for reduction.

        Args:
            invalid: Number of records that were incomplete.
            valid: The summation from valid records.
        """
        self._invalid = invalid
        self._valid = valid

    def get_invalid(self) -> float:
        """Get the count of records that did not have ratios available.

        Returns:
            Count of invalid records.
        """
        return self._invalid

    def get_valid(self) -> float:
        """Get the running ratio sum from valid records.

        Returns:
            The sum of ratios from valid records.
        """
        return self._valid

    def combine(self, other: 'RatioReduceRecord') -> 'RatioReduceRecord':
        """Add two reduction records together.

        Args:
            other: The record to add to this one.

        Returns:
            Record that describes the sum of this record and the other record.
        """
        return RatioReduceRecord(
            self.get_invalid() + other.get_invalid(),
            self.get_valid() + other.get_valid()
        )


class NormalizingIndexedObservationsDecorator(data_struct.IndexedObservations):
    """Decorator for an IndexedObservations structure which normalizes records as they return."""

    def __init__(self, inner: data_struct.IndexedObservations):
        """Create a new decorator.

        Args:
            inner: The indexed observations structure to decorate.
        """
        self._inner = inner

    def get_record(self, year: int, region: str, subtype: str) -> OBSERVATION_MAYBE:
        """Normalize a record when returning it.

        Args:
            year: The year like 2024 for which a record is desired.
            region: The region like "NAFTA" for which a record is desired.
            subtype: The subtype like "Transportation" for which a record is desired.

        Returns:
            Record with normalized ratio or None if an original record was not present or did not
            have a ratio itself.
        """
        # Ensure there is a record which can be normalized.
        unnormalized = self._inner.get_record(year, region, subtype)
        if unnormalized is None:
            return None

        unnormalized_ratio = unnormalized.get_ratio()
        if unnormalized_ratio is None:
            return None

        # Only normalize within the same series (goods or resin) as modeling predicts net goods and
        # resin separately.
        is_goods = subtype in const.SECTORS
        target_subtypes = const.SECTORS if is_goods else const.POLYMERS

        # Sum up the other ratios within the same series.
        all_subtypes_maybe = map(lambda x: self._inner.get_record(year, region, x), target_subtypes)
        all_subtypes = self._filter_for_valid(all_subtypes_maybe)
        sum_ratios = self._get_sum_ratios(all_subtypes)

        # Return normalized value
        return data_struct.Observation(
            unnormalized_ratio / sum_ratios,
            unnormalized.get_gdp(),
            unnormalized.get_population()
        )

    def get_change(self, year: int, region: str, subtype: str,
        years: int) -> typing.Optional[data_struct.Change]:
        return self._inner.get_change(year, region, subtype, years)

    def add(self, year: int, region: str, subtype: str, record: data_struct.Observation):
        self._inner.add(year, region, subtype, record)

    def get_years(self) -> typing.Iterable[int]:
        return self._inner.get_years()

    def has_year(self, target: int) -> bool:
        return self._inner.has_year(target)

    def get_regions(self) -> typing.Iterable[str]:
        return self._inner.get_regions()

    def has_region(self, target: str) -> bool:
        return self._inner.has_region(target)

    def get_subtypes(self) -> typing.Iterable[str]:
        return self._inner.get_subtypes()

    def has_subtype(self, target: str) -> bool:
        return self._inner.has_subtype(target)

    def _filter_for_valid(self, target: OBSERVATIONS_MAYBE) -> OBSERVATIONS:
        """Filter for only found observations.

        Args:
            target: The records to check.

        Returns:
            The target iterator but with invalid or missing records removed.
        """
        return filter(lambda x: x is not None, target)  # type: ignore

    def _get_sum_ratios(self, target: OBSERVATIONS) -> float:
        """Get the sum of all ratios in a collection of observations.

        Args:
            target: The observations whose ratios should be summed, ignoring those without ratios.

        Returns:
            The sum of ratios found in target.
        """
        all_ratios_maybe = map(lambda x: x.get_ratio(), target)
        all_ratios_described = map(lambda x: RatioReduceRecord(
            1 if x is None else 0,
            0 if x is None else x
        ), all_ratios_maybe)
        summed = functools.reduce(
            lambda a, b: a.combine(b),
            all_ratios_described
        )

        if summed.get_invalid() > 0:
            raise RuntimeError('Enountered incomplete year / region data.')

        return summed.get_valid()
