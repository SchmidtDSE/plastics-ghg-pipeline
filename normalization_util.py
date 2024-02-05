import typing

import const
import data_struct

OBSERVATION_MAYBE = typing.Optional[data_struct.Observation]
OBSERVATIONS = typing.Iterable[data_struct.Observation]
OBSERVATIONS_MAYBE = typing.Iterable[typing.Optional[data_struct.Observation]]


class NormalizingIndexedObservationsDecorator(data_struct.IndexedObservations):
    """Decorator for an IndexedObservations structure which normalizes records as they return."""

    def __init__(self, inner: data_struct.IndexedObservations):
        """Create a new decorator.

        Args:
            inner: The indexed observations structure to decorate.
        """
        self._inner = inner

    def get_record(self, year: int, region: str, sector: str) -> OBSERVATION_MAYBE:
        unnormalized = self._inner.get_record(year, region, sector)
        if unnormalized is None:
            return None

        unnormalized_ratio = unnormalized.get_ratio()
        if unnormalized_ratio is None:
            return None

        all_sectors_maybe = map(lambda x: self._inner.get_record(year, region, x), const.SECTORS)
        all_sectors = self._filter_for_valid(all_sectors_maybe)
        sum_ratios = self._get_sum_ratios(all_sectors)

        return data_struct.Observation(
            unnormalized_ratio / sum_ratios,
            unnormalized.get_gdp(),
            unnormalized.get_population()
        )

    def get_change(self, year: int, region: str, sector: str,
        years: int) -> typing.Optional[data_struct.Change]:
        return self._inner.get_change(year, region, sector, years)

    def add(self, year: int, region: str, sector: str, record: data_struct.Observation):
        self._inner.add(year, region, sector, record)

    def get_years(self) -> typing.Iterable[int]:
        return self._inner.get_years()

    def has_year(self, target: int) -> bool:
        return self._inner.has_year(target)

    def get_regions(self) -> typing.Iterable[str]:
        return self._inner.get_regions()

    def has_region(self, target: str) -> bool:
        return self._inner.has_region(target)

    def get_sectors(self) -> typing.Iterable[str]:
        return self._inner.get_sectors()

    def has_sector(self, target: str) -> bool:
        return self._inner.has_sector(target)

    def _filter_for_valid(self, target: OBSERVATIONS_MAYBE) -> OBSERVATIONS:
        return filter(lambda x: x is not None, target)  # type: ignore

    def _get_sum_ratios(self, target: OBSERVATIONS) -> float:
        all_ratios_maybe = map(lambda x: x.get_ratio(), target)
        all_ratios = filter(lambda x: x is not None, all_ratios_maybe)
        return sum(all_ratios)  # type: ignore
