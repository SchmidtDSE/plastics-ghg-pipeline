"""Helper functions for projecting metrics forward.

License: BSD
"""
import statistics
import typing

import numpy

import const
import data_struct


class Predictor:

    def predict(self, input_change: data_struct.Change) -> float:
        raise NotImplementedError('Use implementor.')


class OnnxPredictor(Predictor):

    def __init__(self, model):
        self._model = model

    def preidct(self, input_change: data_struct.Change) -> float:
        input_vector = numpy.array(input_change.to_vector())  # Allows for type inference

        input_name = self._model.get_inputs()[0].name
        label_name = self._model.get_outputs()[0].name
        after_ratio_nest = self._model.run(
            [label_name],
            {input_name: input_vector.astype(numpy.float32)}
        )

        return float(after_ratio_nest[0])


class PredictionObservationIndexDecorator(data_struct.ObservationIndexable):

    def __init__(self, inner: data_struct.ObservationIndexable, model: Predictor):
        self._inner = inner
        self._model = model

    def get_record(self, year: int, region: str,
        sector: str, add_to_cache: bool = True) -> typing.Optional[data_struct.Observation]:
        cached = self._inner.get_record(year, region, sector)
        if cached is None:
            return None

        if cached.get_ratio() is not None:
            return cached

        if not self._query_in_range(year, region, sector):
            return None

        prior_years_offset = range(1, 6)
        prior_years = map(lambda x: {'year': year - x, 'years': x}, prior_years_offset)
        prior_changes = map(lambda x: self.get_change(
            x['year'],
            region,
            sector,
            x['years'],
            add_to_cache=False
        ), prior_years)
        prior_changes_allowed = filter(lambda x: x is not None, prior_changes)
        after_ratios = map(lambda x: x.get_after_ratio(), prior_changes_allowed)  # type: ignore

        try:
            after_ratio = statistics.mean(after_ratios)  # type: ignore
        except statistics.StatisticsError:  # Encountered if empty or invalid value
            return None

        new_obs = self._add_inference_to_cache(year, region, sector, after_ratio)  # type: ignore

        return new_obs

    def get_change(self, year: int, region: str, sector: str,
        years: int, add_to_cache: bool = True) -> typing.Optional[data_struct.Change]:
        before = self.get_record(year, region, sector)
        after = self.get_record(year + years, region, sector)

        if before is None or after is None:
            return None

        before_ratio = before.get_ratio()
        if before_ratio is None:
            return None

        after_ratio = after.get_ratio()
        if after_ratio is None:
            input_change = data_struct.Change(
                region,
                sector,
                year,
                years,
                self._calculate_change(before.get_gdp(), after.get_gdp()),
                self._calculate_change(before.get_population(), after.get_population()),
                before_ratio,
                None
            )
            after_ratio = self._model.predict(input_change)

            if add_to_cache:
                self._add_inference_to_cache(year + years, region, sector, after_ratio)

        return data_struct.Change(
            region,
            sector,
            year,
            years,
            self._calculate_change(before.get_gdp(), after.get_gdp()),
            self._calculate_change(before.get_population(), after.get_population()),
            before_ratio,
            after_ratio
        )

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

    def _add_inference_to_cache(self, year: int, region: str, sector: str,
        ratio: float) -> data_struct.Observation:
        cached = self.get_record(year, region, sector)
        if cached is None:
            raise RuntimeError('Socioeconomic values not available.')

        new_observation = data_struct.Observation(
            ratio,
            cached.get_gdp(),
            cached.get_population()
        )

        self.add(year, region, sector, new_observation)

        return new_observation

    def _query_in_range(self, year: int, region: str, sector: str) -> bool:
        if year < const.MIN_YEAR:
            return False

        if not self.has_region(region):
            return False

        if not self.has_sector(sector):
            return False

        return True
