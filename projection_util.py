"""Helper functions for projecting metrics forward.

License: BSD
"""
import statistics
import typing

import numpy

import const
import data_struct


class Predictor:
    """Interface for models which can infer trade ratios."""

    def predict(self, input_change: data_struct.Change) -> float:
        """Predict a ratio of sector to overall trade.

        Args:
            input_change: The change for which the after ratio needs to be predicted.

        Returns:
            Inferred value.
        """
        raise NotImplementedError('Use implementor.')


class OnnxPredictor(Predictor):
    """Predictor which infers trade ratios through an onnx model."""

    def __init__(self, model):
        """Create a new Onnx-backed predictor.

        Args:
            model: The onnx model to service this predictor.
        """
        self._model = model

    def predict(self, input_change: data_struct.Change) -> float:
        input_vector = input_change.to_vector()

        # Allows for type inference
        num_feature = len(input_vector)  # type: ignore
        input_array = numpy.array(input_vector).astype(numpy.float32).reshape(1, num_feature)

        input_name = self._model.get_inputs()[0].name
        label_name = self._model.get_outputs()[0].name
        after_ratio_nest = self._model.run(
            [label_name],
            {input_name: input_array}
        )

        return float(after_ratio_nest[0])


class InferringIndexedObservationsDecorator(data_struct.IndexedObservations):
    """Decorator for an IndexedObservations structure which infers unknown sector trade ratios."""

    def __init__(self, inner: data_struct.IndexedObservations, model: Predictor):
        """Create a new decorator.

        Args:
            inner: The indexed observations structure to decorate.
            model: The model to use to make inferences.
        """
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

        prior_years_offset = range(1, const.NUM_YEARS_INFERENCE_WINDOW + 1)
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
        before = self.get_record(year, region, sector)  # Require the before to have a value
        after = self._inner.get_record(year + years, region, sector)  # Allow after to be inferred

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
        """Add a new record to the decorated index.

        Args:
            year: The year like 2024 for which a record should be added into the underlying index.
            region: The region like "China" for which a record should be added into the decorated
                index.
            sector: The sector like "Transportation" for which a record should be added into the
                decorated index.
            ratio: The ratio of sector to overall net trade.

        Returns:
            Newly added observation.
        """
        cached = self._inner.get_record(year, region, sector)
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
        """Determine if a ratio can be inferred.

        Args:
            year: The year for which an inferrence would be required.
            region: The region like "Row" in which inference would be required.
            sector: The sector like "Transportation" in which inference would be required.

        Returns:
            True if inferrable and false otherwise.
        """
        if year < const.MIN_YEAR:
            return False

        if not self.has_region(region):
            return False

        if not self.has_sector(sector):
            return False

        return True
