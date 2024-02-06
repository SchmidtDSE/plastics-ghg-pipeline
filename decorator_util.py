"""Utilities for running tasks with IndexedObservations decorators.

Utilities for running tasks with IndexedObservations decorators as in the OOP pattern for stackable
behavior (https://en.wikipedia.org/wiki/Decorator_pattern).

License: BSD
"""

import csv
import itertools
import typing

import luigi  # type: ignore

import const
import data_struct


class DecoratedIndexedObservationsTask(luigi.Task):
    """Template task which reads indexed observations, decorates them, and writes them out again."""

    def run(self):
        """Project data."""
        decorated_index = self._build_index()

        tasks_tuple = itertools.product(
            decorated_index.get_years(),
            decorated_index.get_regions(),
            decorated_index.get_subtypes()
        )
        output_dict = map(
            lambda x: self._get_observation_dict(decorated_index, x[0], x[1], x[2]),
            tasks_tuple
        )
        output_dict_valid = filter(
            lambda x: x is not None,
            output_dict
        )

        self._write_output(output_dict_valid)

    def _build_index(self) -> data_struct.IndexedObservations:
        """Create an index over the raw data file.

        Returns:
            Index over the raw file with the decorator(s) applied.
        """
        index = data_struct.build_index_from_file(
            self.input()['data'].path,
            require_response=self._get_require_response()
        )
        return self._add_decorator(index)

    def _get_observation_dict(self, index: data_struct.IndexedObservations, year: int, region: str,
        subtype: str) -> typing.Optional[typing.Dict]:
        """Create a dictionary describing an observation from an IndexedObservations structure.

        Args:
            index: The index from which the dictionary should be derived.
            year: The year for which an observation dictionary is requested.
            region: The region for which an observation is requested.
            subtype: The subtype for which an observation is requested.

        Returns:
            Dictionary representation of the observation or None if the dictionary could not be
            made.
        """
        observation = index.get_record(year, region, subtype)
        if observation is None:
            return None

        observation_dict = observation.to_dict()

        observation_dict['year'] = year
        observation_dict['region'] = region
        observation_dict['subtype'] = subtype

        return observation_dict

    def _write_output(self, target: typing.Iterable[typing.Dict]):
        """Write CSV file from dicts representing Observations where some are inferred.

        Args:
            target: The dictionaries to write.
        """
        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=const.EXPECTED_PROJECTION_COLS)
            writer.writeheader()
            writer.writerows(target)

    def _add_decorator(self,
        index: data_struct.IndexedObservations) -> data_struct.IndexedObservations:
        """Add the decorator(s) to the base index.

        Args:
            index: The index to decorate.

        Returns:
            The decorated index.
        """
        raise NotImplementedError('Use implementor.')

    def _get_require_response(self) -> bool:
        """Determine if "incomplete" records are allowed.

        Returns:
            True if the index created from observations on disk should include those without subtype
            to overall trade ratios or false if those ratios are required.
        """
        raise NotImplementedError('Use implementor.')
