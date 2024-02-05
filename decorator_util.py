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
            decorated_index.get_sectors()
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
        """Create an index over the raw data file."""
        index = data_struct.build_index_from_file(
            self.input()['data'].path,
            require_response=self._get_require_response()
        )
        return self._add_decorator(index)

    def _get_observation_dict(self, index: data_struct.IndexedObservations, year: int, region: str,
        sector: str) -> typing.Optional[typing.Dict]:
        """Create a dictionary describing an observation from an IndexedObservations structure."""
        observation = index.get_record(year, region, sector)
        if observation is None:
            return None

        observation_dict = observation.to_dict()

        observation_dict['year'] = year
        observation_dict['region'] = region
        observation_dict['sector'] = sector

        return observation_dict

    def _write_output(self, target: typing.Iterable[typing.Dict]):
        """Write CSV file from dicts representing Observations where some are inferred."""
        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=const.EXPECTED_PROJECTION_COLS)
            writer.writeheader()
            writer.writerows(target)

    def _add_decorator(self,
        index: data_struct.IndexedObservations) -> data_struct.IndexedObservations:
        raise NotImplementedError('Use implementor.')

    def _get_require_response(self) -> bool:
        raise NotImplementedError('Use implementor.')
