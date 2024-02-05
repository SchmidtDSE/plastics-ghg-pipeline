"""Logic for longitudinally projecting goods trade ratios.

License: BSD
"""
import csv
import itertools
import os
import typing

import luigi  # type: ignore
import onnxruntime  # type: ignore

import const
import data_struct
import goods_ml_prod
import prepare
import projection_util


class GoodsProjectionTask(luigi.Task):
    """Project data without normalization."""

    def requires(self):
        """Require data to preprocess."""
        return {
            'data': prepare.CheckTradeDataFileTask(),
            'model': goods_ml_prod.TrainProdModelTask()
        }

    def run(self):
        """Project data."""
        inferring_index = self._build_inferring_index()

        tasks_tuple = itertools.product(
            inferring_index.get_years(),
            inferring_index.get_regions(),
            inferring_index.get_sectors()
        )
        output_dict = map(
            lambda x: self._get_observation_dict(inferring_index, x[0], x[1], x[2]),
            tasks_tuple
        )
        output_dict_valid = filter(
            lambda x: x is not None,
            output_dict
        )

        self._write_output(output_dict_valid)

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'projected.csv'))

    def _build_index(self) -> data_struct.IndexedObservations:
        """Create an index over the raw data file."""
        return data_struct.build_index_from_file(self.input()['data'].path, require_response=False)

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

    def _build_inferring_index(self) -> data_struct.IndexedObservations:
        """Load observations from a CSV file into an index.

        Load observations from a CSV file and insert them into an index which attempts to infer
        records requested from client code if they are not present in the index.
        """
        inner_model = onnxruntime.InferenceSession(
            self.input()['model'].path,
            providers=['CPUExecutionProvider']
        )
        model = projection_util.OnnxPredictor(inner_model)
        index = self._build_index()
        inferring_index = projection_util.InferringIndexedObservationsDecorator(index, model)
        return inferring_index

    def _write_output(self, target: typing.Iterable[typing.Dict]):
        """Write CSV file from dicts representing Observations where some are inferred."""
        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=const.EXPECTED_PROJECTION_COLS)
            writer.writeheader()
            writer.writerows(target)
