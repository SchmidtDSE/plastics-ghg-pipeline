"""Luigi tasks for data preprocessing.

License: BSD
"""
import csv
import itertools
import os
import typing

import luigi  # type: ignore

import data_struct
import const
import prepare


class PreprocessDataTask(luigi.Task):
    """Preprocess data for use in machine learning training.
    
    Preprocess data including converting raw data to changes between pairs of years as required for
    machine learning, filtering out instances for which actual ratios are not known as they are not
    usable for training or evaluation (to be inferred in later projection task).
    """

    def requires(self):
        """Require data to preprocess."""
        return prepare.CheckTradeDataFileTask()

    def run(self):
        """Preprocess data."""
        indexed_records = self._build_index()
        tasks = self._build_tasks(indexed_records)

        output_changes = map(
            lambda x: indexed_records.get_change(
                x['baseYear'],
                x['region'],
                x['sector'],
                x['yearDelta']
            ),
            tasks
        )
        output_changes_dict = map(lambda x: x.to_dict(), output_changes)

        self._write_changes(output_changes_dict)

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'preprocessed.csv'))

    def _build_index(self) -> data_struct.ObservationIndexable:
        """Create an index over the raw data file."""
        return data_struct.build_index_from_file(self.input().path, require_response=True)

    def _build_tasks(self, index: data_struct.ObservationIndexable) -> typing.Iterable[typing.Dict]:
        """Build placeholders for the changes that need to be calculated."""
        years = index.get_years()
        year_delta = filter(lambda x: x != 0, range(-5, 6))
        regions = index.get_regions()
        sectors = index.get_sectors()

        tasks_tuple = itertools.product(years, year_delta, regions, sectors)
        tasks = map(lambda x: {
            'baseYear': x[0],
            'yearDelta': x[1],
            'region': x[2],
            'sector': x[3]
        }, tasks_tuple)

        tasks_with_displaced_year = map(
            lambda x: (x['baseYear'] + x['yearDelta'], x),  # type: ignore
            tasks
        )
        tasks_with_included_year = filter(
            lambda x: index.has_year(x[0]),
            tasks_with_displaced_year
        )
        tasks_valid = map(lambda x: x[1], tasks_with_included_year)

        return tasks_valid

    def _write_changes(self, target: typing.Iterable[typing.Dict]):
        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=const.CHANGE_COLS)
            writer.writeheader()
            writer.writerows(target)
