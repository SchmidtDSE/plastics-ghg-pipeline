"""Luigi tasks for data preprocessing.

License: BSD
"""
import csv
import os
import typing

import luigi

import data_struct
import const
import prepare


class PreprocessDataTask(luigi.Task):

    def requires(self):
        """Require data to preprocess."""
        return prepare.CheckTradeDataFile()

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

    def _build_index(self) -> data_struct.ObservationIndex:
        """Create an index over the raw data file."""
        ret_index = data_struct.ObservationIndex()

        with self.input().open('r') as f:
            records_raw = csv.DictReader(f)

            for record_raw in records_raw:
                record = data_struct.Observation.from_dict(record_raw)
                ret_index.add(
                    int(record_raw['year']),
                    str(record_raw['region']),
                    str(record_raw['sector']),
                    record
                )

        return ret_index

    def _build_tasks(self, index: data_struct.ObservationIndex) -> typing.Iterable[typing.Dict]:
        """Build placeholders for the changes that need to be calculated."""
        tasks = []

        for base_year in index.get_years():
            for year_delta in range(-5, 6):
                for region in index.get_regions():
                    for sector in index.get_sectors():
                        tasks.append({
                            'baseYear': base_year,
                            'region': region,
                            'sector': sector,
                            'yearDelta': year_delta
                        })

        tasks_non_zero = filter(lambda x: x['yearDelta'] != 0, tasks)
        tasks_with_displaced_year = map(
            lambda x: (x['baseYear'] + x['yearDelta'], x),
            tasks_non_zero
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
