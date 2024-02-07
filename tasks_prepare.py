"""Luigi tasks for checking / preparing the environment.

License: BSD
"""
import codecs
import csv
import json
import os
import typing

import luigi  # type: ignore
import requests

import const
import ml_util


class CheckFileTask(luigi.Task):
    """Template task for checking that a file exists."""

    def run(self):
        """Simply check that a file exists."""
        if not os.path.exists(self.get_path()):
            raise RuntimeError('Could not find %s.' % self.get_path())

    def output(self):
        """The input file is the output."""
        return luigi.LocalTarget(self.get_path())

    def get_path(self) -> str:
        """Get the path of the file to check.

        Returns:
            File path
        """
        raise NotImplementedError('Use implementor.')


class CheckConfigFileExistsTask(CheckFileTask):
    """Check that the job JSON file is present."""

    def get_path(self) -> str:
        return os.path.join(const.TASK_DIR, const.CONFIG_NAME)


class CheckConfigFileTask(luigi.Task):
    """Check that the config file has expected values."""

    def requires(self):
        """Require data to check."""
        return CheckConfigFileExistsTask()

    def run(self):
        """Validate the JSON contents."""
        with self.input().open() as f:
            content = json.load(f)

        definition = ml_util.ModelDefinition.from_dict(content['model'])
        assert definition.is_valid()

        with self.output().open('w') as f:
            json.dump(content, f)

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'task_checked.json'))


class GetTradeDataFileTask(luigi.Task):
    """Task to download and check the trade data file."""

    def run(self):
        """Download and check file."""
        # Open stream
        response = requests.get(const.TRADE_INPUTS_URL, stream=True)
        line_iterator = response.iter_lines()
        reader = codecs.iterdecode(
            csv.DictReader(line_iterator),
            quotechar='"',
            delimiter=','
        )

        # Check row values are expected
        validated_rows = map(lambda x: self._parse_and_validate_row(x), reader)

        # Remove the "check" polymer type which warns if there are unknown polymers but is not
        # useful for the machine learning or front end after ensuring the check passes.
        rows_no_other = filter(lambda x: x['subtype'] != const.OTHER_SUBTYPE, validated_rows)

        # Output
        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=const.EXPECTED_RAW_DATA_COLS)
            writer.writeheader()
            writer.writerows(rows_no_other)

        # Close exhausted stream
        response.close()

    def output(self) -> str:
        """Write to the deploy directory."""
        full_path = os.path.join(const.DEPLOY_DIR, const.TRADE_FRAME_NAME)
        return luigi.LocalTarget(full_path)

    def parse_and_validate_row(self, target: typing.Dict) -> typing.Dict:
        """Parse an input row and check that its values are as expected."""
        # Check region is known
        if target['region'] not in const.REGIONS:
            raise RuntimeError('Unknown region: ' + target['region'])

        # Internal check for unknown polymer
        subtype_is_other = target['subtype'] == const.OTHER_SUBTYPE

        # Check subtype is known
        subtype_tracked = target['subtype'] in const.SUBTYPES
        subtype_valid = subtype_tracked or subtype_is_other
        if not subtype_valid:
            raise RuntimeError('Subtype not known: ' + target['subtype'])

        # Parse year and determine if additional checks are required
        year = self._parse_int_maybe(target['year'])
        if year is None:
            raise RuntimeError('Could not parse year: %s' % target['year'])

        actuals_required_min = year >= const.ACTUALS_REQUIRED_MIN_YEAR
        actuals_required_max = year <= const.ACTUALS_REQUIRED_MAX_YEAR
        actuals_required = actuals_required_min and actuals_required_max

        # Perform checks on ratio if required
        ratio = self._parse_float_maybe(target['ratioSubtype'])
        if actuals_required:

            if ratio is None:
                raise RuntimeError('Ratio not available on required year %d.' % year)

            if subtype_is_other and abs(ratio - 1) > const.UNKNOWN_RATIO_TOLLERANCE:
                raise RuntimeError('Unknown ratio exceeds allowance in year %d.' % year)

        # Ensure GDP
        gdp = self._parse_float_maybe(target['gdp'])
        if gdp is None:
            raise RuntimeError('Missing GDP for year %d.' % year)

        # Ensure population
        population = self._parse_float_maybe(target['population'])
        if population is None:
            raise RuntimeError('Missing population for year %d.' % year)

        # Create an output string for ratio
        ratio_str = '' if ratio is None else ratio

        # Create resulting record
        return {
            'year': year,
            'region': target['region'],
            'subtype': target['subtype'],
            'ratioSubtype': ratio_str,
            'gdp': gdp,
            'population': population
        }

    def _parse_float_maybe(self, target: str) -> typing.Optional[float]:
        """Try parsing a float but, if encountering unparseable value, return None."""
        try:
            return float(target)
        except ValueError:
            return None

    def _parse_int_maybe(self, target: str) -> typing.Optional[int]:
        """Try parsing an int but, if encountering unparseable value, return None."""
        try:
            return int(target)
        except ValueError:
            return None
