"""Tests for pipline preparation and data retrieval.

License: BSD
"""
import unittest

import tasks_prepare


class GetTradeDataFileTests(unittest.TestCase):

    def setUp(self):
        self._task = tasks_prepare.GetTradeDataFileTask()

    def test_parse_and_validate_row_valid_known(self):
        result = self._task.parse_and_validate_row({
            'year': '2013',
            'region': 'NAFTA',
            'subtype': 'Transportation',
            'ratioSubtype': '1',
            'gdp': '12',
            'population': '34'
        })
        self.assertAlmostEqual(result['ratioSubtype'], 1)

    def test_parse_and_validate_row_valid_unknown(self):
        result = self._task.parse_and_validate_row({
            'year': '2040',
            'region': 'NAFTA',
            'subtype': 'Transportation',
            'ratioSubtype': '',
            'gdp': '12',
            'population': '34'
        })
        self.assertEqual(result['ratioSubtype'], '')

    def test_parse_and_validate_row_invalid_region(self):
        with self.assertRaises(RuntimeError):
            self._task.parse_and_validate_row({
                'year': '2013',
                'region': 'NAFT',
                'subtype': 'Transportatio',
                'ratioSubtype': '1',
                'gdp': '12',
                'population': '34'
            })

    def test_parse_and_validate_row_invalid_sector(self):
        with self.assertRaises(RuntimeError):
            self._task.parse_and_validate_row({
                'year': '2013',
                'region': 'NAFTA',
                'subtype': 'Transportatio',
                'ratioSubtype': '1',
                'gdp': '12',
                'population': '34'
            })

    def test_parse_and_validate_row_invalid_ratio(self):
        with self.assertRaises(RuntimeError):
            self._task.parse_and_validate_row({
                'year': '2013',
                'region': 'NAFTA',
                'subtype': 'Transportation',
                'ratioSubtype': '',
                'gdp': '12',
                'population': '34'
            })

    def test_parse_and_validate_row_invalid_check(self):
        with self.assertRaises(RuntimeError):
            self._task.parse_and_validate_row({
                'year': '2013',
                'region': 'NAFTA',
                'subtype': 'MISC',
                'ratioSubtype': '0.5',
                'gdp': '12',
                'population': '34'
            })

    def test_parse_and_validate_row_invalid_gdp(self):
        with self.assertRaises(RuntimeError):
            self._task.parse_and_validate_row({
                'year': '2013',
                'region': 'NAFTA',
                'subtype': 'Transportation',
                'ratioSubtype': '1',
                'gdp': '',
                'population': '34'
            })

    def test_parse_and_validate_row_invalid_population(self):
        with self.assertRaises(RuntimeError):
            self._task.parse_and_validate_row({
                'year': '2013',
                'region': 'NAFTA',
                'subtype': 'Transportation',
                'ratioSubtype': '1',
                'gdp': '12',
                'population': ''
            })
