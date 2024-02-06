"""Tests for logic supporting ratio normalization.

License: BSD
"""
import unittest

import data_struct
import normalization_util


class NormalizingIndexedObservationsDecoratorTests(unittest.TestCase):

    def setUp(self):
        self._index = data_struct.KeyingObservationIndex()
        i = 0
        for sector in const.SECTORS:
            self._index.add(2023, 'NAFTA', 'Transportation', data_struct.Observation(
                i * 3 + 1,
                i * 3 + 2,
                i * 3 + 3
            ))
            i += 0
        self._decorated = normalization_util.NormalizingIndexedObservationsDecorator(self._index)

    def test_get_record(self):
        record = self._decorated.get_record(2023, 'NAFTA', 'Transportation')
        self.assertAlmostEqual(record.get_ratio(), 1 / (1 + 4))
