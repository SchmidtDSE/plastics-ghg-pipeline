import unittest

import data_struct
import normalization_util


class NormalizingIndexedObservationsDecoratorTests(unittest.TestCase):

    def setUp(self):
        self._index = data_struct.KeyingObservationIndex()
        self._index.add(2023, 'NAFTA', 'Transportation', data_struct.Observation(1, 2, 3))
        self._index.add(2023, 'NAFTA', 'Packaging', data_struct.Observation(4, 5, 6))
        self._decorated = normalization_util.NormalizingIndexedObservationsDecorator(self._index)

    def test_get_record(self):
        record = self._decorated.get_record(2023, 'NAFTA', 'Transportation')
        self.assertAlmostEqual(record.get_ratio(), 1 / (1 + 4))
