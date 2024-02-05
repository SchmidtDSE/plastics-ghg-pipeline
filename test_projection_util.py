import unittest

import data_struct
import projection_util


class TestModel(projection_util.Predictor):

    def __init__(self, value: float):
        self._value = value
    
    def predict(self, input_change: data_struct.Change) -> float:
        return self._value


class ProjectionUtilTests(unittest.TestCase):

    def setUp(self):
        self._index = data_struct.KeyingObservationIndex()
        self._index.add(2023, 'China', 'Transportation', data_struct.Observation(1, 2, 3))
        self._index.add(2024, 'China', 'Transportation', data_struct.Observation(4, 5, 6))
        self._index.add(2025, 'China', 'Transportation', data_struct.Observation(None, 8, 9))
        
        self._decorated_index = projection_util.InferringIndexedObservationsDecorator(
            self._index,
            TestModel(7)
        )

    def test_get_record_passthrough(self):
        record = self._decorated_index.get_record(2023, 'China', 'Transportation')
        self.assertEqual(record.get_ratio(), 1)

    def test_get_change_passthrough(self):
        change = self._decorated_index.get_change(2023, 'China', 'Transportation', 1)
        self.assertEqual(change.get_after_ratio(), 4)

    def test_get_record_infer(self):
        change = self._decorated_index.get_record(2025, 'China', 'Transportation')
        self.assertEqual(change.get_ratio(), 7)

    def test_get_change_infer(self):
        change = self._decorated_index.get_change(2024, 'China', 'Transportation', 1)
        self.assertEqual(change.get_after_ratio(), 7)
