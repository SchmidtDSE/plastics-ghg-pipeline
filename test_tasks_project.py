import unittest

import data_struct
import tasks_project


class ProjectTests(unittest.TestCase):

    def setUp(self):
        self._index = data_struct.KeyingObservationIndex()
        self._index.add(2023, 'NAFTA', 'Transportation', data_struct.Observation(1, 2, 3))
        self._task = tasks_project.ProjectionTask()

    def test_get_observation_dict_none(self):
        serialized = self._task._get_observation_dict(self._index, 2024, 'NAFTA', 'Transportation')
        self.assertIsNone(serialized)

    def test_get_observation_dict_success(self):
        serialized = self._task._get_observation_dict(self._index, 2023, 'NAFTA', 'Transportation')
        self.assertIsNotNone(serialized)
        self.assertEqual(serialized['year'], 2023)
        self.assertEqual(serialized['ratioSubtype'], 1)
