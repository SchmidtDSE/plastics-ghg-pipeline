"""Tests for the preprocessing task.

License: BSD
"""
import unittest

import data_struct
import preprocess


class PreprocessTests(unittest.TestCase):

    def test_build_tasks(self):
        index = data_struct.KeyingObservationIndex()
        index.add(2013, 'China', 'Transportation', data_struct.Observation(1, 2, 3))
        index.add(2014, 'China', 'Transportation', data_struct.Observation(4, 5, 6))

        luigi_task = preprocess.PreprocessDataTask()
        tasks = luigi_task._build_tasks(index)
        count = sum(map(lambda x: 1, tasks))
        self.assertTrue(count > 0)
