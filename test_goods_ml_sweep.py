"""Tests for informational sweep.

License: BSD
"""
import functools
import unittest

import data_struct
import goods_ml_sweep
import ml_util


class SweepTaskTests(unittest.TestCase):

    def setUp(self):
        self._definition = ml_util.ModelDefinition('linear', alpha=0.1)
        self._model = lambda x: [0.1]
        self._trained_model = ml_util.TrainedModel(
            self._model,
            0.2,
            0.3,
            0.4
        )

    def test_get_with(self):
        task = goods_ml_sweep.SweepTask(self._definition)
        task = task.get_with_model(self._model)
        task = task.get_with_trained_model(self._trained_model)

        self.assertIsNotNone(task.get_definition())
        self.assertIsNotNone(task.get_model())
        self.assertIsNotNone(task.get_trained_model())

    def test_get_dict(self):
        task = goods_ml_sweep.SweepTask(
            self._definition,
            trained_model=self._trained_model
        )
        serialized = task.get_sweep_dict()
        self.assertAlmostEqual(serialized['trainMae'], 0.2)


class ModelSweepTaskTests(unittest.TestCase):

    def setUp(self):
        self._task = goods_ml_sweep.ModelSweepTask()

    def test_choose_set(self):
        change = data_struct.Change(
            'NAFTA',
            'Transportation',
            2018,
            -1,
            0.01,
            0.02,
            0.03,
            0.04
        )
        self.assertTrue(self._task._choose_set(change) in ['train', 'test', 'valid'])

    def test_get_model_definitions(self):
        definitions = self._task._get_model_definitions()
        counts = map(lambda x: {
            'valid': 1 if x.is_valid() else 0,
            'invalid': 0 if x.is_valid() else 1
        }, definitions)
        summed = functools.reduce(lambda a, b: {
            'valid': a['valid'] + b['valid'],
            'invalid': a['invalid'] + b['invalid']
        }, counts)

        self.assertEqual(summed['invalid'], 0)
        self.assertTrue(summed['valid'] > 0)
    
    def test_force_str(self):
        target = {'a': 1, 'b': None}
        processed = self._task._force_str(target)
        self.assertEqual(processed['a'], '1')
        self.assertEqual(processed['b'], 'None')
