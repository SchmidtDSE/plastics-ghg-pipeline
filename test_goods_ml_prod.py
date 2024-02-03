"""Tests for production good trade model generation.

License: BSD
"""
import unittest

import data_struct
import goods_ml_prod


class GoodsMlProdTests(unittest.TestCase):

    def setUp(self):
        self._example_change = data_struct.Change(
            'NAFTA',
            'Transportation',
            2018,
            -1,
            0.01,
            0.02,
            0.03,
            0.04
        )

    def test_traditional_choose_set(self):
        task = goods_ml_prod.TraditionalValidateModelTask()
        self.assertTrue(task._choose_set(self._example_change) in ['train', 'test'])

    def test_temporal_choose_set(self):
        task = goods_ml_prod.TemporalValidateModelTask()

        before = data_struct.Change(
            'NAFTA',
            'Transportation',
            2018,
            -1,
            0.01,
            0.02,
            0.03,
            0.04
        )

        after = data_struct.Change(
            'NAFTA',
            'Transportation',
            2018,
            1,
            0.01,
            0.02,
            0.03,
            0.04
        )

        self.assertEqual(task._choose_set(before), 'train')
        self.assertEqual(task._choose_set(after), 'test')

    def test_production_choose_set(self):
        task = goods_ml_prod.TrainProdModelTask()
        self.assertEqual(task._choose_set(self._example_change), 'train')
