"""Tests for logic supporting machine learning utilities.

License: BSD
"""
import unittest

import ml_util


class MlUtilTests(unittest.TestCase):

    def setUp(self):
        self._linear_config = ml_util.ModelDefinition(
            'linear',
            alpha=0.1
        )
        
        self._svr_config = ml_util.ModelDefinition(
            'svr',
            kernel='rbf',
            degree=-1,
            alpha=0.1
        )
        
        self._tree_config = ml_util.ModelDefinition(
            'tree',
            depth=3
        )
        
        self._random_forest_config = ml_util.ModelDefinition(
            'random forest',
            depth=5,
            estimators=10,
            features='log2'
        )
        
        self._adaboost_config = ml_util.ModelDefinition(
            'adaboost',
            depth=5,
            estimators=10
        )

    def test_dict(self):
        serialized = self._linear_config.to_dict()
        deserialized = ml_util.ModelDefinition.from_dict(serialized)
        self.assertEqual(deserialized.get_algorithm(), 'linear')
    
    def test_check_linear_config(self):
        self.assertTrue(self._linear_config.is_valid())
        self.assertFalse(ml_util.check_linear_config(self._tree_config))
    
    def test_check_svr_config(self):
        self.assertTrue(self._svr_config.is_valid())
        self.assertFalse(ml_util.check_svr_config(self._linear_config))
    
    def test_check_tree_config(self):
        self.assertTrue(self._tree_config.is_valid())
        self.assertFalse(ml_util.check_tree_config(self._linear_config))
    
    def test_check_random_forest_config(self):
        self.assertTrue(self._random_forest_config.is_valid())
        self.assertFalse(ml_util.check_random_forest_config(self._linear_config))
    
    def test_check_adaboost_config(self):
        self.assertTrue(self._adaboost_config.is_valid())
        self.assertFalse(ml_util.check_adaboost_config(self._linear_config))
    
    def test_build_linear(self):
        model = ml_util.build_model(self._linear_config)
        self.assertIsNotNone(model)
    
    def test_build_svr(self):
        model = ml_util.build_model(self._svr_config)
        self.assertIsNotNone(model)
    
    def test_build_tree(self):
        model = ml_util.build_model(self._tree_config)
        self.assertIsNotNone(model)
    
    def test_build_random_forest(self):
        model = ml_util.build_model(self._random_forest_config)
        self.assertIsNotNone(model)
    
    def test_build_adaboost(self):
        model = ml_util.build_model(self._adaboost_config)
        self.assertIsNotNone(model)
