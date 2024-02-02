"""Tests for shared data structures.

License: BSD
"""
import unittest

import const
import data_struct


class ChangeTests(unittest.TestCase):

    def setUp(self):
        self._change = data_struct.Change(
            'NAFTA',
            'Transportation',
            2014,
            -1,
            0.01,
            0.02,
            0.03,
            0.04
        )

    def test_dict(self):
        serialized = self._change.to_dict()
        deserialized = data_struct.Change.from_dict(serialized)
        self.assertEqual(deserialized.get_region(), 'NAFTA')
        self.assertEqual(deserialized.get_sector(), 'Transportation')
        self.assertEqual(deserialized.get_year(), 2014)
        self.assertEqual(deserialized.get_years(), -1)
        self.assertAlmostEqual(deserialized.get_gdp_change(), 0.01)
        self.assertAlmostEqual(deserialized.get_population_change(), 0.02)
        self.assertAlmostEqual(deserialized.get_before_ratio(), 0.03)
        self.assertAlmostEqual(deserialized.get_after_ratio(), 0.04)

    def test_to_vector(self):
        vector = self._change.to_vector()
        self.assertEqual(len(vector), len(const.INPUTS))

    def test_hot_encode(self):
        self.assertEqual(self._change._hot_encode('b', 'A'), 0)
        self.assertEqual(self._change._hot_encode('a', 'A'), 1)


class ObservationTests(unittest.TestCase):

    def setUp(self):
        self._index = data_struct.ObservationIndex()
        self._index.add(2013, 'China', 'Transportation', data_struct.Observation(1, 2, 3))
        self._index.add(2014, 'China', 'Transportation', data_struct.Observation(4, 5, 6))

    def test_get_found(self):
        result = self._index.get_record(2013, 'china', 'transportation')
        self.assertNotNone(result)
        self.assertAlmostEqual(result.get_ratio(), 1)

    def test_get_not_found(self):
        result = self._index.get_record(2012, 'china', 'transportation')
        self.assertNone(result)

    def test_get_change_found(self):
        change = self._index.get_change(2013, 'china', 'transportation', 1)
        self.assertNotNone(change)
        self.assertAlmostEqual(change.get_before_ratio(), 1)
        self.assertAlmostEqual(change.get_after_ratio(), 4)

    def test_get_change_not_found(self):
        change = self._index.get_change(2013, 'china', 'transportation', 2)
        self.assertNone(change)

    def test_get_members(self):
        self.assertTrue(2013 in self._index.get_years())
        self.assertTrue(2014 in self._index.get_years())
        self.assertTrue('china' in self._index.get_regions())
        self.assertTrue('transportation' in self._index.get_sectors())

    def test_has_members_found(self):
        self.assertTrue(self._index.has_year(2013))
        self.assertTrue(self._index.has_region('china'))
        self.assertTrue(self._index.has_sectors('transportation'))

    def test_has_members_not_found(self):
        self.assertTrue(self._index.has_year(2012))
        self.assertTrue(self._index.has_region('other'))
        self.assertTrue(self._index.has_sectors('other'))

