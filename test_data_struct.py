"""Tests for shared data structures.

License: BSD
"""
import unittest

import const
import data_struct


class ObservationTests(unittest.TestCase):

    def test_dict(self):
        original = data_struct.Observation(1, 2, 3)
        serialized = original.to_dict()
        parsed = data_struct.Observation.from_dict(serialized)
        self.assertEqual(parsed.get_ratio(), 1)


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
        self.assertEqual(deserialized.get_region(), 'nafta')
        self.assertEqual(deserialized.get_sector(), 'transportation')
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


class ObservationIndexTests(unittest.TestCase):

    def setUp(self):
        self._index = data_struct.ObservationIndex()
        self._index.add(2023, 'China', 'Transportation', data_struct.Observation(1, 2, 3))
        self._index.add(2024, 'China', 'Transportation', data_struct.Observation(4, 5, 6))

    def test_get_found(self):
        result = self._index.get_record(2023, 'china', 'transportation')
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.get_ratio(), 1)

    def test_get_not_found(self):
        result = self._index.get_record(2022, 'china', 'transportation')
        self.assertIsNone(result)

    def test_get_change_found(self):
        change = self._index.get_change(2023, 'china', 'transportation', 1)
        self.assertIsNotNone(change)
        self.assertAlmostEqual(change.get_before_ratio(), 1)
        self.assertAlmostEqual(change.get_after_ratio(), 4)

    def test_get_change_not_found(self):
        change = self._index.get_change(2023, 'china', 'transportation', 2)
        self.assertIsNone(change)

    def test_get_members(self):
        self.assertTrue(2023 in self._index.get_years())
        self.assertTrue(2024 in self._index.get_years())
        self.assertTrue('china' in self._index.get_regions())
        self.assertTrue('transportation' in self._index.get_sectors())

    def test_has_members_found(self):
        self.assertTrue(self._index.has_year(2023))
        self.assertTrue(self._index.has_region('china'))
        self.assertTrue(self._index.has_sector('transportation'))

    def test_has_members_not_found(self):
        self.assertFalse(self._index.has_year(2022))
        self.assertFalse(self._index.has_region('other'))
        self.assertFalse(self._index.has_sector('other'))
