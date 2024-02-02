"""Sanity checks for constants.

License: BSD
"""
import unittest

import const


class ConstTests(unittest.TestCase):

    def test_present(self):
        self.assertTrue(len(const.INPUTS) > 0)
        self.assertTrue(const.RESPONSE != '')
