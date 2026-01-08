#   Copyright (c) 2025, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.

import unittest

from signaloid.distributional.dirac_delta import DiracDelta


class TestDiracDelta(unittest.TestCase):
    def test_dirac_delta_comparisons(self) -> None:
        """
        Test comparing Dirac Deltas
        """

        d1: DiracDelta = DiracDelta(position=1.2, mass=2.3)
        d2: DiracDelta = DiracDelta(position=4.5, mass=5.6)

        self.assertTrue(d1 < d2)
        self.assertTrue(d1 <= d2)
        self.assertFalse(d1 == d2)
        self.assertTrue(d1 != d2)
        self.assertFalse(d1 >= d2)
        self.assertFalse(d1 > d2)

    def test_dirac_delta_sorting(self) -> None:
        """
        Test sorting Dirac Deltas
        """

        d1 = DiracDelta(position=1.0, mass=0.1)
        d2 = DiracDelta(position=2.0, mass=0.2)
        d3 = DiracDelta(position=3.0, mass=0.3)
        d4 = DiracDelta(position=4.0, mass=0.4)

        a: list[DiracDelta] = [d2, d1, d4, d3]
        a.sort()

        b: list[DiracDelta] = [d1, d2, d3, d4]

        for dd_a, dd_b in zip(a, b):
            self.assertTrue(dd_a == dd_b)

    def test_dirac_delta_operations(self) -> None:
        """
        Test adding Dirac Deltas
        """

        d1: DiracDelta = DiracDelta(position=1.2, mass=2.3)
        d2: DiracDelta = DiracDelta(position=4.5, mass=5.6)
        d3 = d1 + d2

        combined_mass: float = 2.3 + 5.6
        combined_position: float = (
            1.2 * 2.3
            + 4.5 * 5.6
        ) / combined_mass
        self.assertTrue(d3.position == combined_position)
        self.assertTrue(d3.mass == combined_mass)


if __name__ == "__main__":
    unittest.main()
