#   Copyright (c) 2024, Signaloid.
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

import csv
from typing import Optional
import unittest

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue


def read_string_bytes_pairs_from_csv(
    csv_filename: str
) -> Optional[list[tuple[str, bytes]]]:
    """
    Reads pairs of Ux strings and Ux bytes(in hex format) from a csv file

    Args:
        csv_filename: The input csv file path
    Returns:
        pairs: list of (string_value, bytearray_value) tuples
    """
    pairs: list[tuple[str, bytes]] = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                string_value = row[0]
                bytearray_value = bytes.fromhex(row[1])
                pairs.append((string_value, bytearray_value))
    return pairs


class TestUxParsing(unittest.TestCase):
    def test_parse_ux_strings_values(
        self,
        input_filename: str = "src/signaloid/distributional/test_ux_value_pairs.csv"
    ) -> None:
        """
        Test parsing Ux string values and converting them to Ux bytes
        """

        ux_pairs = read_string_bytes_pairs_from_csv(input_filename)
        self.assertIsNotNone(ux_pairs)

        if ux_pairs is None:
            return

        for pair in ux_pairs:
            distValueFromUxString = DistributionalValue.parse(pair[0])
            self.assertIsNotNone(distValueFromUxString)
            if distValueFromUxString is not None:
                self.assertEqual(bytes(distValueFromUxString), pair[1])

    def test_parse_ux_bytes_values(
        self,
        input_filename: str = "src/signaloid/distributional/test_ux_value_pairs.csv"
    ) -> None:
        """
        Test parsing Ux bytes and converting them to Ux strings
        """

        ux_pairs = read_string_bytes_pairs_from_csv(input_filename)
        self.assertIsNotNone(ux_pairs)

        if ux_pairs is None:
            return

        for pair in ux_pairs:
            distValueFromBytes = DistributionalValue.parse(pair[1])
            self.assertIsNotNone(distValueFromBytes)
            if distValueFromBytes is not None:
                self.assertEqual(str(distValueFromBytes), pair[0])

    def test_check_is_full_valid_TTR(self):
        """
        Test checking DistributionalValues for Full & Valid TTR
        """
        probabilitiesSlope = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4,
            0.3, 0.2, 0.1
        ]
        dist = DistributionalValue(dirac_deltas=[
            DiracDelta(position=position, mass=mass)
            for position, mass in enumerate(probabilitiesSlope)
        ])
        self.assertFalse(dist.check_is_full_valid_TTR())

        dist = DistributionalValue(dirac_deltas=[
            DiracDelta(position=i, mass=0.1 * (i + 1))
            for i in range(16)
        ])
        self.assertFalse(dist.check_is_full_valid_TTR())

        dist = DistributionalValue(dirac_deltas=[
            DiracDelta(position=i, mass=0.1)
            for i in range(16)
        ])
        self.assertTrue(dist.check_is_full_valid_TTR())

        gaussianMotherTTR16Positions = [
            -2.2194097942437231, -1.5678879053053274, -1.1997100902860450,
            -0.9205473016275229, -0.6859608829556935, -0.4772650338604341,
            -0.2817093825097764, -0.0931705533484249, 0.0931705533484249,
            0.2817093825097764, 0.4772650338604341, 0.6859608829556935,
            0.9205473016275229, 1.1997100902860450, 1.5678879053053274,
            2.2194097942437231
        ]
        gaussianMotherTTR16Probabilities = [
            0.0339789420851602, 0.0520280112620429, 0.0601091352703015,
            0.0663526532241763, 0.0686569819948156, 0.0714941307381180,
            0.0732557829230397, 0.0741243625023456, 0.0741243625023456,
            0.0732557829230397, 0.0714941307381180, 0.0686569819948156,
            0.0663526532241763, 0.0601091352703015, 0.0520280112620429,
            0.0339789420851602
        ]
        dist = DistributionalValue(dirac_deltas=[
            DiracDelta(position=position, mass=mass)
            for position, mass in zip(
                gaussianMotherTTR16Positions,
                gaussianMotherTTR16Probabilities
            )
        ])
        self.assertTrue(dist.check_is_full_valid_TTR())

        exponentialMotherTTR16Positions = [
            0.0463101200943282, 0.1434539976121732, 0.2473298485428934,
            0.3589406008993048, 0.4795601941139194, 0.6107690295987392,
            0.7545735507649249, 0.9136511262522184, 1.0940956889209921,
            1.3020972163662070, 1.5437301020205228, 1.8320038271142292,
            2.1944919248629803, 2.6809449590891815, 3.4180232931306736,
            5.0000000000000000
        ]
        exponentialMotherTTR16Probabilities = [
            0.0898043375672398, 0.0869428363237056, 0.0839866334859530,
            0.0809192993370894, 0.0777683023008333, 0.0744401048312553,
            0.0709620923605228, 0.0672969526219585, 0.0650216515596327,
            0.0606655024127962, 0.0559943437573481, 0.0508626602050525,
            0.0462377199658062, 0.0393104949029426, 0.0314714294791298,
            0.0183156388887342
        ]
        dist = DistributionalValue(dirac_deltas=[
            DiracDelta(position=position, mass=mass)
            for position, mass in zip(
                exponentialMotherTTR16Positions,
                exponentialMotherTTR16Probabilities
            )
        ])
        self.assertTrue(dist.check_is_full_valid_TTR())

        laplaceMotherTTR16Positions = [
            -4.0000000000000000, -2.4180232931306736, -1.6809449590891815,
            -1.1944919248629803, -0.8320038271142292, -0.5437301020205228,
            -0.3020972163662070, -0.0940956889209921, 0.0940956889209921,
            0.3020972163662070, 0.5437301020205228, 0.8320038271142292,
            1.1944919248629803, 1.6809449590891815, 2.4180232931306736,
            4.0000000000000000
        ]
        laplaceMotherTTR16Probabilities = [
            0.0248935341839320, 0.0427741074343744, 0.0534285019812003,
            0.0628435769862145, 0.0691295224912407, 0.0761042035660443,
            0.0824529664115212, 0.0883735869454727, 0.0883735869454727,
            0.0824529664115212, 0.0761042035660443, 0.0691295224912407,
            0.0628435769862145, 0.0534285019812003, 0.0427741074343744,
            0.0248935341839320
        ]
        dist = DistributionalValue(dirac_deltas=[
            DiracDelta(position=position, mass=mass)
            for position, mass in zip(
                laplaceMotherTTR16Positions,
                laplaceMotherTTR16Probabilities
            )
        ])
        self.assertTrue(dist.check_is_full_valid_TTR())

        logisticMotherTTR16Positions = [
            -4.5562387049543025, -2.9418421436602556, -2.1587856510880987,
            -1.6144447065491812, -1.1828706566397532, -0.8138506279227880,
            -0.4770709791304498, -0.1572607957522052, 0.1572607957522052,
            0.4770709791304498, 0.8138506279227880, 1.1828706566397532,
            1.6144447065491812, 2.1587856510880987, 2.9418421436602556,
            4.5562387049543025
        ]
        logisticMotherTTR16Probabilities = [
            0.0281433474818693, 0.0475738959372374, 0.0580005680032459,
            0.0662821885776473, 0.0702935070589577, 0.0743920703246041,
            0.0770073268795160, 0.0783070957369223, 0.0783070957369223,
            0.0770073268795160, 0.0743920703246041, 0.0702935070589577,
            0.0662821885776473, 0.0580005680032459, 0.0475738959372374,
            0.0281433474818693
        ]
        dist = DistributionalValue(dirac_deltas=[
            DiracDelta(position=position, mass=mass)
            for position, mass in zip(
                logisticMotherTTR16Positions,
                logisticMotherTTR16Probabilities
            )
        ])
        self.assertTrue(dist.check_is_full_valid_TTR())

        gumbel1MotherTTR16Positions = [
            -1.3722373849766128, -0.9285915299813774, -0.6394795296057504,
            -0.3977265029274728, -0.1773165596233781, 0.0336306411453416,
            0.2453949715783402, 0.4637578398478052, 0.6941280516995109,
            0.9432140395136739, 1.2205265612985903, 1.5396243403711020,
            1.9288304844468449, 2.4378800713555984, 3.1930831163570588,
            4.7879260278453658
        ]
        gumbel1MotherTTR16Probabilities = [
            0.0476295697963972, 0.0667881624671499, 0.0730630564420902,
            0.0772153676471943, 0.0768815694664325, 0.0772987007741402,
            0.0765686001679291, 0.0749309749136895, 0.0721865495667472,
            0.0689189559863393, 0.0648623956919026, 0.0599040616210838,
            0.0552629494040690, 0.0475407218497456, 0.0384337406899792,
            0.0225146235151103
        ]
        dist = DistributionalValue(dirac_deltas=[
            DiracDelta(position=position, mass=mass)
            for position, mass in zip(
                gumbel1MotherTTR16Positions,
                gumbel1MotherTTR16Probabilities
            )
        ])
        self.assertTrue(dist.check_is_full_valid_TTR())


if __name__ == "__main__":
    unittest.main()
