#   Copyright (c) 2026, Signaloid.
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

import struct
import unittest

import pandas as pd

from signaloid.benchmarking.config import CoreLibraryRepresentationTypes
from signaloid.benchmarking.equivalent_mc.load import _dist_value_from_row

# UR_type byte for Athens
_ATHENS_UR_TYPE = CoreLibraryRepresentationTypes.ATHENS.value


def _build_ux_string(
    ur_type: int,
    positions: list[float],
    raw_masses: list[int],
    particle_value: float = 0.0,
    mean: float = 0.0,
) -> str:
    """Build a minimal double-precision Ux string for testing.

    The format mirrors DistributionalValue.export(to_str=True):
      <particle>Ux<UR_type><sample_count><mean><UR_order><(pos,mass)...>
    All multi-byte fields are big-endian (STRUCT_FORMATS["str"]).
    """
    n = len(positions)
    buf = b""
    buf += struct.pack(">B", ur_type)  # UR_type  (1 byte)
    buf += struct.pack(">Q", n)  # sample_count (8 bytes, unused)
    buf += struct.pack(">d", mean)  # mean (8 bytes)
    buf += struct.pack(">I", n)  # UR_order (4 bytes)
    for pos, raw_mass in zip(positions, raw_masses, strict=True):
        buf += struct.pack(">d", pos)  # position (8 bytes, double)
        buf += struct.pack(">Q", raw_mass)  # mass (8 bytes, fixed-point)
    return f"{particle_value}Ux{buf.hex().upper()}"


def _make_row(
    ux_string: str,
    ur_order: int,
    ur_order_core_library: int,
    ur_type_str: str = "Athens",
    correlation_tracking: str = "Independent",
    value_id: str = "v0",
    particle_value: float = 0.0,
) -> pd.Series:
    """Build a pd.Series matching the columns consumed by _dist_value_from_row."""
    return pd.Series(
        {
            "Dist_Value": ux_string,
            "Expression_Name": "test_expr",
            "Expression_Subprogram": "main",
            "UR_Type": ur_type_str,
            "UR_Order": ur_order,
            "UR_Order_CoreLibrary": ur_order_core_library,
            "CorrelationTracking_Status": correlation_tracking,
            "ValueId": value_id,
            "Particle_Value": particle_value,
        }
    )


# Equal mass for N user atoms (sum = 1.0)
_FIXED_POINT_ONE = 0x8000000000000000


def _equal_mass(n: int) -> int:
    """Return the fixed-point raw_mass for equal-weight N-atom distribution."""
    return int(_FIXED_POINT_ONE / n)


def _build_uxhw_ttr_row(n: int) -> pd.Series:
    """Build a row for a real UxHw-style Athens-N distribution.

    Real UxHw reports UR_Order == N (the user-facing build size) and
    UR_Order_CoreLibrary == N+3 (it additionally counts 3 internal
    sentinel Diracs). It serialises only the N user atoms into the Ux
    string. The "+3" appears only in the column, never on the wire.
    """
    positions = [float(i) for i in range(n)]
    raw_masses = [_equal_mass(n)] * n
    ux = _build_ux_string(
        ur_type=_ATHENS_UR_TYPE,
        positions=positions,
        raw_masses=raw_masses,
    )
    return _make_row(ux, ur_order=n, ur_order_core_library=n + 3)


def _build_external_ttr_row(n: int) -> pd.Series:
    """Build a row for an external-producer Athens-N distribution.

    External producers carry exactly N atoms with no appended sentinels.
    The CSV-to-tracing-DB path sets both UR_Order and UR_Order_CoreLibrary
    to N.
    """
    positions = [float(i) for i in range(n)]
    raw_masses = [_equal_mass(n)] * n
    ux = _build_ux_string(
        ur_type=_ATHENS_UR_TYPE,
        positions=positions,
        raw_masses=raw_masses,
    )
    return _make_row(ux, ur_order=n, ur_order_core_library=n)


def _build_collapsed_scalar_ttr_row(n: int) -> pd.Series:
    """Build a row for a Athens-N scalar-like output that collapses to a
    single serialised atom (e.g. Value at Risk).

    UR_Order is still the build size N and UR_Order_CoreLibrary is N+3,
    but the Ux string carries only ONE atom. representation_size must be N
    (the build size), not 1 (parsed-atom count) and not N+3.
    """
    ux = _build_ux_string(
        ur_type=_ATHENS_UR_TYPE,
        positions=[42.0],
        raw_masses=[_FIXED_POINT_ONE],
    )
    return _make_row(ux, ur_order=n, ur_order_core_library=n + 3)


class TestDistValueFromRowRepresentationSize(unittest.TestCase):
    """representation_size is determined by UR_Order (build size), not parsed atoms."""

    def test_uxhw_ttr_representation_size_uses_build_size(self) -> None:
        """Real UxHw: UR_Order == N, UR_Order_CoreLibrary == N+3.
        representation_size must be the build size N, not N+3."""
        for n in [4, 8, 16, 32, 64]:
            with self.subTest(n=n):
                row = _build_uxhw_ttr_row(n)
                tagged = _dist_value_from_row(row)
                self.assertEqual(tagged.representation_size, n)

    def test_external_ttr_representation_size_uses_build_size(self) -> None:
        """External producer: UR_Order == UR_Order_CoreLibrary == N.
        representation_size must be N (no over-subtraction)."""
        for n in [4, 8, 16, 32, 64]:
            with self.subTest(n=n):
                row = _build_external_ttr_row(n)
                tagged = _dist_value_from_row(row)
                self.assertEqual(tagged.representation_size, n)

    def test_collapsed_scalar_ttr_representation_size_is_build_size(self) -> None:
        """A Athens-N output that collapses to a single serialised atom (e.g.
        Value at Risk) must still be labelled with the build size N and not the
        parsed-atom count (1) and not N+3. Guards both the parsed-atom-count and
        the sentinel-subtraction approaches."""
        for n in [2, 4, 8, 16]:
            with self.subTest(n=n):
                row = _build_collapsed_scalar_ttr_row(n)
                tagged = _dist_value_from_row(row)
                self.assertEqual(tagged.representation_size, n)


if __name__ == "__main__":
    unittest.main()
