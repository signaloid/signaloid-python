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

import sqlite3
import unittest
from sqlite3 import Connection

from signaloid.benchmarking.equivalent_mc.load import _load_mc_with_con


def _build_test_mc_db() -> Connection:
    """Create an in-memory SQLite database with MonteCarlo test data."""
    db = sqlite3.connect(":memory:")
    cursor = db.cursor()
    cursor.execute("""CREATE TABLE MonteCarlo (ValueId TEXT,
    Expression_Name TEXT,
    Expression_Subprogram TEXT,
    Expression_DeclarationFileName TEXT,
    Expression_DeclarationLineNumber INTEGER,
    Execution_Info_Table_ID INTEGER,
    EmulatedCPU_PC INTEGER,
    Assignment_Index INTEGER,
    MC_Id INTEGER,
    Particle_Value REAL );
    """)
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid1','var1','main','main.c',84,1,134250000,0,0,17.0);")
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid11','var1','main','main.c',84,1,134250000,1,0,1700.0);")
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid2','var2','main','main.c',85,1,134253000,0,0,-100.0);")
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid3','var3','main','main.c',86,1,134254000,0,0,201);")
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid1','var1','main','main.c',84,2,134250000,0,1,18.0);")
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid11','var1','main','main.c',84,2,134250000,1,0,1800.0);")
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid2','var2','main','main.c',85,2,134253000,0,1,-101.0);")
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid3','var3','main','main.c',86,2,134254000,0,1,202);")
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid1','var1','main','main.c',84,3,134250000,0,2,19.0);")
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid11','var1','main','main.c',84,3,134250000,1,0,1900.0);")
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid2','var2','main','main.c',85,3,134253000,0,2,-102.0);")
    cursor.execute("INSERT INTO MonteCarlo VALUES\
            ('valueid3','var3','main','main.c',86,3,134254000,0,2,203);")
    # Row layout mirrors the emulator's distributional-value trace output.
    return db


class TestLoadMcWithCon(unittest.TestCase):
    """_load_mc_with_con correctly loads per-variable MC data from an in-memory DB."""

    db: Connection

    @classmethod
    def setUpClass(cls) -> None:
        cls.db = _build_test_mc_db()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.db.close()

    def test__load_mc_with_con_vars(self) -> None:
        var2_data = _load_mc_with_con(self.db, "MonteCarlo", "var2")

        self.assertEqual(var2_data.representation_type, "MonteCarlo")
        self.assertEqual(var2_data.dv.UR_order, 3)
        self.assertTrue(all([v < 99 for v in var2_data.dv.positions]))
        self.assertEqual(len(var2_data.dv.masses), 3)

        var3_data = _load_mc_with_con(self.db, "MonteCarlo", "var3")

        self.assertEqual(var3_data.representation_type, "MonteCarlo")
        self.assertEqual(var3_data.dv.UR_order, 3)
        self.assertTrue(all([v > 200 for v in var3_data.dv.positions]))
        self.assertEqual(len(var3_data.dv.masses), 3)

        var1_data = _load_mc_with_con(self.db, "MonteCarlo", "var1")

        self.assertEqual(var1_data.representation_type, "MonteCarlo")
        self.assertEqual(var1_data.dv.UR_order, 3)
        # Because of larger assignment index, _load_mc_with_con of var1 should get the
        # bigger three values from the fixture
        self.assertTrue(all([not (15 < v < 25) for v in var1_data.dv.positions]))
        self.assertEqual(len(var1_data.dv.masses), 3)


if __name__ == "__main__":
    unittest.main()
