from main import read_file, extract_params, scale_value, calc_freq_range, load_data
import unittest
from unittest.mock import mock_open, patch
import pandas as pd


class TestApp(unittest.TestCase):

    # def test_read_file(self):
    #     """ ファイルの読み込みが正常に行われることをテスト """
    #     file_contents = "# <20240423_150330> *RST;*CLS;:BAND:RES 10 Hz;:AVER:COUNT 20;:SWE:POIN 1001;:FREQ:CENT 25.5 kHz;:FREQ:SPAN 3.5 kHz;:TRAC1:TYPE WRIT;:TRAC2:TYPE AVER;:TRAC3:TYPE MAXH;:INIT:CONT 0;:FORM REAL,32;:FORM:BORD SWAP;:INIT:IMM;\n0 -58.87 -64.29 -55.36\n1 -57.93 -64.08 -55.18\n2 -63.78 -64.03 -55.90\n"
    #     mock_file = mock_open(read_data=file_contents)
    #     with patch("app.open", mock_file, create=True):
    #         lines = read_file(mock_file.return_value)
    #         self.assertEqual(len(lines), 4)

    def test_extract_params(self):
        # パラメータの抽出が正常に行われることをテスト
        line = "# <20240423_150330> *RST;*CLS;:BAND:RES 10 Hz;:AVER:COUNT 20;:SWE:POIN 1001;:FREQ:CENT 25.5 kHz;:FREQ:SPAN 3.5 kHz;:TRAC1:TYPE WRIT;:TRAC2:TYPE AVER;:TRAC3:TYPE MAXH;:INIT:CONT 0;:FORM REAL,32;:FORM:BORD SWAP;:INIT:IMM;"

        params_dict = extract_params(line)
        self.assertEqual(params_dict["FREQ:CENT"], "25.5 kHz")
        self.assertEqual(params_dict["FREQ:SPAN"], "3.5 kHz")

    def test_scale_value(self):
        """単位変換が正常に行われることをテスト"""
        center = "25.5 kHz"
        val, unit = center.split()
        value = scale_value(float(val), unit)
        self.assertAlmostEqual(value, 25500.0, places=1)

    def test_calc_freq_range(self):
        """周波数の範囲の計算が正常に行われることをテスト"""
        params_dict = {"FREQ:CENT": "25.5 kHz", "FREQ:SPAN": "3.5 kHz"}
        start_freq, stop_freq = calc_freq_range(params_dict)
        self.assertAlmostEqual(start_freq, 22000.00, places=2)
        self.assertAlmostEqual(stop_freq, 29000.00, places=2)

    def test_load_data(self):
        # データの読み込みが正常に行われることをテスト
        lines = [
            "# <20240423_150330> *RST;*CLS;:BAND:RES 10 Hz;:AVER:COUNT 20;:SWE:POIN 1001;:FREQ:CENT 25.5 kHz;:FREQ:SPAN 3.5 kHz;:TRAC1:TYPE WRIT;:TRAC2:TYPE AVER;:TRAC3:TYPE MAXH;:INIT:CONT 0;:FORM REAL,32;:FORM:BORD SWAP;:INIT:IMM;",
            "0 -58.87 -64.29 -55.36", "1 -57.93 -64.08 -55.18",
            "2 -63.78 -64.03 -55.90"
        ]
        _, body = lines[0], lines[1:]
        start_freq, stop_freq = 23.75, 27.25
        df = load_data(body, start_freq, stop_freq)
        self.assertEqual(df.shape, (3, 4))
        self.assertTrue(pd.notnull(df["Frequency (kHz)"]).all())

    def test_display_data(self):
        # データの表示が正常に行われることをテスト
        # このテストは実行されないため、スキップします
        pass

    def test_plot_data(self):
        # グラフの描画が正常に行われることをテスト
        # このテストは実行されないため、スキップします
        pass


if __name__ == '__main__':
    unittest.main()
