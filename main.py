from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import pint


def read_file(file):
    """ファイルの読み込み"""
    try:
        lines = []
        for line in uploaded_file.getvalue().decode("utf-8").split("\n"):
            # for line in  file.readline():
            if line.strip() == "# <eof>":
                break
            lines.append(line.strip())
        return lines
        # file_contents = file.read().decode("utf-8")
        # lines = file_contents.split("\n")
        # return lines
    except Exception as e:
        st.error(f"ファイルの読み込みエラー: {e}")
        raise ValueError("ファイルの読み込みに失敗しました")


def extract_params(line: str):
    """ パラメータの抽出 """
    try:
        filename, header = line.lstrip("# <").split(">")
        header_tuple = [i.split(maxsplit=1) for i in header.split(";")]
        return {i[0].lstrip(":"): i[1] for i in header_tuple if len(i) == 2}
    except Exception as e:
        st.error(f"パラメータの抽出エラー: {e}")
        raise ValueError("パラメータの抽出に失敗しました")


def scale_value(value: float, unit: str) -> float:
    """値と単位を渡して正しい数値を返す"""
    # 単位計算
    ureg = pint.UnitRegistry()
    try:
        value_with_unit = (value * ureg[unit]).to_base_units()
        return value_with_unit.magnitude
    except pint.errors.UndefinedUnitError:
        raise ValueError(f"Unknown unit: {unit}")


def calc_freq_range(params_dict):
    """ 周波数の範囲の計算 """
    try:
        val, unit = params_dict["FREQ:CENT"].split()
        center_freq = scale_value(float(val), unit)
        val, unit = params_dict["FREQ:SPAN"].split()
        span_freq = scale_value(float(val), unit)
        start_freq = center_freq - span_freq
        stop_freq = center_freq + span_freq
        return start_freq, stop_freq
    except Exception as e:
        st.error(f"周波数の範囲の計算エラー: {e}")
        raise ValueError("周波数の範囲の計算に失敗しました")


def load_data(lines, start_freq, stop_freq):
    """データの読み込み"""
    try:
        data = np.array([line.split() for line in lines], dtype=float)
        frequencies = start_freq + data[:, 0] * \
            (stop_freq - start_freq) / (len(data) - 1)
        df = pd.DataFrame({
            "Frequency (kHz)": frequencies / 1000,
            "TRAC1": data[:, 1],
            "TRAC2": data[:, 2],
            "TRAC3": data[:, 3]
        })
        return df
    except Exception as e:
        st.error(f"データの読み込みエラー: {e}")
        raise ValueError("データの読み込みに失敗しました")


def display_data(df):
    """データの表示"""
    try:
        st.write(f"データの形状: {df.shape}")
        st.write(df.head())
    except Exception as e:
        st.error(f"データの表示エラー: {e}")


def plot_data(df):
    """ グラフの描画 """
    try:
        columns = df.columns.tolist()
        y_col = st.selectbox("Y軸の列を選択", columns, index=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["Frequency (kHz)"], df[y_col])
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel(y_col)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"グラフの描画エラー: {e}")


if __name__ == "__main__":
    # タイトルと説明の設定
    st.set_page_config(page_title="データ可視化アプリ",
                       page_icon=":bar_chart:",
                       layout="wide")
    st.title("データ可視化アプリ")
    st.write("ファイルをドラッグ&ドロップしてデータを可視化してください。")

    # ファイルのアップロード
    uploaded_file = st.file_uploader("ファイルをアップロード", type=['txt'])

    if uploaded_file is not None:
        try:
            # テキストファイル読み込み
            lines = read_file(uploaded_file)
            header, body = lines[0], lines[1:]
            # パラメータ読み込み
            params_dict = extract_params(header)
            start_freq, stop_freq = calc_freq_range(params_dict)
            # データ読み込み
            df = load_data(body, start_freq, stop_freq)
            display_data(df)
            plot_data(df)
        except ValueError as e:
            st.error(str(e))
