from typing import Dict
import pint
import numpy as np
import polars as pl
import plotly.express as px
import streamlit as st


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


def extract_params(line: str) -> Dict[str, str]:
    """ パラメータの抽出 """
    try:
        filename, header = line.lstrip("# <").split(">")
        header_tuple = [i.split(maxsplit=1) for i in header.split(";")]
        return {i[0].lstrip(":"): i[1] for i in header_tuple if len(i) == 2}
    except Exception as e:
        st.error(f"パラメータの抽出エラー: {e}")
        raise ValueError("パラメータの抽出に失敗しました")


def scale_value(s: str) -> float:
    """値と単位を渡して正しい数値を返す"""
    # 単位計算
    ureg = pint.UnitRegistry()
    try:
        val, unit = s.split()
        value_with_unit = (float(val) * ureg[unit]).to_base_units()
        return value_with_unit.magnitude
    except pint.errors.UndefinedUnitError:
        raise ValueError(f"Unknown unit: {unit}")


def calc_freq_range(params_dict: Dict[str, str]) -> (float, float):
    """ 周波数の範囲の計算 """
    try:
        # start,stopをparamsから取得
        start_str = params_dict.get("FREQ:START")
        stop_str = params_dict.get("FREQ:STOP")
        if start_str is not None and stop_str is not None:
            return scale_value(start_str), scale_value(stop_str)
        # start, stop どちらかがNoneのとき
        # 中心周波数とスパンからstart, stopを計算
        center_str = params_dict.get("FREQ:CENT")
        span_str = params_dict.get("FREQ:SPAN")
        if center_str is not None and span_str is not None:
            center_freq = scale_value(center_str)
            span_freq = scale_value(span_str)
            start_freq = center_freq - span_freq
            stop_freq = center_freq + span_freq
            return start_freq, stop_freq
        raise KeyError("周波数の範囲の計算エラー: FREQから始まるキーが引数に存在しません")
    except Exception as e:
        st.error(f"周波数の範囲の計算エラー: {e}")
        raise ValueError("周波数の範囲の計算に失敗しました")


def load_data(lines, params: Dict[str, str]) -> pl.DataFrame:
    """データの読み込み"""
    try:
        data = np.array([line.split() for line in lines], dtype=float)
        # 周波数範囲の計算
        start_freq, stop_freq = calc_freq_range(params)
        index = start_freq + data[:, 0] * \
            (stop_freq - start_freq) / (len(data) - 1)
        # TRACで始まるキーを抽出
        trac_keys = [k for k in params.keys() if k.startswith("TRAC")]
        # data = {
        #     "TRAC1": data[:, 1],
        #     "TRAC2": data[:, 2],
        #     "TRAC3": data[:, 3]
        # },
        values = {"Frequency (Hz)": index}
        values.update({
            params[k]: data[:, i]
            for i, k in enumerate(trac_keys, start=1)
        })
        df = pl.DataFrame(values)
        # df.index.name = "Frequency (Hz)"
        return df
    except Exception as e:
        st.error(f"データの読み込みエラー: {e}")
        raise ValueError("データの読み込みに失敗しました")


def display_data(df):
    """データの表示"""
    try:
        print(df)
        st.write(f"データの形状: {df.shape}")
        st.write(df)
    except Exception as e:
        st.error(f"データの表示エラー: {e}")


def plot_data(df):
    """ グラフの描画 """
    try:
        y_col = st.selectbox("Y軸の列を選択", df.columns, index=1)
        title = "Frequency (Hz)"
        fig = px.line(df, x=df[title], y=y_col)
        fig.update_layout(title=y_col, xaxis_title=title, yaxis_title=y_col)
        st.plotly_chart(fig)
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
            # データ読み込み
            df = load_data(body, params_dict)
            plot_data(df)
            display_data(df)
        except (KeyError, ValueError) as e:
            st.error(str(e))
