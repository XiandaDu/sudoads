import os
import pandas as pd


def analyze_parquet(file_path):
    """
    分析单个 Parquet 文件的详细元数据。
    """
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    print(f"\n=== File: {file_path} ===")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {df.shape[1]}")
    print("Detailed column analysis:")

    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        null_rate = series.isnull().mean()
        non_null_count = series.notnull().sum()
        unique_count = series.nunique(dropna=True)

        summary = {
            "type": dtype,
            "null_rate": f"{null_rate:.2%}",
            "non_null_count": non_null_count,
            "unique_count": unique_count,
        }

        # 可比较类型，尝试最大/最小值
        if pd.api.types.is_numeric_dtype(
            series
        ) or pd.api.types.is_datetime64_any_dtype(series):
            summary["min"] = series.min()
            summary["max"] = series.max()
        else:
            summary["min"] = summary["max"] = "N/A"

        # 最常见值（top）
        if non_null_count > 0:
            top_value = series.mode().iloc[0] if not series.mode().empty else None
            top_freq = (
                series.value_counts().iloc[0]
                if not series.value_counts().empty
                else None
            )
            summary["top"] = top_value
            summary["top_freq"] = top_freq
        else:
            summary["top"] = summary["top_freq"] = "N/A"

        print(f"\n- Column: {col}")
        for k, v in summary.items():
            print(f"    {k}: {v}")


def analyze_directory(dir_path):
    """
    遍历文件夹，分析所有 Parquet 文件
    """
    print(f"Scanning directory: {dir_path}")
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                analyze_parquet(file_path)


# ========== 用这个方式控制运行 ==========
if __name__ == "__main__":
    # 模式 1：分析单个文件
    analyze_parquet(
        "./avenir-hku-web/kline_data/train_data/1INCHUSDT.parquet"
    )  # 替换为你的路径

    # 模式 2：分析文件夹
    # analyze_directory("./avenir-hku-web/kline_data")  # 替换为你的路径
