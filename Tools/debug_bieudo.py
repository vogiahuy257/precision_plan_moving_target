#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt


def thongKe(series: pd.Series) -> dict:
    """
    Tính thống kê cho một cột thời gian.

    Input:
        series: pd.Series

    Logic:
        - Ép kiểu numeric
        - Bỏ NaN
        - Tính mean, p95, max

    Output:
        dict
    """
    s = pd.to_numeric(series, errors="coerce").dropna()

    if s.empty:
        return {
            "mean": 0.0,
            "p95": 0.0,
            "max": 0.0
        }

    return {
        "mean": float(s.mean()),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max())
    }


def layCotHopLe(df: pd.DataFrame, columnName: str) -> pd.Series:
    """
    Lấy cột thời gian hợp lệ từ DataFrame.

    Input:
        df: pd.DataFrame
        columnName: str

    Logic:
        - Kiểm tra cột có tồn tại không
        - Ép kiểu numeric
        - Chuyển giá trị âm thành NaN vì đây thường là mẫu invalid

    Output:
        pd.Series
    """
    if columnName not in df.columns:
        raise ValueError(f"Thiếu cột bắt buộc: {columnName}")

    series = pd.to_numeric(df[columnName], errors="coerce")
    series = series.where(series >= 0.0)
    return series


def taoStageDf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo DataFrame chứa các stage latency chính để phân tích.

    Input:
        df: pd.DataFrame

    Logic:
        Lấy các cột delay từ schema mới của timing collector.

    Output:
        pd.DataFrame
    """
    stageDf = pd.DataFrame()

    stageDf["aruco_processing"] = layCotHopLe(df, "aruco_processing_dt")
    stageDf["aruco_send"] = layCotHopLe(df, "aruco_send_dt")
    stageDf["aruco_to_kalman"] = layCotHopLe(df, "dt_aruco_to_kalman")
    stageDf["kalman_processing"] = layCotHopLe(df, "kalman_processing_dt")
    stageDf["kalman_send"] = layCotHopLe(df, "kalman_send_dt")
    stageDf["kalman_to_pl_pose"] = layCotHopLe(df, "dt_kalman_to_pl_pose")
    stageDf["pl_pose_wait"] = layCotHopLe(df, "pl_pose_wait_dt")
    stageDf["pl_control_processing"] = layCotHopLe(df, "pl_control_processing_dt")
    stageDf["pl_send_cmd"] = layCotHopLe(df, "pl_send_cmd_dt")
    stageDf["total_image_to_cmd"] = layCotHopLe(df, "dt_total_image_to_cmd")

    return stageDf


def taoBangThongKe(stageDf: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng thống kê mean/p95/max cho từng stage.

    Input:
        stageDf: pd.DataFrame

    Logic:
        Tính thống kê cho từng cột.

    Output:
        pd.DataFrame
    """
    rows = []
    for columnName in stageDf.columns:
        stats = thongKe(stageDf[columnName])
        rows.append({
            "stage": columnName,
            "mean_ms": stats["mean"] * 1000.0,
            "p95_ms": stats["p95"] * 1000.0,
            "max_ms": stats["max"] * 1000.0
        })

    statsDf = (
        pd.DataFrame(rows)
        .sort_values(by="mean_ms", ascending=False)
        .reset_index(drop=True)
    )

    return statsDf


def veBieuDoBottleneck(statsDf: pd.DataFrame) -> None:
    """
    Vẽ biểu đồ bottleneck dạng barh.

    Input:
        statsDf: pd.DataFrame

    Logic:
        - Bar: mean
        - Dot: p95

    Output:
        Không có.
    """
    plt.figure(figsize=(13, 7))

    yPos = list(range(len(statsDf)))
    meanArray = statsDf["mean_ms"].to_numpy()
    p95Array = statsDf["p95_ms"].to_numpy()
    stageLabels = statsDf["stage"].tolist()

    plt.barh(yPos, meanArray)
    plt.plot(p95Array, yPos, "o")

    plt.yticks(yPos, stageLabels)
    plt.xlabel("Độ trễ (ms)")
    plt.title("Bottleneck độ trễ toàn pipeline theo từng stage")

    for i, row in statsDf.iterrows():
        plt.text(
            float(row["mean_ms"]) + 0.5,
            i,
            f"mean={row['mean_ms']:.2f} ms | p95={row['p95_ms']:.2f} ms | max={row['max_ms']:.2f} ms",
            va="center"
        )

    plt.gca().invert_yaxis()
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.show()


def veBieuDoCongDonTongHe(stageDf: pd.DataFrame) -> pd.DataFrame:
    """
    Vẽ biểu đồ đường cộng dồn các stage để so sánh với total_image_to_cmd.

    Input:
        stageDf: pd.DataFrame

    Logic:
        - Lấy mean từng stage thành phần
        - Cộng dồn theo đúng thứ tự pipeline
        - So sánh với mean của total_image_to_cmd

    Output:
        pd.DataFrame bảng dùng để debug cộng dồn
    """
    orderedStageColumns = [
        "aruco_processing",
        "aruco_send",
        "aruco_to_kalman",
        "kalman_processing",
        "kalman_send",
        "kalman_to_pl_pose",
        "pl_pose_wait",
        "pl_control_processing",
        "pl_send_cmd"
    ]

    componentMeanMs = []
    for columnName in orderedStageColumns:
        stats = thongKe(stageDf[columnName])
        componentMeanMs.append(stats["mean"] * 1000.0)

    cumulativeMeanMs = []
    runningSum = 0.0
    for value in componentMeanMs:
        runningSum += value
        cumulativeMeanMs.append(runningSum)

    totalStats = thongKe(stageDf["total_image_to_cmd"])
    totalMeanMs = totalStats["mean"] * 1000.0
    totalP95Ms = totalStats["p95"] * 1000.0
    totalMaxMs = totalStats["max"] * 1000.0

    compareDf = pd.DataFrame({
        "stage": orderedStageColumns,
        "component_mean_ms": componentMeanMs,
        "cumulative_mean_ms": cumulativeMeanMs
    })

    chenhLechMs = cumulativeMeanMs[-1] - totalMeanMs
    chenhLechPct = 0.0
    if totalMeanMs > 1e-9:
        chenhLechPct = 100.0 * chenhLechMs / totalMeanMs

    plt.figure(figsize=(14, 7))

    xPos = list(range(len(compareDf)))

    plt.plot(
        xPos,
        compareDf["component_mean_ms"].to_numpy(),
        marker="o",
        label="Mean từng stage"
    )

    plt.plot(
        xPos,
        compareDf["cumulative_mean_ms"].to_numpy(),
        marker="s",
        label="Mean cộng dồn"
    )

    plt.axhline(
        y=totalMeanMs,
        linestyle="--",
        label=f"Mean total_image_to_cmd = {totalMeanMs:.2f} ms"
    )

    plt.xticks(xPos, compareDf["stage"], rotation=30, ha="right")
    plt.ylabel("Độ trễ (ms)")
    plt.title("So sánh cộng dồn từng stage với tổng độ trễ toàn hệ")

    for i, row in compareDf.iterrows():
        plt.text(
            i,
            row["cumulative_mean_ms"] + 0.5,
            f"{row['cumulative_mean_ms']:.2f}",
            ha="center"
        )

    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n===== KIEM TRA CONG DON PIPELINE =====")
    print(compareDf.to_string(index=False))
    print(f"\nMean total_image_to_cmd: {totalMeanMs:.3f} ms")
    print(f"P95 total_image_to_cmd : {totalP95Ms:.3f} ms")
    print(f"Max total_image_to_cmd : {totalMaxMs:.3f} ms")
    print(f"Tong cong don cuoi cung : {cumulativeMeanMs[-1]:.3f} ms")
    print(f"Chenh lech             : {chenhLechMs:.3f} ms ({chenhLechPct:.2f} %)")

    return compareDf


def veBieuDoBottleneckDt(csvPath: str) -> pd.DataFrame:
    """
    Hàm chính:
    - đọc CSV
    - tạo stage dataframe
    - vẽ bottleneck
    - vẽ biểu đồ cộng dồn
    - in bảng thống kê

    Input:
        csvPath: str

    Output:
        pd.DataFrame thống kê
    """
    csvPath = os.path.expanduser(csvPath)
    df = pd.read_csv(csvPath)

    stageDf = taoStageDf(df)
    statsDf = taoBangThongKe(stageDf)

    veBieuDoBottleneck(statsDf)
    veBieuDoCongDonTongHe(stageDf)

    print("\n===== BOTTLENECK DT TOAN PIPELINE =====")
    print(statsDf.to_string(index=False))

    return statsDf


if __name__ == "__main__":
    csvPath = "~/precision_landing/tracktor-beam/pipeline_timing_nodes.csv"
    veBieuDoBottleneckDt(csvPath)