import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


class PrecisionLandPlotter:
    """
    Class ve bieu do danh gia controller PrecisionLand tu file CSV debug.

    Input:
        csvPath: duong dan file CSV log controller
        outputDir: thu muc luu anh bieu do

    Logic:
        - Doc CSV
        - Kiem tra cot can thiet
        - Ve tung bieu do rieng de de review
        - Luu anh ra file PNG

    Output:
        Cac file PNG trong thu muc output
    """

    def __init__(self, csvPath: str, outputDir: str | None = None):
        self.csvPath = Path(csvPath)
        self.outputDir = Path(outputDir) if outputDir else self.csvPath.parent / f"{self.csvPath.stem}_plots"
        self.dataFrame: pd.DataFrame | None = None

    def loadData(self) -> None:
        """
        Doc file CSV vao pandas DataFrame.

        Input:
            khong co

        Logic:
            - Doc CSV
            - Sap xep theo time
            - Tao cot thoi gian tuong doi timeRel de de ve

        Output:
            self.dataFrame duoc nap du lieu
        """
        if not self.csvPath.exists():
            raise FileNotFoundError(f"Khong tim thay file CSV: {self.csvPath}")

        self.dataFrame = pd.read_csv(self.csvPath)

        if "time" not in self.dataFrame.columns:
            raise ValueError("CSV khong co cot 'time'")

        self.dataFrame = self.dataFrame.sort_values("time").reset_index(drop=True)
        self.dataFrame["timeRel"] = self.dataFrame["time"] - self.dataFrame["time"].iloc[0]

        self.outputDir.mkdir(parents=True, exist_ok=True)

    def requireColumns(self, columns: list[str]) -> None:
        """
        Kiem tra cac cot bat buoc co ton tai.

        Input:
            columns: danh sach ten cot

        Logic:
            - Neu thieu cot thi bao loi ro rang

        Output:
            Khong co
        """
        assert self.dataFrame is not None
        missingColumns = [column for column in columns if column not in self.dataFrame.columns]
        if missingColumns:
            raise ValueError(f"CSV thieu cot: {missingColumns}")

    def saveCurrentFigure(self, fileName: str, title: str) -> None:
        """
        Luu figure hien tai ra file.

        Input:
            fileName: ten file png
            title: tieu de bieu do

        Logic:
            - Dat title
            - Bat grid
            - Tight layout
            - Luu file

        Output:
            File png trong outputDir
        """
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.outputDir / fileName, dpi=150)
        plt.close()

    def plotRawVsFilteredKalman(self) -> None:
        """
        Ve raw target va filtered target sau Kalman tren tung truc X Y Z.

        Input:
            khong co

        Logic:
            - Ve moi truc 1 hinh rieng
            - So sanh raw, est va prediction

        Output:
            3 file png
        """
        assert self.dataFrame is not None
        self.requireColumns([
            "timeRel",
            "target_raw_x", "target_raw_y", "target_raw_z",
            "target_est_x", "target_est_y", "target_est_z",
            "target_pred_x", "target_pred_y", "target_pred_z",
        ])

        axisList = ["x", "y", "z"]
        for axisName in axisList:
            plt.figure(figsize=(10, 5))
            plt.plot(self.dataFrame["timeRel"], self.dataFrame[f"target_raw_{axisName}"], label=f"target_raw_{axisName}")
            plt.plot(self.dataFrame["timeRel"], self.dataFrame[f"target_est_{axisName}"], label=f"target_est_{axisName}")
            plt.plot(self.dataFrame["timeRel"], self.dataFrame[f"target_pred_{axisName}"], label=f"target_pred_{axisName}")
            plt.xlabel("Time [s]")
            plt.ylabel(f"{axisName.upper()} [m]")
            plt.legend()
            self.saveCurrentFigure(
                f"raw_vs_filtered_kalman_{axisName}.png",
                f"Raw vs Filtered vs Predicted Target ({axisName.upper()})",
            )

    def plotTargetVelocityVsFinalSpVsDroneVelocity(self) -> None:
        """
        Ve target velocity estimate, final setpoint va drone velocity tren X Y.

        Input:
            khong co

        Logic:
            - Ve truc X va Y rieng
            - So sanh kha nang dap ung cua drone voi target

        Output:
            2 file png
        """
        assert self.dataFrame is not None
        self.requireColumns([
            "timeRel",
            "target_vel_x", "target_vel_y",
            "final_sp_x", "final_sp_y",
            "drone_vel_x", "drone_vel_y",
        ])

        axisList = ["x", "y"]
        for axisName in axisList:
            plt.figure(figsize=(10, 5))
            plt.plot(self.dataFrame["timeRel"], self.dataFrame[f"target_vel_{axisName}"], label=f"target_vel_{axisName}")
            plt.plot(self.dataFrame["timeRel"], self.dataFrame[f"final_sp_{axisName}"], label=f"final_sp_{axisName}")
            plt.plot(self.dataFrame["timeRel"], self.dataFrame[f"drone_vel_{axisName}"], label=f"drone_vel_{axisName}")
            plt.xlabel("Time [s]")
            plt.ylabel(f"Velocity {axisName.upper()} [m/s]")
            plt.legend()
            self.saveCurrentFigure(
                f"target_vel_vs_final_sp_vs_drone_vel_{axisName}.png",
                f"Target Velocity vs Final SP vs Drone Velocity ({axisName.upper()})",
            )

    def plotPidVsFfVsFinalSp(self) -> None:
        """
        Ve PID output, feedforward va final setpoint tren X Y.

        Input:
            khong co

        Logic:
            - Tach tung truc de de xem PID va FF dong gop bao nhieu

        Output:
            2 file png
        """
        assert self.dataFrame is not None
        self.requireColumns([
            "timeRel",
            "pid_out_x", "pid_out_y",
            "ff_x", "ff_y",
            "final_sp_x", "final_sp_y",
        ])

        axisList = ["x", "y"]
        for axisName in axisList:
            plt.figure(figsize=(10, 5))
            plt.plot(self.dataFrame["timeRel"], self.dataFrame[f"pid_out_{axisName}"], label=f"pid_out_{axisName}")
            plt.plot(self.dataFrame["timeRel"], self.dataFrame[f"ff_{axisName}"], label=f"ff_{axisName}")
            plt.plot(self.dataFrame["timeRel"], self.dataFrame[f"final_sp_{axisName}"], label=f"final_sp_{axisName}")
            plt.xlabel("Time [s]")
            plt.ylabel(f"Velocity Command {axisName.upper()} [m/s]")
            plt.legend()
            self.saveCurrentFigure(
                f"pid_vs_ff_vs_final_sp_{axisName}.png",
                f"PID vs FF vs Final SP ({axisName.upper()})",
            )

    def plotErrorVsFutureError(self) -> None:
        """
        Ve error hien tai va future error.

        Input:
            khong co

        Logic:
            - Ve truc X
            - Ve truc Y
            - Ve norm tong hop

        Output:
            3 file png
        """
        assert self.dataFrame is not None
        self.requireColumns([
            "timeRel",
            "error_x", "error_y",
            "future_error_x", "future_error_y",
            "error_xy_norm", "future_error_xy_norm",
        ])

        for axisName in ["x", "y"]:
            plt.figure(figsize=(10, 5))
            plt.plot(self.dataFrame["timeRel"], self.dataFrame[f"error_{axisName}"], label=f"error_{axisName}")
            plt.plot(self.dataFrame["timeRel"], self.dataFrame[f"future_error_{axisName}"], label=f"future_error_{axisName}")
            plt.xlabel("Time [s]")
            plt.ylabel(f"Error {axisName.upper()} [m]")
            plt.legend()
            self.saveCurrentFigure(
                f"error_vs_future_error_{axisName}.png",
                f"Error vs Future Error ({axisName.upper()})",
            )

        plt.figure(figsize=(10, 5))
        plt.plot(self.dataFrame["timeRel"], self.dataFrame["error_xy_norm"], label="error_xy_norm")
        plt.plot(self.dataFrame["timeRel"], self.dataFrame["future_error_xy_norm"], label="future_error_xy_norm")
        plt.xlabel("Time [s]")
        plt.ylabel("Norm Error [m]")
        plt.legend()
        self.saveCurrentFigure(
            "error_norm_vs_future_error_norm.png",
            "Error Norm vs Future Error Norm",
        )

    def plotTrajectory3D(self) -> None:
        """
        Ve quy dao 3D cua drone, target estimate va target prediction.

        Input:
            khong co

        Logic:
            - Ve 3 duong trong khong gian 3D
            - De nhin tong quan qua trinh bay bam muc tieu

        Output:
            1 file png
        """
        assert self.dataFrame is not None
        self.requireColumns([
            "drone_pos_x", "drone_pos_y", "drone_pos_z",
            "target_est_x", "target_est_y", "target_est_z",
            "target_pred_x", "target_pred_y", "target_pred_z",
        ])

        figure = plt.figure(figsize=(10, 8))
        axis3D = figure.add_subplot(111, projection="3d")

        axis3D.plot(
            self.dataFrame["drone_pos_x"],
            self.dataFrame["drone_pos_y"],
            self.dataFrame["drone_pos_z"],
            label="drone_pos",
        )
        axis3D.plot(
            self.dataFrame["target_est_x"],
            self.dataFrame["target_est_y"],
            self.dataFrame["target_est_z"],
            label="target_est",
        )
        axis3D.plot(
            self.dataFrame["target_pred_x"],
            self.dataFrame["target_pred_y"],
            self.dataFrame["target_pred_z"],
            label="target_pred",
        )

        axis3D.set_xlabel("X [m]")
        axis3D.set_ylabel("Y [m]")
        axis3D.set_zlabel("Z [m]")
        axis3D.legend()
        self.saveCurrentFigure("trajectory_3d.png", "3D Trajectory: Drone / Target Est / Target Pred")

    def plotAltitudeAndDisarm(self) -> None:
        """
        Ve do cao, dist_bottom, trang thai disarm va land detect.

        Input:
            khong co

        Logic:
            - Huu ich de xem pha cuoi va logic tat dong co

        Output:
            1 file png
        """
        assert self.dataFrame is not None
        self.requireColumns([
            "timeRel",
            "altitude_abs",
            "dist_bottom",
            "should_disarm",
            "land_detected",
            "final_sp_z",
        ])

        plt.figure(figsize=(10, 5))
        plt.plot(self.dataFrame["timeRel"], self.dataFrame["altitude_abs"], label="altitude_abs")
        plt.plot(self.dataFrame["timeRel"], self.dataFrame["dist_bottom"], label="dist_bottom")
        plt.plot(self.dataFrame["timeRel"], self.dataFrame["final_sp_z"], label="final_sp_z")
        plt.plot(self.dataFrame["timeRel"], self.dataFrame["should_disarm"], label="should_disarm")
        plt.plot(self.dataFrame["timeRel"], self.dataFrame["land_detected"], label="land_detected")
        plt.xlabel("Time [s]")
        plt.ylabel("Value")
        plt.legend()
        self.saveCurrentFigure(
            "altitude_disarm_landing.png",
            "Altitude / DistBottom / Final SP Z / Disarm / LandDetected",
        )

    def plotAll(self) -> None:
        """
        Ve toan bo bieu do.

        Input:
            khong co

        Logic:
            - Goi lan luot tung ham ve
            - In ra thu muc ket qua

        Output:
            Tat ca file png trong outputDir
        """
        self.loadData()
        self.plotRawVsFilteredKalman()
        self.plotTargetVelocityVsFinalSpVsDroneVelocity()
        self.plotPidVsFfVsFinalSp()
        self.plotErrorVsFutureError()
        self.plotTrajectory3D()
        self.plotAltitudeAndDisarm()

        print(f"Da luu bieu do tai: {self.outputDir}")


def parseArguments() -> argparse.Namespace:
    """
    Parse tham so dong lenh.

    Input:
        khong co

    Logic:
        - Nhan duong dan csv
        - Nhan tuy chon output dir

    Output:
        argparse.Namespace
    """
    argumentParser = argparse.ArgumentParser(description="Ve bieu do tu CSV debug PrecisionLand")
    argumentParser.add_argument("csvPath", type=str, help="Duong dan file CSV controller log")
    argumentParser.add_argument(
        "--outputDir",
        type=str,
        default=None,
        help="Thu muc luu anh ket qua",
    )
    return argumentParser.parse_args()


def main() -> None:
    """
    Ham main.

    Input:
        khong co

    Logic:
        - Parse argument
        - Tao plotter
        - Ve tat ca bieu do

    Output:
        Cac file png trong outputDir
    """
    arguments = parseArguments()
    plotter = PrecisionLandPlotter(arguments.csvPath, arguments.outputDir)
    plotter.plotAll()


if __name__ == "__main__":
    main()