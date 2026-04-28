import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple


class HelipadPoseEstimator:
    """
    Ước lượng pose 3D của helipad từ 4 điểm góc marker.

    Parameters
    ----------
    obj_w_m : float
        Chiều rộng thật của marker (m).
    obj_h_m : float
        Chiều cao thật của marker (m).
    calib_path : str | Path
        Đường dẫn tới file calibration camera (.npz hoặc .yaml).
    camera_matrix : np.ndarray, optional
        Ma trận K (3×3) — nếu truyền trực tiếp, bỏ qua calib_path.
    """

    # ------------------------------------------------------------------
    # Khởi tạo
    # ------------------------------------------------------------------

    def __init__(
        self,
        obj_w_m: float,
        obj_h_m: float,
        calib_path: Optional[str | Path] = None,
        camera_matrix: Optional[np.ndarray] = None,
    ) -> None:
        # --- Object points (hệ tọa độ marker, tâm = gốc) ---
        # Thứ tự: TL → TR → BR → BL (clockwise, khớp ArUco)
        hw, hh = obj_w_m / 2.0, obj_h_m / 2.0
        self.object_points = np.array(
            [[-hw, -hh, 0.0],
             [ hw, -hh, 0.0],
             [ hw,  hh, 0.0],
             [-hw,  hh, 0.0]],
            dtype=np.float64,
        )

        # --- Camera matrix ---
        if camera_matrix is not None:
            self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        elif calib_path is not None:
            self.camera_matrix = self._load_camera_matrix(Path(calib_path))
        else:
            raise ValueError("Phải cung cấp camera_matrix hoặc calib_path.")

        # --- Recording buffer ---
        self._recording: bool = False
        self._t_buf: list[list[float]] = [[], [], []]
        self._r_buf: list[list[float]] = [[], [], []]

    # ------------------------------------------------------------------
    # Load calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _load_camera_matrix(path: Path) -> np.ndarray:
        """Load camera matrix từ file .npz hoặc .yaml."""
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file calibration: {path}")

        suffix = path.suffix.lower()

        if suffix == ".npz":
            data = np.load(path)
            if "camera_matrix" not in data:
                raise KeyError(f"File {path} không có key 'camera_matrix'.")
            return data["camera_matrix"].astype(np.float64)

        if suffix in (".yaml", ".yml"):
            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            mat = fs.getNode("camera_matrix").mat()
            fs.release()
            if mat is None:
                raise KeyError(f"File {path} không có node 'camera_matrix'.")
            return mat.astype(np.float64)

        raise ValueError(f"Định dạng không hỗ trợ: {suffix}. Dùng .npz hoặc .yaml")

    # ------------------------------------------------------------------
    # Solve PnP
    # ------------------------------------------------------------------

    def solve_pnp(
        self,
        corners_2d: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Tính pose 3D từ 4 góc marker trên ảnh.

        Parameters
        ----------
        corners_2d : np.ndarray, shape (4, 2) hoặc (4, 1, 2)
            Tọa độ pixel của 4 góc (TL, TR, BR, BL).

        Returns
        -------
        rvec_deg : np.ndarray shape (3,) — góc Rodrigues (độ), hoặc None
        tvec_m  : np.ndarray shape (3,) — vị trí (m) trong hệ camera, hoặc None
        """
        img_pts = corners_2d.reshape(4, 2).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            img_pts,
            self.camera_matrix,
            np.zeros(5),                    # ảnh đã undistort → dist = 0
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return None, None

        tvec_m  = tvec.flatten()
        rvec_deg = rvec.flatten() * (180.0 / np.pi)

        if self._recording:
            self._record_sample(tvec_m, rvec_deg)

        return rvec_deg, tvec_m

    # ------------------------------------------------------------------
    # Recording / Statistics
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Bắt đầu thu thập mẫu để tính thống kê."""
        self._recording = True
        for buf in (*self._t_buf, *self._r_buf):
            buf.clear()

    def stop_recording(self) -> None:
        """Dừng thu thập mẫu."""
        self._recording = False

    def _record_sample(self, tvec_m: np.ndarray, rvec_deg: np.ndarray) -> None:
        for i in range(3):
            self._t_buf[i].append(float(tvec_m[i]))
            self._r_buf[i].append(float(rvec_deg[i]))

    def print_statistics(self) -> None:
        """In mean ± std của tvec và rvec từ buffer hiện tại."""
        n = len(self._t_buf[0])
        if n == 0:
            print("Chưa có mẫu nào.")
            return

        print(f"\n===== Statistics ({n} samples) =====")
        for i, ax in enumerate(["x", "y", "z"]):
            t = np.array(self._t_buf[i])
            r = np.array(self._r_buf[i])
            print(f"  t{ax}  mean={np.mean(t):>9.4f}  std={np.std(t):>8.4f}  m")
            print(f"  r{ax}  mean={np.mean(r):>9.4f}  std={np.std(r):>8.4f}  deg")
        print("=" * 36 + "\n")
