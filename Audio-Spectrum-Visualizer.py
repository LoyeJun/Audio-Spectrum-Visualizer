import sys, os, math, subprocess, platform, shutil, threading, requests, re
from pathlib import Path
from time import perf_counter
import logging
import traceback

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FFMPEG_MIRROR = {
    "Windows": "https://mirror.ghproxy.com/https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n6.1-latest-win64-gpl-6.1.zip",
    "Darwin":  "https://mirror.ghproxy.com/https://evermeet.cx/ffmpeg/ffmpeg-7.0.zip",
    "Linux":   "https://mirror.ghproxy.com/https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
}
CHUNK = 1024 * 1024


# ------------  FFmpeg 下载 ------------
class FFmpegDownloader(QObject):
    finished = pyqtSignal(Path)
    error = pyqtSignal(str)

    def download_ffmpeg(self):
        try:
            cache_dir = Path.cwd() / "_ffmpeg_cache"
            cache_dir.mkdir(exist_ok=True)
            exe_name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
            exe = cache_dir / exe_name
            if exe.exists():
                self.finished.emit(exe)
                return

            url = FFMPEG_MIRROR[platform.system()]
            arc = cache_dir / ("ffmpeg" + Path(url).suffix)
            logger.info(f"下载FFmpeg: {url}")
            with requests.get(url, stream=True, timeout=20) as r:
                r.raise_for_status()
                with open(arc, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        f.write(chunk)

            logger.info("解压FFmpeg...")
            if arc.suffix == ".zip":
                import zipfile
                with zipfile.ZipFile(arc) as z:
                    for name in z.namelist():
                        if name.endswith(exe_name):
                            z.extract(name, cache_dir)
                            extracted = cache_dir / name
                            extracted.rename(exe)
                            break
            else:
                import tarfile
                with tarfile.open(arc) as tar:
                    for member in tar.getmembers():
                        if member.name.endswith(exe_name):
                            tar.extract(member, cache_dir)
                            extracted = cache_dir / member.name
                            extracted.rename(exe)
                            break
            os.chmod(exe, 0o755)
            self.finished.emit(exe)
        except Exception as e:
            logger.error(f"下载FFmpeg失败: {str(e)}")
            self.error.emit(f"下载FFmpeg失败: {str(e)}")


# ------------  编码线程 ------------
class EncodeThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    time_info = pyqtSignal(str)
    fps_info = pyqtSignal(str)
    speed_info = pyqtSignal(str)  # 新增导出倍速信号
    finished_ok = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, audio_path: Path, fps: int, bins: int, out_name: str,
                 width: int, height: int, smooth: bool):
        super().__init__()
        self.audio_path = audio_path
        self.fps = fps
        self.bins = bins
        self.out_name = out_name
        self.width = width
        self.height = height
        self.smooth = smooth
        self.exe = None
        self.downloader = FFmpegDownloader()
        self.downloader.finished.connect(self.set_ffmpeg_path)
        self.downloader.error.connect(self.error.emit)
        self.total_frames = 0
        self.ffmpeg_process = None
        self.start_t = 0

    def set_ffmpeg_path(self, path):
        self.exe = path
        self.start_encoding()

    def run(self):
        try:
            if (sys_ffmpeg := shutil.which("ffmpeg")):
                self.exe = Path(sys_ffmpeg)
                self.start_encoding()
            else:
                self.status.emit("正在下载FFmpeg...")
                threading.Thread(target=self.downloader.download_ffmpeg, daemon=True).start()
        except Exception as e:
            self.error.emit(f"初始化失败: {str(e)}")

    def start_encoding(self):
        try:
            self.status.emit("读取音频...")
            data, sr = sf.read(str(self.audio_path), always_2d=True)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            if data.shape[1] > 1:
                data = data.mean(axis=1)
            else:
                data = data.flatten()
            data = data.astype(np.float32)

            total_samples = len(data)
            frame_samples = int(round(sr / self.fps))
            self.total_frames = int(math.ceil(total_samples / frame_samples))
            pad_len = self.total_frames * frame_samples - total_samples
            if pad_len > 0:
                data = np.pad(data, (0, pad_len))

            win = np.hanning(frame_samples)
            safe_name = re.sub(r'[\\/:*?"<>|]', '_', self.out_name)
            out_path = Path.cwd() / f"{safe_name}.mp4"

            cmd = [
                str(self.exe), "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
                "-i", str(self.audio_path),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(out_path)
            ]
            if platform.system() == "Windows":
                cmd = [str(cmd[0])] + cmd[1:]
            self.ffmpeg_process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=10**8)

            threading.Thread(target=self.monitor_ffmpeg_output,
                             args=(self.ffmpeg_process,), daemon=True).start()

            # 计算全局最大幅度
            global_max = 0
            for idx in range(self.total_frames):
                if self.isInterruptionRequested():
                    return
                s = idx * frame_samples
                chunk = data[s: s + frame_samples] * win
                fft = np.fft.rfft(chunk)
                global_max = max(global_max, np.max(np.abs(fft)))
            global_max = max(global_max, 1e-6)

            self.start_t = perf_counter()
            for idx in range(self.total_frames):
                if self.isInterruptionRequested():
                    if self.ffmpeg_process and self.ffmpeg_process.stdin:
                        self.ffmpeg_process.stdin.close()
                    if self.ffmpeg_process:
                        self.ffmpeg_process.terminate()
                        self.ffmpeg_process.wait()
                    return

                s = idx * frame_samples
                chunk = data[s: s + frame_samples] * win
                fft = np.fft.rfft(chunk)
                mag = np.abs(fft)[:self.bins]

                if self.smooth:
                    window_size = max(1, int(self.bins * 0.05))
                    weights = np.ones(window_size) / window_size
                    mag = np.convolve(mag, weights, mode='same')

                mag_scaled = mag / global_max
                img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                bin_width = self.width / len(mag_scaled)

                for i, val in enumerate(mag_scaled):
                    bar_h = int(min(val, 1.0) * self.height)
                    x0 = int(i * bin_width)
                    x1 = int((i + 1) * bin_width)
                    if i == len(mag_scaled) - 1:
                        x1 = self.width
                    if bar_h > 0:
                        img[self.height - bar_h:self.height, x0:x1, 0] = int(255 * min(val, 1))
                        img[self.height - bar_h:self.height, x0:x1, 1] = int(255 * (1 - min(val, 1)))

                if self.ffmpeg_process and self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.write(img.tobytes())

                elapsed = perf_counter() - self.start_t
                progress = int((idx + 1) * 100 / self.total_frames)
                fps_real = (idx + 1) / max(elapsed, 1e-3)
                eta = (self.total_frames - idx - 1) / max(fps_real, 1e-3)
                
                # 计算导出倍速（实时帧率 / 目标帧率）
                speed_factor = fps_real / self.fps if self.fps > 0 else 0.0

                self.progress.emit(progress)
                self.status.emit(f"编码帧 {idx+1}/{self.total_frames}")
                self.time_info.emit(f"已用 {elapsed:.1f}s  |  剩余 {eta:.1f}s")
                self.fps_info.emit(f"实时帧率: {fps_real:.1f} fps")
                self.speed_info.emit(f"倍速: {speed_factor:.2f}x")  # 发射倍速信号

            if self.ffmpeg_process and self.ffmpeg_process.stdin:
                self.ffmpeg_process.stdin.close()
            if self.ffmpeg_process:
                self.ffmpeg_process.wait()
            self.status.emit("完成")
            self.finished_ok.emit()
        except Exception as e:
            logger.exception("编码过程中出错")
            self.error.emit(f"编码错误: {str(e)}\n{traceback.format_exc()}")

    def monitor_ffmpeg_output(self, process):
        frame_pattern = re.compile(r'frame=\s*(\d+)')
        while process.poll() is None:
            try:
                line = process.stdout.readline().decode('utf-8', errors='ignore')
                if not line:
                    continue
                match = frame_pattern.search(line)
                if match:
                    frame_num = int(match.group(1))
                    if self.total_frames > 0:
                        ffmpeg_progress = min(int(frame_num * 100 / self.total_frames), 100)
                        self.progress.emit(ffmpeg_progress)
            except Exception as e:
                logger.error(f"监控FFmpeg输出出错: {str(e)}")
                break


# ------------  主窗口 ------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("音频频谱可视化生成器")
        self.resize(520, 550)  # 增加窗口高度以容纳音频信息
        self.audio_path: Path | None = None
        self.thread = None
        self.build_ui()
        self.set_window_icon()

    def set_window_icon(self):
        icon = QIcon.fromTheme("applications-multimedia")
        if icon.isNull():
            icon = QIcon.fromTheme("audio-x-generic")
        self.setWindowIcon(icon)
        QApplication.setWindowIcon(icon)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)

        # --- 文件选择 ---
        file_group = QGroupBox("音频文件")
        file_layout = QVBoxLayout(file_group)
        h_file = QHBoxLayout()
        self.btn_open = QPushButton("选择音频文件")
        self.btn_open.setIcon(QIcon.fromTheme("folder-open"))
        self.btn_open.clicked.connect(self.choose_audio)
        self.lbl_audio = QLabel("未选择")
        self.lbl_audio.setWordWrap(True)
        h_file.addWidget(self.btn_open)
        h_file.addWidget(self.lbl_audio, 1)
        file_layout.addLayout(h_file)
        
        # 添加音频信息显示区域
        audio_info_layout = QFormLayout()
        self.lbl_duration = QLabel("--")
        self.lbl_sample_rate = QLabel("--")
        self.lbl_channels = QLabel("--")
        
        audio_info_layout.addRow("时长:", self.lbl_duration)
        audio_info_layout.addRow("采样率:", self.lbl_sample_rate)
        audio_info_layout.addRow("声道数:", self.lbl_channels)
        
        file_layout.addLayout(audio_info_layout)
        main_layout.addWidget(file_group)

        # --- 编码参数 ---
        group_cfg = QGroupBox("编码参数")
        form = QFormLayout(group_cfg)
        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(10, 120)
        self.spn_fps.setValue(30)
        self.spn_fps.valueChanged.connect(self.update_output_filename)
        form.addRow("目标帧率(FPS):", self.spn_fps)  # 更明确的标签

        self.spn_bins = QSpinBox()
        self.spn_bins.setRange(8, 1024)
        self.spn_bins.setValue(64)
        self.spn_bins.valueChanged.connect(self.update_output_filename)
        form.addRow("频谱条数(Bins):", self.spn_bins)

        res_layout = QHBoxLayout()
        self.cmb_res = QComboBox()
        self.cmb_res.addItems(["1920x1080", "1280x720", "854x480", "自定义"])
        self.cmb_res.setCurrentText("1280x720")
        self.cmb_res.currentTextChanged.connect(self.toggle_custom_res)
        self.cmb_res.currentTextChanged.connect(self.update_output_filename)
        res_layout.addWidget(self.cmb_res)

        self.custom_width = QSpinBox()
        self.custom_width.setRange(320, 3840)
        self.custom_width.setValue(1920)
        self.custom_width.valueChanged.connect(self.update_output_filename)

        self.custom_height = QSpinBox()
        self.custom_height.setRange(240, 2160)
        self.custom_height.setValue(1080)
        self.custom_height.valueChanged.connect(self.update_output_filename)

        self.custom_res_layout = QHBoxLayout()
        self.custom_res_layout.addWidget(QLabel("宽"))
        self.custom_res_layout.addWidget(self.custom_width)
        self.custom_res_layout.addWidget(QLabel("高"))
        self.custom_res_layout.addWidget(self.custom_height)
        self.custom_res_widget = QWidget()
        self.custom_res_widget.setLayout(self.custom_res_layout)
        self.custom_res_widget.setVisible(False)
        res_layout.addWidget(self.custom_res_widget)
        form.addRow("分辨率:", res_layout)

        self.chk_smooth = QCheckBox("启用频谱平滑")
        self.chk_smooth.setChecked(False)
        self.chk_smooth.stateChanged.connect(self.update_output_filename)
        form.addRow(self.chk_smooth)
        main_layout.addWidget(group_cfg)

        # --- 输出设置 ---
        group_out = QGroupBox("输出设置")
        v_out = QVBoxLayout(group_out)

        self.line_out = QLineEdit("my_spectrum")
        self.line_out.setValidator(QRegularExpressionValidator(QRegularExpression(r'[^\\/:*?"<>|]+')))
        v_out.addWidget(QLabel("文件名:"))
        v_out.addWidget(self.line_out)

        h_btn = QHBoxLayout()
        self.btn_start = QPushButton("开始导出")
        self.btn_start.setIcon(QIcon.fromTheme("media-playback-start"))
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_export)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setIcon(QIcon.fromTheme("process-stop"))
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_export)
        h_btn.addStretch()
        h_btn.addWidget(self.btn_start)
        h_btn.addWidget(self.btn_cancel)
        v_out.addLayout(h_btn)
        main_layout.addWidget(group_out)

        # --- 进度/状态 ---
        self.lbl_status = QLabel("空闲")
        self.lbl_status.setStyleSheet("font-weight:bold")
        main_layout.addWidget(self.lbl_status)

        # 重新设计信息显示行，添加倍速显示
        h_info = QHBoxLayout()
        
        # 时间信息
        time_widget = QWidget()
        time_layout = QVBoxLayout(time_widget)
        self.lbl_time = QLabel("已用 --s  |  剩余 --s")
        self.lbl_speed = QLabel("倍速: --x")  # 新增倍速标签
        time_layout.addWidget(self.lbl_time)
        time_layout.addWidget(self.lbl_speed)
        time_layout.setContentsMargins(0, 0, 0, 0)
        h_info.addWidget(time_widget, 1)
        
        # 帧率信息
        fps_widget = QWidget()
        fps_layout = QVBoxLayout(fps_widget)
        self.lbl_fps = QLabel("实时帧率: -- fps")
        self.lbl_target_fps = QLabel(f"目标帧率: {self.spn_fps.value()} fps")  # 显示目标帧率
        fps_layout.addWidget(self.lbl_fps)
        fps_layout.addWidget(self.lbl_target_fps)
        fps_layout.setContentsMargins(0, 0, 0, 0)
        h_info.addWidget(fps_widget)
        
        main_layout.addLayout(h_info)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        main_layout.addWidget(self.bar)
        
        # 监听目标帧率变化
        self.spn_fps.valueChanged.connect(self.update_target_fps)

    def update_target_fps(self, fps):
        """更新目标帧率显示"""
        self.lbl_target_fps.setText(f"目标帧率: {fps} fps")

    def toggle_custom_res(self, text):
        self.custom_res_widget.setVisible(text == "自定义")

    def update_output_filename(self):
        if not self.audio_path:
            return
        base = self.audio_path.stem
        fps = self.spn_fps.value()
        bins = self.spn_bins.value()
        res = f"{self.custom_width.value()}x{self.custom_height.value()}" if self.cmb_res.currentText() == "自定义" else self.cmb_res.currentText()
        smooth = "smooth" if self.chk_smooth.isChecked() else "nosmooth"
        self.line_out.setText(f"{base}_{fps}fps_{bins}bins_{res}_{smooth}")

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.requestInterruption()
            self.thread.wait(2000)
        event.accept()

    def choose_audio(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择音频", "", "音频文件 (*.wav *.flac *.mp3 *.ogg *.m4a)")
        if file:
            self.audio_path = Path(file)
            self.lbl_audio.setText(self.audio_path.name)
            self.update_output_filename()
            self.btn_start.setEnabled(True)
            
            # 读取并显示音频信息
            self.display_audio_info(self.audio_path)

    def display_audio_info(self, audio_path: Path):
        """读取并显示音频文件的元数据（已移除位深度）"""
        try:
            info = sf.info(str(audio_path))
            
            minutes, seconds = divmod(info.frames / info.samplerate, 60)
            duration = f"{int(minutes)}:{int(seconds):02d}"
            
            self.lbl_duration.setText(duration)
            self.lbl_sample_rate.setText(f"{info.samplerate} Hz")
            self.lbl_channels.setText(f"{info.channels}")
            
        except Exception as e:
            logger.error(f"读取音频信息失败: {str(e)}")
            self.lbl_duration.setText("读取失败")
            self.lbl_sample_rate.setText("读取失败")
            self.lbl_channels.setText("读取失败")

    def start_export(self):
        if not self.audio_path:
            return
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.bar.setValue(0)

        resolution = self.cmb_res.currentText()
        if resolution == "自定义":
            width = self.custom_width.value()
            height = self.custom_height.value()
        else:
            width, height = map(int, resolution.split("x"))

        self.thread = EncodeThread(
            self.audio_path,
            self.spn_fps.value(),
            self.spn_bins.value(),
            self.line_out.text().strip(),
            width,
            height,
            self.chk_smooth.isChecked()
        )
        self.thread.progress.connect(self.bar.setValue)
        self.thread.status.connect(self.lbl_status.setText)
        self.thread.time_info.connect(self.lbl_time.setText)
        self.thread.fps_info.connect(self.lbl_fps.setText)
        self.thread.speed_info.connect(self.lbl_speed.setText)  # 连接倍速信号
        self.thread.finished_ok.connect(self.on_done)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def cancel_export(self):
        if self.thread and self.thread.isRunning():
            self.thread.requestInterruption()
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("正在取消...")

    def on_done(self):
        QMessageBox.information(self, "完成", f"已生成：{self.line_out.text().strip()}.mp4")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.bar.setValue(100)
        self.lbl_fps.setText("实时帧率: -- fps")
        self.lbl_speed.setText("倍速: --x")  # 重置倍速显示

    def on_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_fps.setText("实时帧率: -- fps")
        self.lbl_speed.setText("倍速: --x")  # 重置倍速显示


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())