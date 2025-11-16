# stopmotion_capture.py
# Stop Motion Capture Tool with Live Preview pseudo-frame
# Requirements: pip install opencv-python PyQt5 numpy

import sys, json, time
from pathlib import Path
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

APP_NAME = "StopMotion Capture"
DEFAULT_PROJECTS_ROOT = str(Path.home() / "StopMotionProjects")
LIVE_PREVIEW_LABEL = "Live Preview"


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def list_camera_indices(max_index=8):
    """Return indices of cameras that can be opened."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        if ret:
            available.append(i)
        cap.release()
    return available


def current_time_str():
    return time.strftime("%Y%m%d_%H%M%S")


class Project:
    def __init__(self, path):
        self.path = Path(path)
        ensure_dir(self.path)
        self.metafile = self.path / "project.json"
        if self.metafile.exists():
            try:
                with open(self.metafile, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception:
                self.meta = {}
        else:
            self.meta = {"name": self.path.name, "created": current_time_str()}
            self._save_meta()

    def _save_meta(self):
        with open(self.metafile, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def rename(self, new_name):
        parent = self.path.parent
        new_path = parent / new_name
        if new_path.exists():
            raise FileExistsError("Target project name already exists")
        self.path.rename(new_path)
        self.path = new_path
        self.metafile = self.path / "project.json"
        self.meta["name"] = new_name
        self._save_meta()

    def next_frame_filename(self):
        files = sorted(self.path.glob("frame_*.png"))
        if not files:
            return self.path / "frame_0001.png"
        last = files[-1].stem
        try:
            n = int(last.split("_")[1])
        except Exception:
            n = len(files)
        return self.path / f"frame_{n+1:04d}.png"

    def list_frames(self):
        return sorted(self.path.glob("frame_*.png"))

class FiftySnapSlider(QtWidgets.QSlider):
    """A slider that snaps to 50% when double-clicked."""
    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        middle = int((self.maximum() - self.minimum()) * 0.5 + self.minimum())
        self.setValue(middle)
class StopMotionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1000, 700)

        # State
        self.capture = None
        self.device_index = None
        self.live_timer = QtCore.QTimer()
        self.live_timer.timeout.connect(self.grab_frame)
        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)

        self.previous_captured = None       # last captured frame for onion-skin
        self.overlay_enabled = True
        self.overlay_alpha = 0.5

        self.project = None
        ensure_dir(DEFAULT_PROJECTS_ROOT)
        self.projects_root = Path(DEFAULT_PROJECTS_ROOT)
        self.project_index_file = self.projects_root / "projects_index.json"
        self.projects_index = self._load_projects_index()

        self.selected_frame_index = -1
        self.mode_live_preview = True       # True => live view; False => playback
        self.is_playing = False
        self.manual_frame_selection = True  # distinguish user vs programmatic selection

        self._build_ui()
        self.refresh_camera_list()
        self.live_timer.start(30)

    # ---------------- Project Index ----------------

    def _load_projects_index(self):
        if self.project_index_file.exists():
            try:
                with open(self.project_index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_projects_index(self):
        with open(self.project_index_file, "w", encoding="utf-8") as f:
            json.dump(self.projects_index, f, indent=2)

    # ---------------- UI ----------------

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left preview
        preview_box = QtWidgets.QGroupBox("Preview")
        preview_layout = QtWidgets.QVBoxLayout()
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("background-color: #222;")
        preview_layout.addWidget(self.preview_label)

        # Overlay controls
        overlay_row = QtWidgets.QHBoxLayout()
        self.overlay_checkbox = QtWidgets.QCheckBox("Show previous-frame overlay")
        self.overlay_checkbox.setChecked(True)
        self.overlay_checkbox.stateChanged.connect(self.toggle_overlay)
        overlay_row.addWidget(self.overlay_checkbox)
        overlay_row.addStretch()
        overlay_row.addWidget(QtWidgets.QLabel("Overlay alpha"))

        self.alpha_slider = FiftySnapSlider(QtCore.Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(self.overlay_alpha * 100))
        self.alpha_slider.setFixedWidth(150)
        self.alpha_slider.setFixedHeight(22)
        # Tick marks (we’ll set more below)
        self.alpha_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.alpha_slider.setTickInterval(50)  # 0, 50, 100
        overlay_row.addWidget(self.alpha_slider)

        # Value display label
        self.alpha_value_label = QtWidgets.QLabel("100%")
        self.alpha_value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.alpha_value_label.setMinimumWidth(self.alpha_value_label.sizeHint().width())
        overlay_row.addWidget(self.alpha_value_label)

        # Connect after creating label so both update together
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)

        preview_layout.addLayout(overlay_row)

        preview_box.setLayout(preview_layout)
        main_layout.addWidget(preview_box, 2)

        # Right panel
        right_panel = QtWidgets.QVBoxLayout()

        # Camera selection
        cam_box = QtWidgets.QGroupBox("Camera Device")
        cam_layout = QtWidgets.QHBoxLayout()
        self.cam_combo = QtWidgets.QComboBox()
        self.cam_combo.currentIndexChanged.connect(self.change_camera_from_combo)
        cam_layout.addWidget(self.cam_combo)
        self.refresh_cam_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_cam_btn.clicked.connect(self.refresh_camera_list)
        cam_layout.addWidget(self.refresh_cam_btn)
        cam_box.setLayout(cam_layout)
        right_panel.addWidget(cam_box)

        # Project controls
        proj_box = QtWidgets.QGroupBox("Project")
        proj_layout = QtWidgets.QVBoxLayout()
        row = QtWidgets.QHBoxLayout()
        self.project_combo = QtWidgets.QComboBox()
        self.project_combo.currentIndexChanged.connect(self.on_project_selected)
        row.addWidget(self.project_combo)
        btns = QtWidgets.QVBoxLayout()
        self.new_proj_btn = QtWidgets.QPushButton("New...")
        self.new_proj_btn.clicked.connect(self.create_new_project)
        btns.addWidget(self.new_proj_btn)
        self.open_proj_btn = QtWidgets.QPushButton("Open...")
        self.open_proj_btn.clicked.connect(self.open_project_folder)
        btns.addWidget(self.open_proj_btn)
        self.rename_proj_btn = QtWidgets.QPushButton("Rename...")
        self.rename_proj_btn.clicked.connect(self.rename_project)
        btns.addWidget(self.rename_proj_btn)
        row.addLayout(btns)
        proj_layout.addLayout(row)
        self.project_path_label = QtWidgets.QLabel("No project selected")
        self.project_path_label.setWordWrap(True)
        proj_layout.addWidget(self.project_path_label)
        proj_box.setLayout(proj_layout)
        right_panel.addWidget(proj_box)

        # Controls
        controls_box = QtWidgets.QGroupBox("Playback Controls")
        c_layout = QtWidgets.QHBoxLayout()
        self.capture_btn = QtWidgets.QPushButton("Capture Frame")
        self.capture_btn.clicked.connect(self.capture_frame)
        c_layout.addWidget(self.capture_btn)

        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.prev_btn.clicked.connect(self.goto_prev_frame)
        c_layout.addWidget(self.prev_btn)

        self.next_btn = QtWidgets.QPushButton("Next")
        self.next_btn.clicked.connect(self.goto_next_frame)
        c_layout.addWidget(self.next_btn)

        self.delete_btn = QtWidgets.QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_frame)
        c_layout.addWidget(self.delete_btn)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setCheckable(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        c_layout.addWidget(self.play_btn)

        self.export_gif_btn = QtWidgets.QPushButton("Export GIF")
        self.export_gif_btn.clicked.connect(self.export_gif)
        c_layout.addWidget(self.export_gif_btn)

        # GIF resolution dropdown
        self.gif_res_combo = QtWidgets.QComboBox()
        self.gif_res_combo.addItems([
            "Full Resolution",
            "50% (Half Size)",
            "33% (Third Size)",
            "25% (Quarter Size)",
            "Custom..."
        ])
        self.gif_res_combo.setCurrentIndex(0)
        c_layout.addWidget(self.gif_res_combo)

        c_layout.addStretch()
        c_layout.addWidget(QtWidgets.QLabel("FPS:"))
        self.fps_combo = QtWidgets.QComboBox()
        self.fps_combo.addItems(["8", "12", "16", "24", "48"])
        self.fps_combo.setCurrentText("12")
        self.fps_combo.currentIndexChanged.connect(self.change_play_fps)
        c_layout.addWidget(self.fps_combo)

        controls_box.setLayout(c_layout)
        right_panel.addWidget(controls_box)

        # Frame list
        frames_box = QtWidgets.QGroupBox("Project Frames")
        f_layout = QtWidgets.QVBoxLayout()
        self.frames_list = QtWidgets.QListWidget()
        # Single-click selection:
        self.frames_list.currentRowChanged.connect(self.on_frame_selected)
        f_layout.addWidget(self.frames_list)
        frames_box.setLayout(f_layout)
        right_panel.addWidget(frames_box, 1)

        self.status_label = QtWidgets.QLabel("")
        right_panel.addWidget(self.status_label)

        main_layout.addLayout(right_panel, 1)

        self._refresh_project_combo()

    # ---------------- Camera ----------------

    def refresh_camera_list(self):
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        indices = list_camera_indices(8)
        if not indices:
            self.cam_combo.addItem("No camera devices found")
            self.device_index = None
            self._close_capture()
            self.cam_combo.blockSignals(False)
            return
        for i in indices:
            self.cam_combo.addItem(f"Device {i}", i)
        self.cam_combo.blockSignals(False)
        if self.device_index is None and indices:
            self.set_camera(indices[0])

    def change_camera_from_combo(self, idx):
        data = self.cam_combo.itemData(idx)
        if data is not None:
            self.set_camera(int(data))

    def set_camera(self, index):
        if self.device_index == index:
            return
        self._close_capture()
        cap = cv2.VideoCapture(index)
        # Ask for 1080p (driver may adjust if unsupported)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            self.status_label.setText(f"Failed to open device {index}")
            return
        ret, frame = cap.read()
        if not ret:
            cap.release()
            self.status_label.setText(f"No frames from device {index}")
            return
        self.capture = cap
        self.device_index = index
        h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
        w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
        self.status_label.setText(f"Opened device {index} — {w}x{h}")
        if self.mode_live_preview:
            self.live_timer.start(30)

    def _close_capture(self):
        if self.capture:
            try:
                self.capture.release()
            except Exception:
                pass
        self.capture = None
        self.device_index = None

    # ---------------- Overlay ----------------

    def change_alpha(self, val):
        self.overlay_alpha = val / 100.0

    def toggle_overlay(self, state):
        self.overlay_enabled = bool(state)

    # ---------------- Playback ----------------

    def change_play_fps(self):
        fps = int(self.fps_combo.currentText())
        self.play_timer.setInterval(int(1000 / fps) if fps > 0 else 1000)

    def toggle_playback(self):
        if not self.project:
            return
        frames = self.project.list_frames()
        if not frames:
            return

        # Ensure we are in playback mode
        if self.mode_live_preview:
            self.mode_live_preview = False
            self.live_timer.stop()

            # If currently on Live Preview row, start from first frame
            if self.selected_frame_index >= len(frames):
                self.selected_frame_index = 0
                self.manual_frame_selection = False
                self.frames_list.setCurrentRow(self.selected_frame_index)
                self.manual_frame_selection = True
                self.load_frame_to_preview(frames[0])

        # Toggle play state
        self.is_playing = not self.is_playing
        if self.is_playing:
            fps = int(self.fps_combo.currentText())
            self.play_timer.start(int(1000 / fps))
            self.play_btn.setText("Stop")
        else:
            self.play_timer.stop()
            self.play_btn.setText("Play")

    def play_next_frame(self):
        if not self.project:
            return
        frames = self.project.list_frames()
        if not frames:
            return

        # advance index cyclically
        if self.selected_frame_index >= len(frames) or self.selected_frame_index < 0:
            self.selected_frame_index = 0
        else:
            self.selected_frame_index = (self.selected_frame_index + 1) % len(frames)

        # Update preview and selection without triggering selection logic
        self.manual_frame_selection = False
        self.frames_list.setCurrentRow(self.selected_frame_index)
        self.manual_frame_selection = True

        self.load_frame_to_preview(frames[self.selected_frame_index])
        self.frames_list.scrollToItem(
            self.frames_list.currentItem(),
            QtWidgets.QAbstractItemView.PositionAtCenter
        )

    # ---------------- Frame List & Selection ----------------

    def _refresh_frames_list(self):
        self.frames_list.clear()
        if not self.project:
            return

        frames = self.project.list_frames()
        for p in frames:
            self.frames_list.addItem(p.name)
        # Add pseudo-frame at bottom
        self.frames_list.addItem(LIVE_PREVIEW_LABEL)

        # 🔹 If there are existing frames, use the last one for onion-skin
        if frames:
            last_img = cv2.imread(str(frames[-1]))
            if last_img is not None:
                self.previous_captured = last_img
        else:
            self.previous_captured = None  # no overlay if nothing exists yet

        # Default selection: Live Preview
        self.manual_frame_selection = False
        live_row = self.frames_list.count() - 1
        self.selected_frame_index = live_row
        self.frames_list.setCurrentRow(live_row)
        self.manual_frame_selection = True

        # Be in live mode by default
        self.mode_live_preview = True
        if self.capture:
            self.live_timer.start(30)

    def on_frame_selected(self, row):
        if row < 0 or not self.project:
            return
        if not self.manual_frame_selection:
            return
        item = self.frames_list.item(row)
        if not item:
            return
        self.select_frame_from_list(item)

    def on_alpha_changed(self, val):
        self.change_alpha(val)  # keeps overlay_alpha in sync
        if hasattr(self, "alpha_value_label"):
            self.alpha_value_label.setText(f"{val}%")

    def select_frame_from_list(self, item):
        text = item.text()
        total_items = self.frames_list.count()
        frames = self.project.list_frames()
        live_row = total_items - 1

        if text == LIVE_PREVIEW_LABEL:
            # Switch to live preview mode
            self.mode_live_preview = True
            self.is_playing = False
            self.play_timer.stop()
            self.play_btn.setText("Play")
            if self.capture:
                self.live_timer.start(30)
        else:
            # Saved frame: enter playback mode
            self.mode_live_preview = False
            self.live_timer.stop()
            self.is_playing = False
            self.play_timer.stop()
            self.play_btn.setText("Play")

            idx = self.frames_list.row(item)
            # Ensure idx is within frames range
            if 0 <= idx < len(frames):
                self.selected_frame_index = idx
                self.load_frame_to_preview(frames[idx])

        self.selected_frame_index = self.frames_list.row(item)
        self.frames_list.setCurrentRow(self.selected_frame_index)
        self.frames_list.scrollToItem(
            self.frames_list.currentItem(),
            QtWidgets.QAbstractItemView.PositionAtCenter
        )

    def goto_prev_frame(self):
        if not self.project:
            return
        frames = self.project.list_frames()
        if not frames:
            return

        live_row = self.frames_list.count() - 1

        # If currently on Live Preview, jump to last frame
        if self.selected_frame_index == live_row:
            self.selected_frame_index = len(frames) - 1
        else:
            self.selected_frame_index = max(0, self.selected_frame_index - 1)

        # Enter playback mode
        self.mode_live_preview = False
        self.live_timer.stop()
        self.is_playing = False
        self.play_timer.stop()
        self.play_btn.setText("Play")

        self.manual_frame_selection = False
        self.frames_list.setCurrentRow(self.selected_frame_index)
        self.manual_frame_selection = True

        self.load_frame_to_preview(frames[self.selected_frame_index])
        self.frames_list.scrollToItem(
            self.frames_list.currentItem(),
            QtWidgets.QAbstractItemView.PositionAtCenter
        )

    def goto_next_frame(self):
        if not self.project:
            return
        frames = self.project.list_frames()
        if not frames:
            return

        live_row = self.frames_list.count() - 1

        # If on Live Preview, no next frame beyond last saved
        if self.selected_frame_index >= len(frames):
            return

        self.selected_frame_index = min(len(frames) - 1, self.selected_frame_index + 1)

        # Enter playback mode
        self.mode_live_preview = False
        self.live_timer.stop()
        self.is_playing = False
        self.play_timer.stop()
        self.play_btn.setText("Play")

        self.manual_frame_selection = False
        self.frames_list.setCurrentRow(self.selected_frame_index)
        self.manual_frame_selection = True

        self.load_frame_to_preview(frames[self.selected_frame_index])
        self.frames_list.scrollToItem(
            self.frames_list.currentItem(),
            QtWidgets.QAbstractItemView.PositionAtCenter
        )

    def delete_frame(self):
        if not self.project:
            return

        frames = self.project.list_frames()
        frame_count = len(frames)

        # If in Live Preview mode or selected live preview row → don’t delete
        if self.selected_frame_index >= frame_count:
            self.status_label.setText("Cannot delete Live Preview")
            return

        # Confirm deletion (optional)
        frame_path = frames[self.selected_frame_index]
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Frame",
            f"Delete {frame_path.name}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if reply != QtWidgets.QMessageBox.Yes:
            return

        # Delete the file
        try:
            frame_path.unlink()
        except Exception as e:
            self.status_label.setText(f"Failed to delete: {e}")
            return

        # Adjust overlay source if needed
        if self.previous_captured is not None:
            # If the last frame was deleted, update onion-skin to new last frame
            updated_frames = self.project.list_frames()
            if updated_frames:
                last_img = cv2.imread(str(updated_frames[-1]))
                if last_img is not None:
                    self.previous_captured = last_img
            else:
                self.previous_captured = None

        # Refresh frame list
        self._refresh_frames_list()

        # After refresh, select Live Preview frame automatically
        live_row = self.frames_list.count() - 1
        self.selected_frame_index = live_row
        self.frames_list.setCurrentRow(live_row)
        self.frames_list.scrollToItem(
            self.frames_list.currentItem(),
            QtWidgets.QAbstractItemView.PositionAtCenter
        )

        self.mode_live_preview = True
        self.status_label.setText(f"Deleted {frame_path.name}")

    # ---------------- Capture ----------------

    def capture_frame(self):
        if not self.mode_live_preview:
            self.status_label.setText("Cannot capture in playback mode")
            return
        if not self.capture:
            self.status_label.setText("No camera opened")
            return
        if not self.project:
            self.status_label.setText("No project selected")
            return

        ret, frame = self.capture.read()
        if not ret:
            self.status_label.setText("Failed to read from camera")
            return

        out_path = self.project.next_frame_filename()
        if not cv2.imwrite(str(out_path), frame):
            self.status_label.setText("Failed to save frame")
            return

        # Update onion-skin source to this newly captured frame
        self.previous_captured = frame.copy()

        # Refresh list; selection will go to Live Preview
        self._refresh_frames_list()
        self.status_label.setText(f"Saved {out_path.name} ({frame.shape[1]}x{frame.shape[0]})")

    # ---------------- Live Preview / Playback Drawing ----------------

    def grab_frame(self):
        if not self.capture or not self.mode_live_preview:
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        display_frame = frame.copy()
        if self.overlay_enabled and self.previous_captured is not None:
            overlay = cv2.resize(self.previous_captured, (frame.shape[1], frame.shape[0]))
            display_frame = cv2.addWeighted(
                display_frame, 1 - self.overlay_alpha,
                overlay, self.overlay_alpha,
                0
            )
        self.show_preview_image(display_frame)

    def load_frame_to_preview(self, path: Path):
        img = cv2.imread(str(path))
        if img is None:
            self.status_label.setText("Failed loading frame")
            return
        self.show_preview_image(img)

    def show_preview_image(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(
            pix.scaled(self.preview_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    # ---------------- Project Management ----------------

    def create_new_project(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "New Project", "Project name:")
        if not ok or not text.strip():
            return
        name = text.strip()
        path = self.projects_root / name
        if path.exists():
            QtWidgets.QMessageBox.warning(self, "Exists", "A project with that name already exists.")
            return
        Project(path)
        self.projects_index[name] = str(path)
        self._save_projects_index()
        self._refresh_project_combo()
        self.select_project_by_name(name)

    def open_project_folder(self):
        dirpath = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open project folder", str(self.projects_root)
        )
        if not dirpath:
            return
        p = Project(dirpath)
        name = p.meta.get("name", p.path.name)
        if name in self.projects_index and self.projects_index[name] != str(p.path):
            name = f"{name}_{int(time.time())}"
        self.projects_index[name] = str(p.path)
        self._save_projects_index()
        self._refresh_project_combo()
        self.select_project_by_name(name)

    def rename_project(self):
        current_name = self.project_combo.currentText()
        if current_name not in self.projects_index:
            return
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Project", "New name:", text=current_name
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        old_path = Path(self.projects_index[current_name])
        new_path = old_path.parent / new_name
        if new_path.exists():
            QtWidgets.QMessageBox.warning(self, "Exists", "Target name exists.")
            return
        old_path.rename(new_path)
        del self.projects_index[current_name]
        self.projects_index[new_name] = str(new_path)
        self._save_projects_index()
        self._refresh_project_combo()
        self.select_project_by_name(new_name)

    def _refresh_project_combo(self):
        self.project_combo.clear()
        for name in sorted(self.projects_index.keys()):
            self.project_combo.addItem(name)
        if not self.projects_index:
            self.project_combo.addItem("No projects")

    def select_project_by_name(self, name):
        idx = self.project_combo.findText(name)
        if idx >= 0:
            self.project_combo.setCurrentIndex(idx)
            self.on_project_selected(idx)

    def on_project_selected(self, idx):
        name = self.project_combo.currentText()
        if name not in self.projects_index:
            self.project = None
            self.project_path_label.setText("No project selected")
            self.frames_list.clear()
            return
        self.project = Project(self.projects_index[name])
        self.project_path_label.setText(str(self.project.path))
        self._refresh_frames_list()

    def export_gif(self):
        if not self.project:
            self.status_label.setText("No project selected")
            return

        frames = self.project.list_frames()
        if not frames:
            self.status_label.setText("No frames to export")
            return

        # Ask user where to save
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export GIF",
            str(self.project.path / f"{self.project.meta.get('name', 'project')}.gif"),
            "GIF Files (*.gif)"
        )
        if not save_path:
            return

        # FPS -> frame duration
        fps = int(self.fps_combo.currentText())
        duration_ms = int(1000 / fps)

        from PIL import Image

        # Resolution selection
        res_choice = self.gif_res_combo.currentText()
        scale_factor = 1.0
        custom_size = None

        if "50%" in res_choice:
            scale_factor = 0.5
        elif "33%" in res_choice:
            scale_factor = 1 / 3
        elif "25%" in res_choice:
            scale_factor = 0.25
        elif "Custom" in res_choice:
            # Ask for custom width/height
            w, ok1 = QtWidgets.QInputDialog.getInt(self, "Custom Width", "Width (px):", 800, 10, 20000)
            h, ok2 = QtWidgets.QInputDialog.getInt(self, "Custom Height", "Height (px):", 600, 10, 20000)
            if not (ok1 and ok2):
                return
            custom_size = (w, h)

        pil_frames = []
        for fpath in frames:
            img = Image.open(fpath).convert("RGB")

            # Scale frame
            if custom_size:
                img = img.resize(custom_size, Image.LANCZOS)
            elif scale_factor != 1.0:
                new_w = int(img.width * scale_factor)
                new_h = int(img.height * scale_factor)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            pil_frames.append(img)

        # Save GIF
        try:
            pil_frames[0].save(
                save_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0
            )
            self.status_label.setText(f"Exported GIF: {save_path}")
        except Exception as e:
            self.status_label.setText(f"Failed exporting GIF: {e}")

    # ---------------- Qt Lifecycle ----------------

    def closeEvent(self, e):
        self._close_capture()
        self.play_timer.stop()
        self.live_timer.stop()
        e.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = StopMotionApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
