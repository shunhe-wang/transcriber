#!/usr/bin/env python3
"""
Meeting Transcriber — local Whisper-based recorder for macOS
Records from one or two audio devices (system audio + mic), mixes to mono,
then transcribes offline using OpenAI Whisper.
"""

import json
import os
import queue
import threading
import tempfile
import datetime
import warnings
import wave
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

# Suppress harmless Python 3.13 resource_tracker warning on exit
warnings.filterwarnings("ignore", message=".*resource_tracker.*", category=UserWarning)

# ── lazy imports so the UI loads fast ──────────────────────────────────────
sounddevice = None
whisper = None

def lazy_import():
    global sounddevice, whisper
    try:
        import sounddevice as sd
        sounddevice = sd
    except ImportError:
        return False, "sounddevice"
    try:
        import whisper as w
        whisper = w
    except ImportError:
        return False, "openai-whisper"
    return True, None


# ── Recorder ───────────────────────────────────────────────────────────────
class Recorder:
    """Records from one or two devices, resamples, mixes to mono WAV on disk.

    In dual mode, system audio (e.g. BlackHole) and mic run as separate
    InputStreams.  Both feed a shared queue; a writer thread resamples each
    to a common rate, mixes by averaging, and streams int16 to a WAV file.
    Falls back to single-device mode when mic_device is None.
    """

    QUEUE_MAXSIZE = 400     # room for both sources

    def __init__(self):
        self._q               = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self._sys_stream      = None
        self._mic_stream      = None
        self._running         = False
        self._wav_path        = None
        self._wf              = None
        self._writer_thread   = None
        self._frames_written  = 0
        self._dropped_chunks  = 0
        self._stream_warnings = 0
        self._writer_error    = None
        self._output_rate     = 48000
        self._sys_rate        = None
        self._mic_rate        = None
        self._dual            = False

    # ── device enumeration ─────────────────────────────────────────────────
    def list_devices(self):
        if sounddevice is None:
            return []
        devs = []
        for i, d in enumerate(sounddevice.query_devices()):
            if d["max_input_channels"] > 0:
                devs.append((i, d["name"]))
        return devs

    # ── start / stop ───────────────────────────────────────────────────────
    def start(self, sys_device, mic_device=None):
        """Begin recording.  mic_device=None → single-device mode."""

        # ── system audio device ──
        sys_info       = sounddevice.query_devices(sys_device)
        self._sys_rate = int(sys_info["default_samplerate"])
        sys_channels   = int(sys_info["max_input_channels"])

        # ── mic device (optional) ──
        self._dual = mic_device is not None
        if self._dual:
            mic_info       = sounddevice.query_devices(mic_device)
            self._mic_rate = int(mic_info["default_samplerate"])
            mic_channels   = int(mic_info["max_input_channels"])
            self._output_rate = max(self._sys_rate, self._mic_rate)
        else:
            self._mic_rate    = None
            self._output_rate = self._sys_rate

        # ── open WAV for streaming writes ──
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._wav_path = tmp.name
        tmp.close()

        self._wf = wave.open(self._wav_path, "wb")
        self._wf.setnchannels(1)        # always mono
        self._wf.setsampwidth(2)        # int16
        self._wf.setframerate(self._output_rate)

        self._frames_written  = 0
        self._dropped_chunks  = 0
        self._stream_warnings = 0
        self._writer_error    = None
        self._running         = True

        # ── writer / mixer thread ──
        self._writer_thread = threading.Thread(
            target=self._disk_writer, daemon=True
        )
        self._writer_thread.start()

        # ── open audio streams ──
        self._sys_stream = sounddevice.InputStream(
            samplerate=self._sys_rate,
            channels=sys_channels,
            dtype="float32",
            device=sys_device,
            callback=self._sys_callback,
        )
        self._sys_stream.start()

        if self._dual:
            self._mic_stream = sounddevice.InputStream(
                samplerate=self._mic_rate,
                channels=mic_channels,
                dtype="float32",
                device=mic_device,
                callback=self._mic_callback,
            )
            self._mic_stream.start()

    def stop(self) -> str | None:
        """Stop recording, flush to disk, return WAV path (or None)."""
        self._running = False

        for stream_attr in ("_sys_stream", "_mic_stream"):
            stream = getattr(self, stream_attr)
            if stream:
                stream.stop()
                stream.close()
                setattr(self, stream_attr, None)

        if self._writer_thread:
            self._writer_thread.join(timeout=10)
            if self._writer_thread.is_alive():
                self._writer_thread = None
                raise RuntimeError(
                    "Timed out waiting for audio writer to flush. "
                    "Recording may be incomplete."
                )
            self._writer_thread = None

        if self._wf:
            self._wf.close()
            self._wf = None

        if self._writer_error:
            err = self._writer_error
            self._writer_error = None
            raise RuntimeError(f"Audio writer failed: {err}") from err

        if self._frames_written == 0:
            self._cleanup_wav()
            return None

        return self._wav_path

    # ── audio callbacks (must be fast — just enqueue) ──────────────────────
    def _sys_callback(self, indata, frames, time, status):
        if status:
            self._stream_warnings += 1
        if self._running:
            try:
                self._q.put_nowait(("sys", indata.copy()))
            except queue.Full:
                self._dropped_chunks += 1

    def _mic_callback(self, indata, frames, time, status):
        if status:
            self._stream_warnings += 1
        if self._running:
            try:
                self._q.put_nowait(("mic", indata.copy()))
            except queue.Full:
                self._dropped_chunks += 1

    # ── writer / mixer thread ──────────────────────────────────────────────
    def _disk_writer(self):
        """Dequeue from both sources, resample, mix, write to WAV."""
        try:
            sys_parts, sys_n = [], 0
            mic_parts, mic_n = [], 0

            while self._running or not self._q.empty():
                try:
                    source, data = self._q.get(timeout=0.2)
                except queue.Empty:
                    # If one source has >1 s buffered and the other is empty,
                    # write single-source rather than holding indefinitely
                    # (handles muted mic, dead device, etc.)
                    if self._dual:
                        hold_limit = self._output_rate  # 1 second
                        if sys_n > hold_limit and mic_n == 0:
                            self._write_samples(np.concatenate(sys_parts))
                            sys_parts, sys_n = [], 0
                        elif mic_n > hold_limit and sys_n == 0:
                            self._write_samples(np.concatenate(mic_parts))
                            mic_parts, mic_n = [], 0
                    continue

                mono = self._to_mono(data)

                if source == "sys":
                    resampled = self._resample(mono, self._sys_rate, self._output_rate)
                    sys_parts.append(resampled)
                    sys_n += len(resampled)
                else:
                    resampled = self._resample(mono, self._mic_rate, self._output_rate)
                    mic_parts.append(resampled)
                    mic_n += len(resampled)

                # ── mix and write ──
                if self._dual:
                    if sys_n > 0 and mic_n > 0:
                        n = min(sys_n, mic_n)
                        sys_flat = np.concatenate(sys_parts)
                        mic_flat = np.concatenate(mic_parts)
                        mixed = (sys_flat[:n] + mic_flat[:n]) * 0.5
                        self._write_samples(mixed)
                        # keep remainders
                        if n < sys_n:
                            sys_parts, sys_n = [sys_flat[n:]], sys_n - n
                        else:
                            sys_parts, sys_n = [], 0
                        if n < mic_n:
                            mic_parts, mic_n = [mic_flat[n:]], mic_n - n
                        else:
                            mic_parts, mic_n = [], 0
                else:
                    # single-device: write immediately
                    if sys_n > 0:
                        self._write_samples(np.concatenate(sys_parts))
                        sys_parts, sys_n = [], 0

            # ── final flush after stop ──
            if self._dual and (sys_n > 0 or mic_n > 0):
                sys_flat = np.concatenate(sys_parts) if sys_parts else np.empty(0, dtype=np.float32)
                mic_flat = np.concatenate(mic_parts) if mic_parts else np.empty(0, dtype=np.float32)
                # mix whatever overlaps
                n = min(len(sys_flat), len(mic_flat))
                if n > 0:
                    self._write_samples((sys_flat[:n] + mic_flat[:n]) * 0.5)
                    sys_flat = sys_flat[n:]
                    mic_flat = mic_flat[n:]
                # write any single-source tail
                for leftover in (sys_flat, mic_flat):
                    if len(leftover) > 0:
                        self._write_samples(leftover)
            elif sys_n > 0:
                self._write_samples(np.concatenate(sys_parts))

        except Exception as e:
            self._writer_error = e

    def _write_samples(self, data):
        """Clip, convert to int16, write to WAV, update frame count."""
        clipped   = np.clip(data, -1.0, 1.0)
        audio_i16 = (clipped * 32767).astype(np.int16)
        self._wf.writeframes(audio_i16.tobytes())
        self._frames_written += len(data)

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _to_mono(data):
        """Multi-channel float32 → 1-D mono float32."""
        if data.ndim == 2 and data.shape[1] > 1:
            return data.mean(axis=1)
        return data.flatten()

    @staticmethod
    def _resample(data, from_rate, to_rate):
        """Resample 1-D float32 via linear interpolation (adequate for speech)."""
        if from_rate == to_rate or len(data) == 0:
            return data
        n_out = int(round(len(data) * to_rate / from_rate))
        if n_out == 0:
            return np.empty(0, dtype=np.float32)
        x_in  = np.arange(len(data))
        x_out = np.linspace(0, len(data) - 1, n_out)
        return np.interp(x_out, x_in, data).astype(np.float32)

    @property
    def wav_path(self):
        return self._wav_path

    def cleanup(self):
        self._cleanup_wav()

    def _cleanup_wav(self):
        if self._wav_path and os.path.exists(self._wav_path):
            try:
                os.unlink(self._wav_path)
            except OSError:
                pass
        self._wav_path = None

    @property
    def duration(self):
        return self._frames_written / self._output_rate if self._output_rate else 0

    @property
    def dropped_chunks(self):
        return self._dropped_chunks

    @property
    def stream_warnings(self):
        return self._stream_warnings


# ── GUI ────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    ACCENT   = "#E8FF6B"
    BG       = "#0D0D0D"
    PANEL    = "#161616"
    TEXT     = "#F0F0F0"
    DIM      = "#555555"
    RED      = "#FF4A4A"
    FONT_HDR = ("Georgia", 22, "bold")
    FONT_LBL = ("Menlo", 11)
    FONT_BTN = ("Menlo", 13, "bold")
    FONT_TXT = ("Menlo", 11)

    def __init__(self):
        super().__init__()
        self.title("Meeting Transcriber")
        self.configure(bg=self.BG)
        self.resizable(True, True)
        self.geometry("780x800")
        self.minsize(640, 640)

        self._recorder          = Recorder()
        self._wav_path          = None
        self._model             = None
        self._model_name_loaded = None
        self._model_lock        = threading.Lock()
        self._model_name        = tk.StringVar(value="base")
        self._language_var      = tk.StringVar(value="en")
        self._sys_device_var    = tk.IntVar(value=-1)
        self._mic_device_var    = tk.IntVar(value=-1)   # -1 = mic off
        self._status_var        = tk.StringVar(value="Ready")
        self._title_var         = tk.StringVar(value="")
        self._timer_id          = None
        self._elapsed           = 0
        self._recording         = False
        self._transcribing      = False
        self._segments          = []
        self._rec_duration      = 0.0
        self._rec_started_at    = None
        self._rec_stopped_at    = None

        self._build_ui()
        self._check_deps()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── dependency check ───────────────────────────────────────────────────
    def _check_deps(self):
        ok, missing = lazy_import()
        if not ok:
            messagebox.showerror(
                "Missing dependency",
                f"Please install: pip install {missing}\n\nSee the README for full setup instructions."
            )
            self._set_status(f"⚠  Missing: {missing} — see README", color=self.RED)
        else:
            self._populate_devices()
            self._preload_model()

    def _populate_devices(self):
        devs = self._recorder.list_devices()

        # ── system audio dropdown ──
        sys_menu = self._sys_device_menu["menu"]
        sys_menu.delete(0, "end")
        if not devs:
            sys_menu.add_command(label="No input devices found")
            return
        for i, name in devs:
            lbl = f"{i}: {name}"
            sys_menu.add_command(
                label=lbl,
                command=lambda v=i, l=lbl: self._select_sys_device(v, l)
            )

        # ── mic dropdown (includes "None" option) ──
        mic_menu = self._mic_device_menu["menu"]
        mic_menu.delete(0, "end")
        mic_menu.add_command(
            label="None — mic off",
            command=lambda: self._select_mic_device(-1, "None — mic off")
        )
        for i, name in devs:
            lbl = f"{i}: {name}"
            mic_menu.add_command(
                label=lbl,
                command=lambda v=i, l=lbl: self._select_mic_device(v, l)
            )

        # ── auto-select defaults ──
        # System audio: prefer BlackHole / loopback
        self._sys_device_var.set(devs[0][0])
        self._sys_device_label.set(f"{devs[0][0]}: {devs[0][1]}")
        for di, dn in devs:
            if any(k in dn.lower() for k in ("blackhole", "loopback", "soundflower")):
                self._select_sys_device(di, f"{di}: {dn}")
                break

        # Mic: prefer built-in MacBook mic, default to None
        self._mic_device_var.set(-1)
        self._mic_device_label.set("None — mic off")
        for di, dn in devs:
            if any(k in dn.lower() for k in ("macbook", "built-in")):
                self._select_mic_device(di, f"{di}: {dn}")
                break

    def _select_sys_device(self, device_index, label):
        self._sys_device_var.set(device_index)
        self._sys_device_label.set(label)

    def _select_mic_device(self, device_index, label):
        self._mic_device_var.set(device_index)
        self._mic_device_label.set(label)

    def _preload_model(self):
        model_name = self._model_name.get()
        def _load():
            try:
                with self._model_lock:
                    if self._model_name_loaded != model_name:
                        self._model = whisper.load_model(model_name)
                        self._model_name_loaded = model_name
                self.after(0, lambda: self._set_status("Ready — model loaded", color=self.ACCENT))
            except Exception:
                pass
        self._set_status("Loading Whisper model in background…")
        threading.Thread(target=_load, daemon=True).start()

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=self.BG)
        hdr.pack(fill="x", padx=24, pady=(28, 0))
        tk.Label(hdr, text="●", font=("Menlo", 14), fg=self.ACCENT, bg=self.BG).pack(side="left")
        tk.Label(hdr, text=" MEETING TRANSCRIBER", font=self.FONT_HDR,
                 fg=self.TEXT, bg=self.BG).pack(side="left", padx=(6, 0))

        tk.Label(self, text="Local · Free · Private — powered by OpenAI Whisper",
                 font=("Menlo", 10), fg=self.DIM, bg=self.BG).pack(anchor="w", padx=28, pady=(2, 20))

        # ── Settings panel ──
        panel = tk.Frame(self, bg=self.PANEL, bd=0)
        panel.pack(fill="x", padx=24, pady=(0, 16))

        inner = tk.Frame(panel, bg=self.PANEL)
        inner.pack(fill="x", padx=20, pady=16)
        inner.columnconfigure(0, weight=1)

        # Meeting title
        tk.Label(inner, text="MEETING TITLE", font=("Menlo", 9, "bold"),
                 fg=self.DIM, bg=self.PANEL).grid(row=0, column=0, columnspan=3, sticky="w")
        tk.Entry(inner, textvariable=self._title_var,
                 font=self.FONT_LBL, bg=self.BG, fg=self.TEXT,
                 insertbackground=self.ACCENT, relief="flat",
                 highlightthickness=1, highlightcolor=self.ACCENT,
                 highlightbackground=self.DIM
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 12))

        # ── System audio device + refresh ──
        dev_row = tk.Frame(inner, bg=self.PANEL)
        dev_row.grid(row=2, column=0, sticky="w")
        tk.Label(dev_row, text="SYSTEM AUDIO", font=("Menlo", 9, "bold"),
                 fg=self.DIM, bg=self.PANEL).pack(side="left")
        tk.Button(dev_row, text="↻", font=("Menlo", 9), fg=self.DIM,
                  bg=self.PANEL, relief="flat", cursor="hand2",
                  activebackground=self.PANEL, activeforeground=self.ACCENT,
                  command=self._refresh_devices).pack(side="left", padx=(6, 0))

        self._sys_device_label = tk.StringVar(value="Loading…")
        self._sys_device_menu  = tk.OptionMenu(inner, self._sys_device_label, "Loading…")
        self._style_dropdown(self._sys_device_menu, width=40)
        self._sys_device_menu.grid(row=3, column=0, sticky="w", pady=(4, 0))

        # ── Mic device ──
        tk.Label(inner, text="MIC DEVICE", font=("Menlo", 9, "bold"),
                 fg=self.DIM, bg=self.PANEL).grid(row=4, column=0, sticky="w", pady=(8, 0))

        self._mic_device_label = tk.StringVar(value="None — mic off")
        self._mic_device_menu  = tk.OptionMenu(inner, self._mic_device_label, "None — mic off")
        self._style_dropdown(self._mic_device_menu, width=40)
        self._mic_device_menu.grid(row=5, column=0, sticky="w", pady=(4, 0))

        # ── Model selector ──
        tk.Label(inner, text="WHISPER MODEL", font=("Menlo", 9, "bold"),
                 fg=self.DIM, bg=self.PANEL).grid(row=2, column=1, sticky="w", padx=(32, 0))
        models = ["tiny", "base", "small", "medium", "large"]
        model_menu = tk.OptionMenu(inner, self._model_name, *models)
        self._style_dropdown(model_menu, width=10)
        model_menu.grid(row=3, column=1, sticky="w", padx=(32, 0), pady=(4, 0))

        # ── Language selector ──
        tk.Label(inner, text="LANGUAGE", font=("Menlo", 9, "bold"),
                 fg=self.DIM, bg=self.PANEL).grid(row=2, column=2, sticky="w", padx=(32, 0))
        languages = ["en", "auto-detect", "es", "fr", "de", "zh", "ja"]
        lang_menu = tk.OptionMenu(inner, self._language_var, *languages)
        self._style_dropdown(lang_menu, width=12)
        lang_menu.grid(row=3, column=2, sticky="w", padx=(32, 0), pady=(4, 0))

        # ── Hint text ──
        tk.Label(inner,
                 text="System Audio captures Zoom/Teams.  Mic captures your voice.  Both are mixed for transcription.",
                 font=("Menlo", 8), fg=self.DIM, bg=self.PANEL).grid(
                     row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))

        # ── Timer display ──
        self._timer_label = tk.Label(self, text="00:00:00", font=("Menlo", 48, "bold"),
                                     fg=self.DIM, bg=self.BG)
        self._timer_label.pack(pady=(20, 0))

        # ── Buttons ──
        btn_row = tk.Frame(self, bg=self.BG)
        btn_row.pack(pady=20)

        self._rec_btn = tk.Button(
            btn_row, text="⏺  START RECORDING",
            font=self.FONT_BTN, fg=self.BG, bg=self.ACCENT,
            activebackground="#d4eb55", activeforeground=self.BG,
            relief="flat", padx=24, pady=12, cursor="hand2",
            command=self._toggle_recording
        )
        self._rec_btn.pack(side="left", padx=(0, 12))

        self._trans_btn = tk.Button(
            btn_row, text="✦  TRANSCRIBE",
            font=self.FONT_BTN, fg=self.BG, bg=self.ACCENT,
            activebackground="#d4eb55", activeforeground=self.BG,
            disabledforeground=self.DIM,
            relief="flat", padx=24, pady=12, cursor="hand2",
            state="disabled", command=self._start_transcribe
        )
        self._trans_btn.pack(side="left", padx=(0, 12))

        self._save_btn = tk.Button(
            btn_row, text="↓  SAVE",
            font=self.FONT_BTN, fg=self.BG, bg=self.ACCENT,
            activebackground="#d4eb55", activeforeground=self.BG,
            disabledforeground=self.DIM,
            relief="flat", padx=24, pady=12, cursor="hand2",
            state="disabled", command=self._save_transcript
        )
        self._save_btn.pack(side="left")

        # ── Status bar ──
        self._status_lbl = tk.Label(self, textvariable=self._status_var,
                                    font=("Menlo", 10), fg=self.DIM, bg=self.BG)
        self._status_lbl.pack(pady=(0, 10))

        # ── Transcript area ──
        txt_frame = tk.Frame(self, bg=self.PANEL)
        txt_frame.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        tk.Label(txt_frame, text="TRANSCRIPT", font=("Menlo", 9, "bold"),
                 fg=self.DIM, bg=self.PANEL).pack(anchor="w", padx=16, pady=(12, 0))

        scroll = tk.Scrollbar(txt_frame)
        scroll.pack(side="right", fill="y", pady=(0, 16), padx=(0, 8))

        self._text = tk.Text(
            txt_frame, wrap="word", font=self.FONT_TXT,
            bg=self.PANEL, fg=self.TEXT, insertbackground=self.ACCENT,
            relief="flat", padx=16, pady=12, spacing3=4,
            yscrollcommand=scroll.set
        )
        self._text.pack(fill="both", expand=True, padx=(0, 0), pady=(4, 16))
        scroll.config(command=self._text.yview)

        self._text.insert("1.0", "Transcript will appear here after you record and transcribe…")
        self._text.configure(state="disabled")

    def _style_dropdown(self, widget, width=10):
        """Apply consistent dark styling to an OptionMenu."""
        widget.configure(
            bg=self.BG, fg=self.TEXT, activebackground=self.ACCENT,
            activeforeground=self.BG, highlightthickness=0,
            font=self.FONT_LBL, width=width, anchor="w", relief="flat"
        )
        widget["menu"].configure(bg=self.BG, fg=self.TEXT, font=self.FONT_LBL)

    def _refresh_devices(self):
        if self._recording:
            return
        self._populate_devices()
        self._set_status("Device list refreshed")

    # ── Recording ──────────────────────────────────────────────────────────
    def _toggle_recording(self):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._recorder.cleanup()
        self._wav_path = None

        sys_dev = self._sys_device_var.get()
        mic_dev = self._mic_device_var.get()

        # -1 means mic off
        mic_arg = mic_dev if mic_dev >= 0 else None

        # Prevent selecting the same device for both
        if mic_arg is not None and mic_arg == sys_dev:
            messagebox.showwarning(
                "Same device selected",
                "System Audio and Mic are set to the same device. "
                "Select a different mic, or set Mic to 'None'."
            )
            return

        try:
            self._recorder.start(sys_dev, mic_arg)
        except Exception as e:
            err = str(e)
            messagebox.showerror("Recording Error", err)
            return

        self._recording      = True
        self._elapsed        = 0
        self._rec_started_at = datetime.datetime.now()
        self._rec_stopped_at = None
        self._rec_btn.configure(text="⏹  STOP RECORDING", bg=self.RED, fg=self.TEXT,
                                activebackground="#cc3333")
        self._trans_btn.configure(state="disabled")
        self._save_btn.configure(state="disabled")
        self._timer_label.configure(fg=self.RED)

        mode = "sys + mic" if mic_arg is not None else "single device"
        self._set_status(f"Recording ({mode})…")
        self._tick()

    def _stop_recording(self):
        self._recording = False
        if self._timer_id:
            self.after_cancel(self._timer_id)
            self._timer_id = None

        self._rec_btn.configure(text="⏺  START RECORDING", bg=self.ACCENT, fg=self.BG,
                                activebackground="#d4eb55")
        self._timer_label.configure(fg=self.TEXT)
        self._set_status("Stopping — flushing audio to disk…")
        self.update_idletasks()

        try:
            self._wav_path = self._recorder.stop()
        except Exception as e:
            err = str(e)
            messagebox.showerror("Recording Error", f"Failed to save recording:\n{err}")
            self._set_status("Recording error — audio may be lost.", color=self.RED)
            return

        self._rec_stopped_at = datetime.datetime.now()
        self._rec_duration = self._recorder.duration
        dropped  = self._recorder.dropped_chunks
        warnings = self._recorder.stream_warnings

        if self._wav_path:
            self._trans_btn.configure(state="normal")
            dur_str = self._fmt(int(self._rec_duration))
            msg = f"Recording saved ({dur_str}). Ready to transcribe."
            has_issues = False
            if dropped > 0:
                msg += f" ⚠ {dropped} chunks dropped."
                has_issues = True
            if warnings > 0:
                msg += f" ⚠ {warnings} stream warnings."
                has_issues = True
            self._set_status(msg, color=self.RED if has_issues else None)
        else:
            self._set_status(
                "No audio captured — check that the selected devices are receiving audio.",
                color=self.RED
            )

    def _tick(self):
        self._elapsed += 1
        self._timer_label.configure(text=self._fmt(self._elapsed))
        self._timer_id = self.after(1000, self._tick)

    @staticmethod
    def _fmt(s):
        return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

    # ── Transcription ──────────────────────────────────────────────────────
    def _start_transcribe(self):
        if self._transcribing or not self._wav_path:
            return
        self._transcribing = True
        self._trans_btn.configure(state="disabled")
        self._set_status("Loading Whisper model… (first run downloads the model, may take a minute)")
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", "⏳ Transcribing, please wait…")
        self._text.configure(state="disabled")
        model_name = self._model_name.get()
        language   = self._language_var.get()
        threading.Thread(target=self._transcribe_worker,
                         args=(model_name, language), daemon=True).start()

    def _transcribe_worker(self, model_name, language):
        try:
            with self._model_lock:
                if self._model is None or self._model_name_loaded != model_name:
                    self._model             = whisper.load_model(model_name)
                    self._model_name_loaded = model_name

            self.after(0, lambda: self._set_status("Transcribing audio…"))

            lang = None if language == "auto-detect" else language
            result = self._model.transcribe(
                self._wav_path,
                fp16=False,
                language=lang,
                verbose=False,
                condition_on_previous_text=False,
            )
            transcript = result["text"].strip()
            segments   = result.get("segments", [])
            self.after(0, lambda: self._show_transcript(transcript, segments))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: self._set_status(f"Error: {err}", color=self.RED))
            self.after(0, lambda: self._trans_btn.configure(state="normal"))
        finally:
            self._transcribing = False

    def _show_transcript(self, text, segments):
        self._segments = segments
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")

        if not text:
            self._text.insert("1.0",
                "No speech detected. Check that the selected input devices are "
                "actually receiving audio (e.g. BlackHole is routed in Audio MIDI Setup, "
                "or the mic is picking up sound).")
            self._text.configure(state="disabled")
            self._set_status("⚠ Transcription returned empty — no speech detected", color=self.RED)
            self._trans_btn.configure(state="normal")
            return

        self._text.insert("1.0", text)
        self._text.configure(state="disabled")
        self._save_btn.configure(state="normal")
        words = len(text.split())
        self._set_status(f"Transcription complete — {words} words", color=self.ACCENT)
        self._trans_btn.configure(state="normal")

    # ── Save ───────────────────────────────────────────────────────────────
    def _save_transcript(self):
        content = self._text.get("1.0", "end").strip()
        if not content:
            return

        title = self._title_var.get().strip()
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title).strip().replace(" ", "_")
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{safe_title}_{ts}" if safe_title else f"transcript_{ts}"

        # Default save location: transcriber/output/
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialdir=output_dir,
            initialfile=f"{base_name}.json",
            filetypes=[
                ("JSON (with segments)", "*.json"),
                ("SRT subtitles", "*.srt"),
                ("Plain text", "*.txt"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".json":
                self._save_json(path, content)
            elif ext == ".srt":
                self._save_srt(path)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            self._set_status(f"Saved to {os.path.basename(path)}", color=self.ACCENT)
        except OSError as e:
            err = str(e)
            messagebox.showerror("Save Error", f"Could not save file:\n{err}")
            self._set_status("Save failed.", color=self.RED)

    def _save_json(self, path, plain_text):
        data = {
            "title":    self._title_var.get().strip() or None,
            "model":    self._model_name_loaded,
            "language": self._language_var.get(),
            "recording_started": self._rec_started_at.isoformat() if self._rec_started_at else None,
            "recording_stopped": self._rec_stopped_at.isoformat() if self._rec_stopped_at else None,
            "duration_seconds": round(self._rec_duration, 1),
            "text":     plain_text,
            "segments": [
                {
                    "start": round(s.get("start", 0), 2),
                    "end":   round(s.get("end", 0), 2),
                    "text":  s.get("text", "").strip(),
                }
                for s in self._segments
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_srt(self, path):
        lines = []
        for i, seg in enumerate(self._segments, 1):
            start = self._srt_ts(seg.get("start", 0))
            end   = self._srt_ts(seg.get("end", 0))
            text  = seg.get("text", "").strip()
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _srt_ts(seconds):
        h  = int(seconds // 3600)
        m  = int((seconds % 3600) // 60)
        s  = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def _on_close(self):
        if self._transcribing:
            if not messagebox.askyesno(
                "Transcription in progress",
                "A transcription is still running. Close anyway?"
            ):
                return
        if self._recording:
            try:
                self._recorder.stop()
            except Exception:
                pass
        self._recorder.cleanup()
        self.destroy()

    # ── Helpers ────────────────────────────────────────────────────────────
    def _set_status(self, msg, color=None):
        self._status_var.set(msg)
        self._status_lbl.configure(fg=color or self.DIM)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
