import streamlit as st
import threading
import time
import traceback
import logging
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Optional, List

from agents.VLM import VLM
from agents.brain import Brain
from stt.stt_model import STT
from tts.tts_model import TTS


# -----------------------------
# Logging -> Queue Handler
# -----------------------------
class QueueLogHandler(logging.Handler):
    def __init__(self, q: Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        msg = self.format(record)
        self.q.put(msg)


# -----------------------------
# Runner state
# -----------------------------
@dataclass
class SkyState:
    running: bool = False
    vlm_recording: bool = False
    last_user_text: str = ""
    last_ai_text: str = ""
    last_vlm_text: str = ""
    error: Optional[str] = None
    series_ids: List[str] = None


def parse_series_ids(text: str) -> List[str]:
    """
    Accepts:
      - "2629390"
      - "2629390,2629391"
      - multiline: one per line
    Returns unique, stripped list preserving order.
    """
    if not text:
        return []
    raw = text.replace("\n", ",")
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]

    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# -----------------------------
# Core runner
# -----------------------------
class SkyRunner:
    """
    Runs STT -> Brain -> TTS in a background loop
    and also can start/stop VLM screen recording.
    """
    def __init__(self, log_q: Queue):
        self.log_q = log_q
        self.state = SkyState(series_ids=[])

        self.stop_event = threading.Event()

        # Components
        self.stt: Optional[STT] = None
        self.tts: Optional[TTS] = None
        self.vlm: Optional[VLM] = None
        self.brain: Optional[Brain] = None

        # Threads
        self.main_thread: Optional[threading.Thread] = None
        self.vlm_thread: Optional[threading.Thread] = None

        self.vlm_stop_event = threading.Event()

    def log(self, s: str):
        self.log_q.put(s)

    def init_components(self, series_ids: List[str]):
        """
        Initialize components once.
        If the user changes series_ids while stopped, re-init Brain/DataAgent to apply new IDs.
        """
        if self.stt is None:
            self.log("Loading STT...")
            self.stt = STT()

        if self.vlm is None:
            self.log("Loading VLM...")
            self.vlm = VLM()

        # If Brain exists but series_ids changed and system is not running, rebuild Brain
        if self.brain is None:
            self.log("Loading Brain...")
            self.brain = Brain(vlm=self.vlm)

        # Apply series IDs into DataAgent if possible
        if series_ids:
            self.state.series_ids = series_ids
            if hasattr(self.brain, "data_agent") and self.brain.data_agent is not None:
                da = self.brain.data_agent

                # If your DataAgent supports multi-series, set series_ids
                if hasattr(da, "series_ids"):
                    try:
                        da.series_ids = series_ids
                        self.log(f"✅ Set DataAgent.series_ids = {series_ids}")
                    except Exception:
                        pass

                # If DataAgent only supports a single series_id, pick the first
                if hasattr(da, "series_id"):
                    try:
                        da.series_id = series_ids[0]
                        self.log(f"✅ Set DataAgent.series_id = {series_ids[0]}")
                    except Exception:
                        pass
            else:
                self.log("⚠️ Brain has no data_agent to apply series IDs.")

        if self.tts is None:
            try:
                self.log("Loading TTS...")
                self.tts = TTS()
            except FileNotFoundError:
                self.log("WARNING: TTS model files not found. Speech disabled.")
                self.tts = None

    # -----------------------------
    # VLM capture controls
    # -----------------------------
    def start_vlm_recording(self):
        if self.state.vlm_recording:
            return

        self.vlm_stop_event.clear()
        self.state.vlm_recording = True
        self.log("✅ VLM recording ON")

        def _vlm_loop():
            try:
                if hasattr(self.vlm, "producer_loop"):
                    try:
                        # If your VLM supports stop_event
                        self.vlm.producer_loop(stop_event=self.vlm_stop_event)
                        return
                    except TypeError:
                        # producer_loop() without stop_event param
                        self.vlm.producer_loop()
                else:
                    self.log("ERROR: VLM has no producer_loop()")
            except Exception as e:
                self.log(f"VLM capture thread crashed: {e}\n{traceback.format_exc()}")
            finally:
                self.state.vlm_recording = False
                self.log("🛑 VLM recording OFF")

        self.vlm_thread = threading.Thread(target=_vlm_loop, daemon=True)
        self.vlm_thread.start()

    def stop_vlm_recording(self):
        if not self.state.vlm_recording:
            return

        self.vlm_stop_event.set()
        self.state.vlm_recording = False
        self.log("🛑 Stopping VLM recording...")

        try:
            if self.vlm and hasattr(self.vlm, "running"):
                setattr(self.vlm, "running", False)
        except Exception:
            pass

    # -----------------------------
    # Main loop controls
    # -----------------------------
    def start_sky_loop(self, series_ids: List[str]):
        if self.state.running:
            return

        self.stop_event.clear()
        self.state.running = True
        self.log("✅ Sky loop started")

        def _loop():
            try:
                self.init_components(series_ids=series_ids)

                while not self.stop_event.is_set():
                    # Optional: pull VLM events for display
                    try:
                        if self.vlm is not None and hasattr(self.vlm, "detect_events"):
                            raw = self.vlm.detect_events()
                            self.state.last_vlm_text = str(raw)
                    except Exception:
                        pass

                    # STT listen
                    user_text = None
                    try:
                        user_text = self.stt.listen(timeout=0.5) if self.stt else None
                    except Exception:
                        user_text = None

                    if user_text:
                        self.state.last_user_text = user_text
                        self.log(f"👤 User: {user_text}")

                        # Ask brain
                        try:
                            response = self.brain.ask(user_text, vlm_phase=None) if self.brain else "Brain not loaded."
                        except Exception as e:
                            response = f"Brain error: {e}\n{traceback.format_exc()}"

                        self.state.last_ai_text = response
                        self.log(f"🤖 Sky: {response}")

                        # Speak
                        if self.tts:
                            try:
                                self.tts.speak(response)
                            except Exception:
                                pass

                    time.sleep(0.05)

            except Exception as e:
                self.state.error = f"{e}\n{traceback.format_exc()}"
                self.log(f"❌ Sky loop crashed:\n{self.state.error}")
            finally:
                self.state.running = False
                self.log("🛑 Sky loop stopped")

        self.main_thread = threading.Thread(target=_loop, daemon=True)
        self.main_thread.start()

    def stop_sky_loop(self):
        if not self.state.running:
            return
        self.stop_event.set()
        self.state.running = False
        self.log("🛑 Stopping Sky loop...")

    # -----------------------------
    # Combined controls (ONE button start/stop)
    # -----------------------------
    def start_all(self, series_ids: List[str]):
        self.start_vlm_recording()
        self.start_sky_loop(series_ids=series_ids)

    def stop_all(self):
        self.stop_sky_loop()
        self.stop_vlm_recording()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Sky UI", layout="wide")

if "log_q" not in st.session_state:
    st.session_state.log_q = Queue()

if "runner" not in st.session_state:
    st.session_state.runner = SkyRunner(st.session_state.log_q)

if "logs" not in st.session_state:
    st.session_state.logs = []

# Attach queue logger once
if "logger_setup" not in st.session_state:
    handler = QueueLogHandler(st.session_state.log_q)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    st.session_state.logger_setup = True

runner: SkyRunner = st.session_state.runner

st.title("🎮 Sky - Streamlit Control Panel")

# -----------------------------
# Series ID input UI
# -----------------------------
st.markdown("### Series IDs")
series_text = st.text_area(
    "Enter series_id(s) (comma or newline separated)",
    value="2629390",
    height=90,
    help="Example: 2629390\n2629391\n2629392  OR  2629390,2629391",
)
series_ids = parse_series_ids(series_text)

if not series_ids:
    st.warning("Please enter at least one series_id before starting.")

st.divider()

# -----------------------------
# One-button Start/Stop
# -----------------------------
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    if st.button("🟢 Start Sky + VLM", use_container_width=True, disabled=(not series_ids)):
        runner.start_all(series_ids=series_ids)

with c2:
    if st.button("🔴 Stop Sky + VLM", use_container_width=True):
        runner.stop_all()

with c3:
    rec = runner.state.vlm_recording
    running = runner.state.running

    if running and rec:
        color = "green"
        text = "SKY + VLM RUNNING"
    elif running and not rec:
        color = "orange"
        text = "SKY RUNNING, VLM OFF"
    elif (not running) and rec:
        color = "orange"
        text = "VLM ON, SKY OFF"
    else:
        color = "red"
        text = "ALL STOPPED"

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;">
          <div style="width:14px;height:14px;border-radius:50%;background:{color};"></div>
          <div style="font-weight:700;font-size:16px;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# -----------------------------
# VLM Output
# -----------------------------
st.markdown("### Latest VLM Output (raw)")
st.text_area("VLM detect_events()", value=runner.state.last_vlm_text, height=120)

st.divider()

# -----------------------------
# Console logs + Conversation
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown("### Live Console Output (PowerShell-like)")

    # drain logs queue
    try:
        while True:
            st.session_state.logs.append(st.session_state.log_q.get_nowait())
    except Empty:
        pass

    N = 300
    log_text = "\n".join(st.session_state.logs[-N:])
    st.text_area("Logs", value=log_text, height=420)

with right:
    st.markdown("### Latest Conversation")
    st.markdown("**Series IDs:**")
    st.write(", ".join(runner.state.series_ids or []) or "—")
    st.markdown("**User:**")
    st.write(runner.state.last_user_text or "—")
    st.markdown("**Sky:**")
    st.write(runner.state.last_ai_text or "—")

st.divider()

# -----------------------------
# ✅ Always auto-refresh every 0.5s
# -----------------------------
time.sleep(0.5)
st.rerun()
