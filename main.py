# main.py
import logging
import sys
import json
import time
import threading
import traceback

from stt.stt_model import STT
from agents.brain import Brain
from tts.tts_model import TTS
from agents.VLM import VLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ Hardcoded (no PowerShell needed)
API_KEY = "gDVqIdMead2zTW1DKehu8PicvVXStT2xtmbYHK7b"

# ✅ Your CSV sheet that contains series_id / game_id
# Put it in the same folder as main.py OR change this to an absolute path.
SERIES_SHEET_CSV = r"alive_players_midgame.csv"


def main() -> None:
    """
    Main entry point for the Sky Final application.
    Initializes STT, Brain (Router), VLM, and TTS, then enters a loop to
    listen for user input and monitor game events.
    """
    print("--- Sky Final: VALORANT Decision Support AI ---")

    api_key = API_KEY
    if not api_key:
        logger.error("API key is missing.")
        sys.exit(1)

    # -------------------------
    # Init core components once
    # -------------------------
    try:
        stt = STT()
        vlm = VLM()

        # ✅ IMPORTANT:
        # Brain must already support multi-series internally (DataAgent loads SERIES_SHEET_CSV and polls all series).
        # Do NOT pass series_sheet_csv unless you actually added it to Brain.__init__.
        brain = Brain(vlm=vlm, api_key=api_key)

        # ✅ If your Brain exposes a multi-series DataAgent that can load from CSV, set it here.
        # This avoids "unexpected keyword" errors.
        if hasattr(brain, "data_agent") and brain.data_agent is not None:
            # API key
            if hasattr(brain.data_agent, "set_api_key"):
                brain.data_agent.set_api_key(api_key)
            elif hasattr(brain.data_agent, "api_key"):
                brain.data_agent.api_key = api_key

            # Load series list from sheet (only if your DataAgent supports it)
            if hasattr(brain.data_agent, "load_series_ids_from_csv"):
                brain.data_agent.load_series_ids_from_csv(SERIES_SHEET_CSV)
            elif hasattr(brain.data_agent, "series_sheet_csv"):
                brain.data_agent.series_sheet_csv = SERIES_SHEET_CSV

        # Start VLM background capture thread
        vlm_thread = threading.Thread(target=vlm.producer_loop, daemon=True)
        vlm_thread.start()

        # TTS is optional
        try:
            tts = TTS()
        except FileNotFoundError:
            logger.warning("TTS model files not found. Speech output will be disabled.")
            tts = None

    except Exception:
        logger.error("Failed to initialize system:\n" + traceback.format_exc())
        sys.exit(1)

    print("\n✅ System ready. Sky is listening for your questions or watching for game events.")
    print("Press Ctrl+C to exit.")

    # -------------------------
    # Main loop
    # -------------------------
    try:
        while True:
            user_text = stt.listen(timeout=0.5)

            if user_text:
                print(f"\n👤 User: {user_text}")
                print("🧠 Thinking...")

                # Capture screen and ask VLM (interpret output as phase)
                phase = {"mid": None}
                try:
                    vlm_json = vlm.detect_events()
                    vlm_out = json.loads(vlm_json)

                    if isinstance(vlm_out, dict):
                        if "mid" in vlm_out:
                            phase["mid"] = bool(vlm_out["mid"])
                        elif "mid_game" in vlm_out:
                            phase["mid"] = bool(vlm_out["mid_game"])
                        else:
                            phase["mid"] = None
                except Exception:
                    phase = {"mid": None}

                response = brain.ask(user_text, vlm_phase=phase)
                print(f"🤖 Sky: {response}")

                if tts:
                    try:
                        tts.speak(response)
                    except Exception:
                        pass

                continue

            # -------------------------
            # Watch for autonomous VLM events
            # -------------------------
            try:
                events_json = vlm.detect_events()
                events = json.loads(events_json)

                if isinstance(events, dict):
                    for event_type, occurred in events.items():
                        if occurred:
                            print(f"\n🔔 Event detected: {event_type}")

                            response = brain.handle_event(event_type)
                            if response:
                                print(f"🤖 Sky: {response}")
                                if tts:
                                    try:
                                        tts.speak(response)
                                    except Exception:
                                        pass

                            time.sleep(5)
                            break

                # -------------------------
                # ALSO check for GRID tactical events (multi-series safe)
                # -------------------------
                if hasattr(brain, "data_agent") and brain.data_agent is not None:
                    latest_grid_events = brain.data_agent.get_latest_events()

                    if latest_grid_events:
                        for ge in latest_grid_events:
                            # supports both object-style and dict-style events
                            if isinstance(ge, dict):
                                event_type = ge.get("event_type") or ge.get("type") or "UNKNOWN"
                                metadata = ge.get("metadata")
                            else:
                                event_type = getattr(ge, "event_type", None) or getattr(ge, "type", None) or "UNKNOWN"
                                metadata = getattr(ge, "metadata", None)

                            print(f"\n🔔 GRID Event detected: {event_type}")
                            response = brain.handle_event(event_type, metadata=metadata)

                            if response:
                                print(f"🤖 Sky (GRID): {response}")
                                if tts:
                                    try:
                                        tts.speak(response)
                                    except Exception:
                                        pass

                        # ✅ Clear shared event log (multi-series DataAgent version)
                        try:
                            if hasattr(brain.data_agent, "tactical_logger") and hasattr(brain.data_agent.tactical_logger, "event_log"):
                                brain.data_agent.tactical_logger.event_log = []
                        except Exception:
                            pass

            except Exception:
                pass

    except KeyboardInterrupt:
        print("\n👋 Exiting Sky Final. Good luck in your games!")


if __name__ == "__main__":
    main()
