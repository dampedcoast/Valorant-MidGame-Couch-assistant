import mss
import cv2
import numpy as np
import base64
import requests
import time
import threading
import json
from queue import Queue, Empty

# Target Ollama configuration for local VLM inference
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-vl:2b"

# VALORANT UI Regions (assuming 1920x1080 resolution)
# Defined to capture only the minimal necessary context
KILLFEED_REGION = {
    "top": 40,
    "left": 1240,
    "width": 640,
    "height": 260
}

ROUND_END_REGION = {
    "top": 260,
    "left": 350,
    "width": 1220,
    "height": 340
}

# Performance Tuning
SCALE_FACTOR = 0.5  # Downscale image to reduce VLM processing latency and payload size
INFERENCE_FPS = 2  # Target frequency for VLM queries to balance accuracy and system load
EVENT_COOLDOWN = 2.0  # Seconds to wait before reporting the same event type

# System Prompt as required
SYSTEM_PROMPT = """You are a visual referee for a professional VALORANT match.

Classify exactly ONE label:
- KILL
- DEATH
- ROUND_END
- NO_EVENT

Only output the label."""


class VLM:
    """
    A visual processing agent that monitors game events (kills, deaths, round ends) 
    from screenshots in real-time using a Vision-Language Model.
    """
    def __init__(self):
        """
        Initializes the VLM agent, sets up screen capture, and pre-calculates image scaling.
        """
        self.sct = mss.mss()
        # Thread-safe queue for sharing the latest frame between threads
        self.frame_queue = Queue(maxsize=1)
        self.running = True
        self.last_event_time = {
            "KILL": 0,
            "DEATH": 0,
            "ROUND_END": 0
        }

        # Pre-calculate stitching and scaling parameters for loop efficiency
        self.target_width = ROUND_END_REGION["width"]
        self.kf_scale = self.target_width / KILLFEED_REGION["width"]
        self.kf_target_h = int(KILLFEED_REGION["height"] * self.kf_scale)

        # Final dimensions for the VLM payload
        self.final_w = int(self.target_width * SCALE_FACTOR)
        self.final_h = int((ROUND_END_REGION["height"] + self.kf_target_h) * SCALE_FACTOR)

    def producer_loop(self):
        """
        Asynchronous capture thread.
        Continuously grabs and processes screen regions at high frequency.
        """
        while self.running:
            try:
                # 1. Capture regions using mss (very low overhead)
                # We convert to numpy and slice out the alpha channel immediately
                kf_raw = np.array(self.sct.grab(KILLFEED_REGION))[:, :, :3]
                re_raw = np.array(self.sct.grab(ROUND_END_REGION))[:, :, :3]

                # 2. Efficiently stitch regions vertically
                # Resize killfeed to match roundend width using fast interpolation
                kf_resized = cv2.resize(kf_raw, (self.target_width, self.kf_target_h), interpolation=cv2.INTER_NEAREST)
                combined = np.vstack((re_raw, kf_resized))

                # 3. Downscale for VLM processing efficiency
                # AREA interpolation is better for downscaling text/fine details
                processed_frame = cv2.resize(combined, (self.final_w, self.final_h), interpolation=cv2.INTER_AREA)

                # 4. Push to queue, dropping old frames if necessary to ensure low latency
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put(processed_frame)

                # Throttle capture loop to reasonable game-refresh rates
                time.sleep(0.01)
            except Exception as e:
                # Critical for preventing thread death on transient errors
                time.sleep(1)

    def detect_events(self, image_obj=None):
        """
        Analyzes the screen or a provided image to detect game events.

        :param image_obj: Optional PIL image to analyze. If None, uses the latest frame from the queue.
        :return: A JSON string containing detected events (player_killed_enemy, player_died, round_ended, mid).
        """
        if image_obj is not None:
            # Convert PIL Image to numpy array (RGB to BGR for cv2)
            img = cv2.cvtColor(np.array(image_obj), cv2.COLOR_RGB2BGR)
            
            # Since main.py passes a full screenshot, we need to crop it or process it 
            # similar to how producer_loop does, but for simplicity of 'plugging it in',
            # we can just use the full image if it's already cropped, or assume we need to crop.
            # However, OptimizedValorantAgent expects specific regions.
            
            # Let's handle the full screenshot case by cropping regions
            try:
                # We need to ensure the image_obj is at least as large as our regions
                # mss might use different coordinate system or capture might be different size
                # but we'll try to crop according to our defined regions
                kf_raw = img[KILLFEED_REGION["top"]:KILLFEED_REGION["top"]+KILLFEED_REGION["height"], 
                             KILLFEED_REGION["left"]:KILLFEED_REGION["left"]+KILLFEED_REGION["width"]]
                re_raw = img[ROUND_END_REGION["top"]:ROUND_END_REGION["top"]+ROUND_END_REGION["height"], 
                             ROUND_END_REGION["left"]:ROUND_END_REGION["left"]+ROUND_END_REGION["width"]]
                
                # Check if we actually got something (cropping might return empty if out of bounds)
                if kf_raw.size == 0 or re_raw.size == 0:
                     raise ValueError("Cropped region is empty")

                kf_resized = cv2.resize(kf_raw, (self.target_width, self.kf_target_h), interpolation=cv2.INTER_NEAREST)
                combined = np.vstack((re_raw, kf_resized))
                processed_frame = cv2.resize(combined, (self.final_w, self.final_h), interpolation=cv2.INTER_AREA)
            except Exception as e:
                # Fallback to just resizing the input if cropping fails
                processed_frame = cv2.resize(img, (self.final_w, self.final_h), interpolation=cv2.INTER_AREA)
        else:
            # Try to get the latest frame from the queue if no image is provided
            try:
                processed_frame = self.frame_queue.get_nowait()
            except Empty:
                return json.dumps({"NO_EVENT": True})

        event = self.query_vlm(processed_frame)
        
        # Map VLM labels to what main.py and brain.py expect
        # main.py expects: json with keys like "mid", "round_ended", "player_killed_enemy", "player_died"
        # Optimized labels: "KILL", "DEATH", "ROUND_END", "NO_EVENT"
        
        events = {
            "player_killed_enemy": event == "KILL",
            "player_died": event == "DEATH",
            "round_ended": event == "ROUND_END",
            "mid": event == "NO_EVENT" # If nothing special, assume mid-game
        }
        
        return json.dumps(events)

    def query_vlm(self, img_array):
        """
        Sends the processed image to the Ollama VLM for inference.

        :param img_array: The numpy array representing the processed image.
        :return: The detected label as a string.
        """
        # Compress to JPEG to reduce network overhead for Ollama
        _, buffer = cv2.imencode('.jpg', img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        b64_image = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "model": MODEL_NAME,
            "prompt": SYSTEM_PROMPT,
            "images": [b64_image],
            "stream": False,
            "options": {
                "temperature": 0.0,  # Zero temperature for maximum classification consistency
                "num_predict": 10  # Short responses for classification
            }
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=5)
            response.raise_for_status()
            text = response.json().get("response", "NO_EVENT").strip().upper()
            # Strict filtering to ensure we only return expected labels
            for label in ["KILL", "DEATH", "ROUND_END", "NO_EVENT"]:
                if label in text:
                    return label
            return "NO_EVENT"
        except requests.exceptions.RequestException as e:
            return f"ERROR: {str(e)}"
        except Exception as e:
            return f"ERROR: {str(e)}"

    def run(self):
        """
        Main execution loop for the VLM agent. 
        Starts the capture thread and performs continuous inference.
        """
        # Start capture in a background thread to overlap I/O with VLM inference
        capture_thread = threading.Thread(target=self.producer_loop, daemon=True)
        capture_thread.start()

        print("--- Optimized VALORANT VLM Agent Started ---")
        print(f"Targeting {INFERENCE_FPS} Hz inference on {MODEL_NAME}")
        print("Monitoring Killfeed and Round Events. Press CTRL+C to exit.")

        try:
            while self.running:
                loop_start = time.time()

                # Get latest frame from producer
                try:
                    frame = self.frame_queue.get(timeout=2)
                except Empty:
                    continue

                # Execute VLM classification
                event = self.query_vlm(frame)

                # Event debouncing and reporting logic
                current_time = time.time()
                if event in self.last_event_time:
                    if current_time - self.last_event_time[event] > EVENT_COOLDOWN:
                        print(f"[{time.strftime('%H:%M:%S')}] DETECTED: {event}")
                        self.last_event_time[event] = current_time
                elif event == "NO_EVENT":
                    # Explicitly printing NO_EVENT to confirm model health as requested
                    print(f"[{time.strftime('%H:%M:%S')}] {event}")
                elif "ERROR" in event:
                    print(f"[{time.strftime('%H:%M:%S')}] {event}")

                # Calculate remaining time to maintain target FPS
                elapsed = time.time() - loop_start
                delay = (1.0 / INFERENCE_FPS) - elapsed
                if delay > 0:
                    time.sleep(delay)

        except KeyboardInterrupt:
            self.running = False
            print("\nShutting down gracefully...")
        finally:
            print("Cleanup complete.")


if __name__ == "__main__":
    agent = VLM()
    agent.run()
