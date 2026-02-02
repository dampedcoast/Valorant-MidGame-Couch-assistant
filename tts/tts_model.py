import os
import logging
import sounddevice as sd
from kokoro_onnx import Kokoro

# Setup logger for the module
logger = logging.getLogger(__name__)

class TTS:
    """
    A class to handle Text-to-Speech using the Kokoro ONNX model.
    """
    def __init__(self, model_path="tts_models/kokoro.onnx", voices_path="tts_models/voices.bin", default_voice="am_adam"):
        """
        Initialize the Kokoro TTS model.
        
        :param model_path: Path to the kokoro onnx model file.
        :param voices_path: Path to the voices bin file.
        :param default_voice: The default voice to use for speech generation.
        """
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            error_msg = f"Model or voices file not found at {model_path} or {voices_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading Kokoro TTS model from {model_path}...")
        try:
            self.kokoro = Kokoro(model_path, voices_path)
            self.default_voice = default_voice
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro model: {e}")
            raise

    def speak(self, text, voice=None, speed=1.0, wait=True):
        """
        Convert text to speech and play it.
        
        :param text: The text to speak.
        :param voice: The voice to use (optional, defaults to self.default_voice).
        :param speed: The speed of speech (default 1.0).
        :param wait: Whether to wait for the audio to finish playing (default True).
        """
        if not text or not text.strip():
            return

        voice_to_use = voice or self.default_voice
        logger.info(f"🔊 Speaking: {text}")
        
        try:
            samples, sample_rate = self.kokoro.create(
                text, 
                voice=voice_to_use, 
                speed=speed, 
                lang="en-us"
            )
            sd.play(samples, sample_rate)
            if wait:
                sd.wait()
        except Exception as e:
            logger.error(f"❌ Error during TTS: {e}")

    def stop(self):
        """Stop any currently playing audio."""
        sd.stop()

if __name__ == "__main__":
    # Configure basic logging for the test script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test the TTS class
    try:
        tts = TTS()
        tts.speak("Hello, my name is AI_val,and i am here to help you in your game, and this is the first try for the model")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
