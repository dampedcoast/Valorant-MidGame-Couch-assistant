import speech_recognition as sr
import whisper
import os

class STT:
    """
    Hands-free interaction using OpenAI Whisper, allowing players to speak naturally
    and converting their speech to text.
    """
    def __init__(self, model_name="tiny.en"):
        """
        Initializes the STT module, loads the Whisper model, and adjusts for ambient noise.

        :param model_name: The name/size of the Whisper model to load (e.g., 'tiny.en').
        """
        print(f"Loading Whisper model '{model_name}'...")
        self.model = whisper.load_model(model_name)
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.temp_file = "temp.wav"
        
        # Adjust for ambient noise once at initialization
        with self.mic as source:
            print("🔊 Adjusting for ambient noise... (Please be quiet for 1s)")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def listen(self, timeout=None, phrase_time_limit=None):
        """
        Listens for audio input from the microphone and transcribes it using Whisper.

        :param timeout: Maximum number of seconds to wait for a phrase to start.
        :param phrase_time_limit: Maximum number of seconds to allow a phrase to continue.
        :return: The transcribed text, or an empty string if no speech was detected or an error occurred.
        """
        with self.mic as source:
            # pause_threshold is the number of seconds of silence after speech to consider the phrase finished.
            self.recognizer.pause_threshold = 2.0

            print(f"🎤 Listening... (Timeout: {timeout}s)")
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                print("⌛ Processing audio...")
                with open(self.temp_file, "wb") as f:
                    f.write(audio.get_wav_data())

                result = self.model.transcribe(self.temp_file)
                text = result["text"].strip()
                return text
            except sr.WaitTimeoutError:
                return ""
            except Exception as e:
                print(f"❌ Error during STT: {e}")
                return ""
            finally:
                if os.path.exists(self.temp_file):
                    os.remove(self.temp_file)

if __name__ == "__main__":
    stt = STT()
    print("✅ Ready for listening... ")
    try:
        while True:
            text = stt.listen()
            if text:
                print(f"💬 {text}")
    except KeyboardInterrupt:
        print("\n🛑 Exiting...")