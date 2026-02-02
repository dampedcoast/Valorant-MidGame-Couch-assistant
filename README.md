# Sky Final: VALORANT Decision Support AI

An AI-powered decision support system for professional VALORANT players and coaches. It provides real-time tactical advice (Mid-Round) and post-round analysis (Post-Game) using LLMs, speech-to-text, and text-to-speech.

## 🚀 Overview

Sky Final is a multi-agent system designed to assist VALORANT teams in high-pressure environments:
- **Brain (Router)**: The central coordinator that interprets user intent and routes queries to the appropriate specialized agent.
- **Mid-Game Agent**: A live-round tactical advisor. It takes real-time round data and provides exactly two actionable options for the IGL (In-Game Leader) under strict time constraints.
- **Post-Game Agent**: A tactical analyst for post-round or general strategic queries. It evaluates claims and explains trade-offs between different tactical approaches.
- **VLM (Vision-Language Model)**: A visual processing agent that monitors game events (kills, deaths, round ends) from screenshots in real-time.
- **GRID Pipeline**: Integrates real-time game data via GRID's GraphQL API, tracking player states, inventory, and tactical events.
- **STT (Speech-to-Text)**: Hands-free interaction using OpenAI Whisper, allowing players to speak naturally.
- **TTS (Text-to-Speech)**: High-quality voice feedback using Kokoro ONNX, enabling the AI to "talk back" to the team.
- **Data Agent**: Orchestrates data retrieval, combining visual data (VLM) with real-time API data (GRID) to provide context to the tactical agents.

## 🤖 Agent Workflow

1.  **Input**: The system listens for voice input via `STT`.
2.  **Routing**: The `Brain` uses an LLM to classify the input as `mid_game` (immediate tactical) or `post_game` (strategic analysis).
3.  **Context**: 
    -   If data is needed, the `Data Agent` fetches current game state from the GRID pipeline.
    -   The `VLM` continuously monitors the screen for autonomous events (e.g., a kill).
4.  **Inference**:
    -   `MidGameAgent` provides two concise tactical options based on live round data.
    -   `PostGameAgent` provides a detailed analysis using historical snapshot data.
5.  **Output**: The response is printed and spoken back to the user via `TTS`.

## 🛠 Stack

- **Language**: Python 3.12+
- **LLM Framework**: [LangChain](https://www.langchain.com/) (LCEL)
- **Local LLM**: [Ollama](https://ollama.com/) (default: `llama3.2:1b`)
- **Speech-to-Text**: [OpenAI Whisper](https://github.com/openai/whisper)
- **Text-to-Speech**: [Kokoro ONNX](https://github.com/theodoregit/kokoro-onnx)
- **Computer Vision**: [OpenCV](https://opencv.org/) & [mss](https://github.com/boboTIKI/python-mss)
- **Data API**: [GRID GraphQL API](https://grid.gg/)
- **Package Management**: Pip
- **Audio I/O**: PyAudio, SoundDevice, and SpeechRecognition

## 📋 Requirements

### System Dependencies
- **Python 3.12+**
- **PortAudio**: Required for `PyAudio` and `sounddevice`.
  - MacOS: `brew install portaudio`
  - Linux: `sudo apt-get install libportaudio2`
- **Ollama**: Must be installed and running locally.

### Model Files
- **Ollama Model**: `llama3.2:1b` (ensure it's pulled)
- **Whisper Model**: `tiny.en` (downloaded automatically on first run)
- **Kokoro TTS**:
  - `kokoro.onnx`
  - `voices.bin`
  - Both files must be placed in the `tts_models/` directory.

## ⚙️ Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sky-final
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # MacOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Model Initialization**:
   - Pull the Ollama model:
     ```bash
     ollama pull llama3.2:1b
     ```
   - Ensure `tts_models/kokoro.onnx` and `tts_models/voices.bin` are present. if not pleas dawload them from [here](https://github.com/nazdridoy/kokoro-tts)

## 🔐 Environment Variables

The project uses several environment variables for configuration. Some defaults are currently hardcoded in the agents but can be overridden.

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `GRID_API_KEY` | Your GRID API key for GraphQL queries. | `gDVqIdMead...` |
| `GRID_SERIES_ID` | The specific series ID to monitor. | `2629390` |
| `OLLAMA_HOST` | Host address for Ollama API. | `localhost` |

## 🏃 Scripts & Entry Points

### Main Applications
- **CLI Mode**: The primary entry point that runs the full multi-agent system in the terminal.
  ```bash
  python main.py
  ```
- **GUI Mode**: An experimental application runner with logging and component initialization.
  ```bash
  python app.py
  ```

### Data Tools
- **Snapshot Generator**: Fetches live data for series/games listed in `alive_players_midgame.csv` and saves snapshots to `Data/`.
  ```bash
  python grid_pipeline/snapshot_live.py
  ```

### Component Tests & Debugging
You can run individual modules to verify specific functionalities:
- **Mid-Game Agent**: `python agents/mid_game.py`
- **STT (Microphone Test)**: `python stt/stt_model.py`
- **TTS (Voice Output Test)**: `python tts/tts_model.py`
- **VLM (Vision Test)**: `python agents/VLM.py`
- **GRID Pipeline**: `python grid_pipeline/polling.py`

## 📁 Project Structure

```text
.
├── agents/             # AI Agent Logic
│   ├── brain.py        # Central Router and coordinator
│   ├── mid_game.py     # Live round tactical advisor
│   ├── post_game.py    # Post-round analyst
│   ├── data_agent.py   # Orchestrator for VLM and GRID data
│   └── VLM.py          # Vision-Language Model for screen analysis
├── grid_pipeline/      # GRID Data Integration
│   ├── polling.py      # Real-time data fetching logic
│   ├── datause.py      # GraphQL queries and data transformations
│   ├── schemas.py      # Pydantic models for game state
│   ├── event_log.py    # Tactical event detection and logging
│   └── snapshot_live.py # Live data collection tool
├── stt/                # Speech-to-Text (Whisper)
│   └── stt_model.py    
├── tts/                # Text-to-Speech (Kokoro)
│   └── tts_model.py    
├── tts_models/         # Storage for ONNX/Bin model files
├── Data/               # Local data storage (e.g., history.json)
├── main.py             # Main system entry point (CLI)
├── app.py              # Application runner / GUI
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🧪 Tests

### Integration Tests
Run the following script to test the interaction between different agents and the data pipeline:
```bash
python test_agents_flow.py
```

### Other Tests
- `test_json_history.py`: Tests the persistence and loading of game state from `Data/history.json`.
- **Modular Tests**: Most files in `agents/`, `stt/`, `tts/`, and `grid_pipeline/` can be run directly as scripts for unit-level verification.

## 📝 TODO

- [ ] (High) Move hardcoded `API_KEY` and `SERIES_ID` from `main.py` and `agents/brain.py` to a `.env` file.
- [ ] (High) Implement a formal test suite using `pytest`.
- [ ] Connect `data_agent.py` to a production-ready GRID series ID dynamically.
- [ ] Refine `VLM.py` classification logic with more robust vision models.
- [ ] Implement an asynchronous event bus to reduce latency in `main.py`.
- [ ] Improve `app.py` UI and error handling.

