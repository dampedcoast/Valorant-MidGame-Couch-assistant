
import os
import json
import time
import logging
from grid_pipeline.polling import GRIDPoller
from agents.data_agent import DataAgent
from agents.brain import Brain
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_save_and_load_history():
    print("Starting test: Save and Load History from JSON")
    
    # Clean up previous data
    history_file = os.path.join("DATA", "history.json")
    if os.path.exists(history_file):
        os.remove(history_file)
    
    # 1. Initialize DataAgent (starts polling thread)
    data_agent = DataAgent(api_key="test-key", series_id="test-series")
    
    # 2. Let it poll a few times
    print("Waiting for polling to generate data...")
    time.sleep(2)
    
    # 3. Check if file exists and has data
    if os.path.exists(history_file):
        print(f"✅ Success: {history_file} exists.")
        with open(history_file, 'r') as f:
            data = json.load(f)
            print(f"✅ Success: Found {len(data)} snapshots in JSON file.")
            if len(data) > 0:
                print(f"Sample snapshot timestamp: {data[0]['timestamp']}")
    else:
        print(f"❌ Failure: {history_file} does NOT exist.")
        return

    # 4. Mock PostGameAgent to avoid LLM call
    brain = Brain()
    brain.data_agent = data_agent
    brain.post_game_agent.ask = MagicMock(return_value="Mocked Analysis")
    
    # 5. Call brain.ask with a post-game query
    print("Calling brain.ask(post_game query)...")
    response = brain.ask("Tell me about the last round history")
    
    # 6. Verify PostGameAgent was called with data from file
    args, kwargs = brain.post_game_agent.ask.call_args
    data_history = kwargs.get('data_history', '')
    
    if "Snapshot 0" in data_history:
        print("✅ Success: brain.ask passed data history from file to PostGameAgent.")
        # print(f"Data history passed to agent:\n{data_history[:200]}...")
    else:
        print("❌ Failure: brain.ask did NOT pass expected data history.")
        print(f"Data history was: {data_history}")

    # 7. Test round reset
    print("Testing round reset...")
    # Simulate round change in mock data
    data_agent.grid_poller._mock_counter = 0 # This might not trigger it if it doesn't change gameId
    # Manually trigger a clear by changing last_snapshot game_id
    data_agent.grid_poller.last_snapshot.game_id = "old-game"
    
    # Poll once more
    data_agent.grid_poller.poll_snapshot()
    
    if len(data_agent.grid_poller.snapshot_history) == 1: # One for the new poll
        print("✅ Success: snapshot_history reset on round change.")
    else:
        print(f"❌ Failure: snapshot_history not reset. Count: {len(data_agent.grid_poller.snapshot_history)}")

    with open(history_file, 'r') as f:
        data = json.load(f)
        if len(data) == 1:
            print("✅ Success: JSON file cleared/reset on round change.")
        else:
            print(f"❌ Failure: JSON file NOT reset. Count: {len(data)}")

    data_agent.grid_poller.running = False
    print("Test complete.")

if __name__ == "__main__":
    test_save_and_load_history()
