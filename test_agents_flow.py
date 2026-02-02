import json
import os
from agents.brain import Brain
from agents.data_agent import DataAgent
from grid_pipeline.schemas import Snapshot, Player, Position, Team
from unittest.mock import MagicMock, patch

def setup_mock_data():
    if not os.path.exists("DATA"):
        os.makedirs("DATA")
    
    # Create a mock history.json
    mock_history = [
        {
            "series_id": "test-series",
            "game_id": "test-game",
            "timestamp": "2026-02-01T00:00:00",
            "players": {
                "Player1": {"alive": True, "hp_bucket": "80-100", "weapon": "Vandal"},
                "Player2": {"alive": False, "hp_bucket": "0", "weapon": "Classic"}
            }
        }
    ]
    with open("DATA/history.json", "w") as f:
        json.dump(mock_history, f)

def test_data_agent():
    print("--- Testing DataAgent ---")
    setup_mock_data()
    
    # Mock GRIDPoller to avoid API calls
    with patch('agents.data_agent.GRIDPoller') as MockPoller:
        mock_poller_inst = MockPoller.return_value
        mock_poller_inst.series_id = "test-series"
        mock_poller_inst.history_file = "DATA/history.json"
        mock_poller_inst.last_snapshot = Snapshot(
            series_id="test-series",
            game_id="test-game",
            players={
                "Player1": Player(id="Player1", name="Player1", team_name="TeamA", side="ATTACKER", agent="Jett", alive=True, hp_bucket="80-100", armor_bucket="full", weapon="Vandal")
            }
        )
        
        agent = DataAgent(api_key="mock", series_id="test-series")
        
        # Test fetching stats
        res = agent.fetch_data("Show me the stats")
        print(f"Stats Query: {res}")
        assert "GRID Snapshot" in res
        
        # Test fetching tactical conclusions
        agent.tactical_logger.conclusions = ["Enemy spotted at A Site"]
        res = agent.fetch_data("tactical conclusion")
        print(f"Tactical Query: {res}")
        assert "Enemy spotted at A Site" in res

def test_brain_routing():
    print("\n--- Testing Brain Routing ---")
    
    # Mock LLM and Chains in Brain
    with patch('agents.brain.Ollama'), \
         patch('agents.brain.PromptTemplate'), \
         patch('agents.brain.JsonOutputParser'), \
         patch('agents.brain.DataAgent'):
        
        brain = Brain()
        
        # Mock the router_chain entirely
        brain.router_chain = MagicMock()
        brain.router_chain.invoke.return_value = {"agent": "mid_game", "needs_data": True}
        
        route = brain.route("What should I do now?")
        print(f"Route for 'What should I do now?': {route}")
        assert route['agent'] == 'mid_game'
        assert route['needs_data'] is True

def test_full_flow():
    print("\n--- Testing Full Flow (Mocked Agents) ---")
    
    with patch('agents.brain.Ollama'), \
         patch('agents.brain.PromptTemplate'), \
         patch('agents.brain.JsonOutputParser'), \
         patch('agents.brain.DataAgent') as MockDataAgent:
        
        brain = Brain()
        
        # Mock Router
        brain.router_chain = MagicMock()
        brain.router_chain.invoke.return_value = {"agent": "mid_game", "needs_data": True}
        
        # Mock Agents
        brain.mid_game_agent = MagicMock()
        brain.mid_game_agent.ask.return_value = "Mocked Mid-Game Advice: Buy Armor."
        
        mock_data = MockDataAgent.return_value
        mock_data.fetch_data.return_value = "Mocked Round Data: 5v5."
        brain.data_agent = mock_data
        
        response = brain.ask("What is my strategy?")
        print(f"Brain Response: {response}")
        assert "Mocked Mid-Game Advice" in response

if __name__ == "__main__":
    try:
        test_data_agent()
        test_brain_routing()
        test_full_flow()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
