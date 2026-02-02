# agents/brain.py

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from agents.mid_game import MidGameAgent
from agents.post_game import PostGameAgent
from agents.data_agent import DataAgent

import os


# ✅ Hardcoded defaults (you can keep these here)
DEFAULT_API_KEY = "gDVqIdMead2zTW1DKehu8PicvVXStT2xtmbYHK7b"
DEFAULT_SERIES_ID = "2629390"


class Brain:
    """
    The central coordinator (Router) that interprets user intent and routes queries
    to the appropriate specialized agent (Mid-Round or Post-Game).
    """

    def __init__(
        self,
        model: str = "llama3.2:1b",
        temperature: float = 0,
        vlm=None,
        api_key: str = "",
        series_id: str = "",
    ):
        """
        Initializes the Brain with routing logic and specialized agents.

        Priority order for api_key/series_id:
        1) passed in arguments
        2) module defaults (hardcoded)
        3) environment variables (GRID_API_KEY / GRID_SERIES_ID)
        """

        # ✅ Resolve credentials safely (NO UnboundLocalError)
        resolved_api_key = (api_key or DEFAULT_API_KEY or os.getenv("GRID_API_KEY", "")).strip()
        resolved_series_id = (series_id or DEFAULT_SERIES_ID or os.getenv("GRID_SERIES_ID", "")).strip()

        if not resolved_api_key:
            raise ValueError("Missing GRID API key (pass api_key or set GRID_API_KEY).")
        if not resolved_series_id:
            raise ValueError("Missing GRID series id (pass series_id or set GRID_SERIES_ID).")

        self.api_key = resolved_api_key
        self.series_id = resolved_series_id

        # Initialize agents
        self.llm = Ollama(model=model, temperature=temperature)
        self.mid_game_agent = MidGameAgent(model=model, temperature=temperature)
        self.post_game_agent = PostGameAgent(model=model, temperature=temperature)

        self.data_agent = DataAgent(
            vlm=vlm,
            api_key=self.api_key,
            series_id=self.series_id,
        )

        self.router_prompt = PromptTemplate(
            input_variables=["input"],
            template="""
Task: Classify the VALORANT question.

"agent":
- "mid_game" if it's about what to do RIGHT NOW in a round.
- "post_game" if it's about what happened in the PAST or a general claim.

"needs_data":
- true if it needs live round data.
- false otherwise.

Input: {input}

Return ONLY JSON: {{"agent": "...", "needs_data": ...}}
""",
        )

        # Chain (LCEL) with improved parsing
        self.router_chain = self.router_prompt | self.llm | JsonOutputParser()

    def route(self, user_input: str):
        """
        Classifies the user input to determine the appropriate agent and data needs.
        """
        try:
            return self.router_chain.invoke({"input": user_input})
        except Exception:
            # Fallback logic if JSON parsing fails
            return {"agent": "mid_game", "needs_data": True}

    def ask(self, user_input: str, vlm_phase=None):
        """
        Processes a user question by routing it and invoking the appropriate agent.
        """
        route_info = self.route(user_input)

        data = None
        if route_info.get("needs_data", False):
            data = self.data_agent.fetch_data(user_input)

        if route_info.get("agent") == "mid_game":
            round_data = data if data else "No live data available."
            return self.mid_game_agent.ask(round_data=round_data, question=user_input)
        else:
            history = self.data_agent.get_snapshot_history_from_file()
            history_str = ""

            if history:
                for i, snap_dict in enumerate(history):
                    history_str += f"Snapshot {i} ({snap_dict.get('timestamp', 'N/A')}):\n"
                    players = snap_dict.get("players", {})
                    for pid, pstate in players.items():
                        history_str += (
                            f"  - {pid}: "
                            f"HP={pstate.get('hp_bucket', 'N/A')}, "
                            f"Weapon={pstate.get('weapon', 'N/A')}, "
                            f"Alive={pstate.get('alive', 'N/A')}\n"
                        )
            else:
                history_str = "No GRID data history available."

            return self.post_game_agent.ask(claim=user_input, data_history=history_str)

    def handle_event(self, event_type: str, metadata=None):
        """
        Handles autonomous events detected by the VLM or GRID pipeline.
        """
        if event_type == "round_ended":
            return self.post_game_agent.ask(
                "The round has ended. Provide a brief, constructive feedback about the game performance "
                "and suggest improvements for the next round."
            )

        elif event_type in ("player_killed_enemy", "PLAYER_KILLED"):
            stats = self.data_agent.fetch_data("current performance snapshot")
            tactical = self.data_agent.fetch_data("tactical conclusion")
            return f"Good job on that kill! {tactical} Here is a quick snapshot: {stats}"

        elif event_type in ("player_died", "PLAYER_DIED"):
            stats = self.data_agent.fetch_data("round end statistics")
            tactical = self.data_agent.fetch_data("tactical conclusion")
            return f"Tough break. {tactical} Here are your stats for the round: {stats}"

        elif event_type == "FIRST_DEATH":
            pos = metadata.get("position") if metadata else "Unknown"
            return (
                f"Warning: First death of the round. Current position context: {pos}. "
                "IGL should consider rotation."
            )

        elif event_type == "WEAPON_DISADVANTAGE_ENGAGEMENT":
            return "Tactical Alert: Engagement initiated with weapon disadvantage. Play for trades."

        return None
