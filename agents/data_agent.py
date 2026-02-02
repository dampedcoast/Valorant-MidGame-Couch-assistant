from grid_pipeline.polling import GRIDPoller
from grid_pipeline.event_log import TacticalEventLogger
import threading
import time
import logging
import json
import os

logger = logging.getLogger(__name__)

class DataAgent:
    """
    Orchestrates data retrieval, combining visual data from VLM with 
    GRID real-time data to provide context to the tactical agents.
    """
    def __init__(self, vlm=None, api_key="gDVqIdMead2zTW1DKehu8PicvVXStT2xtmbYHK7b", series_id="2629390"):
        """
        Initializes the DataAgent with a VLM instance and GRID pipeline.
        """
        # Validate series_id
        if not series_id:
            raise ValueError("The 'series_id' must be provided and cannot be None.")

        self.vlm = vlm
        if self.vlm is None:
            try:
                from agents.VLM import VLM
                self.vlm = VLM()
            except ImportError:
                logger.warning("VLM module not found or dependencies missing. Visual features disabled.")
                self.vlm = None
            
        self.grid_poller = GRIDPoller(api_key=api_key, series_id=series_id)
        self.tactical_logger = TacticalEventLogger()
        
        # Start GRID polling in a background thread
        self.polling_thread = threading.Thread(target=self._run_polling, daemon=True)
        self.polling_thread.start()

    def _run_polling(self):
        """Background thread for GRID polling and tactical event logging."""
        self.grid_poller.running = True
        logger.info(f"GRID Polling started for series {self.grid_poller.series_id}")
        
        while self.grid_poller.running:
            try:
                snapshot = self.grid_poller.poll_snapshot()
                if snapshot:
                    self.grid_poller.snapshot_history.append(snapshot)
                    self.grid_poller._save_history() # Save to DATA/history.json
                    changes = self.grid_poller.differ.diff(self.grid_poller.last_snapshot, snapshot)
                    for change in changes:
                        # Feed changes to tactical logger
                        self.tactical_logger.process_change(change, snapshot)
                    
                    self.grid_poller.last_snapshot = snapshot
                
                time.sleep(self.grid_poller.poll_interval)
            except Exception as e:
                logger.error(f"Error in DataAgent polling loop: {e}")
                time.sleep(1)

    def fetch_data(self, query):
        """
        Fetches relevant game data based on the query, prioritizing GRID data.
        
        :param query: The user's query or a string indicating the type of data needed.
        :return: A string containing game data or tactical conclusions.
        """
        query_lower = query.lower()
        
        if "tactical" in query_lower or "conclusion" in query_lower:
            conclusions = self.tactical_logger.get_tactical_conclusions()
            return "\n".join(conclusions) if conclusions else "No significant tactical events logged yet."

        if "stats" in query_lower or "performance" in query_lower or "snapshot" in query_lower:
            snapshot = self.grid_poller.last_snapshot
            if snapshot:
                player_count = len(snapshot.players)
                alive_players = [p for p in snapshot.players.values() if p.alive]
                
                # Dynamic summary
                summary = f"GRID Snapshot (Game: {snapshot.game_id}): {len(alive_players)}/{player_count} players alive. "
                if alive_players:
                    top_player = alive_players[0]
                    summary += f"Example: {top_player.name} is at {top_player.position.region_rc} with {top_player.weapon}."
                return summary
            return "No live GRID data available for stats."
            
        elif "round" in query_lower:
            snapshot = self.grid_poller.last_snapshot
            if snapshot:
                alive_count = len([p for p in snapshot.players.values() if p.alive])
                return f"Round Status: {alive_count} players alive. Game ID: {snapshot.game_id}."
            return "No live GRID data available for round status."
        
        return f'Current state: {query}. VLM and GRID are analyzing the game. ' + \
               (f"GRID active for {self.grid_poller.series_id}." if self.grid_poller.last_snapshot else "GRID waiting for data.")

    def get_latest_events(self):
        """Returns the most recent tactical events."""
        return self.tactical_logger.event_log[-5:] if self.tactical_logger.event_log else []

    def get_snapshot_history_from_file(self):
        """Reads snapshot history from the JSON file."""
        try:
            if os.path.exists(self.grid_poller.history_file):
                with open(self.grid_poller.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading history from JSON: {e}")
        return []

    def get_snapshot_history(self, limit: int = 10):
        """Returns the most recent GRID snapshots."""
        return self.grid_poller.snapshot_history[-limit:] if self.grid_poller.snapshot_history else []
