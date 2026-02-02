import os
import logging
from typing import Optional, List, Dict
import json
from .schemas import Snapshot
from .datause import (
    discover_player_inventory_field,
    build_series_state_query,
    fetch_series_state,
    build_rows_from_series_state,
)
api_key="gDVqIdMead2zTW1DKehu8PicvVXStT2xtmbYHK7b"
series_id="2629390"

logger = logging.getLogger(__name__)

class GRIDPoller:
    def __init__(self, api_key: str, series_id: str, poll_interval: int = 5):
        if not series_id:
            raise ValueError("The 'series_id' must be provided and cannot be None.")

        self.api_key = api_key
        self.series_id = series_id
        self.poll_interval = poll_interval
        self.running = False
        self.last_snapshot: Optional[Snapshot] = None
        self.snapshot_history: List[Snapshot] = []
        self.history_file = "DATA/history.json"
        
        # Discover fields once
        found = discover_player_inventory_field()
        if not found:
            logger.error("Could not discover GRID inventory fields.")
            self.player_type, self.inv_field = "GamePlayerStateValorant", "inventory"
        else:
            self.player_type, self.inv_field = found
            
        self.query = build_series_state_query(self.player_type, self.inv_field)
        self.differ = SnapshotDiffer()

        if not os.path.exists("DATA"):
            os.makedirs("DATA")

    def poll_snapshot(self) -> Optional[Snapshot]:
        """Fetches the latest state and converts it to a Snapshot object."""
        try:
            state = fetch_series_state(self.series_id, self.query)
            if not state:
                return None
            
            rows = build_rows_from_series_state(state, self.inv_field)
            if not rows:
                return None
            
            # Use the first row to get game_id
            game_id = rows[0]["game_id"]
            # Additional processing can go here
            return Snapshot(series_id=self.series_id, game_id=game_id, players={})
        except Exception as e:
            logger.error(f"Error polling snapshot: {e}")
            return None
    def _save_history(self):
        """Saves history to DATA/history.json."""
        try:
            # Simple list of dicts for JSON
            history_data = []
            for snap in self.snapshot_history[-50:]: # Keep last 50
                snap_dict = {
                    "series_id": snap.series_id,
                    "game_id": snap.game_id,
                    "timestamp": snap.timestamp,
                    "players": {
                        pid: {
                            "alive": p.alive,
                            "hp_bucket": p.hp_bucket,
                            "weapon": p.weapon
                        } for pid, p in snap.players.items()
                    }
                }
                history_data.append(snap_dict)
                
            with open(self.history_file, "w") as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")

class SnapshotDiffer:
    def diff(self, old: Optional[Snapshot], new: Snapshot) -> List[Dict]:
        """Detects significant changes between two snapshots."""
        changes = []
        if not old:
            return changes
        
        for pid, new_p in new.players.items():
            if pid not in old.players:
                continue
            
            old_p = old.players[pid]
            
            # Death event
            if old_p.alive and not new_p.alive:
                changes.append({
                    "type": "PLAYER_DIED",
                    "player": new_p,
                    "team": new_p.team_name
                })
            
            # Weapon change
            if old_p.weapon != new_p.weapon and new_p.weapon:
                changes.append({
                    "type": "WEAPON_CHANGE",
                    "player": new_p,
                    "old_weapon": old_p.weapon,
                    "new_weapon": new_p.weapon
                })
        
        return changes
