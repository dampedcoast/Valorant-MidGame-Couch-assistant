import logging
from typing import List, Dict, Optional
from .schemas import TacticalEvent, Snapshot, Player

logger = logging.getLogger(__name__)

class TacticalEventLogger:
    def __init__(self):
        self.event_log: List[TacticalEvent] = []
        self.conclusions: List[str] = []

    def process_change(self, change: Dict, current_snapshot: Snapshot):
        """Analyzes a change and logs tactical events based on patterns."""
        event_type = change.get("type")
        player: Player = change.get("player")
        
        if event_type == "PLAYER_DIED":
            # Pattern: First death of the round
            alive_count = len([p for p in current_snapshot.players.values() if p.alive])
            total_players = len(current_snapshot.players)
            
            if alive_count == total_players - 1:
                event = TacticalEvent(
                    event_type="FIRST_DEATH",
                    description=f"First death of the round: {player.name} ({player.team_name})",
                    metadata={
                        "player": player.name,
                        "team": player.team_name,
                        "position": f"{player.position.region_rc} ({player.position.quadrant})",
                        "side": player.side
                    }
                )
                self.event_log.append(event)
                self._add_conclusion(f"Entry engagement lost by {player.team_name} at {player.position.region_rc}.")

        elif event_type == "WEAPON_CHANGE":
            # Pattern: Weapon disadvantage engagement (could be inferred if we had combat events, 
            # but here we just log significant upgrades/downgrades)
            old_w = change.get("old_weapon")
            new_w = change.get("new_weapon")
            
            # Simple heuristic for "Eco" vs "Full"
            if new_w in ["Vandal", "Phantom", "Operator"]:
                self._add_conclusion(f"{player.name} upgraded to {new_w}. Strength increased.")

    def get_tactical_conclusions(self) -> List[str]:
        """Returns summarized tactical insights."""
        return self.conclusions[-5:] # Last 5 insights

    def _add_conclusion(self, text: str):
        if text not in self.conclusions:
            self.conclusions.append(text)
            logger.info(f"Tactical Conclusion: {text}")
