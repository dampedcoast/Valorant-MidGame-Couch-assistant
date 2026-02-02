from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class Position:
    x: Optional[float] = None
    y: Optional[float] = None
    region_rc: str = "Unknown"
    x_band: str = "Unknown"
    y_band: str = "Unknown"
    quadrant: str = "Unknown"

@dataclass
class Player:
    id: str
    name: str
    team_name: str
    side: str
    agent: str
    alive: bool
    hp_bucket: str
    armor_bucket: str
    weapon: Optional[str] = None
    position: Position = field(default_factory=Position)

@dataclass
class Team:
    team_name: str
    side: str
    players: List[Player] = field(default_factory=list)

@dataclass
class Snapshot:
    series_id: str
    game_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    teams: List[Team] = field(default_factory=list)
    players: Dict[str, Player] = field(default_factory=dict) # Convenience mapping by player id or name

@dataclass
class TacticalEvent:
    event_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    metadata: Dict = field(default_factory=dict)
