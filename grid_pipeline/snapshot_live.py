# snapshot_live.py
# Uses ALL (series_id, game_id) pairs from a CSV sheet and writes one snapshot per pair.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from datause import fetch_alive_df

# =========================================
# CONFIG
# =========================================
OUT_DIR = Path("Data")
PREFIX = "snapshot_"
PLAYERS_PER_TEAM = 5

# how many series pages to fetch from central
PAGES = 3
PAGE_SIZE = 20

# ✅ Your sheet export (must contain: series_id, game_id)
PAIR_SHEET_CSV = Path("alive_players_midgame.csv")  # change if your file name differs


# =========================================
# HELPERS
# =========================================
def load_pairs_from_csv(path: Path) -> List[Tuple[str, str]]:
    """
    Loads (series_id, game_id) pairs from a CSV with columns: series_id, game_id
    Removes duplicates while preserving order.
    """
    if not path.exists():
        raise SystemExit(f"Missing CSV: {path}. Put it next to snapshot_live.py or fix PAIR_SHEET_CSV.")

    df_pairs = pd.read_csv(path, dtype=str)

    if "series_id" not in df_pairs.columns or "game_id" not in df_pairs.columns:
        raise SystemExit(
            f"{path} must have columns: series_id, game_id. Found columns: {list(df_pairs.columns)}"
        )

    df_pairs = df_pairs[["series_id", "game_id"]].dropna()
    df_pairs["series_id"] = df_pairs["series_id"].astype(str).str.strip()
    df_pairs["game_id"] = df_pairs["game_id"].astype(str).str.strip()
    df_pairs = df_pairs[(df_pairs["series_id"] != "") & (df_pairs["game_id"] != "")]

    # dedupe but keep order
    seen: Set[Tuple[str, str]] = set()
    ordered: List[Tuple[str, str]] = []
    for sid, gid in df_pairs.itertuples(index=False, name=None):
        key = (sid, gid)
        if key not in seen:
            seen.add(key)
            ordered.append(key)

    return ordered


def safe_filename(s: str) -> str:
    # make filenames safe on Windows
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def group_into_teams(df_game: pd.DataFrame, players_per_team: int = 5) -> List[Dict[str, Any]]:
    teams_out: List[Dict[str, Any]] = []

    for (team_name, side), g in df_game.groupby(["team_name", "side"], dropna=False):
        g = g.copy().head(players_per_team)

        players = []
        for _, r in g.iterrows():
            players.append(
                {
                    "player_name": r.get("player_name"),
                    "agent": r.get("agent_raw"),
                    "weapon": r.get("weapon_current"),
                    "alive": bool(r.get("alive")) if pd.notna(r.get("alive")) else None,
                    "hp_bucket": r.get("hp_bucket"),
                    "armor_bucket": r.get("armor_bucket"),
                    "position": {
                        "x": None if pd.isna(r.get("pos_x")) else float(r.get("pos_x")),
                        "y": None if pd.isna(r.get("pos_y")) else float(r.get("pos_y")),
                        "region_rc": r.get("region_rc"),
                        "x_band": r.get("x_band"),
                        "y_band": r.get("y_band"),
                        "quadrant": r.get("pos_quadrant"),
                    },
                }
            )

        teams_out.append({"team_name": team_name, "side": side, "players": players})

    return teams_out


def make_snapshot(df: pd.DataFrame, series_id: str, game_id: str) -> Dict[str, Any]:
    df_game = df[(df["series_id"] == series_id) & (df["game_id"] == game_id)].copy()

    return {
        "id": None,
        "series_id": series_id,
        "game_id": game_id,
        "teams": group_into_teams(df_game, players_per_team=PLAYERS_PER_TEAM),
    }


def write_snapshot(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# =========================================
# MAIN
# =========================================
def main() -> None:
    # 0) Load all pairs from your sheet
    pairs = load_pairs_from_csv(PAIR_SHEET_CSV)
    if not pairs:
        raise SystemExit(f"No (series_id, game_id) pairs found in {PAIR_SHEET_CSV}.")

    print(f"Loaded {len(pairs)} unique (series_id, game_id) pairs from {PAIR_SHEET_CSV.name}.")

    # 1) Fetch LIVE alive players from series-state
    df = fetch_alive_df(pages=PAGES, page_size=PAGE_SIZE)
    if df.empty:
        raise SystemExit("No rows returned from fetch_alive_df().")

    # normalize types
    df["series_id"] = df["series_id"].astype(str).str.strip()
    df["game_id"] = df["game_id"].astype(str).str.strip()

    # 2) Filter df to only the pairs we care about
    pair_set = set(pairs)
    df = df[df.apply(lambda r: (r["series_id"], r["game_id"]) in pair_set, axis=1)].copy()

    if df.empty:
        raise SystemExit(
            "No live rows matched your sheet pairs. "
            "This usually means those series/games are not present in the pages you fetched (PAGES/PAGE_SIZE), "
            "or they are no longer live."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3) Write one snapshot per pair (only if we have rows for it)
    wrote = 0
    missing = 0

    for (series_id, game_id) in pairs:
        df_game = df[(df["series_id"] == series_id) & (df["game_id"] == game_id)]
        if df_game.empty:
            missing += 1
            continue

        snap = make_snapshot(df, series_id, game_id)

        fname = f"{PREFIX}{safe_filename(series_id)}_{safe_filename(game_id)}.json"
        out_path = OUT_DIR / fname
        write_snapshot(out_path, snap)

        wrote += 1
        print(f"Wrote: {out_path.name} | Teams: {len(snap.get('teams', []))}")

    print(f"\nDone. Snapshots written: {wrote} | Pairs with no live rows: {missing}")


if __name__ == "__main__":
    main()
