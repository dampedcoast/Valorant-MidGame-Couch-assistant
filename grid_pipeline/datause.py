import time
import requests
import pandas as pd
import json
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# CONFIG
# ============================================================
API_KEY = "gDVqIdMead2zTW1DKehu8PicvVXStT2xtmbYHK7b"

HEADERS = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json",
}

URL_CENTRAL = "https://api-op.grid.gg/central-data/graphql"
URL_SERIES_STATE = "https://api-op.grid.gg/live-data-feed/series-state/graphql"

PAGES = 3
PAGE_SIZE = 20

GRID_N = 8
SLEEP_BETWEEN_SERIES = 0.15


# ============================================================
# HELPERS
# ============================================================
def run_gql(url: str, query: str, operation_name: Optional[str], variables: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"query": query, "variables": variables}
    if operation_name:
        payload["operationName"] = operation_name

    r = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")

    out = r.json()
    if "errors" in out and out["errors"]:
        raise RuntimeError(json.dumps(out["errors"], indent=2)[:2000])
    return out


def unwrap_named_type(t: Dict[str, Any]) -> Optional[str]:
    """
    GraphQL types can be wrapped in NON_NULL / LIST. This unwraps down to the named type.
    """
    cur = t
    for _ in range(10):
        if not isinstance(cur, dict):
            return None
        name = cur.get("name")
        kind = cur.get("kind")
        if name:
            return name
        cur = cur.get("ofType")
        if kind is None and cur is None:
            break
    return None


# ============================================================
# INVENTORY DISCOVERY (critical)
# ============================================================
INTROSPECT_TYPE_FIELDS = """
query IntrospectType($name: String!) {
  __type(name: $name) {
    name
    fields {
      name
      type { kind name ofType { kind name ofType { kind name ofType { kind name }}}}
    }
  }
}
"""

INTROSPECT_SCHEMA_TYPE_NAMES = """
query IntrospectSchemaTypeNames {
  __schema {
    types { name }
  }
}
"""

def discover_player_inventory_field() -> Optional[Tuple[str, str]]:
    """
    Finds (playerTypeName, inventoryFieldName) where inventoryFieldName's named type == "PlayerInventory".
    Searches likely player types first, then falls back to scanning schema names.
    """
    # 1) Try the known player type that you're already using
    try_types = ["GamePlayerStateValorant"]

    for tn in try_types:
        try:
            out = run_gql(URL_SERIES_STATE, INTROSPECT_TYPE_FIELDS, "IntrospectType", {"name": tn})
            fields = (((out.get("data") or {}).get("__type") or {}).get("fields")) or []
            for f in fields:
                named = unwrap_named_type(f.get("type") or {})
                if named == "PlayerInventory":
                    return tn, f["name"]
        except Exception:
            pass

    # 2) If not on GamePlayerStateValorant, find other Valorant player-ish types and scan them
    out = run_gql(URL_SERIES_STATE, INTROSPECT_SCHEMA_TYPE_NAMES, "IntrospectSchemaTypeNames", {})
    type_names = [t["name"] for t in (out["data"]["__schema"]["types"] or []) if t.get("name")]

    # Heuristic: focus on types that look like Valorant player states
    candidates = [
        n for n in type_names
        if "Player" in n and "Valorant" in n
    ]

    # scan candidates until we find PlayerInventory
    for tn in candidates:
        try:
            out = run_gql(URL_SERIES_STATE, INTROSPECT_TYPE_FIELDS, "IntrospectType", {"name": tn})
            fields = (((out.get("data") or {}).get("__type") or {}).get("fields")) or []
            for f in fields:
                named = unwrap_named_type(f.get("type") or {})
                if named == "PlayerInventory":
                    return tn, f["name"]
        except Exception:
            continue

    return None


# ============================================================
# SERIES LIST (central)
# ============================================================
QUERY_SERIES_LIST = """
query GetValorantSeriesList($first: Int, $after: String) {
  allSeries(
    first: $first
    after: $after
    filter: { titleId: "6" }
    orderBy: ID
    orderDirection: ASC
  ) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        startTimeScheduled
        type
        updatedAt
        tournament { name nameShortened }
      }
    }
  }
}
"""

def fetch_valorant_series_ids(pages: int = 2, page_size: int = 20) -> pd.DataFrame:
    after = None
    rows: List[Dict[str, Any]] = []

    for _ in range(pages):
        out = run_gql(URL_CENTRAL, QUERY_SERIES_LIST, "GetValorantSeriesList", {"first": page_size, "after": after})
        block = out["data"]["allSeries"]
        edges = block.get("edges") or []

        for e in edges:
            n = e["node"]
            rows.append({
                "series_id": n["id"],
                "startTimeScheduled": n.get("startTimeScheduled"),
                "type": n.get("type"),
                "updatedAt": n.get("updatedAt"),
                "tournament_name": (n.get("tournament") or {}).get("name"),
                "tournament_short": (n.get("tournament") or {}).get("nameShortened"),
            })

        after = (block.get("pageInfo") or {}).get("endCursor")
        if not (block.get("pageInfo") or {}).get("hasNextPage"):
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["series_id"])
    return df


# ============================================================
# POSITION / BUCKETS (unchanged)
# ============================================================
def to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if isinstance(v, float) and (v != v):  # NaN
            return None
        return float(v)
    except Exception:
        try:
            s = str(v).strip()
            return float(s) if s else None
        except Exception:
            return None

def hp_bucket(current: Optional[float], maximum: Optional[float]) -> str:
    if current is None or maximum in (None, 0):
        return "unknown"
    ratio = float(current) / float(maximum)
    if ratio > 0.80:
        return "full"
    if ratio > 0.30:
        return "damaged"
    return "critical"

def armor_bucket(armor: Optional[float]) -> str:
    if armor is None:
        return "unknown"
    a = float(armor)
    if a <= 0:
        return "none"
    if a <= 25:
        return "light"
    return "heavy"

def clamp01(v: float) -> float:
    return min(max(v, 0.0), 0.999999)

def bin_index(v: float, vmin: float, vmax: float, n: int) -> int:
    r = (v - vmin) / (vmax - vmin) if abs(vmax - vmin) > 1e-12 else 0.0
    r = clamp01(r)
    return int(r * n)

def compute_game_bounds(teams: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for t in teams:
        if t.get("__typename") != "GameTeamStateValorant":
            continue
        for p in (t.get("players") or []):
            pos = p.get("position") or {}
            x = to_float(pos.get("x"))
            y = to_float(pos.get("y"))
            if x is not None and y is not None:
                xs.append(x); ys.append(y)

    if not xs or not ys:
        return None

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    if abs(maxx - minx) < 1e-6:
        maxx = minx + 1.0
    if abs(maxy - miny) < 1e-6:
        maxy = miny + 1.0

    return minx, maxx, miny, maxy

def region_labels(x: Optional[float], y: Optional[float],
                  bounds: Optional[Tuple[float, float, float, float]],
                  n: int = 8) -> Dict[str, str]:
    if x is None or y is None or bounds is None:
        return {"region_rc": "Unknown", "x_band": "Unknown", "y_band": "Unknown", "pos_quadrant": "Unknown"}

    minx, maxx, miny, maxy = bounds
    cx = bin_index(x, minx, maxx, n)
    cy = bin_index(y, miny, maxy, n)

    region_rc = f"R{cy+1}C{cx+1}"
    x_band = f"B{cx+1}"
    y_band = f"B{cy+1}"

    mx = (minx + maxx) / 2.0
    my = (miny + maxy) / 2.0
    east = x >= mx
    north = y >= my
    quad = "NE" if (north and east) else "NW" if (north and not east) else "SE" if (not north and east) else "SW"

    return {"region_rc": region_rc, "x_band": x_band, "y_band": y_band, "pos_quadrant": quad}


# ============================================================
# INVENTORY EXTRACTION
# ============================================================
def extract_weapon_from_inventory(inv: Any) -> Optional[str]:
    """
    inv is PlayerInventory: { items: [ItemStack!]! }
    ItemStack has: id, name, quantity, equipped, stashed
    Choose best equipped item name.
    """
    if not isinstance(inv, dict):
        return None
    items = inv.get("items")
    if not isinstance(items, list) or not items:
        return None

    best = None  # (equipped, quantity, name)
    fallback = None

    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name")
        if isinstance(name, str) and name.strip():
            fallback = fallback or name.strip()
        else:
            continue

        try:
            equipped = int(it.get("equipped") or 0)
        except Exception:
            equipped = 0
        try:
            quantity = int(it.get("quantity") or 0)
        except Exception:
            quantity = 0

        if equipped > 0:
            cand = (equipped, quantity, name.strip())
            if best is None or cand > best:
                best = cand

    return best[2] if best else fallback


# ============================================================
# BUILD series-state QUERY USING DISCOVERED FIELD PATH
# ============================================================
def build_series_state_query(player_type: str, inv_field: str) -> str:
    # Inventory selection is fixed now that we know ItemStack fields
    inv_block = f"""
              {inv_field} {{
                items {{
                  id
                  name
                  quantity
                  equipped
                  stashed
                }}
              }}
    """

    return f"""
query MidRoundState($seriesId: ID!) {{
  seriesState(id: $seriesId) {{
    id
    games {{
      id
      teams {{
        __typename
        ... on GameTeamStateValorant {{
          id
          name
          side
          players {{
            __typename
            ... on {player_type} {{
              id
              name
              alive
              participationStatus
              currentHealth
              maxHealth
              currentArmor
              position {{ x y }}
              character {{ name }}
{inv_block}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""


def fetch_series_state(series_id: str, query: str) -> Optional[Dict[str, Any]]:
    try:
        out = run_gql(URL_SERIES_STATE, query, "MidRoundState", {"seriesId": series_id})
        return (out.get("data") or {}).get("seriesState") or None
    except Exception as e:
        print(f"[WARN] series-state failed for {series_id}: {str(e)[:350]}")
        return None


def build_rows_from_series_state(series_state: Dict[str, Any], inv_field: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    series_id = series_state.get("id")

    for g in (series_state.get("games") or []):
        game_id = g.get("id")
        teams = g.get("teams") or []
        bounds = compute_game_bounds(teams)

        for t in teams:
            if t.get("__typename") != "GameTeamStateValorant":
                continue

            for p in (t.get("players") or []):
                pos = p.get("position") or {}
                x = to_float(pos.get("x"))
                y = to_float(pos.get("y"))

                agent_raw = (p.get("character") or {}).get("name")
                chp = to_float(p.get("currentHealth"))
                mhp = to_float(p.get("maxHealth"))
                arm = to_float(p.get("currentArmor"))

                reg = region_labels(x, y, bounds, n=GRID_N)

                inv = p.get(inv_field)  # <-- use discovered field name
                weapon_name = extract_weapon_from_inventory(inv)

                rows.append({
                    "series_id": series_id,
                    "game_id": game_id,
                    "team_name": t.get("name"),
                    "side": t.get("side"),
                    "player_name": p.get("name"),
                    "alive": p.get("alive"),
                    "participationStatus": p.get("participationStatus"),
                    "agent_raw": agent_raw,
                    "hp_bucket": hp_bucket(chp, mhp),
                    "armor_bucket": armor_bucket(arm),
                    "pos_x": x,
                    "pos_y": y,
                    "region_rc": reg["region_rc"],
                    "x_band": reg["x_band"],
                    "y_band": reg["y_band"],
                    "pos_quadrant": reg["pos_quadrant"],
                    "weapon_current": weapon_name,
                })

    return rows


def fetch_alive_df(
    pages: int = 1,
    page_size: int = 10,
    grid_n: int = 8,
) -> pd.DataFrame:
    """
    Fetch alive players from series-state.
    """
    found = discover_player_inventory_field()
    if not found:
        raise RuntimeError("Could not find PlayerInventory field in series-state schema.")
    player_type, inv_field = found

    ss_query = build_series_state_query(player_type, inv_field)

    series_ids = fetch_valorant_series_ids(pages, page_size)
    series_ids = series_ids["series_id"].astype(str).dropna().unique().tolist()

    all_rows: List[Dict[str, Any]] = []
    for sid in series_ids:
        ss = fetch_series_state(sid, ss_query)
        if ss is None:
            continue
        all_rows.extend(build_rows_from_series_state(ss, inv_field))
        time.sleep(SLEEP_BETWEEN_SERIES)

    df = pd.DataFrame(all_rows)

    # keep only alive players
    if "alive" in df.columns:
        df = df[df["alive"] == True].copy()

    return df


def get_latest_game_key(df: pd.DataFrame) -> Optional[tuple]:
    """
    Picks the first (series_id, game_id) found (you can improve this later).
    """
    if df.empty:
        return None
    first = df[["series_id", "game_id"]].dropna().astype(str).iloc[0]
    return (first["series_id"], first["game_id"])


# MAIN
if __name__ == "__main__":
    found = discover_player_inventory_field()
    if not found:
        raise SystemExit(
            "Could not find a PlayerInventory field on any Valorant Player type in this endpoint.\n"
            "That means series-state likely does not expose inventory, or it is behind another type/field name."
        )

    player_type, inv_field = found
    print(f"[INFO] Discovered inventory at: players ... on {player_type} {{ {inv_field}: PlayerInventory }}")

    ss_query = build_series_state_query(player_type, inv_field)

    df_series = fetch_valorant_series_ids(pages=PAGES, page_size=PAGE_SIZE)
    series_ids = df_series["series_id"].astype(str).dropna().unique().tolist()
    print(f"Fetched {len(series_ids)} series IDs.")

    all_rows: List[Dict[str, Any]] = []
    ok = 0
    skipped = 0

    for sid in series_ids:
        ss = fetch_series_state(sid, ss_query)
        if ss is None:
            skipped += 1
            continue
        ok += 1
        all_rows.extend(build_rows_from_series_state(ss, inv_field))
        time.sleep(SLEEP_BETWEEN_SERIES)

    df_players = pd.DataFrame(all_rows)
    print(f"series-state OK: {ok}, skipped: {skipped}")
    print("player rows:", df_players.shape)

    alive_df = df_players[df_players["alive"] == True].copy()  # noqa: E712

    cols = [
        "series_id","game_id","team_name","player_name",
        "agent_raw","weapon_current",
        "hp_bucket","armor_bucket","pos_x","pos_y",
        "region_rc","x_band","y_band","pos_quadrant"
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\nAlive players (first 50 rows):")
    print(alive_df[cols].head(50))

   
    alive_df.to_csv("alive_players_midgame.csv", index=False)
    print("Wrote: alive_players_midgame.csv")

