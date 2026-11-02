"""
SureStats Bot - central configuration.

Everything is environment-driven so the same code runs locally, in Docker
and on Render.com without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(_env(key, str(default))))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- paths -----------------------------------------------------------
    base_dir: Path = BASE_DIR
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "store")
    db_path: Path = field(default_factory=lambda: BASE_DIR / "data" / "store" / "surestats.db")
    log_dir: Path = field(default_factory=lambda: BASE_DIR / "logs")

    # --- API keys --------------------------------------------------------
    odds_api_key: str = field(default_factory=lambda: _env("ODDS_API_KEY"))
    tennis_api_key: str = field(default_factory=lambda: _env("TENNIS_API_KEY"))
    balldontlie_api_key: str = field(default_factory=lambda: _env("BALLDONTLIE_API_KEY"))

    # --- endpoints -------------------------------------------------------
    odds_base_url: str = field(default_factory=lambda: _env("ODDS_BASE_URL", "https://api.the-odds-api.com/v4"))
    tennis_base_url: str = field(default_factory=lambda: _env("TENNIS_BASE_URL", "https://v1.tennis.api-sports.io"))
    nba_base_url: str = field(default_factory=lambda: _env("NBA_BASE_URL", "https://api.balldontlie.io/v1"))

    # --- behaviour -------------------------------------------------------
    odds_refresh_seconds: int = field(default_factory=lambda: _env_int("ODDS_REFRESH_SECONDS", 600))
    http_timeout: int = field(default_factory=lambda: _env_int("HTTP_TIMEOUT", 12))
    cache_ttl_seconds: int = field(default_factory=lambda: _env_int("CACHE_TTL_SECONDS", 600))
    live_data: bool = field(default_factory=lambda: _env_bool("LIVE_DATA", True))
    demo_mode: bool = field(default_factory=lambda: _env_bool("DEMO_MODE", False))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())

    # --- model hyper-parameters -----------------------------------------
    elo_k_tennis: float = field(default_factory=lambda: _env_float("ELO_K_TENNIS", 32.0))
    elo_k_nba: float = field(default_factory=lambda: _env_float("ELO_K_NBA", 20.0))
    elo_scale: float = field(default_factory=lambda: _env_float("ELO_SCALE", 400.0))
    home_advantage_nba: float = field(default_factory=lambda: _env_float("HOME_ADVANTAGE_NBA", 65.0))
    home_advantage_tennis: float = field(default_factory=lambda: _env_float("HOME_ADVANTAGE_TENNIS", 12.0))

    # value thresholds (percent)
    value_green: float = field(default_factory=lambda: _env_float("VALUE_GREEN", 8.0))
    value_yellow: float = field(default_factory=lambda: _env_float("VALUE_YELLOW", 3.0))
    min_edge_to_flag: float = field(default_factory=lambda: _env_float("MIN_EDGE_TO_FLAG", 0.0))

    @property
    def has_odds_key(self) -> bool:
        return bool(self.odds_api_key) and not self.demo_mode

    @property
    def has_tennis_key(self) -> bool:
        return bool(self.tennis_api_key) and not self.demo_mode

    @property
    def has_nba_key(self) -> bool:
        return bool(self.balldontlie_api_key) and not self.demo_mode


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model weights (the "True Probability" blend requested in the spec).
# Positive signals are normalised to sum to 1.0, then the fatigue/injury
# block is applied as a bounded penalty on the final probability.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "elo": 0.35,
    "form": 0.30,
    "surface_or_home": 0.20,
    "h2h": 0.10,
    "fatigue_injury": 0.05,   # negative adjustment block
}

# Bookmaker margin assumptions used when a market has too few quotes
DEFAULT_MARGIN = 0.045
