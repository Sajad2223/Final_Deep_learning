# LLMDriver/safety_wrapper.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SafetyConfig:
    """
    Heuristic safety parameters for the long-tail highway scenarios.

    Encodes the lessons:
      * maintain proper speed and distance
      * be cautious when changing lanes or adjusting speed
    """

    # --- headway / TTC thresholds (current lane) ---
    emergency_gap: float = 8.0        # [m] if front gap < this -> hard brake
    emergency_ttc: float = 1.5        # [s] same but using TTC

    cautious_gap: float = 25.0        # [m] medium risk region
    cautious_ttc: float = 3.0         # [s] medium risk region

    # --- lane-change constraints (target lane) ---
    lane_front_gap_min: float = 15.0  # [m] minimum front gap in target lane
    lane_rear_gap_min: float = 10.0   # [m] minimum rear gap in target lane
    lane_front_ttc_min: float = 3.0   # [s] TTC to front car in target lane
    lane_rear_ttc_min: float = 2.0    # [s] TTC to rear car in target lane

    # --- misc ---
    max_speed_kmh: float = 115.0      # hard speed ceiling for FASTER
    lane_width: float = 4.0           # approximate lane width in y
    lane_index_round: bool = True     # use round() instead of floor()


class SafetyWrapper:
    """
    Rule-based safety shield around the LLM actions.

    It uses gap + TTC to:
      * brake in clearly dangerous states
      * discourage acceleration / lane change in medium-risk states
      * only override when necessary (not "always go IDLE")
    """

    VALID = {"IDLE", "FASTER", "SLOWER", "LANE_LEFT", "LANE_RIGHT"}

    def __init__(self, cfg: SafetyConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _ego_state(self, obs: np.ndarray) -> Tuple[float, float, float]:
        """Return ego (x, y, speed_kmh) from kinematics obs."""
        ego = obs[0]
        x, y, vx = float(ego[1]), float(ego[2]), float(ego[3])
        return x, y, vx * 3.6  # speed in km/h

    def _lane_index(self, y: float) -> int:
        if self.cfg.lane_index_round:
            return int(round(y / self.cfg.lane_width))
        else:
            return int(np.floor(y / self.cfg.lane_width))

    def _lane_front_rear(
        self, obs: np.ndarray, lane_idx: int, ego_x: float, ego_vx: float
    ) -> Tuple[float, float, float, float]:
        """
        For a given lane, return:
            (front_gap, front_ttc, rear_gap, rear_ttc)

        TTC is gap / relative_speed when ego (or rear car) is faster,
        otherwise +inf (no closing).
        """
        INF = 1e9
        front_gap = INF
        rear_gap = INF
        front_ttc = INF
        rear_ttc = INF

        for row in obs[1:]:
            if row[0] <= 0.0:
                continue
            _, x, y, vx, _ = row[:5]
            lane = self._lane_index(float(y))
            if lane != lane_idx:
                continue

            dx = float(x) - ego_x
            v = float(vx)

            if dx > 0:  # car in front
                gap = dx
                if gap < front_gap:
                    front_gap = gap
                    rel_v = ego_vx - v  # positive if ego is faster
                    if rel_v > 1e-2:
                        front_ttc = gap / rel_v
                    else:
                        front_ttc = INF
            else:  # car behind (dx <= 0)
                gap = -dx
                if gap < rear_gap:
                    rear_gap = gap
                    rel_v = v - ego_vx  # positive if rear car is faster
                    if rel_v > 1e-2:
                        rear_ttc = gap / rel_v
                    else:
                        rear_ttc = INF

        return front_gap, front_ttc, rear_gap, rear_ttc

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------
    def filter_action(self, proposed: str, obs: np.ndarray) -> str:
        """
        Given the LLM's proposed high-level action, return a (possibly
        modified) safe action string.
        """
        # 0) Normalize & sanitize
        original = (proposed or "IDLE").upper().strip()
        if original not in self.VALID:
            # Unknown token -> keep lane, we'll reason only from state
            original = "IDLE"
        p = original

        ego_x, ego_y, ego_speed_kmh = self._ego_state(obs)
        ego_lane = self._lane_index(ego_y)
        ego_vx = obs[0][3]

        # current-lane gaps & TTC
        cur_front_gap, cur_front_ttc, _, _ = self._lane_front_rear(
            obs, ego_lane, ego_x, ego_vx
        )

        # Risk classification for current lane
        high_risk = (
            cur_front_gap < self.cfg.emergency_gap
            or cur_front_ttc < self.cfg.emergency_ttc
        )
        med_risk = (
            not high_risk
            and (
                cur_front_gap < self.cfg.cautious_gap
                or cur_front_ttc < self.cfg.cautious_ttc
            )
        )

        # ------------------------------------------------------------------
        # 1) Emergency brake: if something is very close ahead, always SLOWER
        # ------------------------------------------------------------------
        if high_risk:
            return "SLOWER"

        # ------------------------------------------------------------------
        # 2) Handle FASTER: respect headway, TTC, and max speed
        # ------------------------------------------------------------------
        if p == "FASTER":
            # Speed limit
            if ego_speed_kmh >= self.cfg.max_speed_kmh:
                return "IDLE"

            # Medium risk: headway/TTC already tight -> don't accelerate
            if med_risk:
                # stay in lane but do not increase speed
                return "IDLE"

        # ------------------------------------------------------------------
        # 3) Lane changes
        # ------------------------------------------------------------------
        if p in ("LANE_LEFT", "LANE_RIGHT"):
            target_lane = ego_lane + (1 if p == "LANE_LEFT" else -1)

            # Estimate plausible lane range from all vehicles (including ego)
            lane_indices = [
                self._lane_index(float(row[2]))
                for row in obs
                if row[0] > 0.0
            ]
            min_lane, max_lane = min(lane_indices), max(lane_indices)

            # If target lane is outside road bounds, we can't change lanes.
            if target_lane < min_lane or target_lane > max_lane:
                # If we tried lane change while also in a tight situation,
                # slowing down is safer than just keeping lane.
                return "SLOWER" if med_risk else "IDLE"

            # Evaluate safety in target lane
            (
                t_front_gap,
                t_front_ttc,
                t_rear_gap,
                t_rear_ttc,
            ) = self._lane_front_rear(obs, target_lane, ego_x, ego_vx)

            target_front_safe = (
                t_front_gap >= self.cfg.lane_front_gap_min
                and t_front_ttc >= self.cfg.lane_front_ttc_min
            )
            target_rear_safe = (
                t_rear_gap >= self.cfg.lane_rear_gap_min
                and t_rear_ttc >= self.cfg.lane_rear_ttc_min
            )
            target_safe = target_front_safe and target_rear_safe

            if not target_safe:
                # Wanted to change lane but target lane is not safe.
                # If our current lane is already medium-risk, brake a bit
                # instead of blindly staying at speed.
                return "SLOWER" if med_risk else "IDLE"

            # If target lane is clearly safer than current (more space / TTC),
            # allow the lane change; otherwise, no strong reason to move.
            better_front = t_front_gap > cur_front_gap + 5.0
            better_ttc = t_front_ttc > cur_front_ttc + 0.5

            if better_front or better_ttc or med_risk:
                # lane change is beneficial (or we're trying to escape medium risk)
                return p
            else:
                # Not much benefit -> keep lane calmly
                return "IDLE"

        # ------------------------------------------------------------------
        # 4) IDLE and SLOWER in medium-risk region
        # ------------------------------------------------------------------
        if p == "IDLE" and med_risk:
            # We're already converging to a front car; gently prefer SLOWER
            # instead of holding speed, to open up spacing.
            return "SLOWER"

        # Otherwise, accept the proposed action
        return p
