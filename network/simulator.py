"""
Network simulation using Linux `tc` (Traffic Control) / netem.
Applies bandwidth limits and packet loss to a network interface.

Requires: iproute2 (tc), Linux only, usually needs sudo/root.

Usage:
    sim = NetworkSimulator(interface="eth0")
    sim.apply(bandwidth_mbps=100, loss_percent=0)   # High bandwidth
    sim.apply(bandwidth_mbps=1, loss_percent=5)     # Low BW + loss
    sim.clear()                                      # Remove rules
"""

import subprocess
import shutil
import platform
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NetworkProfile:
    name: str
    bandwidth_mbps: float          # e.g. 100.0 or 0.5
    loss_percent: float = 0.0      # e.g. 5.0 for 5%
    delay_ms: float = 0.0          # Additional latency in ms
    delay_jitter_ms: float = 0.0   # Jitter (variation in delay)

    def describe(self) -> str:
        parts = [f"{self.bandwidth_mbps} Mbps"]
        if self.loss_percent > 0:
            parts.append(f"{self.loss_percent}% loss")
        if self.delay_ms > 0:
            jitter = f" ±{self.delay_jitter_ms}ms" if self.delay_jitter_ms > 0 else ""
            parts.append(f"{self.delay_ms}ms latency{jitter}")
        return ", ".join(parts)


# Predefined test scenarios from the spec
PROFILES = {
    "high_bandwidth": NetworkProfile(
        name="High Bandwidth",
        bandwidth_mbps=100.0,
        loss_percent=0.0,
    ),
    "low_bandwidth": NetworkProfile(
        name="Low Bandwidth",
        bandwidth_mbps=1.0,
        loss_percent=0.0,
    ),
    "very_low_bandwidth": NetworkProfile(
        name="Very Low Bandwidth",
        bandwidth_mbps=0.1,
        loss_percent=0.0,
    ),
    "unstable": NetworkProfile(
        name="Unstable Network",
        bandwidth_mbps=10.0,
        loss_percent=7.5,
        delay_ms=20.0,
        delay_jitter_ms=10.0,
    ),
    "worst_case": NetworkProfile(
        name="Worst Case (Low BW + High Loss)",
        bandwidth_mbps=0.5,
        loss_percent=10.0,
        delay_ms=50.0,
        delay_jitter_ms=20.0,
    ),
}


class NetworkSimulator:
    """
    Wraps `tc` commands to apply/remove netem + tbf shaping on a network interface.
    """

    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self._active = False
        self._current_profile: Optional[NetworkProfile] = None

        if not self._check_available():
            logger.warning(
                "[NetSim] `tc` not found or not on Linux. "
                "Network simulation disabled. Install iproute2 on Linux."
            )
            self._available = False
        else:
            self._available = True

    def _check_available(self) -> bool:
        return (
            platform.system() == "Linux"
            and shutil.which("tc") is not None
        )

    @property
    def available(self) -> bool:
        return self._available

    def apply(self, bandwidth_mbps: float, loss_percent: float = 0.0,
              delay_ms: float = 0.0, delay_jitter_ms: float = 0.0) -> bool:
        """Apply traffic shaping. Returns True on success."""
        if not self._available:
            logger.warning("[NetSim] tc not available — simulation skipped")
            return False

        # Clear existing rules first
        self.clear(quiet=True)

        # Build tc commands
        # 1) Add root qdisc: tbf (token bucket filter) for bandwidth
        bw_kbps = int(bandwidth_mbps * 1000)
        burst = max(bw_kbps * 2, 1600)  # burst bytes

        cmds = []

        if loss_percent > 0 or delay_ms > 0:
            # Use netem for loss/delay + tbf for bandwidth
            # netem as child of tbf
            cmds.append(
                f"tc qdisc add dev {self.interface} root handle 1: "
                f"tbf rate {bw_kbps}kbit burst {burst} latency 50ms"
            )
            netem_parts = [f"tc qdisc add dev {self.interface} parent 1:1 handle 10: netem"]
            if delay_ms > 0:
                netem_parts.append(f"delay {int(delay_ms)}ms {int(delay_jitter_ms)}ms")
            if loss_percent > 0:
                netem_parts.append(f"loss {loss_percent}%")
            cmds.append(" ".join(netem_parts))
        else:
            # Just bandwidth limiting
            cmds.append(
                f"tc qdisc add dev {self.interface} root tbf "
                f"rate {bw_kbps}kbit burst {burst} latency 50ms"
            )

        success = True
        for cmd in cmds:
            ret = self._run(cmd)
            if ret != 0:
                success = False

        if success:
            self._active = True
            logger.info(
                f"[NetSim] Applied: {bandwidth_mbps} Mbps, {loss_percent}% loss, {delay_ms}ms delay"
            )
        return success

    def apply_profile(self, profile_name: str) -> bool:
        """Apply a named profile from PROFILES."""
        if profile_name not in PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(PROFILES.keys())}")
        p = PROFILES[profile_name]
        self._current_profile = p
        return self.apply(p.bandwidth_mbps, p.loss_percent, p.delay_ms, p.delay_jitter_ms)

    def clear(self, quiet: bool = False) -> bool:
        """Remove all tc rules from interface."""
        if not self._available:
            return False
        ret = self._run(f"tc qdisc del dev {self.interface} root", quiet=quiet)
        self._active = False
        self._current_profile = None
        return ret == 0

    def status(self) -> str:
        """Return current tc rules as string."""
        if not self._available:
            return "tc not available"
        result = subprocess.run(
            ["tc", "qdisc", "show", "dev", self.interface],
            capture_output=True, text=True
        )
        return result.stdout.strip()

    def _run(self, cmd: str, quiet: bool = False) -> int:
        if not quiet:
            logger.debug(f"[NetSim] Running: {cmd}")
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True
        )
        if result.returncode != 0 and not quiet:
            logger.warning(f"[NetSim] Command failed: {cmd}\n  stderr: {result.stderr.strip()}")
        return result.returncode

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.clear()
