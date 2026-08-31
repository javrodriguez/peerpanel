"""Panel orchestration: blind fan-out, adversarial verification, convergence."""

from .panel import PanelProviders, detect_conflicts, run_panel

__all__ = ["PanelProviders", "detect_conflicts", "run_panel"]
