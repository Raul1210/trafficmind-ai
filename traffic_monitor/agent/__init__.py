"""
Q-learning agent package.

Re-exports the public surface so callers can write:
    from traffic_monitor.agent import TrafficSignalAgent, build_state
"""
from traffic_monitor.agent.q_learning import (
    CONFIG,
    PHASE_DIRECTIONS,
    PHASE_NAMES,
    FixedTimeController,
    SimpleIntersectionEnv,
    TrafficSignalAgent,
    build_state,
    discretize_queue,
    train,
    benchmark,
)

__all__ = [
    "CONFIG",
    "PHASE_DIRECTIONS",
    "PHASE_NAMES",
    "FixedTimeController",
    "SimpleIntersectionEnv",
    "TrafficSignalAgent",
    "build_state",
    "discretize_queue",
    "train",
    "benchmark",
]
