"""
Q-Learning Traffic Signal Optimizer
=====================================
Bachelor's Thesis: Real-Time Intelligent Traffic Monitoring and Optimization System

This agent reads vehicle counts from your YOLOv8 detection system and decides
which signal phase to activate at each step to minimize total vehicle waiting time.

USAGE:
    Standalone simulation:   python q_agent.py --mode simulate
    Connect to your Flask:   python q_agent.py --mode live --host 192.168.1.131 --port 5000

REQUIRES:
    pip install numpy requests
    (SUMO for simulation mode — see README below)
"""

import ast
import math
import numpy as np
import random
import json
import time
import argparse
from collections import defaultdict

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    # Q-learning hyperparameters
    "learning_rate":    0.1,    # Alpha — how fast the agent learns
    "discount_factor":  0.95,   # Gamma — how much future rewards matter
    "epsilon_start":    1.0,    # Start fully exploratory
    "epsilon_end":      0.05,   # End mostly greedy
    "epsilon_decay":    0.995,  # Decay per episode

    # Traffic signal phases
    # Phase 0: North-South green,  East-West red
    # Phase 1: North-South red,    East-West green
    # Phase 2: North-South green,  East-West red  (left-turn protected)
    # Phase 3: North-South red,    East-West green (left-turn protected)
    "num_phases": 4,
    "min_phase_duration": 5,    # Minimum seconds before switching
    "max_phase_duration": 60,   # Maximum green time (prevents starvation)

    # State discretization — queue lengths binned into levels
    # 0 = empty, 1 = light (1-3), 2 = moderate (4-8), 3 = heavy (9+)
    "queue_bins": [0, 3, 8, float("inf")],

    # Reward function weights
    "wait_time_penalty":  -1.0,  # Per vehicle waiting
    "throughput_reward":   2.0,  # Per vehicle that clears the intersection
    "queue_penalty":      -0.5,  # Per vehicle in queue (encourages clearing)

    # Training
    "episodes":          1000,
    "steps_per_episode":  500,
    "save_every":          100,  # Save Q-table every N episodes
}

PHASE_NAMES = {
    0: "N-S Green / E-W Red",
    1: "N-S Red / E-W Green",
    2: "N-S Left-turn",
    3: "E-W Left-turn",
}

# Maps each phase to the pair of directions it clears.
# Changing phase semantics only requires updating this dict.
PHASE_DIRECTIONS = {
    0: ("north", "south"),  # N-S Green
    1: ("east",  "west"),   # E-W Green
    2: ("north", "south"),  # N-S Left-turn protected
    3: ("east",  "west"),   # E-W Left-turn protected
}


# ─────────────────────────────────────────────
#  STATE REPRESENTATION
# ─────────────────────────────────────────────

def discretize_queue(count: int) -> int:
    """Convert raw vehicle count to discrete bin (0–3)."""
    bins = CONFIG["queue_bins"]
    for i, threshold in enumerate(bins[1:]):
        if count <= threshold:
            return i
    return len(bins) - 2


def build_state(north: int, south: int, east: int, west: int, current_phase: int) -> tuple:
    """
    Build a discrete state tuple from vehicle counts.

    State = (binned_north, binned_south, binned_east, binned_west, current_phase)
    This gives 4^4 * num_phases = 1024 possible states — small enough for Q-table.

    Args:
        north, south, east, west: vehicle queue counts per lane
        current_phase: which signal phase is currently active

    Returns:
        tuple: hashable state for Q-table lookup
    """
    return (
        discretize_queue(north),
        discretize_queue(south),
        discretize_queue(east),
        discretize_queue(west),
        current_phase,
    )


# ─────────────────────────────────────────────
#  Q-LEARNING AGENT
# ─────────────────────────────────────────────

class TrafficSignalAgent:
    """
    Q-Learning agent for adaptive traffic signal control.

    The agent observes vehicle queue lengths in all 4 directions and
    decides which signal phase to activate to minimize waiting time.

    State space:  4 directions × 4 queue bins × 4 phases = 1024 states
    Action space: 4 signal phases
    Reward:       Negative total waiting time + throughput bonus
    """

    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(CONFIG["num_phases"]))
        self.epsilon = CONFIG["epsilon_start"]
        self.learning_rate = CONFIG["learning_rate"]
        self.discount = CONFIG["discount_factor"]

        # Tracking
        self.episode_rewards = []
        self.episode_wait_times = []
        self.current_phase = 0
        self.phase_duration = 0
        self.total_steps = 0

    def choose_action(self, state: tuple) -> int:
        """
        Epsilon-greedy action selection.
        During training: explore randomly with probability epsilon.
        During inference: always pick the best known action.
        """
        if random.random() < self.epsilon:
            return random.randint(0, CONFIG["num_phases"] - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: tuple, action: int, reward: float, next_state: tuple):
        """
        Q-table update — the core learning step.

        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

        This is the Bellman equation update. The agent learns:
        "How good was my action, given what happened next?"
        """
        current_q = self.q_table[state][action]
        best_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.discount * best_next_q
        td_error = td_target - current_q
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(
            CONFIG["epsilon_end"],
            self.epsilon * CONFIG["epsilon_decay"]
        )

    def compute_reward(self,
                       waiting_vehicles: int,
                       cleared_vehicles: int,
                       queue_length: int) -> float:
        """
        Reward function design — the most important thesis decision.

        Negative reward for waiting (agent learns to minimize it).
        Positive reward for throughput (agent learns to clear queues).
        Small penalty for queue length (encourages proactive clearing).
        """
        reward = (
            CONFIG["wait_time_penalty"] * waiting_vehicles +
            CONFIG["throughput_reward"] * cleared_vehicles +
            CONFIG["queue_penalty"] * queue_length
        )
        return reward

    def should_switch_phase(self, action: int) -> bool:
        """
        Enforce minimum and maximum phase durations.
        Prevents flickering (switching too fast) and starvation (one direction blocked too long).
        """
        if self.phase_duration < CONFIG["min_phase_duration"]:
            return False
        if self.phase_duration >= CONFIG["max_phase_duration"]:
            return True
        return action != self.current_phase

    def save(self, filepath: str = "q_table.json"):
        """Persist Q-table between runs."""
        data = {
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "episode_wait_times": self.episode_wait_times,
        }
        try:
            with open(filepath, "w") as f:
                json.dump(data, f)
            print(f"[Agent] Q-table saved → {filepath} ({len(self.q_table)} states explored)")
        except IOError as e:
            print(f"[Agent] ERROR: Failed to save Q-table to {filepath}: {e}")
            raise

    def load(self, filepath: str = "q_table.json"):
        """Load a previously trained Q-table."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            for k, v in data["q_table"].items():
                self.q_table[ast.literal_eval(k)] = np.array(v)
            self.epsilon = data.get("epsilon", CONFIG["epsilon_end"])
            self.episode_rewards = data.get("episode_rewards", [])
            self.episode_wait_times = data.get("episode_wait_times", [])
            print(f"[Agent] Q-table loaded ← {filepath} ({len(self.q_table)} states)")
        except FileNotFoundError:
            print(f"[Agent] No saved Q-table found, starting fresh.")

    def get_stats(self) -> dict:
        """Return current agent statistics for the dashboard."""
        recent = self.episode_rewards[-50:] if self.episode_rewards else [0]
        return {
            "epsilon": round(self.epsilon, 3),
            "states_explored": len(self.q_table),
            "total_steps": self.total_steps,
            "episodes": len(self.episode_rewards),
            "avg_reward_last50": round(float(np.mean(recent)), 2),
            "current_phase": self.current_phase,
            "phase_name": PHASE_NAMES[self.current_phase],
            "phase_duration": self.phase_duration,
        }


# ─────────────────────────────────────────────
#  SIMULATION ENVIRONMENT (no SUMO needed)
# ─────────────────────────────────────────────

class SimpleIntersectionEnv:
    """
    Lightweight traffic intersection simulator.

    Simulates vehicle arrivals, queuing, and departures for 4 directions.
    Use this to train and benchmark your agent without SUMO.
    For your thesis, also run in SUMO for the proper results chapter.

    Traffic scenarios:
        "low":    Low arrival rate, all directions equal
        "medium": Moderate traffic, slight N-S dominance
        "peak":   Rush hour — heavy N-S, moderate E-W
    """

    SCENARIOS = {
        "low":    {"north": 0.2, "south": 0.2, "east": 0.15, "west": 0.15},
        "medium": {"north": 0.4, "south": 0.35, "east": 0.25, "west": 0.2},
        "peak":   {"north": 0.7, "south": 0.65, "east": 0.35, "west": 0.3},
    }

    def __init__(self, scenario: str = "medium"):
        self.scenario = scenario
        self.arrival_rates = self.SCENARIOS[scenario]
        self.reset()

    def reset(self):
        """Reset intersection to empty state."""
        self.queues = {"north": 0, "south": 0, "east": 0, "west": 0}
        self.waiting_time = 0
        self.total_cleared = 0
        self.step_count = 0
        self.current_phase = 0
        return self._get_state()

    def _get_state(self) -> tuple:
        return build_state(
            self.queues["north"],
            self.queues["south"],
            self.queues["east"],
            self.queues["west"],
            self.current_phase,
        )

    def step(self, action: int) -> tuple:
        """
        Advance simulation by one time step.

        1. New vehicles arrive (Poisson process)
        2. Active phase clears vehicles in green directions
        3. Waiting time accumulates for queued vehicles
        4. Reward is computed

        Returns: (next_state, reward, done, info)
        """
        self.step_count += 1
        self.current_phase = action

        # Vehicle arrivals — Poisson random process
        for direction, rate in self.arrival_rates.items():
            arrivals = np.random.poisson(rate)
            self.queues[direction] = min(self.queues[direction] + arrivals, 30)

        # Vehicle departures — green phase clears vehicles
        # N-S phases clear north/south; E-W phases clear east/west
        cleared = 0
        discharge_rate = 3  # vehicles per step that can pass

        if action in PHASE_DIRECTIONS:
            dir_a, dir_b = PHASE_DIRECTIONS[action]
            total_clear = min(self.queues[dir_a] + self.queues[dir_b], discharge_rate)
            a_clear = min(self.queues[dir_a], math.ceil(total_clear / 2))
            b_clear = min(self.queues[dir_b], total_clear - a_clear)
            self.queues[dir_a] -= a_clear
            self.queues[dir_b] -= b_clear
            cleared += a_clear + b_clear

        self.total_cleared += cleared

        # Waiting time = sum of all vehicles currently queued
        total_waiting = sum(self.queues.values())
        self.waiting_time += total_waiting

        # Compute reward
        agent_reward = (
            CONFIG["wait_time_penalty"] * total_waiting +
            CONFIG["throughput_reward"] * cleared +
            CONFIG["queue_penalty"] * total_waiting
        )

        done = self.step_count >= CONFIG["steps_per_episode"]

        info = {
            "queues": dict(self.queues),
            "cleared": cleared,
            "waiting": total_waiting,
            "total_cleared": self.total_cleared,
        }

        return self._get_state(), agent_reward, done, info


# ─────────────────────────────────────────────
#  BASELINE: Fixed-time controller
# ─────────────────────────────────────────────

class FixedTimeController:
    """
    Traditional fixed-phase traffic controller.
    This is your BASELINE — the "before" in your thesis comparison.
    Cycles through phases on a fixed timer regardless of actual traffic.
    """

    def __init__(self, phase_duration: int = 30):
        self.phase_duration = phase_duration
        self.current_phase = 0
        self.timer = 0

    def choose_action(self, state=None) -> int:
        """Ignore state — just follow the fixed schedule."""
        self.timer += 1
        if self.timer >= self.phase_duration:
            self.timer = 0
            self.current_phase = (self.current_phase + 1) % CONFIG["num_phases"]
        return self.current_phase


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────

def train(scenario: str = "medium", verbose: bool = True):
    """
    Train the Q-learning agent on a given traffic scenario.

    Runs CONFIG["episodes"] episodes. Each episode simulates a full
    traffic day (CONFIG["steps_per_episode"] steps).

    Returns agent with trained Q-table and reward history.
    """
    agent = TrafficSignalAgent()
    env = SimpleIntersectionEnv(scenario)

    print(f"\n{'='*50}")
    print(f"Training Q-agent on '{scenario}' traffic scenario")
    print(f"Episodes: {CONFIG['episodes']} | Steps/episode: {CONFIG['steps_per_episode']}")
    print(f"{'='*50}\n")

    for episode in range(CONFIG["episodes"]):
        state = env.reset()
        episode_reward = 0
        episode_wait = 0

        for step in range(CONFIG["steps_per_episode"]):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
            episode_wait += info["waiting"]
            agent.total_steps += 1
            if done:
                break

        agent.episode_rewards.append(episode_reward)
        agent.episode_wait_times.append(episode_wait)
        agent.decay_epsilon()

        if verbose and (episode + 1) % 100 == 0:
            recent_wait = np.mean(agent.episode_wait_times[-100:])
            print(f"Episode {episode+1:4d} | "
                  f"Avg wait: {recent_wait:7.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"States: {len(agent.q_table)}")

        if (episode + 1) % CONFIG["save_every"] == 0:
            agent.save(f"q_table_{scenario}_ep{episode+1}.json")

    agent.save(f"q_table_{scenario}_final.json")
    return agent


# ─────────────────────────────────────────────
#  BENCHMARK: RL vs Fixed-time
# ─────────────────────────────────────────────

def benchmark(num_eval_episodes: int = 50):
    """
    Compare Q-learning agent vs fixed-time controller.
    Run this after training to generate your thesis results.

    Outputs:
        - Average waiting time (RL vs Fixed)
        - Average throughput (RL vs Fixed)
        - Improvement percentage
    """
    results = {}

    for scenario in ["low", "medium", "peak"]:
        print(f"\n--- Evaluating scenario: {scenario.upper()} ---")
        env = SimpleIntersectionEnv(scenario)

        # Load trained agent (or train fresh)
        agent = TrafficSignalAgent()
        try:
            agent.load(f"q_table_{scenario}_final.json")
            agent.epsilon = 0.0  # Pure greedy for evaluation
        except:
            print(f"  No trained model for '{scenario}', training now...")
            agent = train(scenario, verbose=False)
            agent.epsilon = 0.0

        fixed = FixedTimeController(phase_duration=30)

        rl_waits, fixed_waits = [], []
        rl_throughputs, fixed_throughputs = [], []

        for _ in range(num_eval_episodes):
            # Evaluate RL agent
            state = env.reset()
            ep_wait, ep_through = 0, 0
            for _ in range(CONFIG["steps_per_episode"]):
                action = agent.choose_action(state)
                state, _, done, info = env.step(action)
                ep_wait += info["waiting"]
                ep_through += info["cleared"]
                if done: break
            rl_waits.append(ep_wait)
            rl_throughputs.append(ep_through)

            # Evaluate Fixed controller (same scenario reset)
            state = env.reset()
            ep_wait, ep_through = 0, 0
            fixed.timer = 0; fixed.current_phase = 0
            for _ in range(CONFIG["steps_per_episode"]):
                action = fixed.choose_action(state)
                state, _, done, info = env.step(action)
                ep_wait += info["waiting"]
                ep_through += info["cleared"]
                if done: break
            fixed_waits.append(ep_wait)
            fixed_throughputs.append(ep_through)

        rl_avg_wait    = np.mean(rl_waits)
        fixed_avg_wait = np.mean(fixed_waits)
        improvement    = (fixed_avg_wait - rl_avg_wait) / fixed_avg_wait * 100

        results[scenario] = {
            "rl_avg_wait":       round(rl_avg_wait, 1),
            "fixed_avg_wait":    round(fixed_avg_wait, 1),
            "rl_avg_throughput": round(float(np.mean(rl_throughputs)), 1),
            "fixed_avg_throughput": round(float(np.mean(fixed_throughputs)), 1),
            "wait_improvement_pct": round(improvement, 1),
        }

        print(f"  Fixed-time  → Avg wait: {fixed_avg_wait:.1f} | Throughput: {np.mean(fixed_throughputs):.1f}")
        print(f"  Q-Learning  → Avg wait: {rl_avg_wait:.1f}  | Throughput: {np.mean(rl_throughputs):.1f}")
        print(f"  Improvement: {improvement:.1f}% reduction in waiting time")

    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*50}")
    for scenario, r in results.items():
        print(f"{scenario.upper():8s} | Wait: {r['fixed_avg_wait']} → {r['rl_avg_wait']} "
              f"({r['wait_improvement_pct']}% better) | "
              f"Throughput: {r['fixed_avg_throughput']} → {r['rl_avg_throughput']}")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved → benchmark_results.json")
    return results


# ─────────────────────────────────────────────
#  LIVE MODE: Connect to your Flask backend
# ─────────────────────────────────────────────

def run_live(host: str = "192.168.1.131", port: int = 5000):
    """
    Connect to your existing Flask backend and run the Q-agent
    using real YOLOv8 detections as the state input.

    Your Flask /api/counts endpoint should return:
    {
        "north": 5,
        "south": 3,
        "east": 7,
        "west": 2
    }

    The agent outputs a recommended phase which you display
    on the dashboard (and eventually send to a signal controller).
    """
    import requests

    agent = TrafficSignalAgent()
    agent.load("q_table_medium_final.json")  # Load best trained model
    agent.epsilon = 0.05  # Mostly greedy, slight exploration

    print(f"\n[Live] Connecting to {host}:{port}")
    print("[Live] Press Ctrl+C to stop\n")

    prev_counts = {"north": 0, "south": 0, "east": 0, "west": 0}

    while True:
        try:
            resp = requests.get(f"http://{host}:{port}/api/counts", timeout=2)
            counts = resp.json()

            state = build_state(
                counts.get("north", 0),
                counts.get("south", 0),
                counts.get("east", 0),
                counts.get("west", 0),
                agent.current_phase,
            )

            action = agent.choose_action(state)

            if agent.should_switch_phase(action):
                agent.current_phase = action
                agent.phase_duration = 0
                print(f"[Signal] → Switching to Phase {action}: {PHASE_NAMES[action]}")
            else:
                agent.phase_duration += 1

            # Estimate reward from real data (approximate)
            total_queue = sum(counts.values())
            cleared_estimate = max(0, sum(prev_counts.values()) - total_queue)
            reward = agent.compute_reward(total_queue, cleared_estimate, total_queue)
            agent.total_steps += 1

            stats = agent.get_stats()
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"Queues: N={counts.get('north',0)} S={counts.get('south',0)} "
                  f"E={counts.get('east',0)} W={counts.get('west',0)} | "
                  f"Phase: {stats['phase_name']} | "
                  f"Reward: {reward:.1f}")

            prev_counts = counts
            time.sleep(3)  # Match your detection refresh rate

        except requests.exceptions.ConnectionError:
            print(f"[Live] Cannot reach {host}:{port} — retrying in 5s...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n[Live] Stopped.")
            agent.save("q_table_live_session.json")
            break


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Learning Traffic Signal Optimizer")
    parser.add_argument("--mode", choices=["train", "benchmark", "live"],
                        default="benchmark", help="Run mode")
    parser.add_argument("--scenario", choices=["low", "medium", "peak"],
                        default="medium", help="Traffic scenario (train mode)")
    parser.add_argument("--host", default="192.168.1.131", help="Flask backend host")
    parser.add_argument("--port", type=int, default=5000, help="Flask backend port")
    args = parser.parse_args()

    if args.mode == "train":
        agent = train(scenario=args.scenario)
        print(f"\nTraining complete. Run with --mode benchmark to evaluate.")

    elif args.mode == "benchmark":
        print("Training all scenarios then benchmarking...")
        for scenario in ["low", "medium", "peak"]:
            train(scenario=scenario, verbose=True)
        benchmark()

    elif args.mode == "live":
        run_live(host=args.host, port=args.port)