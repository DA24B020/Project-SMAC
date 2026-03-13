

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize, LinearConstraint, Bounds
import json
import time
import warnings

warnings.filterwarnings("ignore")


NUM_SCENARIOS = 100
ARENA_WIDTH = 800
ARENA_HEIGHT = 600
DT = 0.1
DAMPING = 0.98
MAX_STEPS = 300
GOAL_THRESHOLD = 15.0
SAFETY_MARGIN = 20.0



class Rectangle:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def signed_distance(self, x, y):
        dx = max(self.x_min - x, 0, x - self.x_max)
        dy = max(self.y_min - y, 0, y - self.y_max)

        if dx > 0 or dy > 0:
            return np.sqrt(dx * dx + dy * dy)

        return -min(x - self.x_min, self.x_max - x, y - self.y_min, self.y_max - y)

    def gradient(self, x, y):
        cx = np.clip(x, self.x_min, self.x_max)
        cy = np.clip(y, self.y_min, self.y_max)

        dx = x - cx
        dy = y - cy
        dist = np.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            if x < (self.x_min + self.x_max) / 2:
                return np.array([-1.0, 0.0])
            else:
                return np.array([1.0, 0.0])

        return np.array([dx / dist, dy / dist])



def particle_dynamics(state, control):
    px, py, vx, vy = state
    ax, ay = control

    px_new = px + vx * DT
    py_new = py + vy * DT
    vx_new = vx * DAMPING + ax * DT
    vy_new = vy * DAMPING + ay * DT

    if px_new < 20:
        px_new = 20
        vx_new = -vx_new * 0.5
    elif px_new > ARENA_WIDTH - 20:
        px_new = ARENA_WIDTH - 20
        vx_new = -vx_new * 0.5

    if py_new < 20:
        py_new = 20
        vy_new = -vy_new * 0.5
    elif py_new > ARENA_HEIGHT - 20:
        py_new = ARENA_HEIGHT - 20
        vy_new = -vy_new * 0.5

    return np.array([px_new, py_new, vx_new, vy_new])



class SimpleController:

    def __init__(self, goal_pos):
        self.goal_pos = np.array(goal_pos)
        self.Kp_pos = 0.8
        self.Kp_vel = 1.5
        self.max_accel = 100.0

    def get_control(self, state):
        px, py, vx, vy = state
        pos = np.array([px, py])
        vel = np.array([vx, vy])

        vel_des = self.Kp_pos * (self.goal_pos - pos)

        accel = self.Kp_vel * (vel_des - vel)

        norm = np.linalg.norm(accel)
        if norm > self.max_accel:
            accel = accel * self.max_accel / norm

        return accel


class CBFController:

    def __init__(self, obstacles, goal_pos):
        self.obstacles = obstacles
        self.goal_pos = np.array(goal_pos)
        self.Kp_pos = 0.8
        self.Kp_vel = 1.5
        self.max_accel = 100.0
        self.cbf_alpha = 0.5

    def get_control(self, state):
        px, py, vx, vy = state
        pos = np.array([px, py])
        vel = np.array([vx, vy])

        vel_des = self.Kp_pos * (self.goal_pos - pos)
        accel_nom = self.Kp_vel * (vel_des - vel)

        norm = np.linalg.norm(accel_nom)
        if norm > self.max_accel:
            accel_nom = accel_nom * self.max_accel / norm

        h_vals = []
        Lf_h_vals = []
        Lg_h_vals = []

        for obs in self.obstacles:
            h = dist - SAFETY_MARGIN
            dist = obs.signed_distance(px, py)
            h = dist - SAFETY_MARGIN
            h_vals.append(h)

            grad_h = obs.gradient(px, py)

            Lf_h = grad_h[0] * vx + grad_h[1] * vy

            Lg_h = np.array([grad_h[0] * DT, grad_h[1] * DT])

            Lf_h_vals.append(Lf_h)
            Lg_h_vals.append(Lg_h)

        
        active_constraints = [i for i, h in enumerate(h_vals) if h < 100]

        if not active_constraints:
            return accel_nom, 0.0, min(h_vals) if h_vals else 1000.0

        def objective(u):
            return np.sum((u - accel_nom) ** 2)

        def grad_obj(u):
            return 2 * (u - accel_nom)
        A_ineq = []
        b_ineq = []

        for i in active_constraints:
            A_ineq.append(Lg_h_vals[i])
            b_ineq.append(-Lf_h_vals[i] - self.cbf_alpha * h_vals[i])

        if A_ineq:
            A_ineq = np.array(A_ineq)
            b_ineq = np.array(b_ineq)

            constraints = LinearConstraint(A_ineq, b_ineq, np.inf)
            bounds = Bounds(-self.max_accel, self.max_accel)

            result = minimize(
                objective,
                accel_nom,
                method="SLSQP",
                jac=grad_obj,
                constraints=constraints,
                bounds=bounds,
                options={"maxiter": 50, "ftol": 1e-4, "disp": False},
            )

            if result.success:
                return result.x, 0.0, min(h_vals)

        return accel_nom, 0.0, min(h_vals) if h_vals else 1000.0



class Simulation:
    def __init__(self, start_pos, start_vel, goal_pos, obstacles, use_cbf=False):
        self.state = np.array([start_pos[0], start_pos[1], start_vel[0], start_vel[1]])
        self.goal_pos = np.array(goal_pos)
        self.obstacles = obstacles
        self.use_cbf = use_cbf

        if use_cbf:
            self.controller = CBFController(obstacles, goal_pos)
        else:
            self.controller = SimpleController(goal_pos)

        self.trajectory = [self.state[:2].copy()]
        self.velocity_history = [self.state[2:].copy()]
        self.control_history = []
        self.barrier_history = []

        self.reached_goal = False
        self.collided = False
        self.step = 0

    def check_collision(self):
        px, py = self.state[:2]
        for obs in self.obstacles:
            if obs.signed_distance(px, py) < -5:
                return True
        return False

    def check_goal(self):
        px, py = self.state[:2]
        return (
            np.linalg.norm([px - self.goal_pos[0], py - self.goal_pos[1]])
            < GOAL_THRESHOLD
        )

    def step_sim(self):
        if self.use_cbf:
            control, slack, h_min = self.controller.get_control(self.state)
            self.barrier_history.append(h_min)
        else:
            control = self.controller.get_control(self.state)

        self.control_history.append(control)
        self.state = particle_dynamics(self.state, control)
        self.trajectory.append(self.state[:2].copy())
        self.velocity_history.append(self.state[2:].copy())

        self.reached_goal = self.check_goal()
        self.collided = self.check_collision()
        self.step += 1

    def run(self):
        while self.step < MAX_STEPS and not self.reached_goal and not self.collided:
            self.step_sim()

        traj = np.array(self.trajectory)
        path_length = (
            np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
            if len(traj) > 1
            else 0
        )

        min_dist = float("inf")
        for pos in self.trajectory:
            for obs in self.obstacles:
                d = abs(obs.signed_distance(pos[0], pos[1]))
                min_dist = min(min_dist, d)

        return {
            "trajectory": traj,
            "velocity": np.array(self.velocity_history),
            "reached_goal": self.reached_goal,
            "collided": self.collided,
            "steps": self.step,
            "path_length": path_length,
            "min_distance": min_dist,
            "barrier_history": self.barrier_history,
            "control_history": np.array(self.control_history),
        }



def generate_scenario(seed):
    np.random.seed(seed)

    start_pos = [np.random.uniform(50, 200), np.random.uniform(100, 500)]
    start_vel = [np.random.uniform(5, 15), np.random.uniform(-5, 5)]
    goal_pos = [np.random.uniform(600, 750), np.random.uniform(100, 500)]

    num_obs = np.random.randint(1, 3)
    obstacles = []

    for _ in range(num_obs):
        w = np.random.uniform(60, 120)
        h = np.random.uniform(80, 180)
        cx = np.random.uniform(250, 550)
        cy = np.random.uniform(100, 500)

        obs = Rectangle(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)

        if (
            obs.signed_distance(start_pos[0], start_pos[1]) > 30
            and obs.signed_distance(goal_pos[0], goal_pos[1]) > 30
        ):
            obstacles.append(obs)

    return start_pos, start_vel, goal_pos, obstacles



def run_batch(num_scenarios):
    print(f"\n{'='*70}")
    print(f"Running {num_scenarios} scenarios...")
    print(f"{'='*70}\n")

    results = {"simple": [], "cbf": [], "scenarios": []}

    for i in range(num_scenarios):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_scenarios}")

        start_pos, start_vel, goal_pos, obstacles = generate_scenario(i)

        results["scenarios"].append(
            {
                "start_pos": start_pos,
                "start_vel": start_vel,
                "goal_pos": goal_pos,
                "obstacles": [
                    {
                        "x_min": o.x_min,
                        "x_max": o.x_max,
                        "y_min": o.y_min,
                        "y_max": o.y_max,
                    }
                    for o in obstacles
                ],
            }
        )

        sim_simple = Simulation(
            start_pos, start_vel, goal_pos, obstacles, use_cbf=False
        )
        results["simple"].append(sim_simple.run())

        sim_cbf = Simulation(start_pos, start_vel, goal_pos, obstacles, use_cbf=True)
        results["cbf"].append(sim_cbf.run())

    print(f"\nComplete!\n")
    return results



def analyze(results):
    n = len(results["simple"])

    analysis = {
        "simple": {
            "success": sum(r["reached_goal"] for r in results["simple"]) / n,
            "collision": sum(r["collided"] for r in results["simple"]) / n,
            "avg_path": np.mean([r["path_length"] for r in results["simple"]]),
            "avg_steps": np.mean([r["steps"] for r in results["simple"]]),
            "avg_min_dist": np.mean([r["min_distance"] for r in results["simple"]]),
        },
        "cbf": {
            "success": sum(r["reached_goal"] for r in results["cbf"]) / n,
            "collision": sum(r["collided"] for r in results["cbf"]) / n,
            "avg_path": np.mean([r["path_length"] for r in results["cbf"]]),
            "avg_steps": np.mean([r["steps"] for r in results["cbf"]]),
            "avg_min_dist": np.mean([r["min_distance"] for r in results["cbf"]]),
        },
    }

    prevented = sum(
        1
        for i in range(n)
        if results["simple"][i]["collided"] and not results["cbf"][i]["collided"]
    )
    analysis["prevented"] = prevented

    return analysis


def print_summary(analysis):
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nWITHOUT CBF:")
    print(f"  Success: {analysis['simple']['success']:.1%}")
    print(f"  Collision: {analysis['simple']['collision']:.1%}")
    print(f"  Avg Path: {analysis['simple']['avg_path']:.1f}")
    print(f"  Avg Min Dist: {analysis['simple']['avg_min_dist']:.1f}")

    print(f"\nWITH CBF:")
    print(f"  Success: {analysis['cbf']['success']:.1%}")
    print(f"  Collision: {analysis['cbf']['collision']:.1%}")
    print(f"  Avg Path: {analysis['cbf']['avg_path']:.1f}")
    print(f"  Avg Min Dist: {analysis['cbf']['avg_min_dist']:.1f}")

    print(f"\nIMPROVEMENT:")
    print(
        f"  Success Δ: {analysis['cbf']['success'] - analysis['simple']['success']:+.1%}"
    )
    print(
        f"  Collision Δ: {analysis['simple']['collision'] - analysis['cbf']['collision']:.1%} reduction"
    )
    print(
        f"  Safety Margin Δ: {analysis['cbf']['avg_min_dist'] - analysis['simple']['avg_min_dist']:+.1f} px"
    )
    print(f"  Collisions Prevented: {analysis['prevented']}")
    print("=" * 70 + "\n")



def create_plots(results, analysis):
    fig = plt.figure(figsize=(20, 11))
    gs = GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.35)

    simple = results["simple"]
    cbf = results["cbf"]


    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ["Success", "Collision", "Timeout"]
    simple_vals = [
        analysis["simple"]["success"],
        analysis["simple"]["collision"],
        1 - analysis["simple"]["success"] - analysis["simple"]["collision"],
    ]
    cbf_vals = [
        analysis["cbf"]["success"],
        analysis["cbf"]["collision"],
        1 - analysis["cbf"]["success"] - analysis["cbf"]["collision"],
    ]
    x = np.arange(3)
    width = 0.35
    ax1.bar(x - width / 2, simple_vals, width, label="No CBF", color="#ff6b6b")
    ax1.bar(x + width / 2, cbf_vals, width, label="With CBF", color="#51cf66")
    ax1.set_ylabel("Rate", fontsize=11, fontweight="bold")
    ax1.set_title("Performance Metrics", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1])


    ax2 = fig.add_subplot(gs[0, 1])
    simple_dists = [r["min_distance"] for r in simple]
    cbf_dists = [r["min_distance"] for r in cbf]

    bp = ax2.boxplot(
        [simple_dists, cbf_dists],
        labels=["No CBF", "With CBF"],
        patch_artist=True,
        widths=0.6,
    )
    bp["boxes"][0].set_facecolor("#ff6b6b")
    bp["boxes"][1].set_facecolor("#51cf66")
    ax2.axhline(
        SAFETY_MARGIN,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target Margin ({SAFETY_MARGIN}px)",
    )
    ax2.set_ylabel("Min Distance (px)", fontsize=11, fontweight="bold")
    ax2.set_title("Safety Margin Analysis", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)


    ax3 = fig.add_subplot(gs[0, 2])
    simple_paths = [r["path_length"] for r in simple if r["reached_goal"]]
    cbf_paths = [r["path_length"] for r in cbf if r["reached_goal"]]

    if simple_paths and cbf_paths:
        bp = ax3.boxplot(
            [simple_paths, cbf_paths],
            labels=["No CBF", "With CBF"],
            patch_artist=True,
            widths=0.6,
        )
        bp["boxes"][0].set_facecolor("#ff6b6b")
        bp["boxes"][1].set_facecolor("#51cf66")
        ax3.set_ylabel("Path Length (px)", fontsize=11, fontweight="bold")
        ax3.set_title("Path Efficiency (Success Only)", fontsize=12, fontweight="bold")
        ax3.grid(alpha=0.3)


    ax4 = fig.add_subplot(gs[0, 3:])
    prevented = analysis["prevented"]
    labels = [f"CBF Prevented\n{prevented} collisions", f"No Intervention\nNeeded"]
    sizes = [prevented, len(simple) - prevented]
    colors = ["#51cf66", "#a9a9a9"]
    explode = (0.1, 0)
    ax4.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=explode,
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    ax4.set_title("CBF Collision Prevention Impact", fontsize=12, fontweight="bold")


    examples = []


    for i in range(len(simple)):
        if simple[i]["collided"] and not cbf[i]["collided"]:
            examples.append(("CBF PREVENTED COLLISION", i))
            break


    for i in range(len(simple)):
        if (
            simple[i]["reached_goal"]
            and cbf[i]["reached_goal"]
            and i not in [e[1] for e in examples]
        ):
            examples.append(("Both Reached Goal", i))
            break


    for i in range(len(simple)):
        if (
            not simple[i]["reached_goal"]
            and cbf[i]["reached_goal"]
            and i not in [e[1] for e in examples]
        ):
            examples.append(("CBF Success / Simple Fail", i))
            break


    while len(examples) < 5:
        i = np.random.randint(len(simple))
        if i not in [e[1] for e in examples]:
            examples.append(("Random Scenario", i))

    for plot_idx, (title, idx) in enumerate(examples[:5]):
        ax = fig.add_subplot(gs[1, plot_idx])

        scenario = results["scenarios"][idx]
        traj_simple = simple[idx]["trajectory"]
        traj_cbf = cbf[idx]["trajectory"]


        for obs in scenario["obstacles"]:
            rect = patches.Rectangle(
                (obs["x_min"], obs["y_min"]),
                obs["x_max"] - obs["x_min"],
                obs["y_max"] - obs["y_min"],
                linewidth=2,
                edgecolor="black",
                facecolor="#95a5a6",
                alpha=0.7,
            )
            ax.add_patch(rect)


        ax.plot(
            traj_simple[:, 0],
            traj_simple[:, 1],
            "r-",
            linewidth=2.5,
            label="No CBF",
            alpha=0.8,
        )
        ax.plot(
            traj_cbf[:, 0],
            traj_cbf[:, 1],
            "g-",
            linewidth=2.5,
            label="With CBF",
            alpha=0.8,
        )


        ax.plot(
            scenario["start_pos"][0],
            scenario["start_pos"][1],
            "o",
            color="blue",
            markersize=12,
            label="Start",
            zorder=5,
        )
        ax.plot(
            scenario["goal_pos"][0],
            scenario["goal_pos"][1],
            "*",
            color="gold",
            markersize=18,
            label="Goal",
            zorder=5,
        )


        if simple[idx]["collided"]:
            ax.plot(
                traj_simple[-1, 0],
                traj_simple[-1, 1],
                "X",
                color="darkred",
                markersize=14,
                markeredgewidth=2,
                zorder=6,
            )
        if cbf[idx]["collided"]:
            ax.plot(
                traj_cbf[-1, 0],
                traj_cbf[-1, 1],
                "X",
                color="darkgreen",
                markersize=14,
                markeredgewidth=2,
                zorder=6,
            )

        ax.set_xlim(0, ARENA_WIDTH)
        ax.set_ylim(0, ARENA_HEIGHT)
        ax.set_xlabel("X (px)", fontsize=9)
        ax.set_ylabel("Y (px)", fontsize=9)
        ax.set_title(f"{title}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal")


    ax10 = fig.add_subplot(gs[2, 0:2])
    for i in range(min(10, len(cbf))):
        if len(cbf[i]["barrier_history"]) > 10:
            ax10.plot(cbf[i]["barrier_history"], alpha=0.6, linewidth=1.5)
    ax10.axhline(0, color="red", linestyle="--", linewidth=2, label="Safety Boundary")
    ax10.axhline(
        SAFETY_MARGIN,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Target ({SAFETY_MARGIN}px)",
    )
    ax10.set_xlabel("Time Step", fontsize=11, fontweight="bold")
    ax10.set_ylabel("Barrier h(x)", fontsize=11, fontweight="bold")
    ax10.set_title(
        "Barrier Function Evolution (Sample Runs)", fontsize=12, fontweight="bold"
    )
    ax10.legend(fontsize=9)
    ax10.grid(alpha=0.3)


    ax11 = fig.add_subplot(gs[2, 2])
    simple_controls = [
        np.mean(np.linalg.norm(r["control_history"], axis=1))
        for r in simple
        if len(r["control_history"]) > 0
    ]
    cbf_controls = [
        np.mean(np.linalg.norm(r["control_history"], axis=1))
        for r in cbf
        if len(r["control_history"]) > 0
    ]

    bp = ax11.boxplot(
        [simple_controls, cbf_controls],
        labels=["No CBF", "With CBF"],
        patch_artist=True,
        widths=0.6,
    )
    bp["boxes"][0].set_facecolor("#ff6b6b")
    bp["boxes"][1].set_facecolor("#51cf66")
    ax11.set_ylabel("Avg Control (px/s²)", fontsize=11, fontweight="bold")
    ax11.set_title("Control Effort", fontsize=12, fontweight="bold")
    ax11.grid(alpha=0.3)


    ax12 = fig.add_subplot(gs[2, 3:])
    ax12.axis("off")

    table_data = [
        ["Metric", "No CBF", "With CBF", "Δ"],
        [
            "Success",
            f"{analysis['simple']['success']:.1%}",
            f"{analysis['cbf']['success']:.1%}",
            f"{analysis['cbf']['success'] - analysis['simple']['success']:+.1%}",
        ],
        [
            "Collision",
            f"{analysis['simple']['collision']:.1%}",
            f"{analysis['cbf']['collision']:.1%}",
            f"{analysis['simple']['collision'] - analysis['cbf']['collision']:.1%}",
        ],
        [
            "Avg Path",
            f"{analysis['simple']['avg_path']:.0f}px",
            f"{analysis['cbf']['avg_path']:.0f}px",
            f"{((analysis['cbf']['avg_path']/analysis['simple']['avg_path'])-1)*100:+.1f}%",
        ],
        [
            "Min Dist",
            f"{analysis['simple']['avg_min_dist']:.1f}px",
            f"{analysis['cbf']['avg_min_dist']:.1f}px",
            f"{analysis['cbf']['avg_min_dist'] - analysis['simple']['avg_min_dist']:+.1f}px",
        ],
        ["Prevented", "—", "—", f"{analysis['prevented']} cases"],
    ]

    table = ax12.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.28, 0.24, 0.24, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    for i in range(4):
        table[(0, i)].set_facecolor("#2c3e50")
        table[(0, i)].set_text_props(weight="bold", color="white", fontsize=11)

    for i in range(1, len(table_data)):
        for j in range(4):
            table[(i, j)].set_text_props(fontsize=10)
            if j == 3:
                table[(i, j)].set_facecolor("#ecf0f1")

    ax12.set_title("QUANTITATIVE COMPARISON", fontsize=13, fontweight="bold", pad=15)

    plt.suptitle(
        "Control Barrier Function Safety Analysis\nParticle Dynamics with Velocity State",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig("cbf_comparison_results_FIXED.png", dpi=200, bbox_inches="tight")
    print("Saved: cbf_comparison_results_FIXED.png\n")



def generate_poster_text():
    text = """
╔════════════════════════════════════════════════════════════════════╗
║                    POSTER DESCRIPTION BOX                          ║
║            (Dense, Technical, Concise for Poster)                  ║
╚════════════════════════════════════════════════════════════════════╝

CONTROL BARRIER FUNCTIONS FOR SAFE ROBOT NAVIGATION
────────────────────────────────────────────────────

PROBLEM: Autonomous systems must reach goals while avoiding obstacles.
Traditional controllers optimize for goal-reaching but ignore safety.

APPROACH: Control Barrier Functions (CBFs) mathematically guarantee 
safety by enforcing h(x) ≥ 0 where h is a barrier function measuring 
distance to unsafe regions. Real-time quadratic programs (QPs) compute 
minimal deviations from nominal control that satisfy safety constraints.

IMPLEMENTATION: 
• State: Position (px, py) + Velocity (vx, vy) with damping
• Barrier: h(x) = distance_to_obstacle - safety_margin
• Constraint: Ḣ(x,u) ≥ -α·h(x) ensures forward invariance
• QP: min ||u - u_nominal||² subject to CBF constraints

BENCHMARKING METHODOLOGY:
100 randomized scenarios with 1-2 obstacles, identical initial conditions
tested with/without CBF. Metrics: success rate (goal reached), collision 
rate (safety violation), path length (efficiency), minimum obstacle 
distance (safety margin). Statistical comparison via Monte Carlo.

KEY RESULTS:
• Collision reduction: [X]% fewer crashes with CBF
• Safety margin: +[Y] pixels average clearance
• [Z] collision events prevented by CBF intervention
• Trade-off: [W]% longer paths for guaranteed safety

SIGNIFICANCE: Demonstrates formal safety guarantees achieve measurable
risk reduction in practice with acceptable performance overhead.

════════════════════════════════════════════════════════════════════
"""
    return text



def main():
    print("\n" + "=" * 70)
    print("ROBUST CBF COMPARISON - FIXED VERSION")
    print("Proper particle dynamics + Better barriers + Dense visualization")
    print("=" * 70)

    results = run_batch(NUM_SCENARIOS)

    analysis = analyze(results)
    print_summary(analysis)

    create_plots(results, analysis)

    poster_text = generate_poster_text()

    poster_text = poster_text.replace(
        "[X]",
        f"{(analysis['simple']['collision'] - analysis['cbf']['collision'])*100:.0f}",
    )
    poster_text = poster_text.replace(
        "[Y]",
        f"{analysis['cbf']['avg_min_dist'] - analysis['simple']['avg_min_dist']:.0f}",
    )
    poster_text = poster_text.replace("[Z]", str(analysis["prevented"]))

    if analysis["simple"]["avg_path"] > 0:
        overhead = (
            (analysis["cbf"]["avg_path"] / analysis["simple"]["avg_path"]) - 1
        ) * 100
        poster_text = poster_text.replace("[W]", f"{overhead:.0f}")
    else:
        poster_text = poster_text.replace("[W]", "N/A")

    print(poster_text)

    with open("poster_description.txt", "w", encoding="utf-8") as f:
        f.write(poster_text)

    print("\nSaved: poster_description.txt")
    print("\nDONE!\n")


if __name__ == "__main__":
    main()
