import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

# --- HELPER: Circle Barrier ---
def get_circle_barrier(robot_pos, circle_config):
    x, y = robot_pos
    cx, cy = circle_config['center']
    r = circle_config['radius']
    
    h_val = ((x - cx)**2 + (y - cy)**2) - r**2
    grad_h = np.array([2 * (x - cx), 2 * (y - cy)])
    return h_val, grad_h

# --- CONTROLLER CLASS ---
class CLFCBF_SmartVirtual_Controller:
    def __init__(self, rect_config, start_goal, gamma=5.0, alpha=2.0, clf_penalty=1000.0):
        self.rect_config = rect_config
        self.original_goal = np.array(start_goal)
        self.current_target = np.array(start_goal)
        
        # State tracking
        self.mode = "ORIGINAL" # Modes: ORIGINAL, VIRTUAL
        self.stuck_timer = 0   # Debounce timer to prevent flickering

        self.gamma = gamma
        self.alpha = alpha
        self.clf_penalty = clf_penalty
        
        # Heuristic Params
        self.lookahead_dist = 6.0  # How far to place virtual goal
        self.nudge_angle = np.deg2rad(40) # 40 degrees is smoother than 90

    def get_control(self, robot_state):
        x = robot_state
        
        # 1. Barrier Info
        h, grad_h = get_circle_barrier(x, self.rect_config)
        
        # 2. Vector Analysis
        vec_to_target = self.current_target - x
        dist_to_target = np.linalg.norm(vec_to_target)
        
        # Normalize for dot product
        if dist_to_target > 0:
            u_norm = vec_to_target / dist_to_target
            grad_norm = grad_h / (np.linalg.norm(grad_h) + 1e-6)
            dot = np.dot(u_norm, grad_norm)
        else:
            dot = 0.0

        # 3. STRATEGY LOGIC
        
        # Detect if we are stuck (Close to wall + Facing it directly)
        is_stuck = (h < 0.8) and (dot < -0.7)
        
        if self.mode == "ORIGINAL":
            if is_stuck:
                self.stuck_timer += 1
                # Only switch if we've been stuck for a few frames (stability)
                if self.stuck_timer > 5:
                    print("Stuck detected. Generating Smooth Virtual Goal.")
                    self.mode = "VIRTUAL"
                    self.stuck_timer = 0
                    
                    # --- CALCULATE SMOOTH VIRTUAL GOAL ---
                    # 1. Vector from Robot to Obstacle Center
                    obs_center = np.array(self.rect_config['center'])
                    vec_to_center = obs_center - x
                    
                    # 2. Rotate this vector by 40 degrees (Nudge)
                    # We pick direction based on where the goal is relative to obstacle
                    # Cross product tells us if goal is 'left' or 'right' of obstacle center
                    cross_val = np.cross(vec_to_center, self.original_goal - x)
                    direction = 1.0 if cross_val > 0 else -1.0
                    
                    theta = direction * self.nudge_angle
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array(((c, -s), (s, c)))
                    
                    nudge_vec = np.dot(R, vec_to_center)
                    nudge_vec = nudge_vec / np.linalg.norm(nudge_vec) # Normalize
                    
                    # 3. Place goal far out along this "nudge" line
                    self.current_target = x + (nudge_vec * self.lookahead_dist)

        elif self.mode == "VIRTUAL":
            # Switch back if:
            # A. We are clear of the obstacle (grad_h points somewhat towards goal?)
            # B. Or we just got close enough to the virtual point
            
            dist_virtual = np.linalg.norm(x - self.current_target)
            
            # Simple reset: If we moved away from the perpendicular "danger zone"
            # Logic: If the obstacle is now "behind" or "beside" us, not "in front"
            # We check dot product with original goal again.
            
            vec_to_orig = self.original_goal - x
            dot_orig = np.dot(vec_to_orig / np.linalg.norm(vec_to_orig), grad_norm)
            
            # If obstacle not blocking path to original goal (-0.2 is lenient threshold)
            path_clear = dot_orig > -0.2 
            
            if dist_virtual < 1.0 or path_clear:
                print("Path clear. Returning to Original Goal.")
                self.mode = "ORIGINAL"
                self.current_target = self.original_goal

        # --- QP SOLVER (Standard) ---
        u_ref = 2.0 * (self.current_target - x)
        
        diff_goal = x - self.current_target
        V = np.dot(diff_goal, diff_goal)
        grad_V = 2 * diff_goal

        P = matrix(np.diag([1.0, 1.0, self.clf_penalty]), tc='d')
        q = matrix(np.array([-u_ref[0], -u_ref[1], 0.0]), tc='d')
        
        G = matrix(np.array([
            [grad_V[0], grad_V[1], -1.0],
            [-grad_h[0], -grad_h[1], 0.0]
        ]), tc='d')
        
        h_qp = matrix(np.array([
            -self.alpha * V,
            self.gamma * h
        ]), tc='d')

        try:
            sol = solvers.qp(P, q, G, h_qp)
            z = np.array(sol['x']).flatten()
            return z[:2], z[2], self.current_target
        except ValueError:
            return np.array([0.0, 0.0]), 0.0, self.current_target

# --- SIMULATION LOOP ---
def run_simulation():
    start_pos = np.array([0.0, 5.0])    
    goal_pos = np.array([12.0, 5.0])    
    
    # Obstacle
    rect_config = {'center': [6.0, 5.0], 'radius': 2.5}

    controller = CLFCBF_SmartVirtual_Controller(rect_config, goal_pos)
    
    dt = 0.05
    steps = 400
    
    current_pos = start_pos.copy()
    trajectory = [current_pos.copy()]
    target_history = [] 
    
    for i in range(steps):
        u_safe, slack, active_target = controller.get_control(current_pos)
        current_pos = current_pos + u_safe * dt
        
        trajectory.append(current_pos.copy())
        target_history.append(active_target.copy())
        
        if np.linalg.norm(current_pos - goal_pos) < 0.1:
            print(f"Goal Reached at step {i}!")
            break

    # --- VISUALIZATION ---
    trajectory = np.array(trajectory)
    target_history = np.array(target_history)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    circle_patch = patches.Circle(
        rect_config['center'], rect_config['radius'], 
        linewidth=2, edgecolor='r', facecolor='salmon', alpha=0.5, label='Obstacle'
    )
    ax1.add_patch(circle_patch)
    
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Robot Path')
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax1.plot(goal_pos[0], goal_pos[1], 'g*', markersize=15, label='Original Goal')
    
    # Plot Virtual Goal Locations
    unique_targets = np.unique(target_history, axis=0)
    for t in unique_targets:
        if np.linalg.norm(t - goal_pos) > 0.1:
            ax1.plot(t[0], t[1], 'mx', markersize=10, label='Virtual Goal')

    ax1.set_aspect('equal')
    ax1.set_xlim(-1, 14)
    ax1.set_ylim(0, 10)
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("Smoother 'Nudge' Strategy (40 deg deviation)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()