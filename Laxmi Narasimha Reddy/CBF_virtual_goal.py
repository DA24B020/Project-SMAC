import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cvxopt import matrix, solvers

# Suppress solver output
solvers.options['show_progress'] = False

class SimConfig:
    DT = 0.05
    STEPS = 500
    
    # QP Tuning
    ALPHA = 2.0          # How aggressively we pursue the goal
    GAMMA = 2.0          # Safety margin (Lower = allows getting closer)
    CLF_PENALTY = 1000.0 # High cost to ensure we don't give up on the goal
    
    # Ghost Target Tuning
    LOOKAHEAD = 4.0      # How far to project the ghost target sideways
    BLEND_RATE = 0.05    # Very slow return to original goal (prevents jitter)
    SENSE_DIST = 5.0     # Distance to start reacting

class CircleObstacle:
    def __init__(self, x, y, r):
        self.center = np.array([x, y], dtype=np.float64)
        self.radius = float(r)
    
    def get_barrier(self, robot_pos):
        # h(x) = ||x - c||^2 - r^2
        diff = robot_pos - self.center
        dist_sq = np.dot(diff, diff)
        dist = np.sqrt(dist_sq)
        
        # Barrier function h(x)
        h = dist_sq - self.radius**2
        
        # Gradient of h(x)
        grad_h = 2 * diff
        return h, grad_h, dist

class CLFCBF_Controller:
    def __init__(self, start_pos, goal_pos, obstacle):
        self.goal = np.array(goal_pos, dtype=np.float64)
        self.obstacle = obstacle
        self.ghost_target = np.array(goal_pos, dtype=np.float64)
        
    def get_control(self, x):
        # 1. Update Ghost Target using Vector Blending
        self._update_ghost_target(x)
        
        # 2. Nominal Control (P-Controller)
        u_ref = -1.5 * (x - self.ghost_target)
        
        # 3. Standard QP Setup (Same as before)
        P_np = np.diag([1.0, 1.0, SimConfig.CLF_PENALTY])
        q_np = np.array([-u_ref[0], -u_ref[1], 0.0])
        
        diff = x - self.ghost_target
        V = np.dot(diff, diff)
        grad_V = 2 * diff
        
        G_list = [[grad_V[0], grad_V[1], -1.0]]
        h_list = [-SimConfig.ALPHA * V]
        
        h_val, grad_h, dist = self.obstacle.get_barrier(x)
        if dist < SimConfig.SENSE_DIST:
            G_list.append([-grad_h[0], -grad_h[1], 0.0])
            h_list.append(SimConfig.GAMMA * h_val)
            
        P = matrix(P_np, tc='d')
        q = matrix(q_np, tc='d')
        G = matrix(np.array(G_list), tc='d')
        h_qp = matrix(np.array(h_list), tc='d')
        
        try:
            sol = solvers.qp(P, q, G, h_qp)
            z = np.array(sol['x']).flatten()
            return z[:2], self.ghost_target
        except ValueError:
            return np.array([0.0, 0.0]), self.ghost_target

    def _update_ghost_target(self, x):
        vec_to_real = self.goal - x
        dist_real = np.linalg.norm(vec_to_real)
        u_goal = vec_to_real / (dist_real + 1e-6)

        vec_to_obs = self.obstacle.center - x
        dist_obs = np.linalg.norm(vec_to_obs)
        u_obs = vec_to_obs / (dist_obs + 1e-6)

        # 1. Base mixing weight based on distance
        safe_margin = self.obstacle.radius + 1.0
        start_blend_dist = SimConfig.SENSE_DIST + 1.0
        
        if dist_obs > start_blend_dist:
            base_weight = 0.0
        elif dist_obs < safe_margin:
            base_weight = 1.0
        else:
            base_weight = (start_blend_dist - dist_obs) / (start_blend_dist - safe_margin)

        # --- THE FIX: DIRECTIONAL FADING ---
        # Dot product checks if the obstacle is in front of us.
        # 1.0 = dead ahead, 0.0 = to the side, negative = behind us.
        obs_in_front = np.dot(u_goal, u_obs)
        
        # We clip it so that if the obstacle is beside or behind us, factor is 0.
        directional_factor = np.clip(obs_in_front, 0.0, 1.0)
        
        # Multiply base weight by directional factor
        # This smoothly turns off avoidance as we pass the obstacle!
        weight = base_weight * directional_factor

        # 2. Tangent Vector
        cross_prod = np.cross(vec_to_obs, vec_to_real)
        direction = 1.0 if cross_prod > 0 else -1.0
        if abs(cross_prod) < 0.1: direction = -1.0 

        theta = direction * np.pi / 2
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        u_tangent = np.dot(R, u_obs)
        
        # 3. Blend vectors
        u_mixed = (1 - weight) * u_goal + weight * u_tangent
        u_mixed = u_mixed / (np.linalg.norm(u_mixed) + 1e-6)
        
        # 4. Smooth Lookahead
        # We cap the lookahead so the ghost target never overshoots the true goal
        dynamic_lookahead = min(SimConfig.LOOKAHEAD, dist_real)
        
        # Set final target
        self.ghost_target = x + u_mixed * dynamic_lookahead

def run():
    start = np.array([0.0, 5.0])
    goal = np.array([12.0, 5.0])
    
    # One large obstacle in the middle
    obstacle = CircleObstacle(6.0, 5.0, 1.5)
    
    controller = CLFCBF_Controller(start, goal, obstacle)
    
    path = [start.copy()]
    ghost_history = [goal.copy()]
    curr_x = start.copy()
    
    print("Simulating Single Obstacle Avoidance...")
    
    for k in range(SimConfig.STEPS):
        u, g_target = controller.get_control(curr_x)
        
        curr_x = curr_x + u * SimConfig.DT
        path.append(curr_x.copy())
        ghost_history.append(g_target.copy())
        
        if np.linalg.norm(curr_x - goal) < 0.1:
            print("Goal Reached!")
            break

    # --- PLOTTING ---
    path = np.array(path)
    ghost_history = np.array(ghost_history)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Draw Obstacle
    c = patches.Circle(obstacle.center, obstacle.radius, fc='salmon', ec='darkred', label='Obstacle')
    ax.add_patch(c)
    
    # Draw Paths
    ax.plot(path[:,0], path[:,1], 'b-', linewidth=2, label='Robot Path')
    ax.scatter(ghost_history[::10,0], ghost_history[::10,1], c='purple', s=15, alpha=0.5, label='Ghost Target')
    
    # Start/Goal
    ax.plot(start[0], start[1], 'go', label='Start')
    ax.plot(goal[0], goal[1], 'g*', markersize=12, label='Goal')
    
    ax.set_aspect('equal')
    ax.set_xlim(-1, 13)
    ax.set_ylim(0, 10)
    ax.legend()
    ax.grid(True)
    ax.set_title("CBF with Ghost Target")
    
    plt.show()

if __name__ == "__main__":
    run()