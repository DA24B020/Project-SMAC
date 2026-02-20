import pygame
import numpy as np
import cvxpy as cp
import heapq
import math

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1200, 800
GRID_SIZE = 40  # Coarser grid for faster A* replanning
DT = 1.0/60.0   # 60 Hz
ROBOT_RADIUS = 15
OBSTACLE_RADIUS = 30
MAX_THRUST = 30.0
DRAG_COEFF = 0.12  # Higher drag for easier control
SAFE_MARGIN = 10.0 # Extra buffer for safety

# Colorsa
WHITE = (220, 220, 220)
BLACK = (20, 20, 30)
RED = (255, 80, 80)
GREEN = (80, 255, 80)
BLUE = (50, 150, 255)
YELLOW = (255, 200, 50)
GRAY = (50, 50, 60)
DARK_GRAY = (30, 30, 40)

# ==========================================
# 1. HELPER: GLOBAL PLANNER (A*)
# ==========================================
class GlobalPlanner:
    def __init__(self, width, height, grid_size):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.cols = width // grid_size
        self.rows = height // grid_size

    def get_grid_pos(self, pos):
        return (int(pos[0] // self.grid_size), int(pos[1] // self.grid_size))

    def plan(self, start_pos, end_pos, obstacles):
        start = self.get_grid_pos(start_pos)
        end = self.get_grid_pos(end_pos)
        
        # Build blocked grid
        blocked = set()
        for obs in obstacles:
            ox, oy = obs.pos
            r = obs.radius + ROBOT_RADIUS + 5 # Inflate by robot size
            
            # Simple bounding box blockage
            min_x = max(0, int((ox - r) // self.grid_size))
            max_x = min(self.cols-1, int((ox + r) // self.grid_size))
            min_y = max(0, int((oy - r) // self.grid_size))
            max_y = min(self.rows-1, int((oy + r) // self.grid_size))
            
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    blocked.add((x, y))

        pq = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        
        while pq:
            _, current = heapq.heappop(pq)
            if current == end:
                path = []
                while current in came_from:
                    px = current[0] * self.grid_size + self.grid_size // 2
                    py = current[1] * self.grid_size + self.grid_size // 2
                    path.append(np.array([px, py]))
                    current = came_from[current]
                path.reverse()
                # Optimize: Remove intermediate points for straighter lines
                return path[::2] if len(path) > 10 else path

            neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
            for dx, dy in neighbors:
                nx, ny = current[0]+dx, current[1]+dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) not in blocked:
                    # Diagonal cost is 1.4, Straight is 1.0
                    move_cost = 1.414 if dx!=0 and dy!=0 else 1.0
                    tentative_g = g_score[current] + move_cost
                    
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        # Heuristic: Euclidean
                        h = math.hypot(nx-end[0], ny-end[1])
                        heapq.heappush(pq, (tentative_g + h, (nx, ny)))
        return []

    def get_rabbit(self, robot_pos, path):
        """Finds the 'Carrot' point on the path."""
        if not path: return robot_pos
        # Find closest point on path
        dists = [np.linalg.norm(p - robot_pos) for p in path]
        closest_idx = np.argmin(dists)
        
        # Look ahead 3-5 nodes (approx 100-200 pixels)
        target_idx = min(closest_idx + 3, len(path) - 1)
        return path[target_idx]

# ==========================================
# 2. HELPER: LOCAL MPC (Trajectory)
# ==========================================
class LocalMPC:
    def __init__(self, horizon=10):
        self.N = horizon
        self.u = cp.Variable((2, self.N))
        self.x = cp.Variable((4, self.N + 1))
        self.x_init = cp.Parameter(4)
        self.x_target = cp.Parameter(2)
        
        # Model: Simple Point Mass (No Drag in prediction for speed)
        A = np.eye(4)
        A[0, 2] = DT; A[1, 3] = DT
        B = np.zeros((4, 2))
        B[2, 0] = DT; B[3, 1] = DT
        
        cost = 0
        constraints = [self.x[:, 0] == self.x_init]
        
        for k in range(self.N):
            # Cost: Distance to Target (High) + Control Effort (Low)
            pos_err = self.x[0:2, k+1] - self.x_target
            cost += cp.sum_squares(pos_err) * 10.0
            cost += cp.sum_squares(self.u[:, k]) * 0.01

            vel_state = self.x[2:4, k+1] 
            cost += cp.sum_squares(vel_state) * 0.5
            
            constraints += [self.x[:, k+1] == A @ self.x[:, k] + B @ self.u[:, k]]
            # Max Thrust
            constraints += [cp.norm(self.u[:, k], "inf") <= MAX_THRUST]

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, state, target_pos):
        self.x_init.value = state
        self.x_target.value = target_pos
        try:
            # Solve with OSQP (Fast, reliable for QPs)
            self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-3, eps_rel=1e-3)
            if self.u.value is None: return np.zeros(2)
            return self.u[:, 0].value
        except:
            return np.zeros(2)

# ==========================================
# 3. HELPER: SAFETY FILTER (HOCBF)
# ==========================================
class SafetyFilter:
    def filter(self, u_des, state, obstacles):
        px, py, vx, vy = state
        u = cp.Variable(2)
        slack = cp.Variable(1)
        
        constraints = [
            cp.norm(u, "inf") <= MAX_THRUST,
            slack >= 0
        ]
        
        # --- HOCBF Parameters ---
        k1, k2 = 3.0, 3.0 # Tuning: Higher = brakes later/harder
        margin = ROBOT_RADIUS + 5.0
        
        active_obs = 0
        
        # --- WALL CONSTRAINTS (The "Geo-Fence") ---
        # We treat each wall as a flat obstacle with infinite radius.
        # HOCBF formulation: h_ddot + k1*h_dot + k2*h >= -slack

        # 1. LEFT WALL (x = 0)
        # h = px - radius
        # h_dot = vx
        h_left = px - ROBOT_RADIUS
        h_dot_left = vx
        Lf_left = -DRAG_COEFF * vx  # Drag contribution
        # Constraint: (u_x) >= -Lf - k2*h - k1*h_dot
        constraints.append(u[0] >= -Lf_left - k2 * h_left - k1 * h_dot_left - slack)

        # 2. RIGHT WALL (x = WIDTH)
        # h = (WIDTH - radius) - px
        # h_dot = -vx
        h_right = (WIDTH - ROBOT_RADIUS) - px
        h_dot_right = -vx
        Lf_right = -DRAG_COEFF * (-vx) # Drag acts opposite to motion
        # Constraint: (-u_x) >= ...  -->  u_x <= ...
        # (u_x * -1) >= RHS  =>  u_x <= -RHS
        constraints.append(-u[0] >= -Lf_right - k2 * h_right - k1 * h_dot_right - slack)

        # 3. TOP WALL (y = 0)
        # h = py - radius
        # h_dot = vy
        h_top = py - ROBOT_RADIUS
        h_dot_top = vy
        Lf_top = -DRAG_COEFF * vy
        constraints.append(u[1] >= -Lf_top - k2 * h_top - k1 * h_dot_top - slack)

        # 4. BOTTOM WALL (y = HEIGHT)
        # h = (HEIGHT - radius) - py
        # h_dot = -vy
        h_bottom = (HEIGHT - ROBOT_RADIUS) - py
        h_dot_bottom = -vy
        Lf_bottom = -DRAG_COEFF * (-vy)
        constraints.append(-u[1] >= -Lf_bottom - k2 * h_bottom - k1 * h_dot_bottom - slack)

        for obs in obstacles:
            ox, oy = obs.pos
            ovx, ovy = obs.vel
            r = obs.radius
            
            dx, dy = px - ox, py - oy
            dist = np.sqrt(dx**2 + dy**2)
            
            # Optimization: Ignore far obstacles
            if dist > r + margin + 200: continue
            
            # Normal Vector (Pointing AWAY from obstacle)
            nx, ny = dx/dist, dy/dist
            
            # 1. HOCBF State: b (Distance Safety)
            b = dist - (r + margin)
            
            # 2. HOCBF Velocity: b_dot (Relative Velocity Safety)
            # Crucial: Must subtract obstacle velocity!
            rel_vx = vx - ovx
            rel_vy = vy - ovy
            b_dot = rel_vx * nx + rel_vy * ny
            
            psi1 = b_dot + k1 * b
            
            # 3. HOCBF Acceleration: Lie Derivatives
            # Lf (Drift): Effect of current velocity/drag on safety
            acc_drag_x = -DRAG_COEFF * vx
            acc_drag_y = -DRAG_COEFF * vy
            
            Lf_psi1 = acc_drag_x * nx + acc_drag_y * ny
            Lg_psi1 = np.array([nx, ny]) # Effect of Thrust on safety
            
            # Constraint: Lf + Lg*u + k2*psi1 >= -slack
            constraints.append(Lg_psi1 @ u >= -Lf_psi1 - k2 * psi1 - slack)
            active_obs += 1

        # Objective: Track u_des, minimize slack^2 heavily
        obj = cp.Minimize(cp.sum_squares(u - u_des) + 100000 * slack**2)
        
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if u.value is None: return u_des
            return u.value
        except:
            return u_des

# ==========================================
# 4. HELPER: DYNAMIC OBSTACLE & PHYSICS
# ==========================================
class DynamicObstacle:
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.vel = np.random.uniform(-40, 40, size=2)
        self.radius = OBSTACLE_RADIUS
        self.mass = 2.0 # Assume heavier than robot
    
    def update(self, dt):
        self.pos += self.vel * dt
        
        # Screen Bounce
        if self.pos[0] < self.radius: 
            self.pos[0] = self.radius; self.vel[0] *= -1
        if self.pos[0] > WIDTH - self.radius: 
            self.pos[0] = WIDTH - self.radius; self.vel[0] *= -1
            
        if self.pos[1] < self.radius: 
            self.pos[1] = self.radius; self.vel[1] *= -1
        if self.pos[1] > HEIGHT - self.radius: 
            self.pos[1] = HEIGHT - self.radius; self.vel[1] *= -1

def resolve_collisions(obstacles):
    """Simple O(N^2) Elastic Collision Resolver"""
    n = len(obstacles)
    for i in range(n):
        for j in range(i + 1, n):
            p1, v1 = obstacles[i].pos, obstacles[i].vel
            p2, v2 = obstacles[j].pos, obstacles[j].vel
            
            dist_vec = p1 - p2
            dist = np.linalg.norm(dist_vec)
            min_dist = obstacles[i].radius + obstacles[j].radius
            
            if dist < min_dist:
                # 1. Position Correction (prevent sticking)
                overlap = min_dist - dist
                n_vec = dist_vec / (dist + 1e-6)
                obstacles[i].pos += n_vec * (overlap / 2.0)
                obstacles[j].pos -= n_vec * (overlap / 2.0)
                
                # 2. Velocity Exchange (Elastic 1D projected)
                # v1' = v1 - 2*m2/(m1+m2) * <v1-v2, n> * n
                dv = v1 - v2
                m1, m2 = obstacles[i].mass, obstacles[j].mass
                dot = np.dot(dv, n_vec)
                
                # Only bounce if moving towards each other
                if dot < 0:
                    obstacles[i].vel -= (2*m2 / (m1+m2)) * dot * n_vec
                    obstacles[j].vel += (2*m1 / (m1+m2)) * dot * n_vec

# ==========================================
# 5. MAIN APPLICATION (NO FONTS VERSION)
# ==========================================
def draw_arrow(screen, color, start, end, size=10):
    pygame.draw.line(screen, color, start, end, 2)
    # Simple arrow head logic
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    pts = [
        (end[0]+size*math.sin(math.radians(rotation)), end[1]+size*math.cos(math.radians(rotation))),
        (end[0]+size*math.sin(math.radians(rotation-120)), end[1]+size*math.cos(math.radians(rotation-120))),
        (end[0]+size*math.sin(math.radians(rotation+120)), end[1]+size*math.cos(math.radians(rotation+120)))
    ]
    pygame.draw.polygon(screen, color, pts)

def main():
    pygame.init()
    # pygame.font.init()  <-- REMOVED TO PREVENT CRASH
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sandbox (NO TEXT MODE): [L-Click] Goal | [R-Click] Obs | [SPACE] Start | [R] Reset")
    clock = pygame.time.Clock()
    
    # Modules
    planner = GlobalPlanner(WIDTH, HEIGHT, GRID_SIZE)
    mpc = LocalMPC(horizon=12)
    safety = SafetyFilter()
    
    # State Variables
    running = True
    simulation_active = False
    
    # Game Data
    robot_start = np.array([100.0, HEIGHT/2])
    robot_pos = robot_start.copy()
    robot_vel = np.array([0.0, 0.0])
    
    goal_pos = np.array([WIDTH - 100.0, HEIGHT/2])
    obstacles = []
    
    path = []
    rabbit = goal_pos
    u_safe = np.zeros(2)

    while running:
        dt = clock.tick(60) / 1000.0
        
        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulation_active = not simulation_active
                    if simulation_active:
                        path = planner.plan(robot_pos, goal_pos, obstacles)
                
                if event.key == pygame.K_r:
                    # Reset Logic
                    simulation_active = False
                    robot_pos = robot_start.copy()
                    robot_vel = np.zeros(2)
                    obstacles = []
                    path = []
                    u_safe = np.zeros(2)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                
                if event.button == 1: # Left Click: Set Goal
                    goal_pos = np.array([mx, my], dtype=float)
                    if simulation_active:
                        path = planner.plan(robot_pos, goal_pos, obstacles)
                        
                if event.button == 3: # Right Click: Add Obstacle
                    obstacles.append(DynamicObstacle(mx, my))
                    if simulation_active:
                        path = planner.plan(robot_pos, goal_pos, obstacles)

        # --- SIMULATION LOOP ---
        if simulation_active:
            # 1. CHECK GOAL REACHED (New Code)
            dist_to_goal = np.linalg.norm(robot_pos - goal_pos)
            if dist_to_goal < 25.0: # 25 pixels tolerance
                print("GOAL REACHED!")
                simulation_active = False # Stop the simulation
                robot_vel = np.zeros(2)   # Kill velocity
                robot_pos = goal_pos.copy() # Snap to center (optional, looks clean)
                continue
            for obs in obstacles: obs.update(dt)
            resolve_collisions(obstacles)
            
            if not path or pygame.time.get_ticks() % 500 < 20: 
                path = planner.plan(robot_pos, goal_pos, obstacles)
            
            state = np.array([robot_pos[0], robot_pos[1], robot_vel[0], robot_vel[1]])
            rabbit = planner.get_rabbit(robot_pos, path)
            u_mpc = mpc.solve(state, rabbit)
            u_safe = safety.filter(u_mpc, state, obstacles)
            
            acc = u_safe - DRAG_COEFF * robot_vel
            robot_vel += acc * dt
            robot_pos += robot_vel * dt
            
            # Boundary Check
            if robot_pos[0] < ROBOT_RADIUS: robot_pos[0] = ROBOT_RADIUS; robot_vel[0] *= -0.5
            if robot_pos[0] > WIDTH - ROBOT_RADIUS: robot_pos[0] = WIDTH - ROBOT_RADIUS; robot_vel[0] *= -0.5
            if robot_pos[1] < ROBOT_RADIUS: robot_pos[1] = ROBOT_RADIUS; robot_vel[1] *= -0.5
            if robot_pos[1] > HEIGHT - ROBOT_RADIUS: robot_pos[1] = HEIGHT - ROBOT_RADIUS; robot_vel[1] *= -0.5

        # --- RENDERING ---
        screen.fill(DARK_GRAY)
        
        # Draw Goal
        pygame.draw.circle(screen, GREEN, (int(goal_pos[0]), int(goal_pos[1])), 10)
        pygame.draw.circle(screen, GREEN, (int(goal_pos[0]), int(goal_pos[1])), 20, 1)
        
        # Draw Obstacles
        for obs in obstacles:
            pygame.draw.circle(screen, RED, (int(obs.pos[0]), int(obs.pos[1])), int(obs.radius))
            end = obs.pos + obs.vel * 0.3
            pygame.draw.line(screen, (100, 0, 0), obs.pos, end, 2)

        # Draw Path
        if path and len(path) > 1:
            pygame.draw.lines(screen, GRAY, False, path, 2)
            pygame.draw.circle(screen, YELLOW, (int(rabbit[0]), int(rabbit[1])), 5)

        # Draw Robot
        speed = np.linalg.norm(robot_vel)
        angle = math.atan2(robot_vel[1], robot_vel[0]) if speed > 1 else -1.57
        tip = robot_pos + np.array([math.cos(angle), math.sin(angle)]) * ROBOT_RADIUS
        left = robot_pos + np.array([math.cos(angle + 2.5), math.sin(angle + 2.5)]) * ROBOT_RADIUS
        right = robot_pos + np.array([math.cos(angle - 2.5), math.sin(angle - 2.5)]) * ROBOT_RADIUS
        pygame.draw.polygon(screen, BLUE, [tip, left, right])
        
        if simulation_active and speed > 0.5:
            # Draw Velocity Vector (Cyan) - Scaled down slightly (0.5) so it fits
            draw_arrow(screen, (0, 255, 255), robot_pos, robot_pos + robot_vel * 0.5)

        # --- TEXT RENDERING REMOVED ---
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()