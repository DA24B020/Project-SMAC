import pygame
import numpy as np
import cvxpy as cp
import heapq
import math

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1200, 800
GRID_SIZE = 40
DT = 1.0/60.0
ROBOT_RADIUS = 20  # Slightly larger for the cycle graphic
OBSTACLE_RADIUS = 30
MAX_THRUST = 25.0
DRAG_COEFF = 0.08
SAFE_MARGIN = 15.0 # Increased for safety

# Colors
WHITE = (220, 220, 220)
DARK_GRAY = (30, 30, 40)
GREEN = (80, 255, 80)
# Monkey Colors
BROWN = (139, 69, 19)
TAN = (210, 180, 140)
# Cycle Colors
CYAN = (0, 255, 255)
TIRE_GRAY = (50, 50, 50)

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
        
        blocked = set()
        for obs in obstacles:
            ox, oy = obs.pos
            r = obs.radius + ROBOT_RADIUS + 5
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
                return path[::2] if len(path) > 10 else path

            neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
            for dx, dy in neighbors:
                nx, ny = current[0]+dx, current[1]+dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) not in blocked:
                    cost = 1.414 if dx!=0 and dy!=0 else 1.0
                    tentative_g = g_score[current] + cost
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        h = math.hypot(nx-end[0], ny-end[1])
                        heapq.heappush(pq, (tentative_g + h, (nx, ny)))
        return []

    def get_rabbit(self, robot_pos, path):
        if not path: return robot_pos
        dists = [np.linalg.norm(p - robot_pos) for p in path]
        closest_idx = np.argmin(dists)
        target_idx = min(closest_idx + 3, len(path) - 1)
        return path[target_idx]

# ==========================================
# 2. HELPER: MPC & SAFETY
# ==========================================
class LocalMPC:
    def __init__(self, horizon=10):
        self.N = horizon
        self.u = cp.Variable((2, self.N))
        self.x = cp.Variable((4, self.N + 1))
        self.x_init = cp.Parameter(4)
        self.x_target = cp.Parameter(2)
        
        A = np.eye(4); A[0, 2] = DT; A[1, 3] = DT
        B = np.zeros((4, 2)); B[2, 0] = DT; B[3, 1] = DT
        
        cost = 0
        constraints = [self.x[:, 0] == self.x_init]
        
        for k in range(self.N):
            pos_err = self.x[0:2, k+1] - self.x_target
            # Added Velocity Penalty to help braking at goal
            vel_state = self.x[2:4, k+1]
            cost += cp.sum_squares(pos_err) * 10.0
            cost += cp.sum_squares(vel_state) * 0.5  # <--- Braking incentive
            cost += cp.sum_squares(self.u[:, k]) * 0.01
            
            constraints += [self.x[:, k+1] == A @ self.x[:, k] + B @ self.u[:, k]]
            constraints += [cp.norm(self.u[:, k], "inf") <= MAX_THRUST]

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, state, target_pos):
        self.x_init.value = state
        self.x_target.value = target_pos
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-3, eps_rel=1e-3)
            if self.u.value is None: return np.zeros(2)
            return self.u[:, 0].value
        except: return np.zeros(2)

class SafetyFilter:
    def filter(self, u_des, state, obstacles):
        px, py, vx, vy = state
        u = cp.Variable(2)
        slack = cp.Variable(1)
        
        constraints = [cp.norm(u, "inf") <= MAX_THRUST, slack >= 0]
        k1, k2 = 3.0, 3.0
        margin = ROBOT_RADIUS + 10.0 # Safety buffer
        
        # Wall Constraints
        constraints.append(u[0] >= -(-DRAG_COEFF*vx) - k2*(px-20) - k1*vx - slack) # Left
        constraints.append(-u[0] >= -(-DRAG_COEFF*-vx) - k2*((WIDTH-20)-px) - k1*-vx - slack) # Right
        constraints.append(u[1] >= -(-DRAG_COEFF*vy) - k2*(py-20) - k1*vy - slack) # Top
        constraints.append(-u[1] >= -(-DRAG_COEFF*-vy) - k2*((HEIGHT-20)-py) - k1*-vy - slack) # Bottom

        # Obstacle Constraints
        for obs in obstacles:
            ox, oy = obs.pos
            ovx, ovy = obs.vel
            dist = np.hypot(px-ox, py-oy)
            if dist > obs.radius + margin + 100: continue
            
            nx, ny = (px-ox)/dist, (py-oy)/dist
            b = dist - (obs.radius + margin)
            b_dot = (vx-ovx)*nx + (vy-ovy)*ny
            
            Lf = -DRAG_COEFF*vx*nx - DRAG_COEFF*vy*ny
            constraints.append(u[0]*nx + u[1]*ny >= -Lf - k2*b - k1*b_dot - slack)

        obj = cp.Minimize(cp.sum_squares(u - u_des) + 100000 * slack**2)
        try:
            cp.Problem(obj, constraints).solve(solver=cp.OSQP, verbose=False)
            return u.value if u.value is not None else u_des
        except: return u_des

# ==========================================
# 3. HELPER: PHYSICS & GRAPHICS
# ==========================================
class DynamicObstacle:
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.vel = np.random.uniform(-40, 40, size=2)
        self.radius = OBSTACLE_RADIUS
        self.mass = 2.0
    
    def update(self, dt):
        self.pos += self.vel * dt
        # Bounce off walls
        if self.pos[0] < self.radius: self.pos[0] = self.radius; self.vel[0] *= -1
        if self.pos[0] > WIDTH-self.radius: self.pos[0] = WIDTH-self.radius; self.vel[0] *= -1
        if self.pos[1] < self.radius: self.pos[1] = self.radius; self.vel[1] *= -1
        if self.pos[1] > HEIGHT-self.radius: self.pos[1] = HEIGHT-self.radius; self.vel[1] *= -1

def draw_cycle(screen, pos, angle_rad, radius):
    """Draws a Tron-like Cycle oriented by angle"""
    # Rotate points around center
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    
    # Offsets for wheels (Front/Back)
    wheel_dist = radius * 0.8
    front_pos = pos + R @ np.array([wheel_dist, 0])
    back_pos = pos + R @ np.array([-wheel_dist, 0])
    
    # Draw Wheels
    pygame.draw.circle(screen, TIRE_GRAY, front_pos.astype(int), int(radius * 0.4))
    pygame.draw.circle(screen, TIRE_GRAY, back_pos.astype(int), int(radius * 0.4))
    pygame.draw.circle(screen, CYAN, front_pos.astype(int), int(radius * 0.4), 2) # Rim
    pygame.draw.circle(screen, CYAN, back_pos.astype(int), int(radius * 0.4), 2) # Rim
    
    # Draw Body (Line connecting wheels + Rider hump)
    pygame.draw.line(screen, CYAN, back_pos, front_pos, 4)
    
    # Rider (Simple Hump)
    top_pos = pos + R @ np.array([0, -radius*0.5])
    pygame.draw.line(screen, CYAN, back_pos, top_pos, 2)
    pygame.draw.line(screen, CYAN, top_pos, front_pos, 2)

def draw_monkey(screen, pos, radius):
    """Draws a Monkey Face"""
    x, y = int(pos[0]), int(pos[1])
    r = int(radius)
    
    # Ears
    pygame.draw.circle(screen, BROWN, (x - int(r*0.9), y - int(r*0.3)), int(r*0.35))
    pygame.draw.circle(screen, BROWN, (x + int(r*0.9), y - int(r*0.3)), int(r*0.35))
    
    # Head Base
    pygame.draw.circle(screen, BROWN, (x, y), r)
    
    # Face (Tan part)
    # Using multiple circles to make the heart-shaped face
    pygame.draw.circle(screen, TAN, (x - int(r*0.3), y + int(r*0.1)), int(r*0.5))
    pygame.draw.circle(screen, TAN, (x + int(r*0.3), y + int(r*0.1)), int(r*0.5))
    pygame.draw.circle(screen, TAN, (x, y + int(r*0.4)), int(r*0.4))
    
    # Eyes
    pygame.draw.circle(screen, (0,0,0), (x - int(r*0.3), y), int(r*0.1))
    pygame.draw.circle(screen, (0,0,0), (x + int(r*0.3), y), int(r*0.1))

# ==========================================
# 4. MAIN
# ==========================================
def main():
    pygame.init()
    # pygame.font.init() # Disabled for safety on 3.14
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cycle vs Monkeys [SPACE: Start] [L-Click: Goal] [R-Click: Monkey]")
    clock = pygame.time.Clock()
    
    planner = GlobalPlanner(WIDTH, HEIGHT, GRID_SIZE)
    mpc = LocalMPC(horizon=12)
    safety = SafetyFilter()
    
    # Robot State
    robot_start = np.array([100.0, HEIGHT/2])
    robot_pos = robot_start.copy()
    robot_vel = np.array([0.0, 0.0])
    
    # Persistent Angle Logic
    last_known_angle = 0.0 
    
    goal_pos = np.array([WIDTH - 100.0, HEIGHT/2])
    obstacles = []
    path = []
    
    running = True
    simulation_active = False
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulation_active = not simulation_active
                    if simulation_active: path = planner.plan(robot_pos, goal_pos, obstacles)
                if event.key == pygame.K_r:
                    simulation_active = False
                    robot_pos = robot_start.copy()
                    robot_vel = np.zeros(2)
                    last_known_angle = 0.0 # Reset angle only on full reset
                    obstacles = []
                    path = []

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if event.button == 1: 
                    goal_pos = np.array([mx, my], dtype=float)
                    if simulation_active: path = planner.plan(robot_pos, goal_pos, obstacles)
                if event.button == 3: 
                    obstacles.append(DynamicObstacle(mx, my))
                    if simulation_active: path = planner.plan(robot_pos, goal_pos, obstacles)

        # LOGIC
        if simulation_active:
            # Check Win (STOP and SNAP)
            if np.linalg.norm(robot_pos - goal_pos) < 10.0:
                print("GOAL REACHED")
                simulation_active = False
                robot_vel = np.zeros(2)
                robot_pos = goal_pos.copy()
                # We DO NOT reset last_known_angle here, so it stays looking at the win
                continue

            for obs in obstacles: obs.update(dt)
            # Elastic Collisions between monkeys
            for i in range(len(obstacles)):
                for j in range(i+1, len(obstacles)):
                    dvec = obstacles[i].pos - obstacles[j].pos
                    dist = np.linalg.norm(dvec)
                    if dist < obstacles[i].radius*2:
                        n = dvec / (dist + 1e-6)
                        # Pos correction
                        overlap = obstacles[i].radius*2 - dist
                        obstacles[i].pos += n * overlap * 0.5
                        obstacles[j].pos -= n * overlap * 0.5
                        # Vel bounce
                        dv = obstacles[i].vel - obstacles[j].vel
                        if np.dot(dv, n) < 0:
                            obstacles[i].vel -= np.dot(dv, n) * n
                            obstacles[j].vel += np.dot(dv, n) * n

            if not path or pygame.time.get_ticks() % 500 < 20:
                new_path = planner.plan(robot_pos, goal_pos, obstacles)
                if new_path: path = new_path

            # Subsumption / Reflex
            closest_dist = float('inf')
            threat_vec = np.array([0., 0.])
            for obs in obstacles:
                d = np.linalg.norm(robot_pos - obs.pos) - obs.radius
                if d < closest_dist:
                    closest_dist = d
                    if d > 0: threat_vec = (robot_pos - obs.pos)/np.linalg.norm(robot_pos-obs.pos)
            
            # Decide Rabbit
            rabbit = robot_pos
            if closest_dist < 80.0: # Panic
                rabbit = robot_pos + threat_vec * 100.0
            elif path:
                rabbit = planner.get_rabbit(robot_pos, path)

            # Control
            state = np.hstack([robot_pos, robot_vel])
            u_mpc = mpc.solve(state, rabbit)
            u_safe = safety.filter(u_mpc, state, obstacles)
            
            acc = u_safe - DRAG_COEFF * robot_vel
            robot_vel += acc * dt
            robot_pos += robot_vel * dt
            
            # Wall Clamps
            if robot_pos[0] < ROBOT_RADIUS: robot_pos[0] = ROBOT_RADIUS; robot_vel[0] *= -0.5
            if robot_pos[0] > WIDTH-ROBOT_RADIUS: robot_pos[0] = WIDTH-ROBOT_RADIUS; robot_vel[0] *= -0.5
            if robot_pos[1] < ROBOT_RADIUS: robot_pos[1] = ROBOT_RADIUS; robot_vel[1] *= -0.5
            if robot_pos[1] > HEIGHT-ROBOT_RADIUS: robot_pos[1] = HEIGHT-ROBOT_RADIUS; robot_vel[1] *= -0.5

        # RENDER
        screen.fill(DARK_GRAY)
        
        # Draw Goal
        pygame.draw.circle(screen, GREEN, (int(goal_pos[0]), int(goal_pos[1])), 15)
        pygame.draw.circle(screen, GREEN, (int(goal_pos[0]), int(goal_pos[1])), 25, 2)
        
        # Draw Monkeys (Obstacles)
        for obs in obstacles:
            draw_monkey(screen, obs.pos, obs.radius)
        
        # Draw Path
        if path and len(path) > 1:
            pygame.draw.lines(screen, (200, 200, 200), False, path, 2)

        # Update Angle ONLY if moving significantly
        speed = np.linalg.norm(robot_vel)
        if speed > 1.0:
            last_known_angle = math.atan2(robot_vel[1], robot_vel[0])
        
        # Draw Cycle (Robot)
        draw_cycle(screen, robot_pos, last_known_angle, ROBOT_RADIUS)
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()