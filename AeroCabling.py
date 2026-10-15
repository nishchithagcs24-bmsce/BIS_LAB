import math
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString

# -------------------- 1. PARAMETERS --------------------------

START = (5.0, 50.0)
GOAL = (95.0, 50.0)

OBSTACLE_RECTS = [
    (15, 15, 25, 40),
    (15, 60, 25, 85),
    (30, 35, 40, 65),
    (45, 5, 55, 30),
    (45, 70, 55, 95),
    (60, 35, 70, 65),
    (75, 15, 85, 40),
    (75, 60, 85, 85),
]

OBSTACLES = [
    Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    for x1,y1,x2,y2 in OBSTACLE_RECTS
]

N_ANTS = 30
MAX_ITERATIONS = 100
ALPHA = 1.0
BETA = 3.0
RHO = 0.30
Q = 100.0

OBSTACLE_PENALTY = 20.0
BEND_COUNT_PENALTY = 5.0
BEND_ANGLE_PENALTY = 10.0
MAX_TURN_ANGLE = math.pi / 2
EPS = 1e-9

random.seed(42)
np.random.seed(42)

# -------------------- 2. GEOMETRY ---------------------------

def dist(a, b):
    return math.dist(a, b)

def segment_free(a, b):
    line = LineString([a, b])
    for obstacle in OBSTACLES:
        interior = obstacle.buffer(-1e-7)
        if not interior.is_empty and interior.intersects(line):
            return False
    return True

def crossed_obstacles(a, b):
    line = LineString([a, b])
    count = 0
    for obstacle in OBSTACLES:
        interior = obstacle.buffer(-1e-7)
        if not interior.is_empty and interior.intersects(line):
            count += 1
    return count

def turn_angle(a, b, c):
    u = np.array([b[0]-a[0], b[1]-a[1]])
    v = np.array([c[0]-b[0], c[1]-b[1]])
    denom = np.linalg.norm(u) * np.linalg.norm(v)

    if denom < EPS:
        return 0.0

    cosine = np.clip(np.dot(u, v)/denom, -1.0, 1.0)
    return math.acos(cosine)

def bend_penalty(angle):
    if angle >= MAX_TURN_ANGLE:
        return float("inf")
    return BEND_ANGLE_PENALTY * (1-math.cos(angle))

# -------------------- 3. BCLP --------------------------------

def bclp(start, goal):
    active = set()
    primary = LineString([start, goal])

    for i, obstacle in enumerate(OBSTACLES):
        interior = obstacle.buffer(-1e-7)
        if not interior.is_empty and interior.intersects(primary):
            active.add(i)

    changed = True

    while changed:
        changed = False
        vertices = []

        for i in active:
            vertices.extend(
                list(OBSTACLES[i].exterior.coords)[:-1]
            )

        for vertex in vertices:
            for target in (start, goal):
                extension = LineString([vertex, target])

                for j, obstacle in enumerate(OBSTACLES):
                    if j in active:
                        continue

                    interior = obstacle.buffer(-1e-7)

                    if (not interior.is_empty
                            and interior.intersects(extension)):
                        active.add(j)
                        changed = True

    return active

# -------------------- 4. VISIBILITY GRAPH --------------------

def build_graph(active=None):
    nodes = [START, GOAL]

    indices = (
        range(len(OBSTACLES))
        if active is None
        else sorted(active)
    )

    for i in indices:
        nodes.extend(
            list(OBSTACLES[i].exterior.coords)[:-1]
        )

    nodes = list(dict.fromkeys(nodes))
    graph = {i: [] for i in range(len(nodes))}

    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if segment_free(nodes[i], nodes[j]):
                length = dist(nodes[i], nodes[j])
                graph[i].append((j, length))
                graph[j].append((i, length))

    return nodes, graph

# -------------------- 5. PATH COST ---------------------------

def path_cost(path, nodes):
    if len(path) < 2:
        return float("inf")

    length = sum(
        dist(nodes[path[i]], nodes[path[i+1]])
        for i in range(len(path)-1)
    )

    angles = [
        turn_angle(nodes[path[i-1]],
                   nodes[path[i]],
                   nodes[path[i+1]])
        for i in range(1, len(path)-1)
    ]

    if any(a >= MAX_TURN_ANGLE for a in angles):
        return float("inf")

    return (
        length
        + BEND_COUNT_PENALTY * len(angles)
        + sum(bend_penalty(a) for a in angles)
    )

# -------------------- 6. HEURISTIC ---------------------------

def heuristic(i, j, nodes, previous=None):
    estimate = dist(nodes[i], nodes[j])
    estimate += dist(nodes[j], GOAL)
    estimate += OBSTACLE_PENALTY * crossed_obstacles(
        nodes[j], GOAL
    )

    if previous is not None:
        angle = turn_angle(
            nodes[previous], nodes[i], nodes[j]
        )
        if angle >= MAX_TURN_ANGLE:
            return EPS
        estimate += bend_penalty(angle)

    return 1.0 / (estimate + EPS)

# -------------------- 7. IMPROVED ACO ------------------------

def iaco(nodes, graph):
    n = len(nodes)
    pheromone = np.ones((n, n))

    best_path = None
    best_cost = float("inf")
    history = []

    for iteration in range(MAX_ITERATIONS):
        solutions = []

        for _ in range(N_ANTS):
            path = [0]
            visited = {0}

            while path[-1] != 1:
                current = path[-1]
                previous = path[-2] if len(path) > 1 else None
                candidates = []

                for neighbor, _ in graph[current]:
                    if neighbor in visited:
                        continue

                    if previous is not None:
                        angle = turn_angle(
                            nodes[previous],
                            nodes[current],
                            nodes[neighbor]
                        )
                        if angle >= MAX_TURN_ANGLE:
                            continue

                    candidates.append(neighbor)

                if not candidates:
                    break

                weights = []

                for j in candidates:
                    eta = heuristic(
                        current, j, nodes, previous
                    )
                    weights.append(
                        pheromone[current, j]**ALPHA
                        * eta**BETA
                    )

                weights = np.asarray(weights, dtype=float)

                if (not np.all(np.isfinite(weights))
                        or weights.sum() <= 0):
                    probabilities = np.ones(len(candidates))
                    probabilities /= len(candidates)
                else:
                    probabilities = weights / weights.sum()

                nxt = random.choices(
                    candidates,
                    weights=probabilities.tolist(),
                    k=1
                )[0]

                path.append(nxt)
                visited.add(nxt)

                if len(path) > n:
                    break

            if path[-1] != 1:
                continue

            cost = path_cost(path, nodes)

            if not math.isfinite(cost):
                continue

            solutions.append((path, cost))

            if cost < best_cost:
                best_path = path.copy()
                best_cost = cost

        # Evaporate pheromone
        pheromone *= (1-RHO)

        # Deposit pheromone according to composite process cost
        for path, cost in solutions:
            deposit = Q / (cost + EPS)

            for a, b in zip(path[:-1], path[1:]):
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

        # Reinforce the best route found so far
        if best_path is not None:
            deposit = Q / (best_cost + EPS)

            for a, b in zip(best_path[:-1], best_path[1:]):
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

        history.append(best_cost)

        print(
            f"Iteration {iteration+1:03d}/{MAX_ITERATIONS} | "
            f"Best cost: {best_cost:.3f}"
        )

    return best_path, best_cost, history

# -------------------- 8. RUN ALGORITHM -----------------------

print("Starting BCLP graph simplification...")
active = bclp(START, GOAL)

print("Total obstacles:", len(OBSTACLES))
print("Activated obstacles:", len(active))

nodes, graph = build_graph(active)

print("Graph nodes:", len(nodes))
print("Graph edges:", sum(map(len, graph.values()))//2)

print("\nStarting IACO optimization...")
best_path, best_cost, history = iaco(nodes, graph)

# Retry on the full visibility graph if necessary
if best_path is None:
    print("Retrying with full visibility graph...")
    nodes, graph = build_graph(None)
    best_path, best_cost, history = iaco(nodes, graph)

if best_path is None:
    raise RuntimeError(
        "No feasible route found. Check the obstacles "
        "or relax the bending-angle constraint."
    )

route = [nodes[i] for i in best_path]

length = sum(
    dist(route[i], route[i+1])
    for i in range(len(route)-1)
)

angles = [
    math.degrees(turn_angle(route[i-1], route[i], route[i+1]))
    for i in range(1, len(route)-1)
]

print("\n========== RESULTS ==========")
print("Route coordinates:")
for p in route:
    print(f"({p[0]:.2f}, {p[1]:.2f})")

print(f"Path length: {length:.3f} units")
print(f"Bending points: {len(angles)}")
print(f"Total bending angle: {sum(angles):.2f} degrees")
print(f"Process cost: {best_cost:.3f}")

# -------------------- 9. VISUALIZATION -----------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]

for obstacle in OBSTACLES:
    x, y = obstacle.exterior.xy
    ax.fill(x, y, color="lightgray",
            edgecolor="black", alpha=0.9)

rx = [p[0] for p in route]
ry = [p[1] for p in route]

ax.plot(rx, ry, "r-o", linewidth=2, markersize=4,
        label="BCLP-IACO route")
ax.scatter(*START, color="green", s=100, label="Start")
ax.scatter(*GOAL, color="blue", s=100, label="Goal")

ax.set_title("Aerospace Cable Routing")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.legend()

axes[1].plot(history, color="darkblue")
axes[1].set_title("IACO Convergence")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Best process cost")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
