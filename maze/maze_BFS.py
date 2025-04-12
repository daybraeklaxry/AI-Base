# maze_BFS.py
import sys
import math
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# 辅助函数 
def find_path(parent, end_node):
    path = [end_node]
    while end_node in parent:
        end_node = parent[end_node]
        path.append(end_node)
    return path[::-1]

def find_portals(maze):
    portals = {}
    portal_pairs = {}
    rows, cols = len(maze), len(maze[0])
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] >= 4:
                portal_id = maze[r][c]
                if portal_id not in portals:
                    portals[portal_id] = []
                portals[portal_id].append((r, c))
    for portal_id, locations in portals.items():
        if len(locations) == 2:
            p1, p2 = locations[0], locations[1]
            portal_pairs[p1] = p2
            portal_pairs[p2] = p1
    return portal_pairs

def calculate_cost(maze, path, portal_map):
    total_cost = 0
    for i in range(len(path) - 1):
        curr = path[i]
        nxt = path[i+1]
        if curr in portal_map and portal_map[curr] == nxt:
            total_cost += 0
            continue
        dr, dc = nxt[0] - curr[0], nxt[1] - curr[1]
        base_cost = math.sqrt(2) if dr != 0 and dc != 0 else 1
        modifier = 1
        cell_type = maze[nxt[0]][nxt[1]]
        if cell_type == 2: modifier = 2.0
        elif cell_type == 3: modifier = 0.5
        total_cost += base_cost * modifier
    return total_cost

def BFS(maze, start, end, portal_map):
    rows, cols = len(maze), len(maze[0])
    queue = deque([start]) # 使用双端队列
    visited = {start} # 记录访问过的节点
    parent = {} # 记录路径父节点
    visited_order = [] # 记录访问顺序

    path_found = None

    while queue:
        now = queue.popleft() # 取出队首元素
        visited_order.append(now)

        # 检查是否是传送门入口
        if now in portal_map:
            teleport_dest = portal_map[now]
            # 只有当传送目标未被访问时才处理传送
            if teleport_dest not in visited:
                visited.add(teleport_dest)
                parent[teleport_dest] = now # 父节点设为传送门入口
                queue.append(teleport_dest) # 将传送目标加入队列
                continue 

        if now == end:
            path_found = find_path(parent, end)
            break 

        # 探索邻居 
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = now[0] + dr, now[1] + dc
            nxt = (nr, nc)

            # 检查边界、是否是墙、是否已访问
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] != 1 and nxt not in visited:
                visited.add(nxt)
                parent[nxt] = now
                queue.append(nxt) # 将有效邻居加入队列

    return path_found, visited_order

# 可视化
def visualize(maze, path, visited_order, title="BFS"):
    fig, ax = plt.subplots(figsize=(len(maze[0]) * 0.7, len(maze) * 0.7))
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'brown', 'lightblue', 'yellow', 'purple', 'orange'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(maze, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks(np.arange(-.5, len(maze[0]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(maze), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)

    visited_scatter = ax.scatter([], [], s=30, color='blue', alpha=0.6, label='Visited')
    # 如果路径为空，创建一个空的plot对象，避免legend出错
    if path:
        path_line, = ax.plot([], [], marker='o', markersize=5, color='red', linewidth=2, label='Final Path')
    else:
        path_line, = ax.plot([], [], label='Final Path') # 空plot

    total_frames = len(visited_order) + (len(path) if path else 0)

    def update(frame):
        if frame < len(visited_order):
            current_visited = visited_order[:frame+1]
            vy, vx = zip(*current_visited)
            visited_scatter.set_offsets(np.column_stack([vx, vy]))
        else:
            if frame == len(visited_order) and visited_order:
                 vy, vx = zip(*visited_order)
                 visited_scatter.set_offsets(np.column_stack([vx, vy]))

            if path: # 只有找到路径才绘制
                path_frame = frame - len(visited_order)
                if path_frame < len(path):
                    current_path_segment = path[:path_frame+1]
                    py, px = zip(*current_path_segment)
                    path_line.set_data(px, py)

        return visited_scatter, path_line

    ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=True, repeat=False)
    ax.legend()
    plt.title(f"{title} Visualization")
    plt.show()

if __name__ == "__main__":
    lines = sys.stdin.read().splitlines()
    n, m = map(int, lines[0].split())
    maze_data = []
    for i in range(1, n + 1):
        maze_data.append(list(map(int, lines[i].split())))

    start_node = (0, 0)
    end_node = (n - 1, m - 1)
    portal_locations = find_portals(maze_data)

    final_path, visited_nodes = BFS(maze_data, start_node, end_node, portal_locations)

    if final_path:
        path_len = len(final_path) - 1
        path_cost = calculate_cost(maze_data, final_path, portal_locations)
        print(f"BFS - 路径找到!")
        print(f"步数: {path_len}")
        print(f"路径总代价: {path_cost:.4f}")
        visualize(maze_data, final_path, visited_nodes, title="BFS")
    else:
        print("BFS - 未找到路径")
        visualize(maze_data, [], visited_nodes, title="BFS (No Path Found)")