# maze_A_star.py
import sys
import math
import heapq
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

# 启发式函数 Chebyshev距离  
def heuristic(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def astar_solve(maze, start, end, portal_map):
    rows, cols = len(maze), len(maze[0])
    # 优先队列，存储 (f_score, 当前节点)
    # f_score = g_score + h_score
    pq = [(heuristic(start, end), start)] # 初始节点的f_score = 0 + h(start, end)
    # g_score: 从起点到当前节点的实际最低代价
    g_score = {start: 0}
    parent = {}
    visited_order = []
    processed = set() # 记录已处理过的节点

    path_found = None
    final_cost = float('inf')

    while pq:
        # 注意：A* 取出的是 f_score 最小的，但后续比较和更新用 g_score
        f_now, now = heapq.heappop(pq)

        # 如果节点已被处理过，跳过
        if now in processed:
            continue
        processed.add(now)
        visited_order.append(now) # 记录访问顺序（实际处理顺序）

        # 如果取出的节点的 f_score 比基于当前已知最优 g_score 计算的 f_score 要差，跳过
        # (这可以处理同一个节点以不同 f_score 多次入队的情况)
        current_g = g_score.get(now, float('inf'))
        if f_now > current_g + heuristic(now, end):
             continue

        # 检查是否是传送门入口
        if now in portal_map:
            teleport_dest = portal_map[now]
            portal_cost = 0 # 传送代价为0
            new_g = current_g + portal_cost # g_score 更新

            # 如果通过传送门到达目标的代价更低
            if new_g < g_score.get(teleport_dest, float('inf')):
                g_score[teleport_dest] = new_g
                parent[teleport_dest] = now # 父节点是传送门入口
                f_new = new_g + heuristic(teleport_dest, end) # 计算新的 f_score
                heapq.heappush(pq, (f_new, teleport_dest))
            # 传送后，不立即探索传送目标的邻居，让它在优先队列中按 f_score 排序

        if now == end:
            path_found = find_path(parent, end)
            final_cost = current_g # 记录最终成本 (g_score)
            break # 找到终点，结束搜索

        # 探索邻居 (包括斜向)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = now[0] + dr, now[1] + dc
            nxt = (nr, nc)

            # 检查边界和是否是墙
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] != 1:
                # 计算基础移动代价
                base_cost = math.sqrt(2) if dr != 0 and dc != 0 else 1
                # 获取目标单元格类型并计算修正因子
                modifier = 1
                cell_type = maze[nr][nc]
                if cell_type == 2: modifier = 2.0 # 沼泽
                elif cell_type == 3: modifier = 0.5 # 加速器

                step_cost = base_cost * modifier
                new_g = current_g + step_cost # 新的 g_score

                # 如果找到更短的路径到达 nxt
                if new_g < g_score.get(nxt, float('inf')):
                    g_score[nxt] = new_g
                    parent[nxt] = now
                    f_new = new_g + heuristic(nxt, end) # 计算新的 f_score
                    heapq.heappush(pq, (f_new, nxt)) # 将新 f_score 和邻居加入优先队列

    return path_found, visited_order, final_cost

def visualize(maze, path, visited_order, title="A*"):
    fig, ax = plt.subplots(figsize=(len(maze[0]) * 0.7, len(maze) * 0.7))
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'brown', 'lightblue', 'yellow', 'purple', 'orange'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5] # 可根据最大传送门ID调整
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(maze, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks(np.arange(-.5, len(maze[0]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(maze), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)

    visited_scatter = ax.scatter([], [], s=30, color='blue', alpha=0.6, label='Visited (Processed)')
    if path:
        path_line, = ax.plot([], [], marker='o', markersize=5, color='red', linewidth=2, label='Final Path')
    else:
        path_line, = ax.plot([], [], label='Final Path')

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

            if path:
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

    final_path, visited_nodes, path_cost = astar_solve(maze_data, start_node, end_node, portal_locations)

    if final_path:
        path_len = len(final_path) - 1
        print(f"A* - 路径找到!")
        print(f"步数: {path_len}")
        print(f"路径总代价: {path_cost:.4f}") 
        visualize(maze_data, final_path, visited_nodes, title="A*")
    else:
        print("A* - 未找到路径")
        visualize(maze_data, [], visited_nodes, title="A* (No Path Found)")