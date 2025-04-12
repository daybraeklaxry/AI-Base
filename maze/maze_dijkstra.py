# maze_dijkstra.py
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

def dijkstra_solve(maze, start, end, portal_map):
    rows, cols = len(maze), len(maze[0])
    # 优先队列，存储 (当前总代价, 当前节点)
    pq = [(0, start)]
    # 记录到各点的最低代价，初始化起点为0，其余为无穷大
    dist = {start: 0}
    parent = {}
    visited_order = []
    processed = set() # 记录已处理过的节点，避免重复处理

    path_found = None
    final_cost = float('inf')

    while pq:
        now_cost, now = heapq.heappop(pq) # 取出代价最小的节点

        # 如果当前取出的成本比已记录的成本高，说明是旧的、较差的路径，跳过
        if now_cost > dist.get(now, float('inf')):
            continue

        # 如果节点已处理过，跳过（Dijkstra保证第一次处理时成本最低）
        if now in processed:
            continue
        processed.add(now)
        visited_order.append(now) # 记录访问顺序（实际处理顺序）

        # 检查是否是传送门入口
        if now in portal_map:
            teleport_dest = portal_map[now]
            portal_cost = 0 # 传送代价为0
            new_cost = now_cost + portal_cost
            # 如果通过传送门到达目标的代价更低
            if new_cost < dist.get(teleport_dest, float('inf')):
                dist[teleport_dest] = new_cost
                parent[teleport_dest] = now # 父节点是传送门入口
                heapq.heappush(pq, (new_cost, teleport_dest))
            # 传送后，不立即探索传送目标的邻居，让它在优先队列中按成本排序

        if now == end:
            path_found = find_path(parent, end)
            final_cost = now_cost # 记录最终成本
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
                new_cost = now_cost + step_cost

                # 如果找到更短的路径到达 nxt
                if new_cost < dist.get(nxt, float('inf')):
                    dist[nxt] = new_cost
                    parent[nxt] = now
                    heapq.heappush(pq, (new_cost, nxt)) # 将新代价和邻居加入优先队列

    return path_found, visited_order, final_cost

def visualize(maze, path, visited_order, title="Dijkstra"):
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

    final_path, visited_nodes, path_cost = dijkstra_solve(maze_data, start_node, end_node, portal_locations)

    if final_path:
        path_len = len(final_path) - 1
        print(f"Dijkstra - 路径找到!")
        print(f"步数: {path_len}")
        print(f"路径总代价: {path_cost:.4f}") 
        visualize(maze_data, final_path, visited_nodes, title="Dijkstra")
    else:
        print("Dijkstra - 未找到路径")
        visualize(maze_data, [], visited_nodes, title="Dijkstra (No Path Found)")