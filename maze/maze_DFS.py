# maze_DFS.py
import sys
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# 回溯路径
def find_path(parent, end_node):
    path = [end_node]
    while end_node in parent:
        end_node = parent[end_node]
        path.append(end_node)
    return path[::-1]

# 查找传送门对
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
        if len(locations) == 2: # 确保传送门成对出现
            p1, p2 = locations[0], locations[1]
            portal_pairs[p1] = p2
            portal_pairs[p2] = p1
    return portal_pairs

# 计算路径总代价
def calculate_cost(maze, path, portal_map):
    total_cost = 0
    for i in range(len(path) - 1):
        curr = path[i]
        nxt = path[i+1]

        # 检查是否通过传送门移动
        if curr in portal_map and portal_map[curr] == nxt:
            total_cost += 0 # 传送代价为0
            continue

        # 计算基础移动代价
        dr, dc = nxt[0] - curr[0], nxt[1] - curr[1]
        base_cost = math.sqrt(2) if dr != 0 and dc != 0 else 1

        # 获取目标单元格类型并计算修正因子
        modifier = 1
        cell_type = maze[nxt[0]][nxt[1]]
        if cell_type == 2: # 沼泽
            modifier = 2.0
        elif cell_type == 3: # 加速器
            modifier = 0.5

        total_cost += base_cost * modifier
    return total_cost

def dfs(maze, start, end, portal_map):
    rows, cols = len(maze), len(maze[0])
    visited = set()
    parent = {}
    path_found = None
    visited_order = []

    stack = [(start, [start])] # (当前节点, 到当前节点的路径) 

    while stack:
        (now, current_path) = stack.pop()

        if now in visited: # 如果已经访问过，跳过
            continue
        visited.add(now)
        visited_order.append(now)

        # 检查是否是传送门入口
        if now in portal_map:
            teleport_dest = portal_map[now]
            if teleport_dest not in visited:
                 # 记录父节点，用于回溯
                parent[teleport_dest] = now # 父节点设为传送门入口
                # 将传送目标加入栈，路径继承，但此步代价为0 
                new_path = current_path + [teleport_dest]
                stack.append((teleport_dest, new_path))
                continue 

        if now == end:
            path_found = current_path
            break # 找到路径

        # 探索邻居
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = now[0] + dr, now[1] + dc
            nxt = (nr, nc)

            # 检查边界、是否是墙、是否已访问
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] != 1 and nxt not in visited:
                parent[nxt] = now
                new_path = current_path + [nxt]
                stack.append((nxt, new_path)) # 加入栈进行后续探索

    return path_found, visited_order # 返回路径和访问顺序

# 可视化 
def visualize(maze, path, visited_order, title="DFS"):
    fig, ax = plt.subplots(figsize=(len(maze[0]) * 0.7, len(maze) * 0.7)) # 设置图形大小
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'brown', 'lightblue', 'yellow', 'purple', 'orange']) # 定义颜色映射: 0:白, 1:黑, 2:棕(沼泽), 3:浅蓝(加速), 4+:传送门
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5] # 颜色边界 (可根据最大传送门ID调整)
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(maze, cmap=cmap, norm=norm, interpolation='nearest') # 显示迷宫，使用定义的颜色

    # 绘制网格线
    ax.set_xticks(np.arange(-.5, len(maze[0]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(maze), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0) # 隐藏次刻度线
    ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False) # 隐藏主刻度线和标签

    # 初始化访问过的节点（蓝色散点）和最终路径（红色线条）
    visited_scatter = ax.scatter([], [], s=30, color='blue', alpha=0.6, label='Visited')
    # 如果路径为空，创建一个空的plot对象，避免legend出错
    if path:
        path_line, = ax.plot([], [], marker='o', markersize=5, color='red', linewidth=2, label='Final Path')
    else:
        path_line, = ax.plot([], [], label='Final Path') # 空plot

    # 计算总帧数 = 访问顺序帧数 + 路径绘制帧数
    total_frames = len(visited_order) + len(path)

    def update(frame):
        # 更新访问过的节点 
        if frame < len(visited_order):
            # 获取当前帧需要显示的访问节点
            current_visited = visited_order[:frame+1]
            # 提取x, y坐标 (注意 matplotlib 的 scatter 和 plot 使用 (x, y) 即 (列, 行))
            vy, vx = zip(*current_visited)
            # 更新散点图数据
            visited_scatter.set_offsets(np.column_stack([vx, vy]))
        # 访问节点显示完毕后，开始绘制路径 
        else:
            # 确保所有访问过的节点都已显示
            if frame == len(visited_order):
                 vy, vx = zip(*visited_order)
                 visited_scatter.set_offsets(np.column_stack([vx, vy]))

            # 计算当前在路径绘制阶段的帧数
            path_frame = frame - len(visited_order)
            if path_frame < len(path):
                # 获取当前帧需要显示的路径节点
                current_path_segment = path[:path_frame+1]
                # 提取x, y坐标
                py, px = zip(*current_path_segment)
                # 更新路径线条数据
                path_line.set_data(px, py)

        # 返回需要更新的图形元素元组 (对于 blitting)
        return visited_scatter, path_line

    # 创建动画
    ani = FuncAnimation(fig, update, frames=total_frames,
                        interval=100, # 每帧之间的延迟（毫秒）
                        blit=True,    # 使用 blitting 优化绘图速度
                        repeat=False) # 动画不重复播放

    ax.legend() # 显示图例
    plt.title(f"{title} Visualization") # 设置标题
    plt.show()  # 显示动画窗口

if __name__ == "__main__":
    lines = sys.stdin.read().splitlines()
    n, m = map(int, lines[0].split())
    maze_data = []
    for i in range(1, n + 1):
        maze_data.append(list(map(int, lines[i].split())))

    start_node = (0, 0)
    end_node = (n - 1, m - 1)

    portal_locations = find_portals(maze_data)

    final_path, visited_nodes = dfs(maze_data, start_node, end_node, portal_locations)

    if final_path:
        path_len = len(final_path) - 1
        path_cost = calculate_cost(maze_data, final_path, portal_locations)
        print(f"DFS - 路径找到!")
        print(f"步数: {path_len}")
        print(f"路径总代价: {path_cost:.4f}")
        # 可视化
        visualize(maze_data, final_path, visited_nodes, title="DFS")
    else:
        print("DFS - 未找到路径")
        visualize(maze_data, [], visited_nodes, title="DFS (No Path Found)")