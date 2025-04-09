from collections import deque
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation
import numpy as np

# 回溯路径
def find_the_path(lst, now):
    total_path = [now]
    while now in lst:
        now = lst[now]
        total_path.append(now)
    return total_path[::-1]

# BFS算法，支持斜向走
def bfs(maze):
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    start = (0, 0)
    end = (rows - 1, cols - 1)

    queue = deque()
    queue.append(start)
    visited = set()
    visited.add(start)
    visited_order = []
    lst = {}

    while queue:
        now = queue.popleft()
        visited_order.append(now)  
        if now == end:
            return find_the_path(lst, now), visited_order

        # 斜对角走法：除了上下左右，还可以走四个斜对角方向
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            x = now[0] + dx
            y = now[1] + dy
            nxt = (x, y)

            # 确保节点在迷宫范围内并且是可行走的
            if 0 <= x < rows and 0 <= y < cols and maze[x][y] == 0:
                if nxt not in visited:
                    lst[nxt] = now
                    visited.add(nxt)
                    queue.append(nxt)

# 可视化迷宫和路径
def visualize_maze_with_path(maze, path, visited_order):
    fig, ax = plt.subplots(figsize=(len(maze[0]), len(maze))) # 设置图形大小
    ax.imshow(maze, cmap='Greys', interpolation='nearest')  # 使用灰度色图，并关闭插值
    
    # 设置坐标轴刻度和边框
    ax.set_xticks(range(len(maze[0])))
    ax.set_yticks(range(len(maze)))
    ax.set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    
    # 初始化空的散点图和线图
    scatter = ax.scatter([], [], s=10, color='blue', alpha=0.5)
    line, = ax.plot([], [], marker='o', markersize=8, color='red', linewidth=3)
    
    # 动画更新函数
    def update(frame):
        # 显示已访问的节点
        if frame < len(visited_order):
            visited_x, visited_y = zip(*visited_order[:frame+1])
            scatter.set_offsets(np.column_stack([visited_y, visited_x]))
        
        # 显示路径
        if frame >= len(visited_order):
            path_frame = frame - len(visited_order)
            if path_frame < len(path):
                path_x, path_y = zip(*path[:path_frame+1])
                line.set_data(path_y, path_x)
        
        return scatter, line
    
    # 计算总帧数（访问过程+路径绘制）
    total_frames = len(visited_order) + len(path)
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=total_frames, interval=1000, blit=True, repeat=False)
    plt.show()

# 读取输入
input = sys.stdin.read().split()
idx = 0
n = int(input[idx])  
idx += 1
m = int(input[idx])  
idx += 1
    
maze = []
for _ in range(n):
    row = list(map(int, input[idx:idx+m]))
    maze.append(row)
    idx += m

path, visited_order = bfs(maze)

print(f"路径长度：{len(path) - 1}")

total_distance = sum(np.sqrt(2) if (path[i][0] != path[i+1][0] and path[i][1] != path[i+1][1]) else 1 for i in range(len(path)-1))

print(f"实际距离（路径总代价）：{total_distance}")

visualize_maze_with_path(maze, path, visited_order)
