import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation
import numpy as np

def find_the_path(lst, now):
    total_path = [now]
    while now in lst:
        now = lst[now]
        total_path.append(now)
    return total_path[::-1]

def dfs(maze):
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    start = (0, 0)
    end = (rows - 1, cols - 1)
    max_depth = 0

    iterations = []
    final_path = None

    while True:
        visited = set()
        lst = {}
        stack = [(start, 0)]
        found = False
        
        # 记录本次迭代的访问顺序
        current_visited_order = []
        
        while stack:
            now, depth = stack.pop()
            current_visited_order.append(now)
            
            if now == end:
                final_path = find_the_path(lst, end)
                found = True
                break
                
            if depth < max_depth:
                visited.add(now)
                for dx, dy in [(-1, 0), (1, 0), (0, - 1), (0, 1)]:
                    x = now[0] + dx
                    y = now[1] + dy
                    nxt = (x, y)
                    if 0 <= x < rows and 0 <= y < cols and maze[x][y] == 0 and nxt not in visited:
                        lst[nxt] = now
                        stack.append((nxt, depth + 1))

        iterations.append({
            "visited_order": current_visited_order,
            "found": found,
            "path": final_path[:] if found else None
        })
        
        if found:
            break
            
        max_depth += 1

    return iterations

def visualize_maze_with_path(maze, iterations):
    fig, ax = plt.subplots(figsize=(len(maze[0]), len(maze)))
    ax.imshow(maze, cmap='Greys', interpolation='nearest')
    
    # 设置坐标轴
    ax.set_xticks(range(len(maze[0])))
    ax.set_yticks(range(len(maze)))
    ax.set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    
    # 初始化绘图元素
    scatter = ax.scatter([], [], s=10, color='blue', alpha=0.5)
    line, = ax.plot([], [], marker='o', markersize=8, color='red', linewidth=3)
    
    # 计算总帧数
    total_frames = sum(len(iter["visited_order"]) for iter in iterations)
    total_frames += len(iterations[-1]["path"])
    
    def update(frame):
        # 确定当前属于哪个阶段
        cum_frames = 0
        current_stage = 0
        path_stage = False
        
        for i, iter in enumerate(iterations):
            if frame < cum_frames + len(iter["visited_order"]):
                current_stage = i
                break
            cum_frames += len(iter["visited_order"])
        
        if frame >= total_frames - len(iterations[-1]["path"]):
            path_stage = True
            path_frame = frame - (total_frames - len(iterations[-1]["path"]))
        
        if not path_stage:
            current_iter = iterations[current_stage]
            frames_in_stage = frame - cum_frames
            visited_x, visited_y = zip(*current_iter["visited_order"][:frames_in_stage+1])
            scatter.set_offsets(np.column_stack([visited_y, visited_x]))
        else:
            final_iter = iterations[-1]
            all_visited = set()
            for iter in iterations:
                all_visited.update(iter["visited_order"])
            visited_x, visited_y = zip(*all_visited)
            scatter.set_offsets(np.column_stack([visited_y, visited_x]))
            
            if path_frame < len(final_iter["path"]):
                path_x, path_y = zip(*final_iter["path"][:path_frame+1])
                line.set_data(path_y, path_x)
        
        return scatter, line
    
    ani = FuncAnimation(fig, update, frames=total_frames, interval=500, blit=True, repeat=False)
    plt.show()

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

iterations = dfs(maze)
print(len(iterations[-1]['path']) - 1)

visualize_maze_with_path(maze, iterations)