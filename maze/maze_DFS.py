import matplotlib.pyplot as plt
import sys

# 回溯路径
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

    while True:
        visited = set()
        lst = {}
        stack = [(start, 0)]  
        found = False

        while stack:
            now, depth = stack.pop()
            if now == end:
                found = True
                break
            if depth < max_depth:
                visited.add(now)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    x = now[0] + dx
                    y = now[1] + dy
                    nxt = (x, y)
                    if 0 <= x < rows and 0 <= y < cols and maze[x][y] == 0 and nxt not in visited:
                        lst[nxt] = now
                        stack.append((nxt, depth + 1))
        if found:
            path = find_the_path(lst, end)
            return path, visited.union({now for now, _ in stack})
        max_depth += 1

def visualize_maze_with_path(maze, path, visited=None):
    plt.figure(figsize=(len(maze[0]), len(maze))) # 设置图形大小
    plt.imshow(maze, cmap='Greys', interpolation='nearest') # 使用灰度色图，并关闭插值

    # 绘制所有访问过的格子
    if visited is not None:
        visited_x, visited_y = zip(*visited)
        plt.scatter(visited_y, visited_x, s=10, color='blue', alpha=0.5)

    # 绘制路径
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', markersize=8, color='red', linewidth=3)

    # 设置坐标轴刻度和边框
    plt.xticks(range(len(maze[0])))
    plt.yticks(range(len(maze)))
    plt.gca().set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=2)

    plt.axis('on') # 显示坐标轴
    plt.show()


input = sys.stdin.read().split()
idx = 0
n = int(input[idx])
idx +=1
m = int(input[idx])
idx +=1
    
maze = []
for _ in range(n):
    row = list(map(int, input[idx:idx+m]))
    maze.append(row)
    idx += m
path, visited = dfs(maze)
print(len(path) - 1)
visualize_maze_with_path(maze, path, visited)