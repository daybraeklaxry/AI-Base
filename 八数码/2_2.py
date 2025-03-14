from collections import deque
import sys

def main():
    s = sys.stdin.readline().strip().split()
    start = ''.join(s)
    target = '12345678x' # 目标状态
    if start == target: # 特殊情况处理
        print(0)
        return
    q = deque()
    q.append( (start, 0) )
    visited = set()
    visited.add(start)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)] # 定义方向
    while q:
        state, steps = q.popleft()
        if state == target: # 找到目标立即返回步数
            print(steps)
            return
        idx = state.index('x')
        i, j = divmod(idx, 3) # 转换为二维坐标
        for di, dj in dirs:
            ni, nj = i+di, j+dj
            if 0<=ni<3 and 0<=nj<3: # 边界检查
                new_idx = ni*3 + nj
                lst = list(state)
                lst[idx], lst[new_idx] = lst[new_idx], lst[idx]
                new_state = ''.join(lst)
                if new_state not in visited: # 如果未被访问过加入队列
                    visited.add(new_state)
                    q.append( (new_state, steps+1) )
    print(-1)

if __name__ == "__main__":
    main()