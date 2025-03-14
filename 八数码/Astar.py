import sys
import heapq

# 计算曼哈顿距离
def manhattan(s):
    target = '12345678x'
    pos = {c:i for i,c in enumerate(target)} # 映射每个字母对应的位置
    res = 0
    for i in range(9):
        c = s[i]
        if c == 'x': continue
        t = pos[c] # 目标索引
        a, b = divmod(i,3) # 当前坐标
        ta, tb = divmod(t,3) # 目标坐标
        res += abs(a-ta) + abs(b-tb)
    return res

def main():
    s = sys.stdin.readline().strip().split()
    start = ''.join(s) # 形成初始状态的字符串
    target = '12345678x'
    if start == target: # 特判
        print("")
        return
    heap = []
    heapq.heappush(heap, (manhattan(start), 0, start, ""))
    visited = {} # 记录g值
    while heap:
        f, g, state, path = heapq.heappop(heap) # 弹出估计值最小的状态
        if state == target:
            print(path)
            return
        if state in visited and visited[state] <= g:
            continue
        visited[state] = g
        idx = state.index('x')
        i, j = divmod(idx,3)
        dirs = [(-1,0,'u'), (1,0,'d'), (0,-1,'l'), (0,1,'r')] # 对应每一个方向
        for di, dj, d in dirs:
            ni, nj = i+di, j+dj
            if 0<=ni<3 and 0<=nj<3:
                new_idx = ni*3 + nj
                lst = list(state)
                lst[idx], lst[new_idx] = lst[new_idx], lst[idx]
                new_state = ''.join(lst)
                new_g = g + 1 # 实际步数+1
                new_f = new_g + manhattan(new_state) # 新的估计值
                if new_state not in visited or visited.get(new_state, float('inf')) > new_g:
                    heapq.heappush(heap, (new_f, new_g, new_state, path + d))
    print("unsolvable")

if __name__ == "__main__":
    main()