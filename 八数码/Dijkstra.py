import sys
import heapq


def main():
    s = sys.stdin.readline().strip().split()
    start = ''.join(s)
    target = '12345678x'

    if start == target:
        print(0)
        return

    # 优先队列初始化
    heap = []
    heapq.heappush(heap, (0, start))

    dist = {start: 0}

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右移动方向

    while heap:
        steps, state = heapq.heappop(heap)

        # 如果当前状态是目标状态，直接返回步数
        if state == target:
            print(steps)
            return

        # 如果当前步数大于已记录的最小步数，跳过
        if steps > dist.get(state, float('inf')):
            continue

        idx = state.index('x')
        i, j = divmod(idx, 3)

        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < 3 and 0 <= nj < 3:
                new_idx = ni * 3 + nj

                lst = list(state)
                lst[idx], lst[new_idx] = lst[new_idx], lst[idx]
                new_state = ''.join(lst)
                new_steps = steps + 1

                if new_state not in dist or new_steps < dist.get(new_state, float('inf')): # 如果不在或者小于在队列里面的值
                    dist[new_state] = new_steps
                    heapq.heappush(heap, (new_steps, new_state))
    # 如果无法到达目标状态
    print(-1)


if __name__ == "__main__":
    main()