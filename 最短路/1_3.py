import sys
import heapq

def main():
    n, m = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(n+1)]
    for _ in range(m):
        a, b, w = map(int, sys.stdin.readline().split())
        adj[a].append( (b, w) )
    INF = float('inf')
    dist = [INF]*(n+1)
    dist[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1)) # 将距离为0的1号点压入堆中
    while heap:
        d, u = heapq.heappop(heap) # 从堆当中找出距离最小的点
        if d > dist[u]: # 跳过过时记录
            continue
        # 松弛操作
        for v, w in adj[u]:
            if dist[v] > d + w:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    print(dist[n] if dist[n] != INF else -1)

if __name__ == "__main__":
    main()