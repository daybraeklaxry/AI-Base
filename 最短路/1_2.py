import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(n+1)]
    for _ in range(m):
        a, b, w = map(int, sys.stdin.readline().split())
        adj[a].append( (b, w) )
    INF = float('inf')
    dist = [INF]*(n+1) # 初始化为无限大
    dist[1] = 0
    visited = [False]*(n+1)
    for _ in range(n):
        u, min_d = -1, INF
        # 找当前未访问的节点里面距离最小的节点u
        for i in range(1, n+1):
            if not visited[i] and dist[i] < min_d:
                u, min_d = i, dist[i]
        if u == -1: break
        visited[u] = True
        # 松弛操作
        for v, w in adj[u]:
            if dist[v] > dist[u]+w:
                dist[v] = dist[u]+w
    print(dist[n] if dist[n] != INF else -1)

if __name__ == "__main__":
    main()