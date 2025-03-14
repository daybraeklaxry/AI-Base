import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(n+1)]
    for _ in range(m):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b) # 有向边
    dist = [-1]*(n+1) # 初始值设为-1
    q = deque()
    dist[1] = 0
    q.append(1)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u]+1
                q.append(v)
    print(dist[n])

if __name__ == "__main__":
    main()