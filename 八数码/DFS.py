import sys

def get_next_states(state):
    lst = list(state)
    idx = lst.index(0)
    row = idx // 3
    col = idx % 3
    next_states = []
    dire = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向
    for dr, dc in dire:
        new_row = row + dr
        new_col = col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row * 3 + new_col
            new_lst = lst.copy()
            new_lst[idx], new_lst[new_idx] = new_lst[new_idx], new_lst[idx]
            next_states.append(tuple(new_lst))
    return next_states

def is_solvable(init):
    st = []
    vis = set()
    target = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    st.append(init)
    vis.add(init)

    while st:
        current = st.pop()
        if current == target:
            return 1
        for next_state in get_next_states(current):
            if next_state not in vis:
                vis.add(next_state)
                st.append(next_state)
    return 0

def main():
    s = sys.stdin.readline().strip().split()
    grid = []
    for c in s:
        if c == 'x':
            grid.append(0)
        else:
            grid.append(int(c))
    init = tuple(grid)
    print(is_solvable(init))

if __name__ == "__main__":
    main()