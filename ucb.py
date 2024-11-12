import math


def get_ucb(R, N, c):
    # 总的选择次数
    total_plays = sum(N)

    # 计算每个臂的UCB值
    ucb_values = []
    for r, n in zip(R, N):
        if n == 0:
            # 如果某个臂还没有被选择过，则赋予它无限大的UCB值
            ucb_value = float('inf')
        else:
            # 计算UCB值
            ucb_value = r + c * math.sqrt(math.log(total_plays) / n)
        ucb_values.append(ucb_value)

    return ucb_values


def select_clients(ucb_values,a):
    # 对UCB值进行排序，并获取索引
    sorted_indices = sorted(range(len(ucb_values)), key=lambda k: ucb_values[k])
    # 选择前a个最小的UCB值的索引
    selected_indices = sorted_indices[:a]
    return selected_indices
