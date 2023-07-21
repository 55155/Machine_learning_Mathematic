'''
player, case = map(int,input().split())

l = [[-1, -1] for i in range(player+1)]


for i in range(case):
    time, p, s = map(int, input().split())
    l[p][s] = time

for i in l:
    start, end = i
    if start == -1:
        print("NO")
    if end == -1:
        print("NO")
    if (end - start) < 60:
        print("NO")
'''

# a1, a2, a3, a4, a5, a6
# s1, s2, s3, s4, s5, s6

# s2 - s1 = a1
# s5 - s1 = a2 + a3 + a4 + a5 = 0 --> 두수로 나누었을 때 0이 나온다는 뜻

# 1,000,000

N, case = map(int, input().split())
board = []

# rowcol[i][0][]
# 00 : start x
# 01 : start y
# 10 : end x
# 11 : end y

for i in range(N):
    temp = list(map(int,input().split()))
    board.append(temp)


Prefix_sum = [[0] * (N+1) for _ in range(N+1)]
for row in range(1, N+1):
    # ex) 5 5 -> 6,6
    Prefix_sum[row][0] = Prefix_sum[row-1][-1]
    # [row][0] = [row - 1][-1]
    for col in range(1, N+1):
        Prefix_sum[row][col] = Prefix_sum[row][col - 1] + board[row-1][col-1]

result =[]
for i in range(case):
    start_row, start_col, end_row, end_col = map(int, input().split())
    temp = Prefix_sum[end_row][end_col] - Prefix_sum[start_row][start_col-1]
    # start_index 포함한다. 
    result.append(temp)

for i in result:
    print(i)