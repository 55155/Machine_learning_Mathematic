'''
player, case = map(int,input().split()) # 

l = [[-1, -1] for i in range(player+1)]

print(l)
for i in range(case):
    time, p, s = map(int, input().split())
    if l[p][s] != -1 and s == 0:
        print("No")
    # 100 1 0
    # 110 1 0
    l[p][s] = time #[[],[],[]]

for i in l:
    start, end = i
    if start == -1 or end == -1 or (end-start) < 60:
        print("NO")
    else:
        print("Yes")
'''
'''
몇개의 인덱스가 잘못된지 제시하면된다
BBB 011 
BBW 111
BWB 000
'''

def