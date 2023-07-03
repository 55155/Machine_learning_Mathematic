import numpy as np

# 시도 횟수
n = 1

# 성공 확률
p = 0.8

# 랜덤 변수의 개수
size = 5

# 이항 분포에서 랜덤 변수 생성
random_variables = np.random.binomial(n, p, size)

print(random_variables)