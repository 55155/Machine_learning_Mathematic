# 1. 순방향 계산 절차
#     1) 입력에 패딩을 붙인다. 
#     2) 패딩된 입력과 파라미터로부터 출력을 계산한다.
# 2. 역방향 계산에서 입력 기울기를 계산하는 절차
#     1) 출력 기울기에 패딩을 덧붙인다. 
#     2) 패딩된 출력 기울기와 입력, 파라미터로부터 입력 기울기와 파라미터 기울기를 계산한다. 
# 3. 역방향 계산에서 파라미터 기울기를 계산하는 절차
#     1) 입력에 패딩을 덧붙인다.
#     2) 패딩된 입력의 각 요소를 순회하며 파라미터 기울기를 더한다. 

