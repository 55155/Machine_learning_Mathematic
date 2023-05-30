import numpy as np
from numpy import ndarray

X = np.array([1,2,3,4,5])
Y = np.array([2,4,6,8,10])

def square(x:ndarray)-> ndarray: # type signiture
    '''
    인자로 받은 ndarray 배열의 각 요솟값을 제곱한다.
    '''
    return np.power(x,2)

def leaky_relu(x:ndarray)->ndarray:
    '''
    ndarray 배열의 각 요소에 'Leaky relu' 함수를 적용한다.
    '''
    return np.maximum(0.2 * x, x)

def relu(x:ndarray)->ndarray:
    return np.maximum(0, x)

# 도함수의 기하학적 정의
from typing import Callable
def deriv(func:Callable[[ndarray], ndarray],
        input_ : ndarray, delta: float = 0.001)-> ndarray:
    
    return (func(input_ + delta) - func(input_ - delta)) / 2*delta

# 합성합수
from typing import List
# 데이터 타입 정의
Array_Function = Callable[[ndarray], ndarray] # 인자는 ndarray, 반환값도 ndarray
Chain = List[Array_Function] # Chain은 함수의 리스트이다.

def chain_length_2(chain:Chain, a:ndarray)-> ndarray:
    assert len(chain) == 2, \
    "인자 chain의 길이는 2여야함"
    
    
    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(a))

# 연쇄 법칙
def sigmoid(x:ndarray)->ndarray:

    return 1/(1+np.exp(-x))

def chain_deriv_2(chain:Chain, input_range:ndarray) -> ndarray:
    assert len(chain) == 2, \
        "인자 chain의 길이는 2여야함"
    
    assert input_range.ndim == 1, \
        "input_range는 1차원이여아 한다."

    f1 = chain[0]
    f2 = chain[1]

    # df1 / dx
    f1_of_x = f1(input_range)

    # df1 / du
    df1dx = deriv(f1, input_range)

    # df2 / du(f1(x))
    df2du = deriv(f2, f1(input_range))

    # 각점끼리 값을 곱함
    return df1dx * df2du

# 세 합성함수    
def chain_deriv_3(chain:Chain, input_range:ndarray)->ndarray:

    assert len(chain) == 3, \
    "인자 chain의 길이는 3이어야 함"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    # df1 / dx
    f1_of_x = f1(input_range)

    # df1 / du
    df1dx = deriv(f1, input_range)

    # df2 / du(f1(x))
    df2du = deriv(f2, f1(input_range))

    # df3du

    df3du = deriv(f3, f2(input_range))

    # 각점끼리 값을 곱함
    return df3du * df1dx * df2du

# x,y 인 합성함수

def multiple_inputs_add(x:ndarray, y:ndarray, sigma:Array_Function)->ndarray:

    assert x.shape == y.shape
    a = x+y
    return sigma(a)

a = np.array([1,2,3,4,5])
b = np.array([2,4,6,8,10])

c = multiple_inputs_add(a,b,relu)
print(type(c))

# x,y를 input으로 받는 합성함수

def multiple_inputs_add_backward(x:ndarray, y:ndarray, sigma:Array_Function)->float:

    a = x+y
    dsda = deriv(sigma, a)

    dadx, dady = 1,1
    return dsda * dadx, dsda * dady

# dot product 행렬연산으로 찾기
def matmul_forward(X:ndarray, W:ndarray)->ndarray:

    assert X.shape[1] == W.shape[0],\
        ''.format(X.shape[1], W.shape[0])
    
    N = np.dot(X,W)

    return N

# 벡터입력의 도함수화 : 데이터가 함수형태로 주어지지 않고 행렬형태로 주겅지기 때문
# 이론 : x = [x1, x2, x3] 로 이루어져 있을 때
def matmul_backward_first(X:ndarray, W:ndarray) -> ndarray:
    dNdX = np.transpose(W,(1,0))
    return dNdX

# 벡터 함수와 도함수 : forward
def matrix_forward_extra(X:ndarray, W:ndarray, sigma:Array_Function)->ndarray:

    assert X.shape[1] == W.shape[0]

    # 행렬곱
    N = np.dot(X,W)

    # 행렬곱의 출력을 함수 sigma의 입력값으로 전달
    S = sigma(N)

    return S


# 벡터 함수와 도함수 : backward
def matrix_function_backward(X:ndarray, W:ndarray, sigma:Array_Function)->ndarray:
    assert X.shape[1] == W.shape[0]

    # 행렬곱
    N = np.dot(X,W)

    # 행렬곱의 출력을 함수 sigma의 입력값으로 전달
    S = sigma(N)

    # 함수에서 -> 도함수로 변환
    dSdN = deriv(sigma , N)

    # 벡터입력의 도함수화
    dNdX = np.transpose(W,(1,0))
    
    # 계산값을 모두 곱함. 여기서는 dNdX의 모양이 1*1이므로 순서와는 무관함
    return np.dot(dSdN, dNdX)

X = np.array([[0.4723, 0.6151, -1.7162]])
print(X.shape)
W = np.array([[100,100,100]]).transpose(1,0)
a = matrix_function_backward(X, W, sigmoid)
print(a)
1.80829203e-32
def matrix_function_forward_sum(X:ndarray, W:ndarray, sigma:Array_Function)->float:
    assert X.shape[0] == W.shape[1]

    # 행렬곱
    N = np.dot(X,W)

    # 함수 계산
    S = sigma(N)

    # sum
    L = np.sum(S)

    return L

def matrix_function_backward_sum_1(X:ndarray, W:ndarray, sigma:Array_Function)->float:
    assert X.shape[0] == W.shape[1]

    # 행렬곱
    N = np.dot(X,W)

    # 함수 계산
    S = sigma(N)

    # sum
    L = np.sum(S)

    dLdS = np.ones_like(S)

    dSdN = deriv(sigma, N) # sigma에 대한 미분

    dNdX = np.transpose(W, (1,0))
    
    dLdN = dLdS * dSdN

    dLdX = np.dot(dSdN, dNdX)

    return dLdX

