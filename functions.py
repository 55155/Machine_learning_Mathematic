import numpy as np
from numpy import ndarray

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
    
    return (func(input + delta) - func(input - delta)) / 2*delta

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