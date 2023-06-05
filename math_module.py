import numpy as np
from numpy import ndarray
from typing import Callable,List
import matplotlib.pyplot as plt

# 자료형 정의 
Array_Function = Callable[[ndarray], ndarray] # 수학적인 함수 자료형 정의
# ndarray 인자값으로 받고 ndarray return 하는 함수 
Chain = List[Array_Function] # 함수의 연속 정의

def square(x:ndarray):
    return x**2

def relu(x:ndarray):
    return np.maximum(0, x)

def leaky_relu(x:ndarray):
    return np.maximum(0.1*x, x)

def sigmoid(x:ndarray)->ndarray:
    return 1 / (1 + np.exp(-x))

def quick_sort(array):
    # 리스트가 하나 이하의 원소를 가지면 종료
    if len(array) <= 1: return array
    
    # pivot은 첫번째 인덱스
    pivot, tail = array[0], array[1:]
    # leftside는 pivot보다 작은 값을 정렬 
    leftSide = [x for x in tail if x <= pivot]
    # rightside는 pivot보다 큰값을 정렬
    rightSide = [x for x in tail if x > pivot]
    
    # 분할 정복 알고리즘
    return quick_sort(leftSide) + [pivot] + quick_sort(rightSide)

def deriv(func:Callable[[ndarray], ndarray], input_:ndarray, delta:float = 0.001)->ndarray: # ndarray 형식으로 반환
    # 미분식을 return
    return (func(input_ + delta) - func(input_ - delta)) / (2*delta)


# 합성함수 정의
def chain_length_2(chain : Chain, x:ndarray):

    assert len(chain) == 2, \
    "len(chain) != 2"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))

def chain_length_3(chain:Chain, x:ndarray):
    assert len(chain) == 3,\
    "chain != 3"
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    return f3(f2(f1(x)))


# df2 / df1 * df1 / dx
def chain_deriv_2(chain : Chain, x: ndarray):
    assert len(chain) == 2, \
    "len(chain) != 2"
    
    assert x.ndim == 1, \
    "x는 1차원이어야한다."
    
    f1 = chain[0]
    f2 = chain[1]

    deriv_f1 = deriv(f1, x)
    deriv_f2 = deriv(f2, f1(x))

    return deriv_f2 * deriv_f1

def chain_deriv_3(chain:Chain, x:ndarray):
    # df3/df2  *  df2/df1 *  df1/dx
    assert len(chain) == 3, \
    "len(chain) != 3"
    assert x.ndim == 1, \
    "x.ndim != 1"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    f1_x = f1(x)
    f2_f1_x = f2(f1_x)

    df1_dx = deriv(f1, x)
    df2_df1 = deriv(f2, f1_x)
    df3_df2 = deriv(f3, f2_f1_x)

    return df1_dx * df2_df1 * df3_df2

# 변수가 두개인 경우 합성 함수
def multi_inputs_add(x:ndarray, y:ndarray, sigma:Array_Function)->float:
    assert x.shape == y.shape # 
    a = x+y

    return sigma(a)

if __name__ == '__main__':
  
    PLOT_RANGE = np.arange(-3, 3, 0.01)
    chain_1 = [square, sigmoid]
    chain_2 = [sigmoid, square]
    chain_3 = [leaky_relu, sigmoid, square]

    chain_1_ = chain_length_2(chain_1, PLOT_RANGE)
    chain_1_deriv = chain_deriv_2(chain_1, PLOT_RANGE)

    chain_2_ = chain_length_2(chain_2, PLOT_RANGE)
    chain_2_deriv = chain_deriv_2(chain_2, PLOT_RANGE)

    plt.figure(1)
    plt.plot(PLOT_RANGE, chain_1_)
    plt.plot(PLOT_RANGE, chain_1_deriv)
    plt.figure(2)
    plt.plot(PLOT_RANGE, chain_2_)
    plt.plot(PLOT_RANGE, chain_2_deriv)
    plt.figure(3)
    plt.plot(PLOT_RANGE, chain_length_3(chain_3, PLOT_RANGE))
    plt.plot(PLOT_RANGE, chain_deriv_3(chain_3, PLOT_RANGE))
    
    plt.show()