import numpy as np
from numpy import ndarray

input_1d = np.array([1,2,3,4,5])
param_1d = np.array([1,1,1])

def assert_same_shape(output: ndarray, 
                      output_grad: ndarray):
    assert output.shape == output_grad.shape, \
    '''
    두 ndarray의 모양이 같아야 하는데,
    첫 번째 ndarray의 모양은 {0}이고
    두 번째 ndarray의 모양은 {1}이다.
    '''.format(tuple(output_grad.shape), tuple(output.shape))
    return None

# dimention 제한
def assert_dim(t: ndarray,
               dim: ndarray):
    assert len(t.shape) == dim, \
    '''
    이 텐서는 {0}차원이어야 하는데, {1}차원이다.
    '''.format(dim, len(t.shape))
    return None

# padding
def _pad_1d(inp: ndarray, num:int)-> ndarray: # input과 padding을 인자로 받는다. 
    z = np.array([0])
    z = np.repeat(z,num)
    return np.concatenate([z,inp,z])

#convolution
def conv_1d(inp:ndarray, param:int):
    assert_dim(inp,1)
    assert_dim(param, 1)

    param_len = param.shape[0]
    param_mid = param_len // 2
    input_pad = _pad_1d(inp, param_mid)

    # 출력값 초기화 input의 크기만큼 [0,0,0,...,0] 을 생성한다. 
    out = np.zeros(inp.shape) # [0,0,0,0,0]
    
    for o in range(out.shape[0]): # o = 0,1,2,3,4,
        for p in range(param_len): # param_len = kernel_size = 0,1,2
            out[o] += param[p] * input_pad[o+p]
            # out[0] += param[0] * input_pad[0]
            # out[0] += param[1] * input_pad[1]
            # out[0] += param[2] * input_pad[2]

            # out[1] += param[0] * input_pad[1]
            # out[1] += param[1] * input_pad[2]
            # out[1] += param[2] * input_pad[3]

    # 출력 모양이 입력과 동일한지 확인
    assert_same_shape(inp, out)

    return out

def conv_1d_sum(inp:ndarray, param:ndarray)->ndarray:
    out = conv_1d(inp, param)
    return np.sum(out) # 선형결합의 결과 출력

def conv_1d_batch(inp:ndarray, param: ndarray)->ndarray:
    outs = [conv_1d(obs , param) for obs in inp] 
    # 만약 사진이라고 가정하면 1행 conv, 2행 conv, ... 이런 식으로 convolution을 하고 이 행렬을 2차원의 array 로 저장한다.
    return np.stack(outs)
    
def _param_grad_1d(inp: ndarray, 
                   param: ndarray, 
                   output_grad: ndarray = None) -> ndarray:
    
    param_len = param.shape[0]
    param_mid = param_len // 2
    input_pad = _pad_1d(inp, param_mid)
    
    if output_grad is None:
        output_grad = np.ones_like(inp)
    else:
        assert_same_shape(inp, output_grad)

    # 0으로 패딩된 1차원 합성곱
    param_grad = np.zeros_like(param)
    input_grad = np.zeros_like(inp)

    for o in range(inp.shape[0]):
        for p in range(param.shape[0]):
            param_grad[p] += input_pad[o+p] * output_grad[o]
        
    assert_same_shape(param_grad, param)
    
    return param_grad

def _input_grad_1d(inp: ndarray, 
                   param: ndarray, 
                   output_grad: ndarray = None) -> ndarray:
    
    param_len = param.shape[0] # 
    param_mid = param_len // 2
    inp_pad = _pad_1d(inp, param_mid)
    
    if output_grad is None:
        output_grad = np.ones_like(inp)
    else:
        assert_same_shape(inp, output_grad)
    
    output_pad = _pad_1d(output_grad, param_mid)
    
    # 0으로 패딩된 1차원 합성곱
    param_grad = np.zeros_like(param)
    input_grad = np.zeros_like(inp)
    
    # chain rule 에 의한 규칙
    for o in range(inp.shape[0]): # o = 0, 1, 2, 3, 4
        for f in range(param.shape[0]): # f = 0 , 1, 2
            input_grad[o] += output_pad[o+param_len-f-1] * param[f]
            # input_grad[0] += output_grad_pad[2] * param[0]
            # input_grad[0] += output_grad_pad[1] * param[1]
            # input_grad[0] += output_grad_pad[0] * param[2]


            # input_grad[4] += output_grad_pad[6] * param[0]
            # input_grad[4] += output_grad_pad[5] * param[1]
            # input_grad[4] += output_grad_pad[4] * param[2]
        
    assert_same_shape(param_grad, param)
    
    return input_grad

def _pad_1d_batch(inp: ndarray, 
                  num: int) -> ndarray:
    outs = [_pad_1d(obs, num) for obs in inp]
    return np.stack(outs)

def conv_1d_batch(inp:ndarray, param:ndarray) -> ndarray:
    outs = [conv_1d(obs, param) for obs in inp]
    return np.stack(outs)

def input_grad_1d_batch(inp:ndarray, param:ndarray)->ndarray:
    out = conv_1d_batch(inp, param)
    out_grad = np.ones_like(out)
    batch_size = out_grad.shape[0]
    grads = [_input_grad_1d(inp[i], param, out_grad[i]) for i in range(batch_size)]

    return np.stack(grads)

def param_grad_1d_batch(inp:ndarray, param:ndarray)->ndarray:
    output_grad = np.ones_like(param)
    

if __name__ == '__main__':
    input_1d_2 = np.array([1,2,3,4,6])
    param_1d = np.array([1,1,1])

    print(conv_1d_sum(input_1d, param_1d)) 
    print(conv_1d_sum(input_1d_2, param_1d)) # input 이 1만큼 변했는데 기울기에 얼마나 영향을 주는지 확인해야한다. 

    input_1d = np.array([1,2,3,4,5])
    param_1d_2 = np.array([2,1,1])

    # dL / dw_1  = 10
    print(conv_1d_sum(input_1d, param_1d))
    print(conv_1d_sum(input_1d, param_1d_2))

    print(_pad_1d(input_1d, 1))
    print(conv_1d(input_1d, param_1d))

    input_1d_batch = np.array([[0,1,2,3,4,5,6], 
                                [1,2,3,4,5,6,7]])

    print(conv_1d_batch(input_1d_batch, param_1d))
