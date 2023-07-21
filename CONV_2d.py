# 1. 순방향 계산 절차
#     1) 입력에 패딩을 붙인다. 
#     2) 패딩된 입력과 파라미터로부터 출력을 계산한다.
# 2. 역방향 계산에서 입력 기울기를 계산하는 절차
#     1) 출력 기울기에 패딩을 덧붙인다. 
#     2) 패딩된 출력 기울기와 입력, 파라미터로부터 입력 기울기와 파라미터 기울기를 계산한다. 
# 3. 역방향 계산에서 파라미터 기울기를 계산하는 절차
#     1) 입력에 패딩을 덧붙인다.
#     2) 패딩된 입력의 각 요소를 순회하며 파라미터 기울기를 더한다. 

import numpy as np
from numpy import ndarray
from CONV_1d import assert_dim
from CONV_1d import _pad_1d_batch
from CONV_1d import _pad_1d

def _pad_2d_obs(inp:ndarray, num:int):

    inp_pad = _pad_1d_batch(inp, num)
    other = np.zeros((num, inp.shape[0] + num * 2))
    
    # output

def _pad_2d(inp: ndarray, num : int):
    outs = [_pad_2d_obs(obs, num) for obs in inp]

    return np.stack(outs)



def _compute_output_obs(obs: ndarray, param: ndarray)->ndarray:
    '''
    obs : [channels, img_width, img_height]
    param : [in_channels, param_width, param_height]
    '''
    assert_dim(obs, 3) # obs : output.shape
    assert_dim(param, 4)

    param_size = param.shape[2]
    param_mid = param_size // 2
    obs_pad = _pad_2d_channel(obs, param_mid)

    in_channel = fil.shape[0]
    out_channel = fil.shape[1]

    img_size = obs.shape[1]
    omg_size = obs.shape[1]

if __name__ == '__main__':

    input_1d_batch = np.array([[0,1,2,3,4,5,6], [1,2,3,4,5,6,7]])
    print(_pad_1d_batch(input_1d_batch, 1))
    

    
