import numpy as np
from numpy import ndarray

class Operation(object):
    # 연산의 기본 정의 
    # 메소드 : 연산 규칙에 의한 forward, backward
    def __init__(self):
        pass
    def forward(self, input_:ndarray)->ndarray:
        self.input = input_
        self.output = self.output_()

        return self.output

    def backward(self, output_grad:ndarray):
        assert self.ouput.shape == output_grad.shape
        self.input_grad_ = self._input_grad(output_grad) # chain rule
        assert self.input.shape == self.input_grad
        return self.input_grad

    def output_(self):
        raise NotImplementedError()
    def _input_grad(self):
        raise NotImplementedError()

class ParamOperation(Operation):
    def __init__(self, param:ndarray):
        super().__init__() # 부모 클래스의 init 
        self.param = param
    
    def backward(self, output_grad):
        assert self.output.shape == output_grad.shape

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert self.param_grad.shape == self.param
        assert self.input_grad.shape == self.input

        return self.input_grad

    def _param_grad(self):
        raise NotImplementedError()

class WeightMultiply(ParamOperation):
    def __init__(self, W:ndarray):
        '''
        self.param = W
        '''
        super().__init__(W)
    
    def _output(self)->ndarray:
        # X @ W
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray)-> ndarray:
        # 기울기 값 계산
        # chain rule
        return np.dot(output_grad, np.transpose(self.param, (1,0)))

    def _param_grad(self, output_grad:ndarray)->ndarray:
        # 행렬 사이즈로 인해 _input_grad와 곱 순서가 달라진다.
        return np.dot(np.transpose(self.input_, (1,0)), output_grad)

class BiasAdd(ParamOperation):
    def __init__(self, B:ndarray):
        # Bias의 행은 1, 열은 다음 뉴런의 크기
        assert B.shape[0] == 1
        
        super().__init__(B)

    def _output(self)->ndarray:
        return self.input_ + self.param
    
    def _input_grad(self, output_grad:ndarray)->ndarray:
        # 입력에 대한 기울기    
        # [1,1,1,1,1,1,1,1 .... ,1] 
        # Bias.shape은 (1, 다음 뉴런의 개수)
        present = np.ones_like(self.input_)
        return np.dot(output_grad, present)

    def _param_grad(self, output_grad:ndarray)->ndarray:
        param_grad = np.ones_like(self.param) * output_grad
        #