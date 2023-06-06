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

class BiasAdd(ParamOperation):
    def __init__(self):
        pass