from typing import Union, List
Numberable = Union[float, int]

class NumberWithGrad():
    def __init__(self, num:Numberable, depends_on:List[Numberable] = None, creation_op:str = ''):
        self.num = num
        self.grad =  None
        self.depends_on = depends_on or []
        self.creation_op = creation_op
    
    def __add__(self, other:Numberable):
        return NumberWithGrad(self.num + ensure_number(other).num, depends_on = [self, ensure_number(other)], creation_op = 'add')
    
    def __mul__(self, other:Numberable):
        return NumberWithGrad(self.num * ensure_number(other).num, depends_on = [self, ensure_number(other)], creation_op = 'mul')

    def backward(self, backward_grad: Numberable = None):
        if backward_grad is None:
            self.grad = 1
        else:
            if self.grad is None:
                self.grad = backward_grad
            else:
                self.grad += backward_grad 
                # 이미 관여한 gradient가 있으면 해당 값을 더한다. 
            
        if self.creation_op == "add": 
            # greaient 자체는 이미 Numberable 의 성격을 갖는다.
            self.depends_on[0].backward(self.grad)
            self.depends_on[1].backward(self.grad)
        
        if self.creation_op == "mul":
            new = self.depends_on[1] * self.grad
            # 순서를 지켜야한다. 
            self.depends_on[0].backward(new.num)
            
            new = self.depends_on[0] * self.grad 
            # new는 instance 가 된다. -> __mul__ 에 의한 오버라이딩으로 인해
            self.depends_on[1].backward(new.num)

                



def ensure_number(num:Numberable) -> NumberWithGrad:
    if isinstance(num, NumberWithGrad): 
    # 만약 이미 num이 NumberWithGrad의 instance 라면,
        return num
    else:
        return NumberWithGrad(num)

if __name__ == '__main__':
    a = NumberWithGrad(3)
    b = a * 4
    c = b + 5
    c.backward()
    print(a.grad)
    print(b.grad)
