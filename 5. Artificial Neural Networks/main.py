import numpy as np


class Tensor:

    def __init__(self, arr, requires_grad=True):

        self.arr = arr
        self.requires_grad = requires_grad

        # When node is created without predecessor the op is denoted as 'leaf'
        # 'leaf' signifies leaf node
        self.history = ['leaf', None, None]
        # History stores the information of the operation that created the Tensor.
        # Check set_history

        # Gradient of the tensor
        self.zero_grad()
        self.shape = self.arr.shape

    def zero_grad(self):
        
        self.grad = np.zeros_like(self.arr)

    def set_history(self, op, operand1, operand2):
        
        self.history = []
        self.history.append(op)
        self.requires_grad = False
        self.history.append(operand1)
        self.history.append(operand2)

        if operand1.requires_grad or operand2.requires_grad:
            self.requires_grad = True

    
    def __add__(self, other):
        
        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                raise ArithmeticError(
                    f"Shape mismatch for +: '{self.shape}' and '{other.shape}' ")
            out = self.arr + other.arr
            out_tensor = Tensor(out)
            out_tensor.set_history('add', self, other)

        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'")

        return out_tensor

    

    def __matmul__(self, other):
        
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"unsupported operand type(s) for matmul: '{self.__class__}' and '{type(other)}'")
        if self.shape[-1] != other.shape[-2]:
            raise ArithmeticError(
                f"Shape mismatch for matmul: '{self.shape}' and '{other.shape}' ")
        out = self.arr @ other.arr
        out_tensor = Tensor(out)
        out_tensor.set_history('matmul', self, other)

        return out_tensor

    def grad_add(self, gradients=None):
        
        if(gradients is None):
            matmul_left_oper = self.history[1].arr
            matmul_right_oper = self.history[2].arr
            gradients = np.ones((matmul_left_oper.shape[0], matmul_right_oper.shape[1]), dtype=int)
        return (gradients,gradients)

    def grad_matmul(self, gradients=None):
        
        left_operand_of_grad_matmul = self.history[1].arr
        right_operand_of_grad_matmul = self.history[2].arr
        if(gradients is None):
            gradients = np.ones((left_operand_of_grad_matmul.shape[0], right_operand_of_grad_matmul.shape[1]), dtype=int)
        grad1 = np.matmul(gradients, right_operand_of_grad_matmul.transpose())
        grad2 = np.matmul(left_operand_of_grad_matmul.transpose(), gradients)
        return (grad1,grad2)

    def backward(self, gradients=None):
        
        if(self.history[0] == 'leaf'):
            if(self.requires_grad):
                self.grad += gradients
            else:
                self.zero_grad()
            return
        elif(self.history[0] == 'matmul'):
            right_gradients = self.grad_matmul(gradients)
        elif(self.history[0] == 'add'):
            right_gradients = self.grad_add(gradients)
        self.history[1].backward(right_gradients[0])
        self.history[2].backward(right_gradients[1])
