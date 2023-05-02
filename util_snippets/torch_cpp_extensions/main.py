
import time
import math
import torch
import torch.nn.functional as F
import argparse
import numpy as np

import lltm_cpp

class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)
        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cpp.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell
    
class LLTM_CPP(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM_CPP, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)

class LLTM_PY(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM_PY, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate

        return new_h, new_cell

def run(rnn, X, h, C, num_iter):
    forwards = []; backwards = []
    F = 0; B = 0
    for _ in range(num_iter):
        t1 = time.time()
        new_h, new_C = rnn(X, (h, C))
        t2 = time.time()
        e = t2 - t1
        forwards.append(e)
        F += e

        t1 = time.time()
        (new_h.sum() + new_C.sum()).backward()
        t2 = time.time()
        e = t2 - t1
        B += e
        backwards.append(e)

    forwards = np.array(forwards)
    backwards = np.array(backwards)

    print(f"Forward: total:{F:.3f} s | mean:{forwards.mean()*1000:.3f} ms | std:{forwards.std()*1000:.3f} ms")
    print(f"Backward: total:{B:.3f} s | mean:{backwards.mean()*1000:.3f} ms | std:{backwards.std()*1000:.3f} ms")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--state_size', type=int, default=128)
    parser.add_argument('--num_iter', type=int, default=100000)
    # parser.add_argument('--backend', type=str, default='py', choices=['py', 'cpp'])
    parser.add_argument('--py', action='store_true', default=False)
    parser.add_argument('--cpp', action='store_true', default=False)
    args = parser.parse_args()

    X = torch.randn(args.batch_size, args.input_size)
    h = torch.randn(args.batch_size, args.state_size)
    C = torch.randn(args.batch_size, args.state_size)

    if args.py:
        rnn = LLTM_PY(args.input_size, args.state_size)
    elif args.cpp:
        rnn = LLTM_CPP(args.input_size, args.state_size)
    else:
        raise ValueError("Unknown backend")

    run(rnn, X, h, C, args.num_iter)

'''
Usage:
python main.py --py --num_iter 10000 --batch_size 16 --input_size 32 --state_size 128
python main.py --cpp --num_iter 10000 --batch_size 16 --input_size 32 --state_size 128
'''
