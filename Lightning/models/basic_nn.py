import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

# with Pytorch, creating a new neural network means creating a new class
# and BasicNN will inherit from PyTorch class called Module
class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        # init the weight and bias
        # Because we don't need to optimize this weight, we'll set requires_grad, which is
        # short for requires gradient, to False
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

    def forward(self, input):
        # Connecting all of them with Inputs, Activation Functions and Outputs...
        # make a forward pass through the neural network that uses weights and bias
        # that we just initialized

        # Given an input value the forward() function does a forward pass through the neural
        # network to calculate and return the output value
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output

class BasicNN_train(nn.Module):
    def __init__(self):
        super().__init__()
        # init the weight and bias
        # Because we don't need to optimize this weight, we'll set requires_grad, which is
        # short for requires gradient, to False
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, input):
        # Connecting all of them with Inputs, Activation Functions and Outputs...
        # make a forward pass through the neural network that uses weights and bias
        # that we just initialized

        # Given an input value the forward() function does a forward pass through the neural
        # network to calculate and return the output value
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output