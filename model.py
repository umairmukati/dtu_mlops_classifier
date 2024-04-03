from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)

    def forward(self, x):
        return self.fc1(x)
