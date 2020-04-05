import torch.nn as nn
import torch
"""# Сетки"""


class Actor(nn.Module):

    def __init__(self, state_dim):  # Для простоты сразу считаю, что action_dim = 1
        super().__init__()
        lin1 = nn.Linear(state_dim, 64)
        nn.init.xavier_uniform_(lin1.weight)

        lin2 = nn.Linear(64, 128)
        nn.init.xavier_uniform_(lin2.weight)

        self.lins = nn.Sequential(lin1, nn.ReLU(), lin2, nn.ReLU())

        mean_layer = nn.Linear(128, 1)
        nn.init.xavier_uniform_(mean_layer.weight)
        self.mean_layer = nn.Sequential(mean_layer, nn.Tanh())

        var_layer = nn.Linear(128, 1)
        nn.init.xavier_uniform_(var_layer.weight)
        self.var_layer = nn.Sequential(var_layer, nn.Softplus())

    def forward(self, input):
        output_lins = self.lins(input)
        mu = self.mean_layer(output_lins)
        sigma = torch.abs(self.var_layer(output_lins))
        return mu.squeeze(0) * 2, sigma.squeeze(0) + 0.00001


class Critic(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        lin1 = nn.Linear(state_dim, 256)
        nn.init.xavier_uniform_(lin1.weight)

        lin2 = nn.Linear(256, 64)
        nn.init.xavier_uniform_(lin2.weight)

        lin3 = nn.Linear(64, 1)
        nn.init.xavier_uniform_(lin3.weight)

        self.lins = nn.Sequential(lin1, nn.ReLU(), lin2, nn.ReLU(), lin3)

    def forward(self, input):
        return self.lins(input)
