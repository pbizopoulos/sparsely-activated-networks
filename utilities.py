from torch import nn
from torch.nn import functional as F


def calculate_inverse_compression_ratio(model, data, num_activations):
    activation_multiplier = 1 + len(model.weights_list[0].shape)
    num_parameters = sum([weights.shape[0] for weights in model.weights_list])
    return (activation_multiplier*num_activations + num_parameters)/(data.shape[-1]*data.shape[-2])


class FNN(nn.Module):
    def __init__(self, sample_data, num_classes):
        super(FNN, self).__init__()
        self.fc = nn.Linear(sample_data.shape[-1]*sample_data.shape[-2], num_classes)

    def forward(self, batch_x):
        x = batch_x.view(batch_x.shape[0], -1)
        out = self.fc(x)
        return out


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.conv2 = nn.Conv1d(3, 16, 5)
        self.fc1 = nn.Linear(656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool1d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
