
from torch import nn
from torch.nn import functional as F
from .architecture_factory import register_architecture
from fastshap.utils import MaskLayer1d
from fastshap import MaskLayer1d_Masked

@register_architecture('mlp2hidden_surrogate')
class MLP2HiddenSurrogate(nn.Module):

    def __init__(self, **kwargs):
        super(MLP2HiddenSurrogate, self).__init__()
        model_params = kwargs['model_params']
        input_dim = model_params['input']
        hidden1_dim = model_params['hidden1']
        hidden2_dim = model_params['hidden2']
        dropout = model_params['dropout']
        output_dim = model_params['output']
        self.mask= MaskLayer1d(value=0, append=True)
        self.fc1 = nn.Linear(2*input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden2_dim, output_dim)

    def forward(self, batch):
        x = F.relu(self.fc1(self.mask(batch)))
        x = F.relu(self.fc2(x))
        #dx = self.drop(x)
        x = self.out(x)
        return x

    def freeze(self):
        self.fc1.requires_grad_(False)

    def freeze_all(self):
        self.fc1.requires_grad_(False)
        self.out.requires_grad_(False)

    def unfreeze_all(self):
        self.fc1.requires_grad_(True)
        self.out.requires_grad_(True)


@register_architecture('mlp3hidden_surrogate')
class MLP3HiddenSurrogate(nn.Module):

    def __init__(self, **kwargs):
        super(MLP3HiddenSurrogate, self).__init__()
        model_params = kwargs['model_params']
        input_dim = model_params['input']
        hidden1_dim = model_params['hidden1']
        hidden2_dim = model_params['hidden2']
        hidden3_dim = model_params['hidden3']
        dropout = model_params['dropout']
        output_dim = model_params['output']
        self.mask= MaskLayer1d(value=0, append=True)
        self.fc1 = nn.Linear(2*input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden3_dim, output_dim)

    def forward(self, batch):
        x = F.relu(self.fc1(self.mask(batch)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = self.out(x)
        return x

    def freeze(self):
        self.fc1.requires_grad_(False)

    def freeze_all(self):
        self.fc1.requires_grad_(False)
        self.out.requires_grad_(False)

    def unfreeze_all(self):
        self.fc1.requires_grad_(True)
        self.out.requires_grad_(True)

@register_architecture('mlp2hidden_explainer')
class MLP2HiddenExplainer(nn.Module):

    def __init__(self, **kwargs):
        super(MLP2HiddenExplainer, self).__init__()
        model_params = kwargs['model_params']
        input_dim = model_params['input']
        hidden1_dim = model_params['hidden1']
        hidden2_dim = model_params['hidden2']
        dropout = model_params['dropout']
        output_dim = model_params['output']
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden2_dim, output_dim*input_dim)

    def forward(self, batch):
        x = F.relu(self.fc1(batch))
        x = F.relu(self.fc2(x))
        #x = self.drop(x)
        x = self.out(x)
        return x

    def freeze(self):
        self.fc1.requires_grad_(False)

    def freeze_all(self):
        self.fc1.requires_grad_(False)
        self.out.requires_grad_(False)

    def unfreeze_all(self):
        self.fc1.requires_grad_(True)
        self.out.requires_grad_(True)

@register_architecture('mlp3hidden_explainer')
class MLP3HiddenExplainer(nn.Module):

    def __init__(self, **kwargs):
        super(MLP3HiddenExplainer, self).__init__()
        model_params = kwargs['model_params']
        input_dim = model_params['input']
        hidden1_dim = model_params['hidden1']
        hidden2_dim = model_params['hidden2']
        hidden3_dim = model_params['hidden3']
        dropout = model_params['dropout']
        output_dim = model_params['output']
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden3_dim, output_dim*input_dim)

    def forward(self, batch):
        x = F.relu(self.fc1(batch))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = self.out(x)
        return x

    def freeze(self):
        self.fc1.requires_grad_(False)

    def freeze_all(self):
        self.fc1.requires_grad_(False)
        self.out.requires_grad_(False)

    def unfreeze_all(self):
        self.fc1.requires_grad_(True)
        self.out.requires_grad_(True)
