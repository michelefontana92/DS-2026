from opacus import PrivacyEngine
from torch.utils.data import DataLoader

class PrivacyWrapper:
    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon) if epsilon is not None else None
        self.privacy_engine = PrivacyEngine()
        self.delta = None

    def reset(self):
        self.privacy_engine = PrivacyEngine()
        self.delta = None

    def get_total_budget(self):
        return self.epsilon

    def get_epsilon(self):
        if self.delta is None:
            raise ValueError("delta non inizializzato: chiama apply_privacy() prima di get_epsilon().")
        return self.privacy_engine.get_epsilon(self.delta)

    def apply_privacy(self, model, optimizer, data_loader, epochs, criterion, max_grad_norm=5.0, delta=None):
        assert model is not None
        assert optimizer is not None
        assert isinstance(data_loader, DataLoader)
        assert criterion is not None, "Passa criterion (es. CrossEntropyLoss) a make_private_with_epsilon."

        if delta is None:
            N = len(data_loader.dataset)
            delta = 1.0 / N
        self.delta = delta

        if self.epsilon is not None and self.epsilon > 0:
            model, optimizer, data_loader = self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                criterion=criterion,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=max_grad_norm,
                epochs=epochs,
            )

        return model, optimizer, data_loader
