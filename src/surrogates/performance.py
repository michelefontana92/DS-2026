from .surrogate_factory import register_surrogate
from torch.nn import CrossEntropyLoss

@register_surrogate('performance')
class PerformanceSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        loss_params = kwargs.get('loss_params',{})
        self.target_groups = None
        self.loss = CrossEntropyLoss(reduction='mean',**loss_params)
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')       
        class_weights = kwargs.get('class_weights')
        if class_weights is not None:
            print(f"Using class weights: {class_weights}")
            self.loss = CrossEntropyLoss(weight=class_weights, reduction='mean', **kwargs.get('loss_params', {}))
        final_loss = self.loss(logits,labels.long().view(-1,)).squeeze()
       
        # Numero di classi
        #C = logits.size(1)

        # Calcola la Normalized Cross Entropy
        #final_loss = loss.mean()# / torch.log(torch.tensor(C, dtype=torch.float))
        return final_loss

@register_surrogate('performance_batch')
class PerformanceSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        loss_params = kwargs.get('loss_params',{})
        self.target_groups = None
        self.group_name = None
        self.loss = CrossEntropyLoss(reduction='none',**loss_params)
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        final_loss = self.loss(logits,labels.long().view(-1,)).squeeze()
       
        # Numero di classi
        #C = logits.size(1)

        # Calcola la Normalized Cross Entropy
        #final_loss = loss.mean()# / torch.log(torch.tensor(C, dtype=torch.float))
        return final_loss
