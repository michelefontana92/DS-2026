from .surrogate_factory import register_surrogate
from .soft_confusion_matrix.performance import *
from torch.nn import functional as F
import torch
from entmax import entmax_bisect

@register_surrogate('binary_accuracy')
class BinaryAccuracySurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.target_groups = None
        self.group_name = None
        
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
       
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        y_hat = probabilities[:,1]
        positive_mask = labels==1
        positive_mask = positive_mask.view(y_hat.shape)
        accuracy = binary_accuracy(y_hat,
                                       positive_mask=positive_mask)#kwargs.get('positive_mask'))    
       
        if self.use_max:
            return torch.max(torch.zeros_like(accuracy),self.upper_bound-accuracy)
        return self.upper_bound-accuracy

@register_surrogate('binary_precision')
class BinaryPrecisionSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.target_groups = None
        self.group_name = None

    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        
        positive_mask = labels==1
        precision = binary_precision(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
        
        if self.use_max:
            return torch.max(torch.zeros_like(precision),self.upper_bound-precision)
        return self.upper_bound-precision



@register_surrogate('binary_recall')
class BinaryRecallSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.target_groups = None
        self.group_name = None
        
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #probabilities = F.softmax(logits/0.2,dim=1)
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
       
        positive_mask = labels==1
        recall = binary_recall(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
       
        if self.use_max:
            return torch.max(torch.zeros_like(recall),self.upper_bound-recall)
        return self.upper_bound-recall


@register_surrogate('binary_f1')
class BinaryF1Surrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.mode = kwargs.get('mode','min')
        #print(f'[BinaryF1Surrogate] Initialized with mode: {self.mode}')
        self.target_groups = None
        self.group_name = None
        
    def __call__(self,**kwargs):
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        logits = kwargs.get('logits')
        #probabilities = entmax_bisect(
        #            logits / 0.25,
        #            alpha=1.5,
        #            dim=-1,
        #        )
        assert probabilities is not None, 'probabilities must be provided'
        # Controllo NaN nei probabilities
        
        if torch.isnan(probabilities).any():
            print('Probabilities contengono NaN!')
            #probabilities = torch.nan_to_num(probabilities, nan=0.0)  # Sostituisci NaN nei logits
            
        
        positive_mask = labels==1
        f1 = binary_f1_score(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
        
        if self.mode == 'min':
            #print('[MODE MIN]: Returning upper bound minus F1 score.')
            if self.use_max:
                return torch.max(torch.zeros_like(f1),self.upper_bound-f1)
            return self.upper_bound-f1
        else:
            #print('[MODE MAX]: Returning F1 score directly.')
            return f1 


@register_surrogate('multiclass_f1')
class MulticlassF1Surrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average','weighted')
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.mode = kwargs.get('mode','min')
        self.target_groups = None
        self.group_name = None
        
    def __call__(self,**kwargs):
        labels = kwargs.get('labels')
        logits = kwargs.get('logits')
        probabilities = kwargs.get('probabilities')
        #probabilities = entmax_bisect(
        #            logits / 0.25,
        #            alpha=1.5,
        #            dim=-1,
        #        )
        #probabilities = kwargs.get('probabilities')
        assert probabilities is not None, 'probabilities must be provided'
        # Controllo NaN nei probabilities
        if torch.isnan(probabilities).any():
            print('Probabilities contengono NaN!')
            probabilities = torch.nan_to_num(probabilities, nan=0.0)  # Sostituisci NaN nei logits
        
    
        f1 = multiclass_f1_score(probabilities,
                                 labels=labels,
                                average=self.average)
        
        if self.mode == 'min':
            if self.use_max:
                return torch.max(torch.zeros_like(f1),self.upper_bound-f1)
            return self.upper_bound-f1
        else:
            return f1
         
@register_surrogate('binary_true_positive')
class BinaryTruePositiveSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.target_groups = None
        self.group_name = None
    
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #probabilities = F.softmax(logits/0.2,dim=1)
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        
        positive_mask = labels==1
        tp = true_positive(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average,
                                       get_probability=True)
         
        
        if self.use_max:
            return torch.max(torch.zeros_like(tp),self.upper_bound-tp)
        return self.upper_bound-tp


@register_surrogate('binary_true_negative')
class BinaryTrueNegativeSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.target_groups = None
        self.group_name = None
    
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #probabilities = F.softmax(logits/0.2,dim=1)
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        
        positive_mask = labels==1
        tn = true_negative(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average,
                                       get_probability=True)
       
  
        if self.use_max:
            return torch.max(torch.zeros_like(tn),self.upper_bound-tn)
        return self.upper_bound-tn


@register_surrogate('binary_false_positive')
class BinaryFalsePositiveSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.target_groups = None
        self.group_name = None

    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #probabilities = F.softmax(logits/0.2,dim=1)
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        
        positive_mask = labels==1
        fp = false_positive(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average,
                                       get_probability=True)
       
       
        if self.use_max:
            return torch.max(torch.zeros_like(fp),self.upper_bound-fp)
        return self.upper_bound-fp


@register_surrogate('binary_false_negative')
class BinaryFalseNegativeSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.target_groups = None
        self.group_name = None

    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #probabilities = F.softmax(logits/0.2,dim=1)
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        
        positive_mask = labels==1
        fn = false_negative(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average,
                                       get_probability=True)
        
       
        if self.use_max:
            return torch.max(torch.zeros_like(fn),self.upper_bound-fn)
        return self.upper_bound-fn