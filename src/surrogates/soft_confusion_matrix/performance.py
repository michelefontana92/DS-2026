import torch

def binary_accuracy(y_hat,**kwargs):
    #print('Binary Accuracy')
    tp = true_positive(y_hat,**kwargs)
    tn = true_negative(y_hat,**kwargs)
    fp = false_positive(y_hat,**kwargs)
    fn = false_negative(y_hat,**kwargs)
    if tp + tn + fp + fn < 1e-3:
        return torch.tensor(0.0)
    return (tp + tn) / (tp + tn + fp + fn)


def true_positive(y_hat, **kwargs):
    positive_mask = kwargs.get('positive_mask')
    get_probability = kwargs.get('get_probability', False)
    assert positive_mask is not None
    
    # Controlla se la maschera è vuota
    if torch.sum(positive_mask).item() == 0:
        return torch.tensor(0.0)
    
    positive_proba = torch.mean(y_hat[positive_mask])
    
    if not get_probability:
        positive_proba *= torch.sum(positive_mask)
        
    return positive_proba

def true_negative(y_hat, **kwargs):
    negative_mask = ~kwargs.get('positive_mask')
    get_probability = kwargs.get('get_probability', False)
    assert negative_mask is not None
    
    # Controlla se la maschera è vuota
    if torch.sum(negative_mask).item() == 0:
        return torch.tensor(0.0)
    
    negative_proba = 1 - torch.mean(y_hat[negative_mask])
    
    if not get_probability:
        negative_proba *= torch.sum(negative_mask)
        
    return negative_proba

def false_positive(y_hat, **kwargs):
    negative_mask = ~kwargs.get('positive_mask')
    get_probability = kwargs.get('get_probability', False)
    assert negative_mask is not None
    
    # Controlla se la maschera è vuota
    if torch.sum(negative_mask).item() == 0:
        return torch.tensor(0.0)
    
    negative_proba = torch.mean(y_hat[negative_mask])
    
    if not get_probability:
        negative_proba *= torch.sum(negative_mask)
        
    return negative_proba

def false_negative(y_hat, **kwargs):
    positive_mask = kwargs.get('positive_mask')
    get_probability = kwargs.get('get_probability', False)
    assert positive_mask is not None
    
    # Controlla se la maschera è vuota
    if torch.sum(positive_mask).item() == 0:
        return torch.tensor(0.0)
    
    positive_proba = 1 - torch.mean(y_hat[positive_mask])
    
    if not get_probability:
        positive_proba *= torch.sum(positive_mask)
        
    return positive_proba

def _precision(y_hat, **kwargs):
    tp = true_positive(y_hat, **kwargs)
    fp = false_positive(y_hat, **kwargs)
    
    # Controllo per NaN e numeri molto piccoli
    if torch.isnan(tp) or torch.isnan(fp) or (tp + fp < 1e-6):
        return torch.tensor(0.0)
    
    return tp / (tp + fp)

def _recall(y_hat, **kwargs):
    tp = true_positive(y_hat, **kwargs)
    fn = false_negative(y_hat, **kwargs)
    
    # Controllo per NaN e numeri molto piccoli
    if torch.isnan(tp) or torch.isnan(fn) or (tp + fn < 1e-6):
        return torch.tensor(0.0)
    
    return tp / (tp + fn)

def binary_precision(y_hat, **kwargs):
    average = kwargs.get('average')
    if average is None:
        return _precision(y_hat, **kwargs)
    elif average == 'weighted':
        return _weighted_precision(y_hat, **kwargs)
    else: 
        raise ValueError(f'{average} method is unknown')
    
def binary_recall(y_hat, **kwargs):
    average = kwargs.get('average')
    if average is None:
        return _recall(y_hat, **kwargs)
    elif average == 'weighted':
        return _weighted_recall(y_hat, **kwargs)
    else: 
        raise ValueError(f'{average} method is unknown')

def _f1_score(y_hat, **kwargs):
    kwargs['average'] = None
    precision = binary_precision(y_hat, **kwargs)
    recall = binary_recall(y_hat, **kwargs)
    
    # Prevenzione di divisioni per zero
    if precision + recall < 1e-6:
        return torch.tensor(0.0)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Controllo per NaN
    if torch.isnan(f1):
        print('F1 is NaN!')
        return torch.tensor(0.0)
    
    return f1

def binary_f1_score(y_hat, **kwargs):
    average = kwargs.get('average')
    if average is None:
        return _f1_score(y_hat, **kwargs)
    elif average == 'weighted':
        return _weighted_f1_score(y_hat, **kwargs)
    else: 
        raise ValueError(f'{average} method is unknown')

def _weighted_f1_score(y_hat, **kwargs):
    positive_mask = kwargs.get('positive_mask')
    assert positive_mask is not None 
    negative_mask = ~positive_mask
    f1_score_class_0 = _f1_score(1 - y_hat, positive_mask=negative_mask)
    f1_score_class_1 = _f1_score(y_hat, positive_mask=positive_mask)
    n_positive = torch.sum(positive_mask)
    n_negative = torch.sum(negative_mask)
    n_records = n_positive + n_negative
    
    return (n_positive * f1_score_class_1 + n_negative * f1_score_class_0) / n_records

def _weighted_precision(y_hat, **kwargs):
    positive_mask = kwargs.get('positive_mask')
    assert positive_mask is not None 
    negative_mask = ~positive_mask
    precision_class_0 = _precision(1 - y_hat, positive_mask=negative_mask)
    precision_class_1 = _precision(y_hat, positive_mask=positive_mask)
    n_positive = torch.sum(positive_mask)
    n_negative = torch.sum(negative_mask)
    n_records = n_positive + n_negative
    
    return (n_positive * precision_class_1 + n_negative * precision_class_0) / n_records

def _weighted_recall(y_hat, **kwargs):
    positive_mask = kwargs.get('positive_mask')
    assert positive_mask is not None 
    negative_mask = ~positive_mask
    recall_class_0 = _recall(1 - y_hat, positive_mask=negative_mask)
    recall_class_1 = _recall(y_hat, positive_mask=positive_mask)
    n_positive = torch.sum(positive_mask)
    n_negative = torch.sum(negative_mask)
    n_records = n_positive + n_negative
    
    return (n_positive * recall_class_1 + n_negative * recall_class_0) / n_records



def multiclass_accuracy(probabilities,**kwargs): 
    n_records = probabilities.shape[0]
    tp_list = true_positive_multiclass(probabilities,**kwargs)
    tp = torch.sum(tp_list)
    accuracy = (tp) / n_records
    return accuracy

def multiclass_precision(probabilities,**kwargs):
    average = kwargs.get('average')
    kwargs['average'] = None
    labels = kwargs.get('labels')
    tp_list = true_positive_multiclass(probabilities,**kwargs)
    fp_list = false_positive_multiclass(probabilities,**kwargs)
    if torch.sum(tp_list + fp_list) < 1e-4:
        return torch.zeros_like(tp_list)
    else:
        precision_list = torch.zeros_like(tp_list)
        for i,(tp,fp) in enumerate(zip(tp_list,fp_list)):
            if tp + fp < 1e-4:
                precision_list[i] = 0
            else:
                precision_list[i] = tp / (tp + fp)
        if average is None:
            return precision_list
        elif average == 'weighted':
            support_list = torch.tensor([torch.sum(labels==i) for i in range(probabilities.shape[1])])
            return torch.sum(precision_list*support_list) / torch.sum(support_list)
        elif average == 'macro':
            return torch.mean(precision_list)
        else:
            raise ValueError(f'{average} method is unknown')
    
def multiclass_recall(probabilities,**kwargs):
    average = kwargs.get('average')
    kwargs['average'] = None
    labels = kwargs.get('labels')
    tp_list = true_positive_multiclass(probabilities,**kwargs)
    fn_list = false_negative_multiclass(probabilities,**kwargs)
    if torch.sum(tp_list + fn_list) < 1e-4:
        return torch.zeros_like(tp_list)
    else:
        recall_list = torch.zeros_like(tp_list)
        for i,(tp,fn) in enumerate(zip(tp_list,fn_list)):
            if tp + fn < 1e-4:
                recall_list[i] = 0
            else:
                recall_list[i] = tp / (tp + fn)
        if average is None:
            return recall_list
        elif average == 'weighted':
            support_list = torch.tensor([torch.sum(labels==i)for i in range(probabilities.shape[1])])
            return torch.sum(recall_list*support_list) / torch.sum(support_list)
        elif average == 'macro':
            return torch.mean(recall_list)
        else:
            raise ValueError(f'{average} method is unknown')

def multiclass_f1_score(probabilities,**kwargs):
    average = kwargs.get('average')
    kwargs['average'] = None
    labels = kwargs.get('labels')
    precision_list = multiclass_precision(probabilities,**kwargs)
    recall_list = multiclass_recall(probabilities,**kwargs)
    
    if torch.sum(precision_list + recall_list) < 1e-4:
        return torch.zeros_like(precision_list)
    else:
        f1_score_list = torch.zeros_like(precision_list)
        for i,(p,r) in enumerate(zip(precision_list,recall_list)):
            if p+ r < 1e-4:
                f1_score_list[i] = 0
            else:
                f1_score_list[i] = 2*(p*r) / (p+r)
        
        if average is None:
            return f1_score_list
        elif average == 'weighted':
            support_list = torch.tensor([torch.sum(labels == i).item() for i in range(probabilities.shape[1])], dtype=torch.float)

            return torch.sum(f1_score_list*support_list) / torch.sum(support_list)   
        elif average == 'macro':
            return torch.mean(f1_score_list)
        else:
            raise ValueError(f'{average} method is unknown')
            
def true_positive_multiclass(probabilities,**kwargs):
    labels = kwargs.get('labels')
    tp_list = torch.zeros(probabilities.shape[1])
    for i in range(probabilities.shape[1]):
        y_hat = probabilities[:,i]
        positive_mask = labels == i
        if torch.sum(positive_mask) == 0:
            tp = 0 
        else: 
            tp = torch.mean(y_hat[positive_mask])*torch.sum(positive_mask)
        tp_list[i] = tp
    return tp_list

def true_negative_multiclass(probabilities,**kwargs):
    labels = kwargs.get('labels')
    tn_list = torch.zeros(probabilities.shape[1])
    for i in range(probabilities.shape[1]):
        y_hat = probabilities[:,i]
        negative_mask = ~(labels == i)
        if torch.sum(negative_mask) == 0:
            tn = 0 
        else: 
            tn = (1-torch.mean(y_hat[negative_mask]))*torch.sum(negative_mask)
        tn_list[i] = tn
    return tn_list

def false_positive_multiclass(probabilities,**kwargs):
    labels = kwargs.get('labels')
    fp_list = torch.zeros(probabilities.shape[1])
    for i in range(probabilities.shape[1]):
        y_hat = probabilities[:,i]
        negative_mask = ~(labels == i)
        if torch.sum(negative_mask) == 0:
            fp = 0 
        else: 
            fp = torch.mean(y_hat[negative_mask])*torch.sum(negative_mask)
        fp_list[i] = fp
    return fp_list

def false_negative_multiclass(probabilities,**kwargs):
    labels = kwargs.get('labels')
    fn_list = torch.zeros(probabilities.shape[1])
    for i in range(probabilities.shape[1]):
        y_hat = probabilities[:,i]
        negative_mask = labels == i
        if torch.sum(negative_mask) == 0:
            fn = 0 
        else: 
            fn = (1-torch.mean(y_hat[negative_mask]))*torch.sum(negative_mask)
        fn_list[i] = fn
    return fn_list

