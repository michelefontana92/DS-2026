import torch

def _binary_class_demographic_parity(probabilities, group_masks, group_ids):
    available_groups = torch.unique(group_masks)
    
    # Escludi il calcolo se uno dei gruppi target non è presente
    if not all(gid in available_groups for gid in group_ids):
        return None  

    mask_0 = group_masks == group_ids[0]
    mask_1 = group_masks == group_ids[1]
    if torch.sum(mask_0) == 0 or torch.sum(mask_1) == 0:
        return None 
    dp = torch.mean(probabilities[mask_0]) - torch.mean(probabilities[mask_1])
    return torch.abs(dp)

   
def demographic_parity(probabilities, **kwargs):
    group_masks = kwargs.get('group_masks')
    group_ids = kwargs.get('target_groups')
    target_class = kwargs.get('target_class')
    probabilities = probabilities[:, target_class]
    return _binary_class_demographic_parity(probabilities, group_masks, group_ids)

def _binary_class_equal_opportunity(probabilities,labels_mask,group_masks,group_ids):
    
    available_groups = torch.unique(group_masks)
    #print(f'[SURROGATE] Available groups: {available_groups}, Group IDs: {group_ids}')
    # Escludi il calcolo se uno dei gruppi target non è presente
    if not all(gid in available_groups for gid in group_ids):
        #print(f'[SURROGATE] One of the target groups {group_ids} is not available in group_masks {available_groups}. Returning None.')
        return None
  
    else:
        positive_mask = (group_masks == group_ids[0]) & (labels_mask)
        negative_mask = (group_masks == group_ids[1]) & (labels_mask)
        if torch.sum(positive_mask) == 0 or torch.sum(negative_mask) == 0:
            #print(f'[SURROGATE] Positive or negative mask is empty for group IDs {group_ids}. Returning None.')
            return None
        eo = torch.mean(probabilities[positive_mask]) - torch.mean(probabilities[negative_mask])
        return torch.abs(eo)
    
def equal_opportunity(probabilities,**kwargs):
    group_masks = kwargs.get('group_masks')
    group_ids = kwargs.get('target_groups')
    labels = kwargs.get('labels')
    target_class = kwargs.get('target_class')
    
    assert group_masks is not None, 'group_masks must be provided'
    assert group_ids is not None, 'group_ids must be provided'
    assert labels is not None, 'labels must be provided'
    assert target_class is not None, 'target_class must be provided'
    #print('Probabilities head:', probabilities[:5])
    #print(f'[INFO] Equal Opportunity on group {group_ids} with target class {target_class} = {probabilities[:5]}')
    probabilities = probabilities[:,target_class]
    labels_mask = (labels == target_class)
    #print(f'[SURROGATE] Group IDs: {group_ids}, Target Class: {target_class}, Labels Mask: {labels_mask[:5]} True Labels: {labels[:5]}')
    return _binary_class_equal_opportunity(probabilities,labels_mask,
                                               group_masks,group_ids)

    

def predictive_equality(probabilities,**kwargs):
    group_masks = kwargs.get('group_masks')
    group_ids = kwargs.get('target_groups')
    labels = kwargs.get('labels')
    target_class = kwargs.get('target_class')
    assert group_masks is not None, 'group_masks must be provided'
    assert group_ids is not None, 'group_ids must be provided'
    assert labels is not None, 'labels must be provided'

    probabilities = probabilities[:,target_class]
    labels_mask = ~(labels == target_class)
    return _binary_class_equal_opportunity(probabilities,labels_mask,
                                               group_masks,group_ids)
    

       
def equalized_odds(probabilities,**kwargs):
    
    eo = equal_opportunity(probabilities,**kwargs)
    pe = predictive_equality(probabilities,**kwargs)
    #print(f'[INFO] Equalized Odds: {eo}, Predictive Equality: {pe}')
    if (eo is None)and (pe is None):
        # Gruppo mancante: non includere vincolo
        return None 
    if eo is None:
        # Solo Predictive Equality disponibile
        return pe
    if pe is None:
        # Solo Equalized Odds disponibile
        return eo
    return torch.max(eo,pe)