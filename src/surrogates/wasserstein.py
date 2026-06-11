from .surrogate_factory import register_surrogate
import torch

@register_surrogate('wasserstein')
class WassersteinSurrogate:
    def __init__(self, **kwargs) -> None:
        self.surrogate_name = kwargs.get('surrogate_name', 'wasserstein')
        self.weight = kwargs.get('weight', 1.0)
        self.group_name = kwargs.get('group_name')
        self.lower_bound = kwargs.get('lower_bound',0.0)
        self.target_groups = kwargs.get('target_groups')
        self.use_local_distance = kwargs.get('use_local_distance', False)
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.teacher_idx = kwargs.get('teacher_idx')
        self.use_avg = kwargs.get('use_avg', False)
        if not self.use_avg:
            assert self.teacher_idx is not None, 'teacher_idx must be provided'
        
        if self.use_local_distance:
            assert self.group_name is not None, 'group_name must be provided'
            assert self.target_groups is not None, 'target_groups must be provided'
            #print(f'Local Wasserstein Distance on group {self.group_name} and target groups {self.target_groups}: lower_bound = {self.lower_bound}')
        

    def _wasserstein_distance_local(self,p,q,group_masks, target_groups):
       
        assert p.shape[1] == q.shape[1], 'Probabilities and target_probabilities must have the same number of classes' 
        positive_mask = torch.isin(group_masks, target_groups)
        if torch.sum(positive_mask) == 0:
            return torch.tensor(0.0, device=p.device)
        #print('p: ',torch.mean(p[positive_mask],dim=0))
        #print('q: ',torch.mean(q[positive_mask],dim=0))
        F_p = torch.cumsum(torch.mean(p[positive_mask],dim=0),dim=0)
        F_q = torch.cumsum(torch.mean(q[positive_mask],dim=0),dim=0)
        wasserstein_distance = torch.abs(F_p - F_q)
        return torch.sum(wasserstein_distance).to(p.device)

    def _wasserstein_distance_global(self,p,q):
    
        assert p.shape[1] == q.shape[1], 'Probabilities and target_probabilities must have the same number of classes' 
        assert p.shape[0] == q.shape[0], 'Probabilities and target_probabilities must have the same number of samples'
        #print('p: ',torch.mean(p,dim=0))
        #print('q: ',torch.mean(q,dim=0))
        F_p = torch.cumsum(torch.mean(p,dim=0),dim=0)
        F_q = torch.cumsum(torch.mean(q,dim=0),dim=0)
        wasserstein_distance = torch.abs(F_p - F_q)
        return torch.sum(wasserstein_distance).to(p.device)
    
    def __call__(self, **kwargs):
        #print('Wasserstein surrogate called')
        probabilities = kwargs.get('probabilities')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')
        ##print(f"[INFO] Computing Wasserstein surrogate for teacher index {self.teacher_idx}."
        #      )
        #print(f"[DEBUG] Probabilities shape: {probabilities.shape if probabilities is not None else 'None'}"
        #      f", Teacher probabilities list length: {len(teacher_probabilities_list) if teacher_probabilities_list is not None else 'None'}")
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_probabilities_list is not None, 'teacher_probabilities must be provided'
        teacher_probabilities = teacher_probabilities_list[self.teacher_idx]
        #assert isinstance(teacher_probabilities_list, list), 'teacher_probabilities must be a list'
        # Controllo NaN nei probabilities
        if torch.isnan(probabilities).any():
            print('Probabilities contengono NaN!')
            raise ValueError('Probabilities contiene NaN!')
        
        if torch.isnan(teacher_probabilities).any():
            print('Teacher probabilities contengono NaN!')
            raise ValueError('Teacher probabilities contiene NaN!')
       
        group_masks = kwargs.get('group_masks')
        assert group_masks is not None, 'group_masks must be provided'
        if self.use_local_distance:
            w1 = self._wasserstein_distance_local(
                probabilities,
                teacher_probabilities,
                group_masks=group_masks[self.group_name],
                target_groups=self.target_groups.to(self.device),
               
            )
        elif self.use_avg:
            distances = []
            for i in range(len(teacher_probabilities_list)):
                teacher_probabilities = teacher_probabilities_list[i]
                distance = self._wasserstein_distance_global(probabilities, 
                                                             teacher_probabilities)
                distances.append(distance)
            distances = torch.stack(distances)
            w1 = torch.mean(distances)
        else:              
            w1 = self._wasserstein_distance_global(probabilities,
                                                    teacher_probabilities)
                            
        
        return torch.max(w1 - self.lower_bound, torch.tensor(0.0, device=w1.device))
