from .surrogate_factory import register_surrogate
from torch.nn import CrossEntropyLoss
import torch 
from .soft_confusion_matrix.performance import binary_f1_score
from entmax import entmax_bisect
@register_surrogate('adaptive_aggregation')
class AdaptiveAggregationSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        loss_params = kwargs.get('loss_params',{})
        self.loss = CrossEntropyLoss(reduction='mean',**loss_params)
    
    def _wasserstein_distance_global(self,p,q):
    
        assert p.shape[1] == q.shape[1], 'Probabilities and target_probabilities must have the same number of classes' 
        #print('p: ',torch.mean(p,dim=0))
        #print('q: ',torch.mean(q,dim=0))
        F_p = torch.cumsum(torch.mean(p,dim=0),dim=0)
        F_q = torch.cumsum(torch.mean(q,dim=0),dim=0)
        wasserstein_distance = torch.abs(F_p - F_q)
        return torch.sum(wasserstein_distance).to(p.device)
    
    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')
       
        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_probabilities_list is not None, 'teacher_probabilities must be provided'
        
        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')

        performance_loss = self.loss(logits, labels.long().view(-1,))
        if len(teacher_probabilities_list) <= 0:
            w1 = 0
        else:
            distances = [
                self._wasserstein_distance_global(probabilities, t_probs)
                for t_probs in teacher_probabilities_list
            ]
            w1 = torch.mean(torch.stack(distances)).item()

        final_loss = performance_loss + w1
        #print(f"Performance loss: {performance_loss}, Wasserstein distance: {w1}")
        #print(f"Final loss: {final_loss}")
        return final_loss


@register_surrogate('adaptive_aggregation_f1')
class AdaptiveAggregationF1Surrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        loss_params = kwargs.get('loss_params',{})
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
    def _wasserstein_distance_global(self,p,q):
    
        assert p.shape[1] == q.shape[1], 'Probabilities and target_probabilities must have the same number of classes' 
        #print('p: ',torch.mean(p,dim=0))
        #print('q: ',torch.mean(q,dim=0))
        F_p = torch.cumsum(torch.mean(p,dim=0),dim=0)
        F_q = torch.cumsum(torch.mean(q,dim=0),dim=0)
        wasserstein_distance = torch.abs(F_p - F_q)
        return torch.sum(wasserstein_distance).to(p.device)
    
    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')
       
        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_probabilities_list is not None, 'teacher_probabilities must be provided'
        
        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')
        
       
        positive_mask = labels==1
        f1 = binary_f1_score(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
        
        if self.use_max:
            f1 =  torch.max(torch.zeros_like(f1),self.upper_bound-f1)
        else: 
            f1 = self.upper_bound-f1
        if len(teacher_probabilities_list) <= 0:
            w1 = 0
        else:
            distances = [
                self._wasserstein_distance_global(probabilities, t_probs)
                for t_probs in teacher_probabilities_list
            ]
            w1 = torch.mean(torch.stack(distances)).item()

        final_loss = f1 + w1
        #print(f"Performance loss: {performance_loss}, Wasserstein distance: {w1}")
        #print(f"Final loss: {final_loss}")
        return final_loss




@register_surrogate('adaptive_aggregation_f1_2')
class AdaptiveAggregationF1Surrogate2:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        loss_params = kwargs.get('loss_params',{})
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
    def _wasserstein_distance_global(self,p,q):
    
        assert p.shape[1] == q.shape[1], 'Probabilities and target_probabilities must have the same number of classes' 
        #print('p: ',torch.mean(p,dim=0))
        #print('q: ',torch.mean(q,dim=0))
        F_p = torch.cumsum(torch.mean(p,dim=0),dim=0)
        F_q = torch.cumsum(torch.mean(q,dim=0),dim=0)
        wasserstein_distance = torch.abs(F_p - F_q)
        return torch.sum(wasserstein_distance).to(p.device)
    
    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')
        output_distribution = kwargs.get('output_distribution')  # softmax dello student
        teacher_softmax_list = kwargs.get('teacher_softmax_list')  # softmax dei teacher
        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_probabilities_list is not None, 'teacher_probabilities must be provided'
        
        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')
        
       
        positive_mask = labels==1
        f1 = binary_f1_score(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
        
        if self.use_max:
            f1 =  torch.max(torch.zeros_like(f1),self.upper_bound-f1)
        else: 
            f1 = self.upper_bound-f1

        if len(teacher_probabilities_list) == 0:
            #print('No teacher probabilities provided')
            return f1
        #print('Teacher probabilities provided: ',len(teacher_probabilities_list))
        global_model_teacher = teacher_probabilities_list[0]
        w1_global = self._wasserstein_distance_global(output_distribution, teacher_softmax_list[0]).item()
        w1_topk = 0 
        if len(teacher_probabilities_list) > 1:
            distances = [
                self._wasserstein_distance_global(output_distribution, t_probs)
                for t_probs in teacher_softmax_list[1:]
            ]
            w1_topk = torch.mean(torch.stack(distances)).item()
            #print(f'F1 surrogate: {f1.item()}, Global distillation loss: {w1_global}, Peer distillation loss: {w1_topk}')
        final_loss = f1 + 1.0*w1_global+ 1.0*w1_topk
        
        #print(f"Performance loss: {performance_loss}, Wasserstein distance: {w1}")
        #print(f"Final loss: {final_loss}")
        return final_loss



@register_surrogate('adaptive_aggregation_2')
class AdaptiveAggregationSurrogate2:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        loss_params = kwargs.get('loss_params',{})
        self.loss = CrossEntropyLoss(reduction='mean',**loss_params)
    
    def _wasserstein_distance_global(self,p,q):
    
        assert p.shape[1] == q.shape[1], 'Probabilities and target_probabilities must have the same number of classes' 
        #print('p: ',torch.mean(p,dim=0))
        #print('q: ',torch.mean(q,dim=0))
        F_p = torch.cumsum(torch.mean(p,dim=0),dim=0)
        F_q = torch.cumsum(torch.mean(q,dim=0),dim=0)
        wasserstein_distance = torch.abs(F_p - F_q)
        return torch.sum(wasserstein_distance).to(p.device)
    
    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')
       
        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_probabilities_list is not None, 'teacher_probabilities must be provided'
        
        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')

        performance_loss = self.loss(logits, labels.long().view(-1,))
        if len(teacher_probabilities_list) <= 0:
            return performance_loss
        
        global_model_teacher = teacher_probabilities_list[0]
        w1_global = self._wasserstein_distance_global(probabilities, global_model_teacher).item()
        w1_topk = 0 
        if len(teacher_probabilities_list) > 1:
            distances = [
                self._wasserstein_distance_global(probabilities, t_probs)
                for t_probs in teacher_probabilities_list[1:]
            ]
            w1_topk = torch.mean(torch.stack(distances)).item()

        final_loss = performance_loss + w1_global + w1_topk
        #print(f"Performance loss: {performance_loss}, Wasserstein distance: {w1}")
        #print(f"Final loss: {final_loss}")
        return final_loss



@register_surrogate('adaptive_aggregation_f1_3')
class AdaptiveAggregationF1Surrogate3:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        loss_params = kwargs.get('loss_params',{})
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
    

    def _correction_loss(self, probabilities, teacher_probabilities, labels):
        assert probabilities.shape[1] == teacher_probabilities.shape[1], \
            'Probabilities and teacher_probabilities must have the same number of classes'
        
        # Convert everything to float
        labels = labels.float()
        student_probs = probabilities[:, 1]
        teacher_probs = teacher_probabilities[:, 1]

        assert labels.shape[0] == student_probs.shape[0], \
            'Labels and probabilities must have the same number of samples'

        # Soft correction score
        #correction_fn_tp = torch.mean(labels * (1 - student_probs) * teacher_probs)
        #correction_fp_tn = torch.mean((1 - labels) * student_probs * (1 - teacher_probs))
        #correction_loss = 0.5*(correction_fn_tp + correction_fp_tn)
        fn_mask = labels * (1 - student_probs)
        fp_mask = (1 - labels) * student_probs

        numerator_fn = torch.sum(fn_mask * teacher_probs)
        denominator_fn = torch.sum(fn_mask) + 1e-8

        numerator_fp = torch.sum(fp_mask * (1 - teacher_probs))
        denominator_fp = torch.sum(fp_mask) + 1e-8

        correction_loss = 0.5 * (
            numerator_fn / denominator_fn +
            numerator_fp / denominator_fp
)
        return correction_loss

        
    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')
       
        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_probabilities_list is not None, 'teacher_probabilities must be provided'
        
        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')
        
       
        positive_mask = labels==1
        f1 = binary_f1_score(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
        
        if self.use_max:
            f1 =  torch.max(torch.zeros_like(f1),self.upper_bound-f1)
        else: 
            f1 = self.upper_bound-f1

        correction_loss = 0
    
        if len(teacher_probabilities_list) > 0:
            
            #distances = [self._correction_loss(probabilities, t_probs, labels)
            #        for t_probs in teacher_probabilities_list]
            #correction_loss = torch.mean(torch.stack(distances))
       
            for t_probs in teacher_probabilities_list:
                correction_loss += self._correction_loss(probabilities, t_probs, labels)
            correction_loss /= len(teacher_probabilities_list)

        final_loss = f1 + correction_loss
        return final_loss
    


@register_surrogate('adaptive_aggregation_f1_4')
class AdaptiveAggregationF1Surrogate4:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        loss_params = kwargs.get('loss_params',{})
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.lambda_global = kwargs.get("lambda_global",1.0)
        self.lambda_peer = kwargs.get("lambda_peer", 1.0)


    def _distillation_loss(self, probabilities, teacher_probabilities, 
                         labels,softmax_distribution,teacher_softmax_distribution):
        
        assert probabilities.shape[1] == teacher_probabilities.shape[1], \
            'Probabilities and teacher_probabilities must have the same number of classes'
        
        # Convert everything to float
        labels = labels.float()
        student_probs = probabilities[:, 1]
        teacher_probs = teacher_probabilities[:, 1]

        student_distribution = softmax_distribution[:, 1].unsqueeze(1)
        teacher_distribution = teacher_softmax_distribution[:, 1].unsqueeze(1)
        assert student_distribution.shape[0] == student_probs.shape[0], \
            'Softmax distribution and probabilities must have the same number of samples'
        assert teacher_distribution.shape[0] == student_probs.shape[0], \
        'Teacher softmax distribution and student probs must match'
        assert labels.shape[0] == student_probs.shape[0], \
        'Labels and probabilities must have the same number of samples'

        fn_mask = labels * (1 - student_probs) * teacher_probs
        fp_mask = (1 - labels) * student_probs * (1 - teacher_probs)
        tp_mask = labels * student_probs * teacher_probs
        tn_mask = (1 - labels) * (1 - student_probs) * (1 - teacher_probs)
        
        #correction_mask = fn_mask + fp_mask +tn_mask+tp_mask # [B]
        #correction_mask = correction_mask.unsqueeze(1)  # [B, 1] 

        weighted_mask = (fn_mask + fp_mask).unsqueeze(1)
        
        assert weighted_mask.shape == student_distribution.shape, \
            'Weighted mask and student distribution must have the same shape'
        assert weighted_mask.shape == teacher_distribution.shape, \
            'Weighted mask and teacher distribution must have the same shape'
        weighted_student_distribution = weighted_mask * student_distribution
        weighted_teacher_distribution = weighted_mask * teacher_distribution

        loss = self._wasserstein_distance_global(weighted_student_distribution, weighted_teacher_distribution)
        return loss

    def _wasserstein_distance_global(self,p,q):
    
        assert p.shape[1] == q.shape[1], 'Probabilities and target_probabilities must have the same number of classes' 
        #print('p: ',torch.mean(p,dim=0))
        #print('q: ',torch.mean(q,dim=0))
        F_p = torch.cumsum(torch.mean(p,dim=0),dim=0)
        F_q = torch.cumsum(torch.mean(q,dim=0),dim=0)
        wasserstein_distance = torch.abs(F_p - F_q)
        return torch.sum(wasserstein_distance).to(p.device)
    
    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')

        output_distribution = kwargs.get('output_distribution')
        teacher_softmax_list = kwargs.get('teacher_softmax_list')
        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_probabilities_list is not None, 'teacher_probabilities must be provided'
        assert output_distribution is not None, 'output_distribution must be provided'
        assert teacher_softmax_list is not None, 'teacher_softmax_list must be provided'

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')
        
       
        positive_mask = labels==1
        f1 = binary_f1_score(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
        
        if self.use_max:
            f1 =  torch.max(torch.zeros_like(f1),self.upper_bound-f1)
        else: 
            f1 = self.upper_bound-f1

        if len(teacher_probabilities_list) == 0:
            #print('No teacher probabilities provided')
            return f1
        #print('Teacher probabilities provided: ',len(teacher_probabilities_list))
    
        global_model_teacher = teacher_probabilities_list[0]
        w1_global = self._distillation_loss(probabilities, global_model_teacher,
                                            labels,output_distribution,teacher_softmax_list[0])
        
        
        w1_topk = 0 
        if len(teacher_probabilities_list) > 1:
            distances = [
                self._distillation_loss(probabilities, t_probs,
                                            labels,output_distribution,t_softmax)
                for t_probs,t_softmax in zip(teacher_probabilities_list[1:],teacher_softmax_list[1:])
            ]
            w1_topk = torch.mean(torch.stack(distances))
        
            #print(f"w1_topk: {w1_topk.item()}")
        #global_close_w1 = self._wasserstein_distance_global(output_distribution, teacher_softmax_list[0]).item()
        final_loss = f1 + (self.lambda_global*w1_global)#+ (self.lambda_peer*w1_topk) #+ 0.9*global_close_w1
        #final_loss = f1 + (self.lambda_peer*w1_topk)
        #print(f"Performance loss: {performance_loss}, Wasserstein distance: {w1}")
        #print(f"Final loss: {final_loss}")
        return final_loss

@register_surrogate('adaptive_aggregation_f1_5')
class AdaptiveAggregationF1Surrogate5:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.average = kwargs.get('average', None)
        self.upper_bound = kwargs.get('upper_bound', 1.0)
        self.use_max = kwargs.get('use_max', False)
        self.lambda_global = kwargs.get("lambda_global", 1.0)
        self.lambda_peer = kwargs.get("lambda_peer", 1.0)

    def _distillation_loss(self, probabilities, teacher_probabilities, 
                       labels, softmax_distribution, teacher_softmax_distribution):

        # Assumiamo softmax_distribution e teacher_softmax_distribution siano già [B, C]
        labels = labels.float()
        student_probs = probabilities[:, 1]
        teacher_probs = teacher_probabilities[:, 1]

        # Distribuzioni [B, C]
        student_distribution = softmax_distribution
        teacher_distribution = teacher_softmax_distribution

        # Check di sicurezza
        assert student_distribution.shape == teacher_distribution.shape, \
            'Softmax distributions must have same shape'
        assert student_distribution.shape[0] == student_probs.shape[0], \
            'Batch size mismatch'

        # Maschera soft: FP/FN corretti dal teacher
        fn_mask = labels * (1 - student_probs) * teacher_probs
        fp_mask = (1 - labels) * student_probs * (1 - teacher_probs)
        tp_mask = labels * student_probs * teacher_probs
        tn_mask = (1 - labels) * (1 - student_probs) * (1 - teacher_probs)
        
        correction_mask = fn_mask + fp_mask +tn_mask+tp_mask # [B]
        correction_mask = correction_mask.unsqueeze(1)  # [B, 1]

        # MSE per classe (non ridotta)
        mse = torch.nn.functional.mse_loss(student_distribution, teacher_distribution, reduction='none')  # [B, C]
        mse = mse.mean(dim=1, keepdim=True)  # [B, 1], loss per esempio

        # Loss finale focalizzata
        loss = torch.sum(mse * correction_mask) / (torch.sum(correction_mask) + 1e-8)

        return loss


    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')
        output_distribution = kwargs.get('output_distribution')  # softmax dello student
        teacher_softmax_list = kwargs.get('teacher_softmax_list')  # softmax dei teacher

        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_probabilities_list is not None, 'teacher_probabilities must be provided'
        assert output_distribution is not None, 'output_distribution must be provided'
        assert teacher_softmax_list is not None, 'teacher_softmax_list must be provided'

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')

        # Calcolo F1 surrogate
        positive_mask = labels == 1
        f1 = binary_f1_score(probabilities[:, 1],
                             positive_mask=positive_mask,
                             average=self.average)

        if self.use_max:
            f1 = torch.max(torch.zeros_like(f1), self.upper_bound - f1)
        else:
            f1 = self.upper_bound - f1

        if len(teacher_probabilities_list) == 0:
            return f1

        # Distillazione dal modello globale
        global_model_teacher = teacher_probabilities_list[0]
        global_softmax = teacher_softmax_list[0]
        loss_global = self._distillation_loss(probabilities, global_model_teacher,
                                              labels, output_distribution, global_softmax)

        
        #print(f'F1 surrogate: {f1.item()}, Global distillation loss: {loss_global.item()}, Mask sum: {mask_sum.item()}')
        loss_topk = 0.0
        # Distillazione dai peer
        
        loss_topk = 0.0
        if len(teacher_probabilities_list) > 1:
            peer_losses = [
                self._distillation_loss(probabilities, t_probs,
                                        labels, output_distribution, t_softmax)
                for t_probs, t_softmax in zip(teacher_probabilities_list[1:], teacher_softmax_list[1:])
            ]
            loss_topk = torch.mean(torch.stack(peer_losses))
            #print(f'F1 surrogate: {f1.item()}, Global distillation loss: {loss_global.item()}, Peer distillation loss: {loss_topk.item()}')
        # Loss finale
        
        final_loss = f1 + self.lambda_global * loss_global #+ self.lambda_peer * loss_topk
        return final_loss



@register_surrogate('adaptive_aggregation_f1_6')
class AdaptiveAggregationF1Surrogate6:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        loss_params = kwargs.get('loss_params',{})
        self.average = kwargs.get('average',None)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.lambda_global = kwargs.get("lambda_global",1.0)
        self.lambda_peer = kwargs.get("lambda_peer", 1.0)



    def _wasserstein_distance_global(self,p,q):
    
        assert p.shape[1] == q.shape[1], 'Probabilities and target_probabilities must have the same number of classes' 
        #print('p: ',torch.mean(p,dim=0))
        #print('q: ',torch.mean(q,dim=0))
        F_p = torch.cumsum(torch.mean(p,dim=0),dim=0)
        F_q = torch.cumsum(torch.mean(q,dim=0),dim=0)
        wasserstein_distance = torch.abs(F_p - F_q)
        return torch.sum(wasserstein_distance).to(p.device)
    
    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')

        output_distribution = kwargs.get('output_distribution')
        teacher_softmax_list = kwargs.get('teacher_softmax_list')
        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_probabilities_list is not None, 'teacher_probabilities must be provided'
        assert output_distribution is not None, 'output_distribution must be provided'
        assert teacher_softmax_list is not None, 'teacher_softmax_list must be provided'

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')
        
       
        positive_mask = labels==1
        f1 = binary_f1_score(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
        
        if self.use_max:
            f1 =  torch.max(torch.zeros_like(f1),self.upper_bound-f1)
        else: 
            f1 = self.upper_bound-f1

        if len(teacher_probabilities_list) == 0:
            #print('No teacher probabilities provided')
            return f1
        #print('Teacher probabilities provided: ',len(teacher_probabilities_list))

        teacher_ensemble_distribution = torch.mean(torch.stack(teacher_softmax_list), dim=0)
        #global_model_teacher = teacher_probabilities_list[0]
        #w1_global = self._distillation_loss(probabilities, global_model_teacher,
        #                                    labels,output_distribution,teacher_softmax_list[0])
        
        w1_global = self._wasserstein_distance_global(output_distribution, teacher_ensemble_distribution)
        
        final_loss = f1 + (self.lambda_global*w1_global)
        #final_loss = f1 + (self.lambda_peer*w1_topk)
        #print(f"Performance loss: {performance_loss}, Wasserstein distance: {w1}")
        #print(f"Final loss: {final_loss}")
        return final_loss


@register_surrogate('adaptive_aggregation_f1_7')
class AdaptiveAggregationF1Surrogate7:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.average = kwargs.get('average', None)
        self.upper_bound = kwargs.get('upper_bound', 1.0)
        self.use_max = kwargs.get('use_max', False)
        self.lambda_global = kwargs.get("lambda_global", 0.3)
        self.lambda_peer = kwargs.get("lambda_peer", 1.0)
        loss_params = kwargs.get('loss_params',{})
        self.loss = CrossEntropyLoss(reduction='mean',**loss_params)
    
    def _wasserstein_distance_global(self, p, q):
        assert p.shape[1] == q.shape[1], 'Distributions must have same number of classes'
        F_p = torch.cumsum(torch.mean(p, dim=0), dim=0)
        F_q = torch.cumsum(torch.mean(q, dim=0), dim=0)
        return torch.sum(torch.abs(F_p - F_q)).to(p.device)

    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_logits_list = kwargs.get('teacher_logits_list')
        output_distribution = kwargs.get('output_distribution')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')
        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_logits_list is not None, 'teacher_logits_list must be provided'
        assert output_distribution is not None, 'output_distribution must be provided'

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contain NaN')
        for t in teacher_logits_list:
            if torch.isnan(t).any():
                raise ValueError("Teacher logits contain NaN")
            assert t.shape == logits.shape, 'Teacher and student logits must match in shape'

        #f1 = self.loss(logits, labels.long().view(-1,))
        #return f1
        
        
        # F1 surrogate
        positive_mask = labels==1
        f1 = binary_f1_score(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
        
        if self.use_max:
            f1 =  torch.max(torch.zeros_like(f1),self.upper_bound-f1)
        else: 
            f1 = self.upper_bound-f1

        if len(teacher_logits_list) == 0:
            return f1
        
        distances = [
            self._wasserstein_distance_global(probabilities, t_probs)
            for t_probs in teacher_probabilities_list
        ]
        # Distillation via Wasserstein-1
        w1 = torch.mean(torch.stack(distances))

        return f1 + self.lambda_global * w1
        
@register_surrogate('adaptive_aggregation_f1_8')
class AdaptiveAggregationF1Surrogate8:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.average = kwargs.get('average', None)
        self.upper_bound = kwargs.get('upper_bound', 1.0)
        self.use_max = kwargs.get('use_max', False)
        self.lambda_global = kwargs.get("lambda_global", 1.0)
        self.lambda_peer = kwargs.get("lambda_peer", 1.0)
        self.temperature = kwargs.get("temperature", 2.0)  
        self.weights_list = []  

    def set_weights(self, weights):
        self.weights_list = weights

    def _wasserstein_distance_global(self, p, q):
        assert p.shape[1] == q.shape[1], 'Distributions must have same number of classes'
        F_p = torch.cumsum(torch.mean(p, dim=0), dim=0)
        F_q = torch.cumsum(torch.mean(q, dim=0), dim=0)
        return torch.sum(torch.abs(F_p - F_q)).to(p.device)

    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_logits_list = kwargs.get('teacher_logits_list')
        output_distribution = kwargs.get('output_distribution')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')

        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_logits_list is not None, 'teacher_logits_list must be provided'
        assert output_distribution is not None, 'output_distribution must be provided'
        assert len(self.weights_list) == len(teacher_logits_list), \
            'weights_list must match length of teacher_logits_list'

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contain NaN')
        for t in teacher_logits_list:
            if torch.isnan(t).any():
                raise ValueError("Teacher logits contain NaN")
            #print('Teacher logits shape:', t.shape)
            #print('Student logits shape:', logits.shape)
            assert t.shape == logits.shape, 'Teacher and student logits must match in shape'
        
        if isinstance(self.weights_list, torch.Tensor):
            raw_weights = self.weights_list
        else:
            assert isinstance(self.weights_list, (list, tuple)), "weights_list must be a list or tuple"
            raw_weights = torch.tensor(self.weights_list, device=teacher_logits_list[0].device)
            raw_weights = raw_weights.to(dtype=teacher_logits_list[0].dtype)

        #raw_weights = torch.tensor(self.weights_list, dtype=torch.float32, device=teacher_logits_list[0].device)
        weights_tensor = torch.softmax(raw_weights / 0.5, dim=0)
        #print('Weights tensor:', weights_tensor)
        #print('Weights:', weights_tensor)
        if isinstance(teacher_logits_list, torch.Tensor):
            stacked_logits = teacher_logits_list  # già [T, B, C]
        else:
            assert isinstance(teacher_logits_list, (list, tuple)), "teacher_logits_list must be a list or tuple"
            stacked_logits = torch.stack(teacher_logits_list, dim=0)

        weighted_logits = torch.einsum('tbc,t->bc', stacked_logits, weights_tensor)  # [B, C]

        # Teacher consensus: softmax over averaged logits
        teacher_ensemble_distribution = torch.softmax(weighted_logits / self.temperature, dim=1)  # [B, C]
        #student_distribution = torch.softmax(logits, dim=1)  # [B, C]
        # Student prediction: log-softmax
        student_log_probs = torch.log_softmax(logits / self.temperature, dim=1)  # [B, C]
        
        loss = torch.nn.functional.kl_div(student_log_probs, teacher_ensemble_distribution, reduction='batchmean',log_target=False) * (self.temperature ** 2)
        #loss = self._wasserstein_distance_global(student_distribution, teacher_ensemble_distribution)
        #print('Loss:', loss.item())
        return loss

@register_surrogate('adaptive_aggregation_f1_8_max')
class AdaptiveAggregationF1Surrogate8Max:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.average = kwargs.get('average', None)
        self.upper_bound = kwargs.get('upper_bound', 1.0)
        self.use_max = kwargs.get('use_max', False)
        self.lambda_global = kwargs.get("lambda_global", 1.0)
        self.lambda_peer = kwargs.get("lambda_peer", 1.0)
        self.temperature = kwargs.get("temperature", 2.0)  
        self.weights_list = []  

    def set_weights(self, weights):
        self.weights_list = weights

    def _wasserstein_distance_global(self, p, q):
        assert p.shape[1] == q.shape[1], 'Distributions must have same number of classes'
        F_p = torch.cumsum(torch.mean(p, dim=0), dim=0)
        F_q = torch.cumsum(torch.mean(q, dim=0), dim=0)
        return torch.sum(torch.abs(F_p - F_q)).to(p.device)

    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_logits_list = kwargs.get('teacher_logits_list')
        output_distribution = kwargs.get('output_distribution')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')

        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_logits_list is not None, 'teacher_logits_list must be provided'
        assert output_distribution is not None, 'output_distribution must be provided'
        assert len(self.weights_list) == len(teacher_logits_list), \
            'weights_list must match length of teacher_logits_list'

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contain NaN')
        for t in teacher_logits_list:
            if torch.isnan(t).any():
                raise ValueError("Teacher logits contain NaN")
            #print('Teacher logits shape:', t.shape)
            #print('Student logits shape:', logits.shape)
            assert t.shape == logits.shape, 'Teacher and student logits must match in shape'
        
        if isinstance(self.weights_list, torch.Tensor):
            raw_weights = self.weights_list
        else:
            assert isinstance(self.weights_list, (list, tuple)), "weights_list must be a list or tuple"
            raw_weights = torch.tensor(self.weights_list, device=teacher_logits_list[0].device)
            raw_weights = raw_weights.to(dtype=teacher_logits_list[0].dtype)

        #raw_weights = torch.tensor(self.weights_list, dtype=torch.float32, device=teacher_logits_list[0].device)
        weights_tensor = torch.softmax(raw_weights / self.temperature, dim=0)
        #print('Weights tensor:', weights_tensor)
        #print('Weights:', weights_tensor)
        if isinstance(teacher_logits_list, torch.Tensor):
            stacked_logits = teacher_logits_list  # già [T, B, C]
        else:
            assert isinstance(teacher_logits_list, (list, tuple)), "teacher_logits_list must be a list or tuple"
            stacked_logits = torch.stack(teacher_logits_list, dim=0)

        weighted_logits = torch.einsum('tbc,t->bc', stacked_logits, weights_tensor)  # [B, C]

        # Teacher consensus: softmax over averaged logits
        teacher_ensemble_distribution = torch.softmax(weighted_logits / self.temperature, dim=1)  # [B, C]
        #student_distribution = torch.softmax(logits, dim=1)  # [B, C]
        # Student prediction: log-softmax
        student_log_probs = torch.log_softmax(logits / self.temperature, dim=1)  # [B, C]
        
        loss = torch.nn.functional.kl_div(student_log_probs, teacher_ensemble_distribution, reduction='batchmean',log_target=False) * (self.temperature ** 2)
        #loss = self._wasserstein_distance_global(student_distribution, teacher_ensemble_distribution)
        #print('Loss:', loss.item())
        return -loss
    
@register_surrogate('adaptive_aggregation_f1_9')
class AdaptiveAggregationF1Surrogate9:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.average = kwargs.get('average', None)
        self.upper_bound = kwargs.get('upper_bound', 1.0)
        self.use_max = kwargs.get('use_max', False)
        self.lambda_global = kwargs.get("lambda_global", 1.0)
        self.temperature = kwargs.get("temperature", 2.0)
        self.use_class_weights = kwargs.get("use_class_weights", True)
        self.weights_list = []

    def set_weights(self, weights):
        self.weights_list = weights

    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        probabilities = kwargs.get('probabilities')  # [B, C]
        teacher_probabilities_list = kwargs.get('teacher_probabilities')  # list of [B, C]

        assert probabilities is not None, 'Student probabilities must be provided'
        assert teacher_probabilities_list is not None, 'Teacher probabilities must be provided'
        assert len(self.weights_list) == len(teacher_probabilities_list), \
            'weights_list must match number of teacher models'

        B, C = probabilities.shape
        num_teachers = len(teacher_probabilities_list)
        device = probabilities.device

        # Calcola i pesi normalizzati via softmax con temperatura
        raw_weights = torch.tensor(self.weights_list, dtype=torch.float32, device=device)
        weights_tensor = torch.softmax(raw_weights / self.temperature, dim=0)  # [T]
        #print('Weights tensor:', weights_tensor)
        # Inizializza i voti
        votes = torch.zeros((B, C), device=device)

        # Voto hard pesato: argmax di ogni teacher
        for i, teacher_probs in enumerate(teacher_probabilities_list):
            pred_classes = torch.argmax(teacher_probs, dim=1)  # [B]
            for b in range(B):
                votes[b, pred_classes[b]] += weights_tensor[i]

        # Calcola la distribuzione finale: one-hot votata
        consensus_indices = torch.argmax(votes, dim=1)  # [B], valori interi ∈ [0, C)

        if self.use_class_weights:
            # Calcola frequenze per classe nel consensus
            class_counts = torch.bincount(consensus_indices, minlength=C).float() + 1e-6
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * C  # normalizza

            loss = torch.nn.functional.cross_entropy(logits, consensus_indices, weight=class_weights.to(device))
        else:
            loss = torch.nn.functional.cross_entropy(logits, consensus_indices)

        return loss

@register_surrogate('adaptive_aggregation_f1_10')
class AdaptiveAggregationF1Surrogate10:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.average = kwargs.get('average', None)
        self.upper_bound = kwargs.get('upper_bound', 1.0)
        self.use_max = kwargs.get('use_max', False)
        self.lambda_global = kwargs.get("lambda_global", 1.0)
        self.temperature = kwargs.get("temperature", 2.0)
        self.weights_list = []
    
    def set_weights(self, weights):
        self.weights_list = weights

    def _distillation_loss(self, logits, teacher_logits_list, 
                           probabilities, teacher_probabilities_list, labels):
        labels = labels.float()
        student_prob = probabilities[:, 1]

        """
        if isinstance(self.weights_list, torch.Tensor):
            raw_weights = self.weights_list
        else:
            assert isinstance(self.weights_list, (list, tuple)), "weights_list must be a list or tuple"
            raw_weights = torch.tensor(self.weights_list, device=teacher_logits_list[0].device)
            raw_weights = raw_weights.to(dtype=teacher_logits_list[0].dtype)

        weights_tensor = torch.softmax(raw_weights / self.temperature, dim=0)
        #print('Weights tensor:', weights_tensor)
        """
        
        # ==== 1. Crea ensemble logits (media)
        if isinstance(teacher_logits_list, torch.Tensor):
            stacked = teacher_logits_list  # [T, B, C]
        else:
            stacked = torch.stack(teacher_logits_list, dim=0)  # [T, B, C]
        ensemble_logits = stacked.mean(dim=0)  # [B, C]
        #ensemble_logits = torch.einsum('tbc,t->bc', stacked, weights_tensor)  # [B, C]

        # ==== 2. Costruisci FP/FN mask
        with torch.no_grad():
            teacher_probs = torch.nn.functional.softmax(ensemble_logits,dim=1)  # [B, C]
            teacher_pred = torch.argmax(teacher_probs, dim=1).float()

        fn_mask = labels * teacher_pred * (1 - student_prob)
        fp_mask = (1 - labels) * (1 - teacher_pred) * student_prob
        correction_mask = (fn_mask + fp_mask).unsqueeze(1)  # [B, 1]

        # ==== 3. Teacher soft + confidenza
        T = self.temperature

        
        student_log_soft = torch.log_softmax(logits / T, dim=1)
        teacher_soft = torch.softmax(ensemble_logits / T, dim=1).detach()
        #teacher_conf = teacher_soft.max(dim=1)[0].unsqueeze(1)  # [B, 1]

        # ==== 4. KL distillazione focalizzata + pesata per confidenza
        kl = torch.nn.functional.kl_div(student_log_soft, teacher_soft, reduction='none').sum(dim=1, keepdim=True)
        kl = kl * (T * T)

        mask = correction_mask #* teacher_conf  # [B, 1], pesata
        loss = (kl * mask).sum() / (mask.sum() + 1e-8)

        return loss

    def __call__(self, **kwargs):
        logits = kwargs.get('logits')                           # [B, C]
        labels = kwargs.get('labels')                           # [B]
        probabilities = kwargs.get('probabilities')             # [B, C]
        teacher_logits_list = kwargs.get('teacher_logits_list')  # list of [B, C]
        teacher_probabilities_list = kwargs.get('teacher_probabilities')  # list of [B, C]

        assert logits is not None and labels is not None
        assert probabilities is not None
        assert teacher_logits_list is not None
        assert teacher_probabilities_list is not None

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')

        # ==== CE pesata per la parte supervisionata
        B, C = probabilities.shape
        labels_long = labels.long()
        class_counts = torch.bincount(labels_long, minlength=C).float() + 1e-6
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * C
        ce_loss = torch.nn.functional.cross_entropy(logits, labels_long, weight=class_weights.to(logits.device))

        # ==== Distillazione
        if len(teacher_logits_list) == 0:
            return ce_loss

        distill_loss = self._distillation_loss(logits, teacher_logits_list,
                                               probabilities, teacher_probabilities_list, labels)

        return ce_loss + self.lambda_global * distill_loss

@register_surrogate('batch_adaptive_aggregation_f1_10')
class BatchAdaptiveAggregationF1Surrogate10:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.average = kwargs.get('average', None)
        self.upper_bound = kwargs.get('upper_bound', 1.0)
        self.use_max = kwargs.get('use_max', False)
        self.lambda_global = kwargs.get("lambda_global", 1.0)
        self.temperature = kwargs.get("temperature", 2.0)

    def _distillation_loss(self, logits, teacher_logits_list, 
                       probabilities, teacher_probabilities_list, labels):
        labels = labels.float()
        student_prob = probabilities[:, 1]

        # Ensemble logits
        if isinstance(teacher_logits_list, torch.Tensor):
            stacked = teacher_logits_list  # [T, B, C]
        else:
            stacked = torch.stack(teacher_logits_list, dim=0)  # [T, B, C]
        ensemble_logits = stacked.mean(dim=0)  # [B, C]

        with torch.no_grad():
            teacher_probs = teacher_probabilities_list.mean(dim=0)  # [B, C]
            teacher_pred = torch.argmax(teacher_probs, dim=1).float()

        fn_mask = labels * teacher_pred * (1 - student_prob)
        fp_mask = (1 - labels) * (1 - teacher_pred) * student_prob
        correction_mask = (fn_mask + fp_mask).unsqueeze(1)  # [B, 1]

        T = self.temperature
        student_log_soft = torch.log_softmax(logits / T, dim=1)
        teacher_soft = torch.softmax(ensemble_logits / T, dim=1).detach()
        #teacher_conf = teacher_soft.max(dim=1)[0].unsqueeze(1)  # [B, 1]

        kl = torch.nn.functional.kl_div(student_log_soft, teacher_soft, reduction='none').sum(dim=1, keepdim=True)
        kl = kl * (T * T)

        mask = correction_mask   # [B, 1]
        masked_kl = (kl * mask).squeeze(1)  # [B]

        denom = (mask.squeeze(1) + 1e-8)  # [B]
        distill_loss_per_sample = masked_kl / denom.clamp(min=1e-8)  # [B]

        return distill_loss_per_sample


    def __call__(self, **kwargs):
        logits = kwargs.get('logits')                           # [B, C]
        labels = kwargs.get('labels')                           # [B]
        probabilities = kwargs.get('probabilities')             # [B, C]
        teacher_logits_list = kwargs.get('teacher_logits_list')  # list of [B, C]
        teacher_probabilities_list = kwargs.get('teacher_probabilities')  # list of [B, C]

        assert logits is not None and labels is not None
        assert probabilities is not None
        assert teacher_logits_list is not None
        assert teacher_probabilities_list is not None

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')

        B, C = probabilities.shape
        labels_long = labels.long()

        # Cross entropy per esempio (senza media)
        class_counts = torch.bincount(labels_long, minlength=C).float() + 1e-6
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * C
        ce_loss_per_sample = torch.nn.functional.cross_entropy(
            logits, labels_long, weight=class_weights.to(logits.device), reduction='none'
        )  # [B]

        # ==== Distillazione
        if len(teacher_logits_list) == 0:
            return ce_loss_per_sample  # shape [B]

        distill_loss_per_sample = self._distillation_loss(
            logits, teacher_logits_list, probabilities, teacher_probabilities_list, labels
        )  # [B]

        return ce_loss_per_sample + self.lambda_global * distill_loss_per_sample  # [B]

@register_surrogate('adaptive_aggregation_f1_11')
class AdaptiveAggregationF1Surrogate11:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'adaptive_aggregation')
        self.lambda_global = kwargs.get("lambda_global", 1.0)
        self.lambda_entropy = kwargs.get("lambda_entropy", 1.0)
        self.temperature = float(kwargs.get("temperature", 2.0))
        self.confidence_threshold = kwargs.get("confidence_threshold", 0.7)
        self.focus_only_high_conf = kwargs.get("focus_only_high_conf", True)
        self.weights_list = []

    def set_weights(self, weights):
        self.weights_list = weights

    def __call__(self, **kwargs):
        logits = kwargs.get('logits')                           # [B, C]
        teacher_logits_list = kwargs.get('teacher_logits_list')  # [T, B, C]

        assert logits is not None
        assert teacher_logits_list is not None
        assert len(self.weights_list) == len(teacher_logits_list)

        # === Normalize teacher weights
        raw_weights = torch.tensor(self.weights_list, device=logits.device, dtype=logits.dtype)
        weights_tensor = torch.softmax(raw_weights / self.temperature, dim=0)  # [T]

        # === Stack and combine teacher logits
        if isinstance(teacher_logits_list, torch.Tensor):
            stacked_logits = teacher_logits_list
        else:
            stacked_logits = torch.stack(teacher_logits_list, dim=0)  # [T, B, C]

        weighted_logits = torch.einsum('tbc,t->bc', stacked_logits, weights_tensor)  # [B, C]
        teacher_probs = torch.softmax(weighted_logits, dim=1)  # [B, C]

        # === Confidence mask (focus distillation only where teachers agree confidently)
        with torch.no_grad():
            confidence = teacher_probs.max(dim=1)[0]  # [B]
            if self.focus_only_high_conf:
                mask = (confidence > self.confidence_threshold).float()  # [B]
            else:
                mask = torch.ones_like(confidence)

        # === Student prediction
        student_log_probs = torch.log_softmax(logits, dim=1)  # [B, C]

        # === KL divergence
        kl_div = torch.nn.functional.kl_div(
            student_log_probs, teacher_probs, reduction='none', log_target=False
        ).sum(dim=1)  # [B]

        masked_kl = kl_div * mask
        loss_kl = masked_kl.sum() / (mask.sum() + 1e-8)

        # === Optional entropy regularization on consensus
        if self.lambda_entropy > 0:
            entropy = -(teacher_probs * teacher_probs.log()).sum(dim=1)  # [B]
            entropy_loss = entropy.mean()
            total_loss = self.lambda_global * loss_kl + self.lambda_entropy * entropy_loss
        else:
            total_loss = self.lambda_global * loss_kl

        return total_loss
