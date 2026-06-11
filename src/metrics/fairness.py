from torchmetrics import StatScores
import torch 
from .metrics_factory import register_metric
from .base_metric import BaseMetric
from surrogates import SurrogateFactory

@register_metric('statistic_scores')
class StatisticScores(BaseMetric):
    def __init__(self,**kwargs):
        task = kwargs.get('task','multiclass')
        num_classes = kwargs.get('num_classes',2)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.stat_scores = StatScores(task=task,
                                      num_classes=num_classes).to(self.device)
    
    def calculate(self, y_pred, y_true):
        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)
        self.stat_scores.update(y_pred, y_true)
    
    def get(self,normalize=False):
        support = self.stat_scores.tp + self.stat_scores.fp + self.stat_scores.tn + self.stat_scores.fn
        stats = {"tp": self.stat_scores.tp, 
            "fp": self.stat_scores.fp,
            "tn": self.stat_scores.tn,
            "fn": self.stat_scores.fn,
            "tpr": self.stat_scores.tp / (self.stat_scores.tp + self.stat_scores.fn) if (self.stat_scores.tp + self.stat_scores.fn) != 0 else 0,
            "fpr": self.stat_scores.fp / (self.stat_scores.fp + self.stat_scores.tn) if (self.stat_scores.fp + self.stat_scores.tn) != 0 else 0,
            "tnr": self.stat_scores.tn / (self.stat_scores.tn + self.stat_scores.fp) if (self.stat_scores.tn + self.stat_scores.fp) != 0 else 0,
            "fnr": self.stat_scores.fn / (self.stat_scores.fn + self.stat_scores.tp) if (self.stat_scores.fn + self.stat_scores.tp) != 0 else 0,
            "base_rate": (self.stat_scores.tp + self.stat_scores.fp) / support if support != 0 else 0,
            }
            
        if normalize:
            stats["tp"] = self.stat_scores.tp / support
            stats["fp"] = self.stat_scores.fp / support
            stats["tn"] = self.stat_scores.tn / support
            stats["fn"] = self.stat_scores.fn / support
        return stats    
       
    def reset(self):
        self.stat_scores.reset()




class GroupFairnessMetric(BaseMetric):
    def __init__(self,**kwargs):
        task = kwargs.get('task','binary')
        self.num_classes = kwargs.get('num_classes',2)
        group_ids = kwargs.get('group_ids')
        self.group_ids = group_ids
        self.group_name = kwargs.get('group_name')
        assert isinstance(group_ids, dict), "group_ids must be a dictionary"
        assert len(group_ids[self.group_name]) > 1, "group_ids must have at least 2 groups"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_multiclass = kwargs.get('use_multiclass',False)
        _REDUCTION_TYPES = {
            'min': torch.min, 
            'mean': torch.mean, 
            'max': torch.max
        }

        self._reduction = _REDUCTION_TYPES.get(kwargs.get('reduction','max'))
    
        self.stats_per_group = {}
        for group_id in group_ids[self.group_name]:
            self.stats_per_group[group_id] = StatisticScores(task='binary',
                                                             num_classes=2)
        self.stats_per_class = {}
        for current_class in range(self.num_classes):
            self.stats_per_class[current_class] = {}
            for group_id in group_ids[self.group_name]:
                self.stats_per_class[current_class][group_id] = StatisticScores(task='binary',
                                                             num_classes=2)

    def calculate(self, y_pred, y_true, group_ids:dict):
        #print('Group ids: ',group_ids.keys())
        current_group_ids:list = group_ids[self.group_name]
        #print('Current group ids: ',current_group_ids)
        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)
        assert len(y_pred) == len(y_true) == len(current_group_ids), "y_pred, y_true and group_ids must have the same length"
        #print(f'Group IDs: { torch.unique(current_group_ids)}')
        if self.num_classes == 2:
            
            for group_id in torch.unique(current_group_ids):
                if group_id != -1:
                    #print('y_pred:',y_pred)
                    #print('group_ids:',group_ids[self.group_name])
                    y_pred_group = y_pred[current_group_ids==group_id.item()]
                    y_true_group = y_true[current_group_ids==group_id.item()]
                    self.stats_per_group[group_id.item()].calculate(y_pred_group, y_true_group)
        else:
            #print(f'[METRIC] Predictions: {y_pred[:5]}, True labels: {y_true[:5]}, Group IDs: {current_group_ids[:5]}')
            for current_class in range(self.num_classes):
                y_pred_class = torch.where(y_pred==current_class,1,0)
                y_true_class = torch.where(y_true==current_class,1,0)
                for group_id in torch.unique(current_group_ids):
                    if group_id != -1:
                        y_pred_group = y_pred_class[current_group_ids==group_id.item()]
                        y_true_group = y_true_class[current_group_ids==group_id.item()]
                        #self.stats_per_class[current_class][group_id.item()].calculate(y_pred_group, y_true_group)
                        if y_pred_group.numel() > 0:
                            self.stats_per_class[current_class][group_id.item()].calculate(y_pred_group, y_true_group)

    def get(self,normalize=False):
       pass

    def get_stats_per_group(self,group_id):     
        pass
             
    def reset(self):
        for _,stats in self.stats_per_group.items():
            stats.reset()
        for _,stats_dict in self.stats_per_class.items():
            for _,stats in stats_dict.items():
                stats.reset()


@register_metric('demographic_parity')
class DemographicParity(GroupFairnessMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats_per_group_diff = []
        self.stats_per_class_group_diff = {}
        self.metrics_per_class = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff[current_class] = []

            
    def get(self):
        if self.num_classes == 2:
           
            #print('Number of groups with valid stats:', len(group_ids))

            for i in range(len(self.stats_per_group.keys())-1):
                for j in range(i+1,len(self.stats_per_group.keys())):
                    stats_group_i = self.stats_per_group[i].get()
                    stats_group_j = self.stats_per_group[j].get()
                    support_i = stats_group_i["tp"] + stats_group_i["fp"] + stats_group_i["tn"] + stats_group_i["fn"]
                    support_j = stats_group_j["tp"] + stats_group_j["fp"] + stats_group_j["tn"] + stats_group_j["fn"]
                    is_empty_i = support_i == 0
                    is_empty_j = support_j == 0
                    if (not is_empty_i) and (not is_empty_j):
                       self.stats_per_group_diff.append(
                           abs(stats_group_i['base_rate'] - stats_group_j['base_rate']))
            return {
                    f'demographic_parity_{self.group_name}':self._reduction(
                        torch.tensor(self.stats_per_group_diff))
                    }
        else: 
            group_ids = [
                gid for gid in self.stats_per_class[0].keys()
                if any(
                    self.stats_per_class[c][gid].stat_scores.tp +
                    self.stats_per_class[c][gid].stat_scores.fp +
                    self.stats_per_class[c][gid].stat_scores.tn +
                    self.stats_per_class[c][gid].stat_scores.fn > 0
                    for c in range(self.num_classes)
                )
            ]

          
            for current_class in range(self.num_classes):
                for i in range(len(group_ids)):
                    for j in range(i+1,len(group_ids)):
                        stats_group_i = self.stats_per_class[current_class][group_ids[i]].get()
                        stats_group_j = self.stats_per_class[current_class][group_ids[j]].get()
                        is_empty_i = stats_group_i["tp"] + stats_group_i["fp"] + stats_group_i["tn"] + stats_group_i["fn"] == 0
                        is_empty_j = stats_group_j["tp"] + stats_group_j["fp"] + stats_group_j["tn"] + stats_group_j["fn"] == 0
                        if (not is_empty_i) and (not is_empty_j):
                         
                            self.stats_per_class_group_diff[current_class].append(
                            abs(stats_group_i['base_rate'] - stats_group_j['base_rate'])
                            )
                
                
                if len(self.stats_per_class_group_diff[current_class]) > 0:
                    dp = self._reduction(torch.tensor(self.stats_per_class_group_diff[current_class]))
                    self.metrics_per_class.append(dp)
           
            return {
                    f'demographic_parity_{self.group_name}':self._reduction(
                        torch.tensor(self.metrics_per_class))
                    }
    def get_stats_per_group(self, group_id):
        return self.stats_per_group[group_id].get()['base_rate'][0].item()
            
    def reset(self):
        super().reset()
        self.stats_per_group_diff = []
        self.metrics_per_class = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff[current_class] = []
    
@register_metric('equal_opportunity')
class EqualOpportunity(GroupFairnessMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats_per_group_diff = []
        self.stats_per_class_group_diff = {}
        self.metrics_per_class = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff[current_class] = []

    def get(self):
        if self.num_classes == 2:
            for i in range(len(self.stats_per_group)-1):
                for j in range(i+1,len(self.stats_per_group)):
                    self.stats_per_group_diff.append(abs(self.stats_per_group[i].get()['tpr'] - self.stats_per_group[j].get()['tpr']))
        
            return {f'equal_opportunity_{self.group_name}':
                    self._reduction(
                        torch.tensor(self.stats_per_group_diff))
                    }
        else:
            group_ids = [
                gid for gid in self.stats_per_class[0].keys()
                if any(
                    self.stats_per_class[c][gid].stat_scores.tp +
                    self.stats_per_class[c][gid].stat_scores.fp +
                    self.stats_per_class[c][gid].stat_scores.tn +
                    self.stats_per_class[c][gid].stat_scores.fn > 0
                    for c in range(self.num_classes)
                )
            ]

          
            for current_class in range(self.num_classes):
                for i in range(len(group_ids)):
                    for j in range(i+1,len(group_ids)):
                        self.stats_per_class_group_diff[current_class].append(
                            abs(self.stats_per_class[current_class][group_ids[i]].get()['tpr'] - self.stats_per_class[current_class][group_ids[j]].get()['tpr']))
                if len(self.stats_per_class_group_diff[current_class]) > 0:
                    eo = self._reduction(torch.tensor(self.stats_per_class_group_diff[current_class]))
                    self.metrics_per_class.append(eo)
      
            return {
                    f'equal_opportunity_{self.group_name}':self._reduction(
                        torch.tensor(self.metrics_per_class))
                    }
        
    def get_stats_per_group(self, group_id):
        return torch.tensor(
            self.stats_per_group[group_id].get()['tpr'][0].item()
            )
    def reset(self):
        super().reset()
        self.stats_per_group_diff = []
        self.metrics_per_class = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff[current_class] = []
    
    
    
@register_metric('equalized_odds')
class EqualizedOdds(GroupFairnessMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats_per_group_diff_tpr = []
        self.stats_per_group_diff_fpr = []
        self.stats_per_class_group_diff_tpr = {}
        self.stats_per_class_group_diff_fpr = {}
        self.metrics_per_class = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff_tpr[current_class] = []
            self.stats_per_class_group_diff_fpr[current_class] = []
       
    def get(self):
        if self.num_classes == 2: 
            for i in range(len(self.stats_per_group.keys())-1):
                for j in range(i+1,len(self.stats_per_group.keys())):
                    stats_group_i = self.stats_per_group[i].get()
                    stats_group_j = self.stats_per_group[j].get()
                    is_empty_i = (stats_group_i['tp'] + stats_group_i['fn'] + stats_group_i['fp'] + stats_group_i['tn']) == 0
                    is_empty_j = (stats_group_j['tp'] + stats_group_j['fn'] + stats_group_j['fp'] + stats_group_j['tn']) == 0
                    n_gi_tpr = stats_group_i["tp"] + stats_group_i["fn"]
                    n_gj_tpr = stats_group_j["tp"] + stats_group_j["fn"]
                    n_gi_fpr = stats_group_i["fp"] + stats_group_i["tn"]
                    n_gj_fpr = stats_group_j["fp"] + stats_group_j["tn"]
                    if (not is_empty_i) and (not is_empty_j):
                        if n_gi_tpr > 0 and n_gj_tpr > 0:
                            #print(f'[METRIC] Diff TPR: {abs(stats_group_i["tpr"] - stats_group_j["tpr"])}, Diff FPR: {abs(stats_group_i["fpr"] - stats_group_j["fpr"])}')
                            self.stats_per_group_diff_tpr.append(
                                abs(stats_group_i['tpr'] - stats_group_j['tpr'])
                            )
                        
                        if n_gi_fpr > 0 and n_gj_fpr > 0:
                            #print(f'[METRIC] Diff FPR: {abs(stats_group_i["fpr"] - stats_group_j["fpr"])}')
                            self.stats_per_group_diff_fpr.append(
                                abs(stats_group_i['fpr'] - stats_group_j['fpr'])
                            )
                    

            return {
                f'equalized_odds_{self.group_name}':
                torch.max(
                self._reduction(
                    torch.tensor(self.stats_per_group_diff_tpr)),
                self._reduction(
                    torch.tensor(self.stats_per_group_diff_fpr))
            )
            }
        else:
            group_ids = [
                gid for gid in self.stats_per_class[0].keys()
                if any(
                    self.stats_per_class[c][gid].stat_scores.tp +
                    self.stats_per_class[c][gid].stat_scores.fp +
                    self.stats_per_class[c][gid].stat_scores.tn +
                    self.stats_per_class[c][gid].stat_scores.fn > 0
                    for c in range(self.num_classes)
                )
            ]
            #print(f'[INFO] Group IDs: {group_ids}')

          
            for current_class in range(self.num_classes):
                for i in range(len(group_ids)):
                    for j in range(i+1,len(group_ids)):
                        #print(f'[METRIC] Class {current_class} Group IDs: {group_ids[i]}, {group_ids[j]}')
                        stats_group_i = self.stats_per_class[current_class][group_ids[i]].get()
                        stats_group_j = self.stats_per_class[current_class][group_ids[j]].get()
                        
                        is_empty_i = (stats_group_i['tp'] + stats_group_i['fn'] + stats_group_i['fp'] + stats_group_i['tn']) == 0
                        is_empty_j = (stats_group_j['tp'] + stats_group_j['fn'] + stats_group_j['fp'] + stats_group_j['tn']) == 0
                        # Verifica se uno dei gruppi ha label==current_class
                        n_gi_tpr = stats_group_i["tp"] + stats_group_i["fn"]
                        n_gj_tpr = stats_group_j["tp"] + stats_group_j["fn"]

                        n_gi_fpr = stats_group_i["fp"] + stats_group_i["tn"]
                        n_gj_fpr = stats_group_j["fp"] + stats_group_j["tn"]
                        if (not is_empty_i) and (not is_empty_j):
                            if n_gi_tpr > 0 and n_gj_tpr > 0:
                                         
                                #print(f'[METRIC] Diff TPR: {abs(stats_group_i["tpr"] - stats_group_j["tpr"])}, Diff FPR: {abs(stats_group_i["fpr"] - stats_group_j["fpr"])}')
                                self.stats_per_class_group_diff_tpr[current_class].append(
                                    abs(stats_group_i['tpr'] - stats_group_j['tpr'])
                                )
                            
                            if n_gi_fpr > 0 and n_gj_fpr > 0:
                                #print(f'[METRIC] Diff FPR: {abs(stats_group_i["fpr"] - stats_group_j["fpr"])}')
                                self.stats_per_class_group_diff_fpr[current_class].append(
                                    abs(stats_group_i['fpr'] - stats_group_j['fpr'])
                                )
                             
                if len(self.stats_per_class_group_diff_tpr[current_class]) > 0  and len(self.stats_per_class_group_diff_fpr[current_class]) > 0: 
                    eod = torch.max(self._reduction(torch.tensor(self.stats_per_class_group_diff_tpr[current_class])),
                               self._reduction(torch.tensor(self.stats_per_class_group_diff_fpr[current_class])))
            
                    self.metrics_per_class.append(eod)
            #for current_class,result in zip(range(self.num_classes),self.metrics_per_class):
            #    print(f'[INFO METRIC] Equalized Odds for class {current_class} on group {self.group_name} = {result}')    
            return {
                    f'equalized_odds_{self.group_name}':self._reduction(
                        torch.tensor(self.metrics_per_class))
                    }
    def reset(self):
        super().reset()
        self.stats_per_group_diff_tpr = []
        self.metrics_per_class = []
        self.stats_per_group_diff_fpr = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff_tpr[current_class] = []
            self.stats_per_class_group_diff_fpr[current_class] = []
      
