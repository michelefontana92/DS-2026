from wrappers import LocalLearner
from dataclasses import dataclass
from callbacks import EarlyStopping, ModelCheckpoint
from surrogates import SurrogateFactory
import torch
import numpy as np
import copy

@dataclass
class SubProblemConfig:
    id: int
    inequality_constraints: list
    equality_constraints: list
    macro_constraints: list
    checkpoints_config: dict
    options: dict
    num_constraints: int
    compute_only_score: bool
    epsilon: float=0.0

    def _compute_active_groups(self):
        self.active_groups = {}
        for constraint in self.inequality_constraints:
            if constraint.target_groups is not None:
                self.active_groups[constraint.group_name] = set()

        for constraint in self.inequality_constraints:
            if constraint.target_groups is not None:
                for group in constraint.target_groups:
                    self.active_groups[constraint.group_name].add(group.item())
       
        
    def __post_init__(self):
        self.reset()
       
    def reset(self):
        self.current_inequality_constraints = self.inequality_constraints
        self.current_macro_constraints = self.macro_constraints
        self.current_num_constraints = self.num_constraints
        self._init_checkpoints()
        
    def _init_checkpoints(self):
        self.checkpoints = [
                EarlyStopping(patience=5, 
                            monitor='val_constraints_score', 
                            mode='max'),
                ModelCheckpoint(save_dir=f"{self.checkpoints_config['checkpoint_dir']}/subproblem_{self.id}", 
                                save_name=f"{self.checkpoints_config['checkpoint_name']}", 
                                monitor='val_constraints_score', 
                                mode='max')
            ]
        
        self.lagrangian_checkpoints = [EarlyStopping(patience=2, 
                            monitor='score', 
                            mode='min') for _ in range(len(self.current_inequality_constraints))]
    
    def add_local_proximity_constraint(self,teacher_idx,group_name,group_id,delta,new_macro_constraint):
        local_constraint = SurrogateFactory.create(name=f'wasserstein', 
                                                    surrogate_name=f'wasserstein', 
                                                    surrogate_weight=1,  
                                                    group_name=group_name, 
                                                    use_local_distance=True,
                                                    lower_bound=delta, 
                                                    teacher_idx=teacher_idx,
                                                    target_groups=torch.tensor(group_id) if isinstance(group_id,list) else torch.tensor([group_id]))
        if new_macro_constraint:
            self.current_macro_constraints.append([self.current_num_constraints])
        else:
            self.current_macro_constraints[-1].append(self.current_num_constraints)
        self.current_inequality_constraints.append(local_constraint)        
        self.current_num_constraints += 1
    
    
    def set_alm(self):
        inequality_lambdas = self.instance.inequality_lambdas 
        additional_values = torch.full(
            (len(self.current_inequality_constraints) - len(inequality_lambdas),), self.instance.inequality_lambdas_0_value
            ).to(inequality_lambdas.device)
        inequality_lambdas = torch.cat([inequality_lambdas, additional_values])
        self.instance.inequality_lambdas = inequality_lambdas
        self.instance.inequality_constraints_fn_list = self.current_inequality_constraints
        self.instance.macro_constraints_list = self.current_macro_constraints
        self._init_checkpoints()
        self.instance.checkpoints = self.checkpoints
        self.instance.lagrangian_checkpoints = self.lagrangian_checkpoints

    def instanciate(self,model):
        self._init_checkpoints()
        config = self.options
        config['inequality_constraints'] = self.current_inequality_constraints
        config['equality_constraints'] = self.equality_constraints
        config['lagrangian_checkpoints'] = self.lagrangian_checkpoints
        config['macro_constraints_list'] = self.current_macro_constraints
        config['checkpoints'] = self.checkpoints
        config['compute_only_score'] = self.compute_only_score
        config['id'] = f'Subproblem {self.id}'
        config['epsilon'] = self.epsilon
        self.instance = LocalLearner(model=copy.deepcopy(model),**config)
    
