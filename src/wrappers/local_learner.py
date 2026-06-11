import numpy as np
from .torch_nn_wrapper import TorchNNWrapper
import torch
import tqdm
from callbacks import EarlyStopping, ModelCheckpoint
import os
from entmax import entmax_bisect
from metrics import Performance,GroupFairnessMetric
import copy
from privacy.privacy_wrapper import PrivacyWrapper
from opacus.grad_sample import GradSampleModule

class EarlyStoppingException(Exception):
    pass

class LocalLearner(TorchNNWrapper):
    """
    LocalLearner is a class that implements a local learner with ALM optimization.
    Methods:
        compute_groups_cardinality(): Computes the cardinality of groups.
        _init_inequality_lambdas(): Initializes the inequality lambdas.
        _init_alm_parameters(): Initializes the ALM parameters.
        update_lambdas_inequality(constraints): Updates the inequality lambdas.
        update_lambdas_equality(constraints): Updates the equality lambdas.
        update_alm_parameters_and_metrics(update_alm=True, **kwargs): Updates the ALM parameters and computes metrics.
        compute_constraints(**kwargs): Computes the constraints.
        compute_score(**kwargs): Computes the score.
        compute_loss_fn(**kwargs): Computes the loss function.
        _compute_metrics(metrics, prefix='val', **kwargs): Computes metrics.
        _training_step(batch, batch_idx): Performs a training step.
        _train_eval_step(**kwargs): Performs a training evaluation step.
        _validation_step(**kwargs): Performs a validation step.
        set_constraints(inequality_constraints_fn_list, equality_constraints_fn_list, macro_constraints_list, inequality_lambdas, equality_lambdas): Sets the constraints.
        evaluate(model_dict, **kwargs): Evaluates the model.
        compute_val_kwargs(model_dict, use_training=False): Computes validation kwargs.
        compute_violations(val_kwargs, **kwargs): Computes the violations.
        fit(**kwargs): Fits the model.
    """
    def __init__(self, *args, **kwargs):
      
        super(LocalLearner, self).__init__(*args, **kwargs)
        self.epsilon = kwargs.get('epsilon')
        self.privacy_wrapper = PrivacyWrapper(epsilon=self.epsilon)
        self.id = kwargs.get('id','LagrangianWrapper')
        self.compute_only_score =kwargs.get('compute_only_score',False)
        self.optimizer_fn: callable = kwargs.get('optimizer_fn')
        self.lagrangian_checkpoints = kwargs.get('lagrangian_checkpoints', [])
        #self.training_group_name: str = kwargs.get('training_group_name')
        
        self.teacher_model = kwargs.get('teacher_model')
        #self.distillation_loss_fn:callable = kwargs.get('distillation_loss_fn')
        self.batch_objective_function = kwargs.get('batch_objective_fn')
        self.original_objective_fn:callable = kwargs.get('original_objective_fn')
        self.objective_fn: callable = kwargs.get('objective_fn')
        self.inequality_constraints_fn_list: list = kwargs.get('inequality_constraints')
        self.equality_constraints_fn_list: list = kwargs.get('equality_constraints')
        
        self.mu_max = kwargs.get('mu_max', 1e3)
        self.nu_max = kwargs.get('nu_max', 100)
        self.lambda_equality_max = kwargs.get('lambda_equality_max', 100)
        self.lambda_inequality_max = kwargs.get('lambda_inequality_max', 100)

        self.rho = kwargs.get('rho', 2)
        self.mu_0 = kwargs.get('mu_0', 2)
        self.damping_factor = kwargs.get('damping_factor', 1.0)  # Valore di damping per rallentare l'aggiornamento

        self.gamma_objective = kwargs.get('gamma_objective', 0.8)
        self.gamma_constraint = kwargs.get('gamma_constraint', 10)

        self.inequality_lambdas_0_value = kwargs.get('inequality_lambdas_0_value', 0.1)
        self.equality_lambdas_0_value = kwargs.get('equality_lambdas_0_value', 0.)
        self.objective_multiplier_0_value = kwargs.get('objective_multiplier_0_value', 1)
        self.macro_constraints_list= kwargs.get('macro_constraints_list')
        # Assicurati che tutti i tensori siano su device
        self.inequality_lambdas_0 = torch.ones(len(self.inequality_constraints_fn_list), device=self.device) * self.inequality_lambdas_0_value
        self.equality_lambdas_0 = torch.ones(len(self.equality_constraints_fn_list), device=self.device) * self.equality_lambdas_0_value
        self.objective_multiplier_0 = torch.tensor(self.objective_multiplier_0_value, device=self.device)
        self.lambda0_max_value = kwargs.get('lambda0_max_value', 0.1)
        self.target_groups = set()
        #self.compute_groups_cardinality()
        
        self.active_groups = {}
        self.teacher_model_list = []
        for constraint in self.inequality_constraints_fn_list:
            if constraint.group_name is not None:
                self.target_groups.add(constraint.group_name)
                if constraint.group_name not in self.active_groups:
                    self.active_groups[constraint.group_name] = []
                for c in constraint.target_groups:
                    if c not in self.active_groups[constraint.group_name]:
                        self.active_groups[constraint.group_name].append(c.item())
                  
        assert self.macro_constraints_list is not None, f'{self.macro_constraints_list} has to be provided'
        self.group_cardinality = None

        self.all_group_ids = kwargs.get('all_group_ids')

        self.teachers_kwargs = {'train':{},
                                'val':{}
                                }

        self.wasserstein_kwargs = {'train':{},
                                   'val':{}
                                  }
        self._init_alm_parameters()


        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)
    
    def set_wasserstein_teachers_kwargs(self,teachers_kwargs):
        #print('Setting Wasserstein kwargs:',teachers_kwargs)
        self.wasserstein_kwargs.update(teachers_kwargs)

    def set_teachers_kwargs(self,teachers_kwargs):
         #print('Setting Teachers kwargs:',teachers_kwargs)
         self.teachers_kwargs.update(teachers_kwargs)
         
    def compute_groups_cardinality(self):
        #print('Computing groups cardinality')
        #print(self.get_group_ids())
        groups_ids = next(iter(self.get_train_loader_eval()))['groups_ids_unique']
        groups = next(iter(self.get_train_loader_eval()))['groups']
        self.group_cardinality = {group_name: {} for group_name in self.target_groups} 
        self.max_cardinality = {group_name: 0 for group_name in self.target_groups}
        for group_name in self.target_groups:
            for group in groups_ids[group_name].unique():
                self.group_cardinality[group_name].update({group.item(): len(groups[group_name][groups[group_name] == group])})               
                if len(groups[group_name][groups[group_name] == group]) > self.max_cardinality[group_name]:
                    self.max_cardinality[group_name] = len(groups[group_name][groups[group_name] == group])
        print('Group cardinality:',self.group_cardinality)
        print('Target groups:',self.target_groups)
    def _init_inequality_lambdas(self):
        self.inequality_lambdas = torch.ones_like(self.inequality_lambdas_0, device=self.device) * self.inequality_lambdas_0_value
    
    def _init_alm_parameters(self):
        self._init_inequality_lambdas()
        self.equality_lambdas = self.equality_lambdas_0
        self.mu = self.mu_0
        self.objective_multiplier = self.objective_multiplier_0

    def update_lambdas_inequality(self, constraints):
       
        if constraints is None:
            return self.inequality_lambdas
        
        new_lambdas = torch.max(
            torch.ones_like(self.inequality_lambdas, device=self.device)*self.inequality_lambdas_0_value,
            self.inequality_lambdas + self.mu * torch.max(constraints,torch.zeros_like(constraints, 
                                                                                       device=self.device))
        )
       
        assert torch.all(new_lambdas >= self.inequality_lambdas_0_value), 'Negative Lagrange multipliers!'
        return new_lambdas

    def update_lambdas_equality(self, constraints):
        
        if constraints is None:
            return self.equality_lambdas
        new_lambdas = self.equality_lambdas + self.mu * constraints * self.damping_factor
        new_lambdas = torch.clamp(new_lambdas, min=self.equality_lambdas_0_value, max=self.lambda_equality_max)
    
        return new_lambdas

    
   
    
    def update_alm_parameters_and_metrics(self, update_alm=True, model=None, **kwargs):
        metrics = {}
        model = self.model if model is None else model
        if isinstance(model, GradSampleModule):
            base = model._module
            model = copy.deepcopy(base)

        model.eval()
        
        if not isinstance(model, GradSampleModule):
            model.to(self.device)

       
        with torch.no_grad():
            
            val_loader = self.data_module.val_loader(batch_size=None)
            val_kwargs = self._compute_kwargs_in_batches(val_loader, model,use_entmax=False,use_training=False)
            train_loader = self.data_module.train_loader_eval(batch_size=None)
            train_kwargs = self._compute_kwargs_in_batches(train_loader, model,use_entmax=False,use_training=True)          
            kwargs = {}   
            kwargs['val_kwargs'] = val_kwargs
            kwargs['train_kwargs'] = train_kwargs
            

            inequality_constraints = train_kwargs['inequality_constraints']
            equality_constraints = train_kwargs['equality_constraints']
            
            if update_alm:
                self._apply_early_stopping(inequality_constraints, equality_constraints)

               
                if inequality_constraints is not None:
                    inequality_constraints = inequality_constraints * self.inequality_mask

                if equality_constraints is not None:
                    equality_constraints = equality_constraints * self.equality_mask

               
                if inequality_constraints is not None:
                    self.inequality_lambdas = self.update_lambdas_inequality(inequality_constraints)
                if equality_constraints is not None:
                    self.equality_lambdas = self.update_lambdas_equality(equality_constraints)

        
            val_score = self.compute_score(**val_kwargs)    
            metrics['val_constraints_score'] = val_score
            val_loss = self.compute_loss_fn(**val_kwargs)
            metrics['val_loss'] = val_loss
            if not self.compute_only_score:
                metrics.update(self._compute_metrics(self.metrics,  prefix='val', **val_kwargs))
            return metrics
        
    def _apply_early_stopping(self, inequality_constraints, equality_constraints):
        n_inequality_constraints = len(self.inequality_constraints_fn_list)
        self.inequality_mask = torch.ones_like(self.inequality_lambdas, device=self.device)
        self.equality_mask = torch.ones_like(self.equality_lambdas, device=self.device)
        cached_scores = {}  
        for i, checkpoint in enumerate(self.lagrangian_checkpoints):
            if isinstance(checkpoint, EarlyStopping):
                if i < n_inequality_constraints:
                    if i not in cached_scores:
                        cached_scores[i] = {'score': inequality_constraints[i]}
                    update, _ = checkpoint(metrics=cached_scores[i])
                    if not update:
                        self.inequality_mask[i] = 0  # Ferma l'aggiornamento per questo vincolo
                    else:
                        checkpoint.reset(keep_best=True)
                else:
                    eq_index = i - n_inequality_constraints
                    if eq_index not in cached_scores:
                        cached_scores[eq_index] = {'score': equality_constraints[eq_index]}
                    update, _ = checkpoint(metrics=cached_scores[eq_index])
                    if not update:
                        self.equality_mask[eq_index] = 0  # Ferma l'aggiornamento per questo vincolo
                    else:
                        checkpoint.reset()
    
    def compute_constraints(self, **kwargs):
        device = self.device
        if len(self.inequality_constraints_fn_list)>0:
            inequality_constraints = torch.stack(
                [torch.clamp(constraint_fn(**kwargs),min=0).to(device) for constraint_fn in self.inequality_constraints_fn_list], dim=0
            ).to(device)
        else:
            inequality_constraints = torch.tensor([], device=device)

        if len(self.equality_constraints_fn_list)>0:
            equality_constraints = torch.stack(
                [constraint_fn(**kwargs) for constraint_fn in self.equality_constraints_fn_list], dim=0
            ).to(device)
        else:
            equality_constraints = torch.tensor([], device=device)

        return inequality_constraints, equality_constraints


    def compute_score(self, **kwargs):
        objective_function = kwargs.get('original_objective_function')
        inequality_constraints = kwargs.get('inequality_constraints')
        equality_constraints = kwargs.get('equality_constraints')
        #print('Original_objective function',objective_function.item())
        # Inizializza lo score con la funzione obiettivo
       
        score = objective_function.clone().to(self.device)
        
        #print(f'[INFO] Score before constraints: {score.item()}')
        # Inizializza una variabile per il conteggio delle violazioni dei vincoli
        total_penalty = torch.tensor(0.0, device=self.device)
        # Penalità per vincoli di disuguaglianza
        if len(inequality_constraints) > 0:
            for _,macro_constraint in enumerate(self.macro_constraints_list):
                if len(macro_constraint) > 0:
                    inequality_penalty = torch.max(torch.clamp(inequality_constraints[macro_constraint], min=0))   
                    total_penalty += inequality_penalty*self.gamma_constraint
                    #print(f'[INFO] Inequality Penalty for macro constraint {macro_constraint}: {inequality_penalty.item()}')
        
        if len(equality_constraints)> 0:
            equality_penalty = torch.max(torch.abs(equality_constraints))
            total_penalty += equality_penalty * self.gamma_constraint
        score -= total_penalty
        
        return score

       
    def compute_loss_fn(self, **kwargs):
        objective_function = kwargs['objective_function']
        batch_objective_function = kwargs['batch_objective_function']
        equality_constraints = kwargs.get('equality_constraints')
        inequality_constraints = kwargs.get('inequality_constraints')
        group_ids = kwargs.get('group_ids')

        assert group_ids is not None, 'Group ids must be provided'
        
        loss = objective_function.clone()
        group_losses = []

        for group_name in self.target_groups:
            group_tensor = group_ids[group_name]  # shape [B]
            unique_group_ids = torch.unique(group_tensor)

            total_weight = 0.0
            for group_id in unique_group_ids:
                mask = group_tensor == group_id
                if mask.sum() > 0:
                    group_loss = batch_objective_function[mask].mean()
                    weight = 1.0 - mask.sum().item() / batch_objective_function.shape[0]
                    group_losses.append(weight * group_loss)
                    total_weight += weight

        if len(group_losses) > 0:
            loss += torch.stack(group_losses).sum() / total_weight  # oppure normalizza: `.sum() / total_weight`

        # === Equality constraints
        if equality_constraints is not None and len(self.equality_constraints_fn_list) > 0:
            equality_penalty = torch.mean(torch.abs(equality_constraints)) * self.mu
            equality_lagrange = (self.equality_lambdas * equality_constraints).sum()
            loss += equality_penalty + equality_lagrange

        # === Inequality constraints
        if inequality_constraints is not None and len(self.inequality_constraints_fn_list) > 0:
            if torch.any(self.inequality_lambdas > 0):
                penalty = torch.sum(torch.clamp(inequality_constraints, min=0))
                lambdas_updated = torch.clamp(self.inequality_lambdas + self.mu * inequality_constraints, min=0)
                lagrange = (lambdas_updated.pow(2) - self.inequality_lambdas.pow(2)).sum() / (2 * self.mu)
                loss += penalty + lagrange

        if torch.isnan(loss).any():
            raise ValueError("NaN trovato nella loss!")

        return loss



    
    def _compute_kwargs_in_batches(self, loader, model, use_entmax=True, use_training=False):
        all_logits = []
        all_labels = []
        all_indices = []
        all_positive_masks = []
        if self.all_group_ids is not None:
            all_group_ids = {group_name: [] for group_name in self.all_group_ids.keys()}
            all_group_ids_list = {group_name: [] for group_name in self.all_group_ids.keys()}
            #print("[INFO] Collected all group IDs and their corresponding lists.")
            
        else:
            if self.target_groups:
                all_group_ids = {group_name: [] for group_name in loader.dataset[0]['groups'].keys() if group_name in self.target_groups}
                all_group_ids_list = {group_name: [] for group_name in loader.dataset[0]['groups_ids_list'].keys() if group_name in self.target_groups}
            else:
                print("[INFO] No target groups specified, skipping group collection.")
                all_group_ids = {}
                all_group_ids_list = {}

        for batch in loader:
            inputs = batch['data'].float().to(self.device)
            outputs = model(inputs)

            all_logits.append(outputs)
            all_labels.append(batch['labels'].to(self.device))
            all_indices.append(batch['index'])
            all_positive_masks.append(batch['positive_mask'].to(self.device))

            if self.target_groups or self.all_group_ids is not None:
                for group_name in all_group_ids:
                    all_group_ids[group_name].append(batch['groups'][group_name].to(self.device))
                for group_name in all_group_ids_list:
                    all_group_ids_list[group_name].append(batch['groups_ids_list'][group_name].to(self.device))

        final_logits = torch.cat(all_logits, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        final_indices = torch.cat(all_indices, dim=0)
        final_positive_masks = torch.cat(all_positive_masks, dim=0)

        if self.all_group_ids is not None:
            final_group_ids = {g: torch.cat(all_group_ids[g], dim=0) for g in all_group_ids} 
            final_group_ids_list = {g: torch.cat(all_group_ids_list[g], dim=0) for g in all_group_ids_list}
        else:
            final_group_ids = {g: torch.cat(all_group_ids[g], dim=0) for g in all_group_ids} if self.target_groups else {}
            final_group_ids_list = {g: torch.cat(all_group_ids_list[g], dim=0) for g in all_group_ids_list} if self.target_groups else {}

        batch_dict = {
            'logits': final_logits,
            'labels': final_labels,
            'groups': final_group_ids,
            'groups_ids_list': final_group_ids_list,
            'positive_mask': final_positive_masks,
            'index': final_indices,
        }

        return self._compute_kwargs(batch_dict, final_logits, use_entmax=use_entmax, use_training=use_training)

    def _compute_kwargs(self, batch, outputs, use_entmax=True, use_training=False):
        device = self.device

        # === Gruppi (fairness): opzionali ===
        if self.all_group_ids is not None:
            group_ids = {g: batch['groups'][g].to(device) for g in self.all_group_ids.keys()} 
            group_ids_list = {g: batch['groups_ids_list'][g].to(device) for g in self.all_group_ids.keys()} 
        else:
            group_ids = {g: batch['groups'][g].to(device) for g in batch.get('groups', {})} if self.target_groups else {}
            group_ids_list = {g: batch['groups_ids_list'][g].to(device) for g in batch.get('groups_ids_list', {})} if self.target_groups else {}

        # === Mask e label: obbligatori ===
        positive_mask = batch.get('positive_mask')
        if positive_mask is None:
            raise ValueError("'positive_mask' non è presente nel batch.")
        positive_mask = positive_mask.to(device)

        labels = batch.get('labels')
        if labels is None:
            raise ValueError("'labels' non è presente nel batch.")
        labels = labels.to(device)

        # === Predizioni ===
        predictions = torch.argmax(outputs, dim=-1)
      
        probabilities = (
            entmax_bisect(outputs*20, alpha=1.5, dim=-1)
            if use_entmax else
            torch.nn.functional.one_hot(predictions, num_classes=outputs.size(-1)).float()
        )
        #print(f'[LL] Outputs head: {outputs[:5,:]}')
        #print(f'[LL] Probabilities head: {probabilities[:5,:]}')
        output_distribution = torch.nn.functional.softmax(outputs, dim=-1)
        #print(f"[INFO] Output distribution: {output_distribution[:5,:]}, Predictions shape: {predictions[:5]}")
        # === Base kwargs ===
        kwargs = {
            'group_ids': group_ids,
            'group_ids_list': group_ids_list,
            'group_masks': group_ids,
            'positive_mask': positive_mask,
            'logits': outputs,
            'labels': labels,
            'probabilities': probabilities,
            'predictions': predictions,
            'output_distribution': output_distribution,
        }
        #print(f'Using training: {use_training}')
        indices = batch.get('index')
        if indices is not None and indices.numel() > 0:
            max_index = indices.max().item()
           
            teacher_kwargs = self.teachers_kwargs['train'] if use_training else self.teachers_kwargs['val']
            for k, v in teacher_kwargs.items():
                #print(f"[INFO] Processing batch with max index: {max_index}")
                #print(f"[DEBUG] shape for key '{k}': {v.shape if isinstance(v, torch.Tensor) else 'N/A'}")
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] > max_index:
                    kwargs[k] = v[:, indices, ...]
        
        empty_shape = (0, outputs.shape[0], outputs.shape[1])  # es: (0, batch_size, num_classes)
        for key in [
            'teacher_logits_list',
            'teacher_probabilities',
            'teacher_predictions_list',
            'teacher_softmax_list',
        ]:
            if key not in kwargs:
                #print(f"[INFO KWARGS] Initializing empty tensor for key: {key} with shape {empty_shape}")
                kwargs[key] = torch.empty(empty_shape, device=device)
        
        #print(f"[INFO KWARGS] Teacher probabilities shape in kwargs: {kwargs['teacher_probabilities'].shape}")
        #print(f"[INFO] Keys in kwargs: {list(kwargs.keys())}")
        # === Funzioni surrogate e constraint ===
        kwargs['objective_function'] = self.objective_fn(**kwargs)
        kwargs['original_objective_function'] = self.original_objective_fn(**kwargs)
        kwargs['batch_objective_function'] = self.batch_objective_function(**kwargs)
        kwargs['inequality_constraints'], kwargs['equality_constraints'] = self.compute_constraints(**kwargs)
        #print(f'Kwargs keys: {list(kwargs.keys())}')
        return kwargs

    def _compute_metrics(self,metrics,prefix='val',**kwargs):
        group_ids = kwargs['group_ids']
        y_pred = kwargs['predictions']
        y_true = kwargs['labels']

        tmp_result = {}
        final_result = {}
        
                
        for metric in metrics:
            metric.reset()
            if issubclass(metric.__class__,GroupFairnessMetric):
                            group_ids_detached = {group_name:group_ids[group_name].detach().cpu() for group_name in group_ids.keys()}
                            metric.calculate(y_pred.detach().cpu(),
                                            y_true.detach().cpu(),
                                            group_ids_detached)
                           
            elif isinstance(metric,Performance):
                metric.calculate(y_pred.detach().cpu(),
                                 y_true.detach().cpu())
            else:
                raise ValueError(f"{metric} is an invalid metric")
            tmp_result.update(metric.get())
            
      
        for key, value in tmp_result.items():
            if prefix == '':
                final_result[key] = value
            else:
                final_result[f'{prefix}_{key}'] = value
        return final_result 

    def _training_step(self, batch, batch_idx):
        self.dp_model.train()
        inputs = batch['data'].float().to(self.device)
        labels = batch["labels"].to(self.device).long()
        self.dp_optimizer.zero_grad()
        outputs = self.dp_model(inputs)
        kwargs = self._compute_kwargs(batch, outputs,use_entmax=True,use_training=True)
       
        loss = self.compute_loss_fn(**kwargs)
        #loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        if torch.isnan(loss).any():
            raise ValueError('Loss contiene NaN!')
      
        loss.backward()
        if self.epsilon <= 0.0:
            torch.nn.utils.clip_grad_norm_(self.dp_model.parameters(), max_norm=5.0)
        self.dp_optimizer.step()

        return loss.item()

    def _train_eval_step(self,**kwargs):
        self.dp_model.eval()
        with torch.no_grad():
            train_kwargs = kwargs['train_kwargs']
            outputs = train_kwargs['logits']
            targets = train_kwargs['labels']
            loss = self.compute_loss_fn(**train_kwargs)
            predictions = torch.argmax(outputs, dim=1)

            return loss.item(), outputs, targets, predictions
            
    def _validation_step(self,**kwargs):
        self.dp_model.eval()
        with torch.no_grad():
            val_kwargs = kwargs['val_kwargs']
            outputs = val_kwargs['logits']
            targets = val_kwargs['labels']
            loss = self.compute_loss_fn(**val_kwargs)
            predictions = torch.argmax(outputs, dim=1)

            return loss.item(), outputs, targets, predictions

    def set_constraints(self, inequality_constraints_fn_list, equality_constraints_fn_list,macro_constraints_list,inequality_lambdas, equality_lambdas):
        self.inequality_constraints_fn_list = inequality_constraints_fn_list
        self.equality_constraints_fn_list = equality_constraints_fn_list
        self.macro_constraints_list = macro_constraints_list
        self.inequality_lambdas=inequality_lambdas
        self.equality_lambdas=equality_lambdas

    def evaluate(self, model_dict, **kwargs):
        eval_model = self._get_base_model_for_eval()
        eval_model.load_state_dict(model_dict)
        eval_model.to(self.device)
        eval_model.eval()
        metrics = self.update_alm_parameters_and_metrics(update_alm=False, model=eval_model)
        return metrics
    
    def compute_val_kwargs(self, model_dict, use_training=False):
        eval_model = self._get_base_model_for_eval()
        eval_model.load_state_dict(model_dict)
        eval_model.to(self.device)
        eval_model.eval()

        if use_training:
            loader = self.data_module.train_loader_eval(batch_size=None)
        else:
            loader = self.data_module.val_loader(batch_size=None)

        kwargs = self._compute_kwargs_in_batches(loader, eval_model, use_entmax=False,use_training=use_training)
        return kwargs

    
    def compute_violations(self, val_kwargs, **kwargs):
        inequality_constraints, equality_constraints = self.compute_constraints(**val_kwargs)
        results = {}

        violations = {k: None for k, _ in enumerate(self.macro_constraints_list)}
        violations_per_group_list = {}
        violations_per_group = {}

        for key, value_dict in self.group_cardinality.items():
            violations_per_group_list[key] = {k: [] for k in value_dict.keys()}

        if inequality_constraints is None:
            # Nessun vincolo: restituisci dizionari vuoti o zeri coerenti
            results['violations_per_group'] = {
                key: {k: 0.0 for k in value_dict.keys()}
                for key, value_dict in self.group_cardinality.items()
            }
            results['inequality_constraints_violations'] = []
            results['macro_constraints_violations'] = [
                [] for _ in self.macro_constraints_list
            ]
            return results

        # === Elaborazione normale se i vincoli sono presenti ===
        for i, constraint_violation in enumerate(inequality_constraints):
            constraint = self.inequality_constraints_fn_list[i]
            target_groups = constraint.target_groups
            group_name = constraint.group_name

            if group_name is not None:
                for group in target_groups:
                    violations_per_group_list[group_name][group.item()].append(constraint_violation)

        for key, value_dict in violations_per_group_list.items():
            try:
                violations_per_group[key] = {
                    k: torch.stack(v).max().item() if v else 0.0
                    for k, v in value_dict.items()
                }
            except RuntimeError:
                violations_per_group[key] = {k: 0.0 for k in value_dict.keys()}

        results['violations_per_group'] = copy.deepcopy(violations_per_group)

        for i, macro_constraint in enumerate(self.macro_constraints_list):
            violations[i] = inequality_constraints[macro_constraint].detach().cpu().numpy()

        results['inequality_constraints_violations'] = inequality_constraints.detach().cpu().numpy()

        macro_constraints_violation = copy.deepcopy(violations)
        for i, macro_constraint in enumerate(self.macro_constraints_list):
            if len(macro_constraints_violation[i]) > 0:
                macro_constraints_violation[i] = [macro_constraints_violation[i].max()]
            else:
                macro_constraints_violation[i] = []

        results['macro_constraints_violations'] = copy.deepcopy(macro_constraints_violation)

        return results

    def _get_base_model_for_eval(self):
        if isinstance(self.model, GradSampleModule):
            base = self.model._module
        else:
            base = self.model
        return copy.deepcopy(base)
        
    
    def fit(self, **kwargs):
        print(f'[{self.id}]:Number of inequality constraints:', len(self.inequality_constraints_fn_list))
        print(f'Macro constraints:', self.macro_constraints_list)

        num_epochs = kwargs.get('num_epochs', -1)
        disable_log = kwargs.get('disable_log', False)
        evaluate_best_model = kwargs.get('evaluate_best_model', True)
        n_rounds = self.num_epochs if num_epochs == -1 else num_epochs

        self.teacher_model_list = kwargs.get('teacher_model_list', [])
        if len(self.teacher_model_list) > 0:
            print(f'[{self.id}]: Using {len(self.teacher_model_list)} teacher models for distillation')
            self.teachers_kwargs['train'] = self.query_teachers(use_training=True)
            self.teachers_kwargs['val'] = self.query_teachers(use_training=False)
        


        start_model_dict = kwargs.get('start_model_dict')

        # -------------------------------------------------------
        # 1) MODELLO PULITO DI RIFERIMENTO (self.base_model)
        # -------------------------------------------------------
        # Non voglio che questo sia MAI un GradSampleModule
        if isinstance(self.model, GradSampleModule):
            base = self.model._module
        else:
            base = self.model

        # Creo una copia "pulita" che userò per ALM / valutazione
        self.base_model = copy.deepcopy(base)

        if start_model_dict is not None:
            self.base_model.load_state_dict(copy.deepcopy(start_model_dict))

        self.base_model.to(self.device)
        self.base_model.train()

        # -------------------------------------------------------
        # 2) MODELLO PER IL TRAINING DP (self.dp_model)
        # -------------------------------------------------------
        # Questo è il SOLO modello che passerà dentro Opacus
        model_for_dp = copy.deepcopy(self.base_model)

        self.optimizer = self.optimizer_fn(model_for_dp.parameters())
       
        if self.epsilon is None or self.epsilon <= 0:
            self.dp_model = self.base_model
            self.dp_optimizer = self.optimizer_fn(self.dp_model.parameters())
            self.train_loader = self.data_module.train_loader()
        else:

            self.privacy_wrapper.reset()
            criterion = torch.nn.CrossEntropyLoss()
            self.dp_model, self.dp_optimizer, self.train_loader = self.privacy_wrapper.apply_privacy(
            model=model_for_dp,
            optimizer=self.optimizer,
            data_loader=self.get_train_loader(),
            epochs=n_rounds,
            criterion=criterion,
        )

        # IMPORTANTE: niente .to(self.device) qui, Opacus l'ha già gestito
        self.dp_model.train()
        #print('Model and optimizer wrapped with privacy')
        #for name, param in self.dp_model.named_parameters():
        #    print(name, param.requires_grad)

        # Debug opzionale: controlla che tutti abbiano _forward_counter
        #for name, p in self.dp_model.named_parameters():
        #    if not hasattr(p, "_forward_counter"):
        #        print("[DEBUG] NO _forward_counter at init on:", name)

        try:
            for epoch in tqdm.tqdm(range(n_rounds), desc=f'Epoch 0/{n_rounds}', total=n_rounds, unit='epoch'):
                if self.epsilon <= 0.0:
                    train_loader = self.data_module.train_loader()
                else: 
                    train_loader = self.train_loader
                batch_iterator = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_rounds}', leave=False)

                for batch_idx, batch in enumerate(batch_iterator):
                    self._training_step(batch, batch_idx)

                # -------------------------------------------------------
                # 3) DOPO OGNI EPOCH: VALUTAZIONE SU MODELLO PULITO
                # -------------------------------------------------------
                with torch.no_grad():
                    # estraggo pesi "plain" dal DP model
                    if isinstance(self.dp_model, GradSampleModule):
                        base_trained = self.dp_model._module
                    else:
                        base_trained = self.dp_model

                    # aggiorno la copia pulita usata per ALM
                    self.base_model.load_state_dict(base_trained.state_dict())
                    self.base_model.to(self.device)
                    self.base_model.eval()

                    metrics = self.update_alm_parameters_and_metrics(
                        update_alm=True,
                        model=self.base_model,   # <-- MAI dp_model qui
                        **kwargs,
                    )

                    # Early stopping + checkpoint
                    for checkpoint in self.checkpoints:
                        if self.epsilon <= 0:
                            if isinstance(checkpoint, EarlyStopping):
                                stop, counter = checkpoint(metrics=metrics)
                                metrics['early_stopping'] = counter
                                if stop:
                                    raise EarlyStoppingException
                        if isinstance(checkpoint, ModelCheckpoint):
                            model_checkpoint = checkpoint(save_fn=self.save, metrics=metrics)
                            metrics['model_checkpoint'] = 1 if model_checkpoint else 0

                    if not disable_log:
                        self.logger.log(metrics)

                    batch_iterator.set_description(f'Epoch {epoch+1}/{n_rounds}')
        except EarlyStoppingException:
            pass

        # -------------------------------------------------------
        # 4) SE CI SONO CHECKPOINT, RICARICO SUL MODELLO PULITO
        # -------------------------------------------------------
        for checkpoint in self.checkpoints:
            if isinstance(checkpoint, ModelCheckpoint):
                if os.path.exists(checkpoint.get_model_path()):
                    self.load(checkpoint.get_model_path())  # questo deve aggiornare self.model o self.base_model

        # -------------------------------------------------------
        # 5) VALUTAZIONE FINALE (ANCORA SU MODELLO PULITO)
        # -------------------------------------------------------
        if evaluate_best_model:
            self.base_model.eval()
            metrics = self.update_alm_parameters_and_metrics(
                update_alm=False,
                model=self.base_model,   # ancora: MAI dp_model
                **kwargs,
            )
            final_metrics = {f'final_{name}': value for name, value in metrics.items()}
            if not disable_log:
                self.logger.log(final_metrics)

        # -------------------------------------------------------
        # 6) COPIO I PESI FINALI SUL MODELLO UFFICIALE self.model
        # -------------------------------------------------------
        # self.model deve restare sempre un modello "normale" (no GradSampleModule)
        self.model.load_state_dict(self.base_model.state_dict())

        return copy.deepcopy(self.model.state_dict())

            
    def save(self, path):
        """
        Salva SEMPRE un modello 'pulito' (no GradSampleModule).
        Se esiste self.base_model (usato nel training DP), salva quello,
        altrimenti ripiega su self.model.
        """
        if hasattr(self, "base_model"):
            state_dict = self.base_model.state_dict()
        else:
            state_dict = self.model.state_dict()

        torch.save(state_dict, path)
        

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)

        # carico sul modello 'ufficiale'
        self.model.load_state_dict(state_dict)

        # se sto usando anche base_model (DP), lo allineo
        if hasattr(self, "base_model"):
            self.base_model.load_state_dict(state_dict)

    def query_teachers(self, use_training=False, **kwargs):
        use_entmax = kwargs.get('use_entmax', True)

        all_teacher_logits = []
        teachers_probabilities_list = []
        teachers_predictions_list = []
        teachers_softmax_list = []
        teachers_logits_list = []
        if self.teacher_model_list is not None and len(self.teacher_model_list) > 0:
            if use_training:
                loader = self.data_module.train_loader_eval()
            else:
                loader = self.data_module.val_loader()
            
            for teacher_model_dict in self.teacher_model_list:
                teacher_model = copy.deepcopy(self.model)
                teacher_model.load_state_dict(copy.deepcopy(teacher_model_dict))
                teacher_model.to(self.device)
                teacher_model.eval()

                teacher_outputs_list = []
                for batch in loader:
                    inputs = batch['data'].float().to(self.device)
                    with torch.no_grad():
                        outputs = teacher_model(inputs) *20
                    teacher_outputs_list.append(outputs)
                    all_teacher_logits.append(outputs)

                if len(teacher_outputs_list) > 0:
                    stacked_outputs = torch.cat(teacher_outputs_list, dim=0)
                else:
                    stacked_outputs = torch.tensor([], device=self.device)

                if stacked_outputs.numel() > 0:
                    teacher_softmax = torch.nn.functional.softmax(stacked_outputs / 1.0, dim=-1)
                    teacher_predictions = torch.argmax(stacked_outputs, dim=-1)

                    if use_entmax:
                        teacher_probabilities = entmax_bisect(stacked_outputs, alpha=1.5, dim=-1)
                    else:
                        teacher_probabilities = torch.nn.functional.one_hot(
                            teacher_predictions, num_classes=stacked_outputs.size(-1)
                        ).float()

                    if torch.isnan(teacher_probabilities).any():
                        raise ValueError("Teacher Probabilities contiene NaN!")

                    teachers_probabilities_list.append(teacher_probabilities)
                    teachers_predictions_list.append(teacher_predictions)
                    teachers_softmax_list.append(teacher_softmax)
                    teachers_logits_list.append(stacked_outputs)

        if len(teachers_probabilities_list)> 0:
            teacher_probabilities = torch.stack(teachers_probabilities_list, dim=0)
            teacher_predictions = torch.stack(teachers_predictions_list, dim=0)
            teacher_softmax = torch.stack(teachers_softmax_list, dim=0)
            teacher_logits = torch.stack(teachers_logits_list, dim=0)
        else:
            teacher_probabilities = torch.tensor([], device=self.device)
            teacher_predictions = torch.tensor([], device=self.device)
            teacher_softmax = torch.tensor([], device=self.device)
            teacher_logits = torch.tensor([], device=self.device)
        
        result = {
            'teacher_probabilities': teacher_probabilities,
            'teacher_softmax_list': teacher_softmax,
            'teacher_logits_list': teacher_logits,
            'teacher_predictions_list': teacher_predictions,
        }
        #print(f'[QUERY TEACHER] Result for using training={use_training}:')
        #print(f"[QUERY TEACHER] Queried {len(self.teacher_model_list)} teacher models.")
        #print(f"[QUERY TEACHER] Teacher probabilities shape: {teacher_probabilities.shape}")
        return result