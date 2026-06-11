from .base_builder import Base_Builder
from metrics import MetricsFactory
from surrogates import SurrogateFactory
from wrappers import OrchestratorWrapper
from dataloaders import DataModule
from torch.optim import Adam
from functools import partial
from callbacks import EarlyStopping, ModelCheckpoint
from loggers import WandbLogger
from torch.nn import CrossEntropyLoss
import copy
import wandb
import pprint
import torch 

class FairLabBuilder(Base_Builder):
    def _assign_resources(self):
        self.num_cpus =  1
        self.num_gpus = 1
         
    def compute_group_cardinality(self,group_name,sensitive_attributes):
        for name,group_dict in sensitive_attributes:
            if name == group_name:
                total = 1
                for key in group_dict.keys():
                    total *= len(group_dict[key])
                return total 
        raise KeyError(f'Group {group_name} not found in sensitive attributes') 
    
    def __init__(self,**kwargs):
        super(FairLabBuilder,self).__init__(**kwargs)
        #print('\n\nInitializing FairLab Builder\n\n')
        self.gpu_devices = kwargs.get('gpu_devices',[0])
        self._assign_resources()
        self.id = kwargs.get('id')
        self.run_dict = kwargs.get('run_dict')
        self.common_client_params  = self._get_common_params(**kwargs)
        self.experiment_name = kwargs.get('experiment_name')
        self.algorithm = kwargs.get('algorithm', 'fairlab')
        self.start_index = kwargs.get('start_index',1)
        self.wrapper = self._build_client(self.id, **kwargs)
        
        
        self.eval_mode = kwargs.get('eval_mode',False)
        self.eval_dir = kwargs.get('checkpoint_path')
        self.eval_prefix = kwargs.get('eval_prefix')
        
    def _get_common_params(self,**kwargs):
        common_params = {}
        common_params['epsilon'] = kwargs.get('epsilon')
        common_params['metrics_list'] = kwargs.get('metrics_list')
        common_params['groups_list'] = kwargs.get('groups_list')
        common_params['threshold_list'] = kwargs.get('threshold_list')
        common_params['lr'] = kwargs.get('lr', 1e-4)
        common_params['loss'] = partial(CrossEntropyLoss)
        common_params['num_lagrangian_epochs'] = kwargs.get('num_lagrangian_epochs', 1)
        common_params['batch_size'] = kwargs.get('batch_size', 128)
        common_params['project_name'] = kwargs.get('project_name')
        common_params['checkpoint_dir'] = kwargs.get('checkpoint_dir', f'checkpoints/{common_params["project_name"]}')
        
        common_params['verbose'] = kwargs.get('verbose', False)
        common_params['optimizer_fn'] = partial(Adam, lr=common_params['lr'])
        
        common_params['monitor'] = kwargs.get('monitor', 'val_constraints_score')
        common_params['mode'] = kwargs.get('mode', 'max')
        
        
        common_params['log_model'] = kwargs.get('log_model', False)
        
        common_params['num_global_iterations'] = kwargs.get('num_global_iterations')
        common_params['num_local_iterations'] = kwargs.get('num_local_iterations')
      
        
        common_params['performance_constraint'] = kwargs.get('performance_constraint')
        common_params['delta'] = kwargs.get('delta', 0.2)
        common_params['max_constraints_in_subproblem'] = kwargs.get('max_constraints_in_subproblem')
        common_params['global_patience'] = kwargs.get('global_patience')
        common_params['local_patience'] = kwargs.get('local_patience')
        common_params['num_classes'] = self.run_dict['num_classes']
        self.num_classes = common_params['num_classes']
        print('Number of classes:', self.num_classes)
        print('Groups: ', common_params['groups_list'])
        # Callbacks
        
        # Metriche
        common_params['metrics'] = [MetricsFactory().create_metric('performance',num_classes=common_params['num_classes'])]

        # Funzione obiettivo e vincoli
        common_params['objective_function'] = SurrogateFactory.create(name='performance', surrogate_name='cross_entropy', weight=1, average='weighted',num_classes=common_params['num_classes'])
        common_params['batch_objective_function'] = SurrogateFactory.create(name='performance_batch', surrogate_name='cross_entropy', weight=1, average='weighted',num_classes=common_params['num_classes'])
        if common_params['num_classes'] > 2:
            common_params['original_objective_fn'] = SurrogateFactory.create(name='multiclass_f1', surrogate_name='multiclass_f1', weight=1, average='weighted',num_classes=common_params['num_classes'],mode='max')
        else:
            common_params['original_objective_fn'] = SurrogateFactory.create(name='binary_f1', surrogate_name='binary_f1', weight=1, average='weighted',num_classes=common_params['num_classes'],mode='max')
        print('Original objective function: ', common_params['original_objective_fn'])
        common_params['equality_constraints'] = []
        common_params['shared_macro_constraints'] = []
        print()

        if common_params['performance_constraint'] < 1.0:
            print('Performance constraint: ', common_params['performance_constraint'])
            if common_params['num_classes'] > 2:
                common_params['inequality_constraints'] = [SurrogateFactory.create(name='multiclass_f1', 
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=common_params['performance_constraint'],
                                    use_max=True)]
            else:
                common_params['inequality_constraints'] = [SurrogateFactory.create(name='binary_f1', 
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=common_params['performance_constraint'],
                                    use_max=True)]
            common_params['lagrangian_callbacks'] = [EarlyStopping(patience=2, 
                                                                   monitor='score', 
                                                                   mode='min')]
            common_params['macro_constraints_list'] = [[0]]
            common_params['shared_macro_constraints'] = [0]
          
        else:
            print('No performance constraint')
            print()
            common_params['inequality_constraints'] = []
            common_params['lagrangian_callbacks'] = []
            common_params['macro_constraints_list'] = []
         
        # Configurazione dei macro vincoli
        
        for key,value in self.run_dict.items():
           
            if key not in common_params:
                common_params[key] = value
        
        all_group_ids = {}
        inequality_constraints = common_params['inequality_constraints']
        macro_constraints = common_params['macro_constraints_list']
        idx_constraint = len(inequality_constraints)
        for metric, group, threshold in zip(common_params['metrics_list'], common_params['groups_list'], common_params['threshold_list']):
            common_params['threshold'] = threshold
            common_params['metric'] = metric
            common_params['training_group_name'] = group
            common_params['num_groups'] = self.compute_group_cardinality(common_params['training_group_name'],common_params['sensitive_attributes'])
            group_ids = {common_params['training_group_name']: list(range(common_params['num_groups']))}
            common_params['group_ids'] = group_ids
            all_group_ids.update(group_ids)
            # Aggiunta della metrica
            common_params['metrics'] += [MetricsFactory().create_metric(metric, group_ids=common_params['group_ids'], group_name=common_params['training_group_name'],
                                                                        num_classes=common_params['num_classes'],)]
        
            group_cardinality = common_params['num_groups']
            macro_constraint = []
            current_group_ids = {group: list(range(group_cardinality))}
            all_group_ids.update(current_group_ids)
            for i in range(group_cardinality):
                for j in range(i+1,group_cardinality):
                    if self.num_classes == 2:
                        constraint = SurrogateFactory.create(name=f'diff_{metric}',
                                                        surrogate_name=f'diff_{metric}_{group}',
                                                        surrogate_weight=1,
                                                        average='weighted',
                                                        group_name=group,
                                                        unique_group_ids={group: list(range(group_cardinality))},
                                                        lower_bound=threshold,
                                                        use_max=False,
                                                        target_groups=torch.tensor([i, j]))
                    
                        inequality_constraints.append(constraint)
                        macro_constraint.append(idx_constraint)
                        idx_constraint += 1
                    else:
                        for c in range(self.num_classes):
                            constraint = SurrogateFactory.create(name=f'diff_{metric}',
                                                            surrogate_name=f'diff_{metric}_{group}',
                                                            surrogate_weight=1,
                                                            average='weighted',
                                                            group_name=group,
                                                            unique_group_ids={group: list(range(group_cardinality))},
                                                            lower_bound=threshold,
                                                            use_max=False,
                                                            target_groups=torch.tensor([i, j]),
                                                            target_class = c
                                                            )
                            inequality_constraints.append(constraint)
                            macro_constraint.append(idx_constraint)
                            idx_constraint += 1
            macro_constraints.append(macro_constraint)  
        
        
        common_params['inequality_constraints'] = inequality_constraints
        common_params['macro_constraints_list'] = macro_constraints
        common_params['all_group_ids'] = all_group_ids   
        common_params['optimizer'] = Adam(copy.deepcopy(self.run_dict['model']).parameters(),
                          lr=common_params['lr']
                          )
    
        
        return common_params

    
    def _build_client(self,client_name,**kwargs):
        client_params = copy.deepcopy(self.common_client_params)
        client_params['client_name'] = client_name
        checkpoint_name = kwargs.get('checkpoint_name', f'{client_name}_local.h5')
        client_params['checkpoint_name'] = checkpoint_name   
        client_params['callbacks'] = [
            EarlyStopping(patience=client_params['global_patience'], monitor=client_params['monitor'], mode=client_params['mode']),
            ModelCheckpoint(save_dir=client_params['checkpoint_dir'], save_name=kwargs.get('checkpoint_name', checkpoint_name),                                                                                  monitor=client_params['monitor'], mode=client_params['mode'])
        ]

        client_params['client_checkpoint_name'] = kwargs.get('client_checkpoint_name', f'{client_name}_local_final.h5')  
        client_params['client_callbacks'] = [
            ModelCheckpoint(save_dir=client_params['checkpoint_dir'], 
                            save_name=client_params['client_checkpoint_name'], 
                            monitor=client_params['monitor'],
                            mode=client_params['mode'])
        ]


       
        
        config = {
            'hidden1': client_params['hidden1'],
            'hidden2': client_params['hidden2'],
            'dropout': client_params['dropout'],
            'lr': client_params['lr'],
            'batch_size': client_params['batch_size'],
            'dataset': client_params['dataset'],
            'optimizer': 'Adam',
            'num_lagrangian_epochs': client_params['num_lagrangian_epochs'],
            'num_epochs': client_params['num_local_iterations'],
            'patience': client_params['global_patience'],
            'monitor': client_params['monitor'],
            'mode': client_params['mode'],
            'log_model': client_params['log_model']
        }
        
        checkpoints_config = {
            'checkpoint_dir': client_params['checkpoint_dir'],
            'checkpoint_name': client_params['checkpoint_name'],
            'monitor': client_params['monitor'],
            'mode': client_params['mode'],
            'patience': client_params['local_patience']
        }
        client_params['checkpoints_config'] = checkpoints_config
        client_params['config'] = config
         # Creazione del DataModule
        if self.experiment_name is None:
            path = f'node_{self.start_index}/{client_params["dataset"]}'
        else:
            path = f'{self.experiment_name}/node_{self.start_index}/{client_params["dataset"]}' 
        #print('Data path:',path)
        client_params['data_module'] = DataModule(dataset=client_params["dataset"], 
                                               root=client_params["data_root"], 
                                               train_set=f'{path}_train.csv',
                                                 val_set=f'{path}_val.csv', 
                                                 test_set=f'{path}_val.csv', 
                                                 batch_size=client_params["batch_size"], 
                                                 num_workers=4, 
                                                 use_local_weights = self.algorithm == 'fedavg_lr',
                                                 sensitive_attributes=client_params["sensitive_attributes"])

        # Configurazione del logger
        client_params['logger'] = WandbLogger(
                                project=client_params["project_name"], 
                                  config=config, 
                                  id=client_name,
                                  checkpoint_dir=client_params["checkpoint_dir"], 
                                  checkpoint_path=client_params["checkpoint_name"],
                                  data_module=client_params["data_module"] if client_params["log_model"] else None
                                  )

        if self.algorithm == 'fairlab':
            #print('Building FairLab Orchestrator Wrapper')
            #for key,value in client_params.items():
            #    print(f'{key}: {value}')
            orchestrator = OrchestratorWrapper(
                model=copy.deepcopy(client_params['model']),
                inequality_constraints=client_params['inequality_constraints'],
                macro_constraints_list=client_params['macro_constraints_list'],
                optimizer_fn=client_params['optimizer_fn'],
                optimizer=client_params['optimizer'],
                objective_function=client_params['objective_function'],
                original_objective_fn=client_params['original_objective_fn'],
                batch_objective_fn=client_params['batch_objective_function'],
                num_classes=client_params['num_classes'],
                equality_constraints=client_params['equality_constraints'],
                metrics=client_params['metrics'],
                num_epochs=client_params['num_local_iterations'],
                loss = client_params['loss'],
                data_module=client_params['data_module'],
                logger=client_params['logger'],
                lagrangian_checkpoints=client_params['lagrangian_callbacks'],
                checkpoints=client_params['callbacks'],
                all_group_ids=client_params['all_group_ids'],
                checkpoints_config=client_params['checkpoints_config'],
                shared_macro_constraints=client_params['shared_macro_constraints'],
                delta=client_params['delta'],
                max_constraints_in_subproblem=client_params['max_constraints_in_subproblem'],
                epsilon=client_params['epsilon'],
                
            )
            

            return orchestrator
        
        else:
            raise ValueError(f'Unknown algorithm: {self.algorithm}. Supported algorithms are fedfairlab.')


    def run(self):
        if self.eval_mode:
            print('Entering EVAL MODE.')
            self.evaluate(checkpoint_dir=self.eval_dir,
                          prefix=self.eval_prefix)
        else:
            self.wrapper.fit(num_global_iterations=self.common_client_params['num_global_iterations'],
                                num_local_epochs=self.common_client_params['num_local_iterations'],
                                num_subproblems=self.common_client_params.get('num_subproblems',5))
           
        
 
    """
    def evaluate(self,checkpoint_dir,prefix='fedavg'):
        ray.init(num_cpus=20,num_gpus=1)
        self.server.setup()
        checkpoint_path = f'{checkpoint_dir}/{prefix}_server_global.h5'
        global_results = self.server.evaluate_model_from_ckpt(checkpoint_path=checkpoint_path,
                                             client_id=None
                                             )
        local_results = []
        for c in range(len(self.clients)):
            try:
                checkpoint_path = f'{checkpoint_dir}/{prefix}_client_{1+c}_local.h5'
                results = self.server.evaluate_model_from_ckpt(checkpoint_path=checkpoint_path,
                                                    client_id=c)
            except FileNotFoundError as e:
                checkpoint_path = f'{checkpoint_dir}/{prefix}_client_{1+c}_local_final.h5'
                results = self.server.evaluate_model_from_ckpt(checkpoint_path=checkpoint_path,
                                                    client_id=c)
            local_results.append(results)

        print('Results')
        print()
        print(50 * '-')
        print('Global Results:')
        pprint.pprint(global_results)
        print(50 * '-')
        print()
        for c in range(len(self.clients)):
            print()
            print(50 * '-')
            print(f'Local Results for Client {1+c}:')
            pprint.pprint(local_results[c])
            print(50 * '-')
            print()
        self.server.shutdown(log_results=False)
        ray.shutdown()
        

    def evaluate_old(self,checkpoint_path,run_id,client_id=None,init_fl =True,shutdown=False):
        project='Folk_Employment_New2'
        run = wandb.init(project=project, 
                         id=run_id, 
                         resume='must')
        
        #print("ID inizializzato:", run.id)
        #print("Stato della run:", run.)
        
        if init_fl:
            ray.init(num_cpus=self.num_cpus,num_gpus=self.num_gpus)
            self.server.setup()
        attributes = ['GenderRace', 'GenderMarital', 'RaceMarital', 'Gender', 'Race', 'Marital']
        attributes = [ 'RaceMarital', 'Race', 'Marital']
        metric_name = 'final_val_demographic_parity_'
       
        
        results = self.server.evaluate_model_from_ckpt(checkpoint_path=checkpoint_path,
                                                       log_results=False,
                                                       client_id=client_id)

        
        results_to_log = {}
        for attribute in attributes:
            results_to_log[f'{metric_name}{attribute}'] = results[f'{metric_name}{attribute}'].item()
        run.log(results_to_log)
        
    
        run.finish()
       
        return results_to_log
    
    def shutdown(self):
        self.server.shutdown(log_results=False)
        ray.shutdown()
    """