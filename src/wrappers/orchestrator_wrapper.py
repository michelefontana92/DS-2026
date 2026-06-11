from wrappers.torch_nn_wrapper import TorchNNWrapper
from wrappers.orchestrator import MainProblemOrchestrator
import copy
from exceptions.early_stopping import EarlyStoppingException


class OrchestratorWrapper(TorchNNWrapper):
    """
    Implementation of the orchestrator.

    Methods:
        fit(num_global_iterations=5, num_local_epochs=5, num_subproblems=5):
            Fits the model using the specified number of global iterations, local epochs, and subproblems.
            Args:
                num_global_iterations (int): Number of global iterations. Default is 5.
                num_local_epochs (int): Number of local epochs. Default is 5.
                num_subproblems (int): Number of subproblems. Default is 5.
            Returns:
                The trained model.
    """
    def __init__(self, *args,**kwargs):
        super(OrchestratorWrapper, self).__init__(*args, **kwargs)
        # Estrarre i parametri necessari da kwargs, con valori di default ove appropriato
        self.loss_fn = kwargs.get('loss')
        self.epsilon = kwargs.get('epsilon')
        self.num_classes = kwargs.get('num_classes',2)
        self.inequality_constraints = kwargs.get("inequality_constraints", [])
        self.macro_constraints_list = kwargs.get("macro_constraints_list", [])
        self.target_groups = kwargs.get("target_groups", [])
        self.min_subproblems = kwargs.get("min_subproblems", 2)
        self.max_subproblems = kwargs.get("max_subproblems", 5)
        self.all_group_ids = kwargs.get("all_group_ids")
        assert self.all_group_ids is not None, 'all_group_ids must be provided'
        
        self.optimizer_fn: callable = kwargs.get('optimizer_fn')
        self.objective_function = kwargs.get("objective_function")
        self.original_objective_fn = kwargs.get("original_objective_fn")
        self.equality_constraints = kwargs.get("equality_constraints")
        self.metrics = kwargs.get("metrics", [])
        self.num_epochs = kwargs.get("num_epochs", 10)
        self.logger = kwargs.get("logger")
        self.lagrangian_checkpoints = kwargs.get("lagrangian_checkpoints", [])
        
        self.checkpoints = kwargs.get("checkpoints")
        self.checkpoints_config = kwargs.get("checkpoints_config")
        self.delta = kwargs.get("delta")
       
        self.current_model = self.model
        self.shared_macro_constraints = kwargs.get("shared_macro_constraints",[])
        self.max_constraints_in_subproblem = kwargs.get("max_constraints_in_subproblem",5)
        self.batch_objective_function = kwargs.get("batch_objective_fn")
        self.options = {
                'optimizer_fn': self.optimizer_fn,
                'objective_fn': self.objective_function,
                'batch_objective_fn': self.batch_objective_function,
                'original_objective_fn': self.original_objective_fn,
                'metrics': self.metrics,
                'num_epochs': self.num_epochs,
                'logger': self.logger,
                'loss': self.loss_fn,
                'optimizer':self.optimizer,
                'optimizer_fn':self.optimizer_fn,
                'data_module':self.data_module,
                'verbose':self.verbose,  
                'inequality_lambdas_0_value': 0,
            }
        
        
   
    def fit(self, num_global_iterations=5,num_local_epochs=5,num_subproblems=5):
        main_problem = MainProblemOrchestrator(
                                            model=copy.deepcopy(self.model),
                                            inequality_constraints=self.inequality_constraints,
                                            equality_constraints=self.equality_constraints,
                                            macro_constraints=self.macro_constraints_list,
                                            checkpoints_config=self.checkpoints_config,
                                            all_group_ids=self.all_group_ids,
                                            num_subproblems=num_subproblems,
                                            options=self.options,
                                            logger=self.logger,
                                            checkpoints=self.checkpoints,
                                            shared_macro_contraints=self.shared_macro_constraints,
                                            delta=self.delta,
                                            max_constraints_in_subproblem=self.max_constraints_in_subproblem,
                                            epsilon=self.epsilon,
                                            num_classes=self.num_classes                                            
                                           )
        
        metrics = main_problem.evaluate(main_problem.model)
        self.logger.log(metrics)
        try:
            for i in range(num_global_iterations):
                print('Iteration',i)
                main_problem.iterate(num_local_epochs=num_local_epochs,
                                    add_proximity_constraints=True,
                                    send_teacher_model=i>0)
        except EarlyStoppingException:
            print('Early stopping')

        main_problem.eval_final_model()
        return self.current_model

    