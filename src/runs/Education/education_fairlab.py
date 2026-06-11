import shutil
from .education_run import EducationRun
from ..run_factory import register_run
from metrics import MetricsFactory
from surrogates import SurrogateFactory
from wrappers import OrchestratorWrapper
from dataloaders import DataModule
from torch.optim import Adam
from functools import partial
from callbacks import EarlyStopping, ModelCheckpoint
from loggers import WandbLogger
from torch.nn import CrossEntropyLoss
import torch
from builder import FairLabBuilder
import wandb

@register_run('education_fairlab')
class EducationHierALMCentralized(EducationRun):
    def __init__(self, **kwargs) -> None:
        super(EducationHierALMCentralized, self).__init__(**kwargs)
        kwargs['run_dict'] = self.to_dict()
        self.builder = FairLabBuilder(**kwargs)
    
    def setUp(self):
        #print(self.builder.clients)
        pass
    
    

    def run(self):
        self.builder.run()
        
    def tearDown(self) -> None:
        # Pulizia finale dei file di checkpoint, se necessario
        pass
        #shutil.rmtree(f'checkpoints/{self.project_name}')
