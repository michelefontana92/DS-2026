import shutil
from .income_3_run import Income3Run
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
@register_run('income_3_fairlab')
class Income3HierALMCentralized(Income3Run):
    def __init__(self, **kwargs) -> None:
        super(Income3HierALMCentralized, self).__init__(**kwargs)
        kwargs['run_dict'] = self.to_dict()
        self.builder = FairLabBuilder(**kwargs)
    
    def setUp(self):
        pass
    
    def run(self):
        self.builder.run()