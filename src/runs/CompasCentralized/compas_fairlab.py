import shutil
from .compas_run import CentralizedCompasRun
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
@register_run('compas_fairlab')
class CompasHierALMCentralized(CentralizedCompasRun):
    def __init__(self, **kwargs) -> None:
        super(CompasHierALMCentralized, self).__init__(**kwargs)
        kwargs['run_dict'] = self.to_dict()
        self.builder = FairLabBuilder(**kwargs)
    
    def setUp(self):
        pass
    
    def run(self):
        self.builder.run()