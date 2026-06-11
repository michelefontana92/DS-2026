from abc import ABC, abstractmethod
from pathlib import Path


class BaseRun(ABC):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self,**kwargs):
        
        self.model = kwargs.get('model')
        self.dataset = kwargs.get('dataset')
        self.sensitive_attributes = kwargs.get('sensitive_attributes')
        self.project_name = kwargs.get('project_name')
        self.data_root = kwargs.get('data_root')
        root_dir = kwargs.get('root_dir')
        if root_dir is None:
            self.root_dir = self.PROJECT_ROOT / 'data'
        else:
            self.root_dir = Path(root_dir).expanduser().resolve()

    def project_path(self, *parts):
        return str(self.PROJECT_ROOT.joinpath(*parts))

    def data_path(self, *parts):
        return str(self.root_dir.joinpath(*parts))

    def compute_group_cardinality(self,group_name):
        for name,group_dict in self.sensitive_attributes:
            if name == group_name:
                total = 1
                for key in group_dict.keys():
                    total *= len(group_dict[key])
                return total 
        raise KeyError(f'Group {group_name} not found in sensitive attributes') 
    
    @abstractmethod
    def setUp(self):
        pass 
    
    @abstractmethod
    def tearDown(self):
        pass

    @abstractmethod
    def run(self,**kwargs):
        pass
    
    def eval(self):
        pass
    
    def __call__(self, **kwargs):
        self.setUp()
        self.run(**kwargs)
        self.tearDown()

  
    def to_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'sensitive_attributes': self.sensitive_attributes,
            'project_name': self.project_name,
            'root_dir': str(self.root_dir),
            'data_root': self.data_root,
            'clean_data_path': getattr(self, 'clean_data_path', None),
            'hidden1': self.hidden1,
            'hidden2': self.hidden2,
            'input': self.input,
            'dropout': self.dropout,
            'num_classes': self.num_classes,
            'output': self.output,
        }
