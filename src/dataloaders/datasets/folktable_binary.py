import os
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset


@register_dataset('folktables_binary')
class FolkTablesBinaryDataset(BaseDataset):

    def __init__(self,**kwargs):
        super(FolkTablesBinaryDataset, self).__init__(**kwargs)
        self.root = kwargs.get('root', 'data/FolkTables')
        data_name = kwargs['filename']

        self.data_path = os.path.join(self.root, data_name)
        
        self.scaler_name = kwargs.get('scaler_name', 
                                      'folktables_binary_scalers.p')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                                [{}])
       
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        
        self.target = 'PINCP'
        self.cat_cols = ['Gender','Race','Job','Marital']
        self.num_cols = []
        self.labels = [0,1]
        self.clean_data_path = os.path.join(self.root,'folktables_CA_binary_clean.csv')
        self.setup()
        