import os
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset

@register_dataset('income_3')
class Income3Dataset(BaseDataset):

    def __init__(self,**kwargs):
        super(Income3Dataset, self).__init__(**kwargs)
        self.root = kwargs.get('root', 'data/Income_3')
        data_name = kwargs['filename']

        self.data_path = os.path.join(self.root, data_name)
        
        self.scaler_name = kwargs.get('scaler_name', 
                                      'income_3_scalers.p')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                                [{}])
       
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        
        self.target = 'PINCP'
        self.cat_cols = ['Gender','Race','COW','Marital',
                        'SCHL',      
                        'RELP',
                        'DIS', 'ESP', 
                         'CIT', 'MIG', 'MIL', 'ANC', 
                         'NATIVITY', 'DEAR', 'DEYE', 
                         'DREM','ESR'
                        ]
        self.num_cols = ['AGEP','WKHP','OCCP','POBP']
        self.labels = [0,1,2]
        self.clean_data_path = os.path.join(self.root,'income_3_clean.csv')
        self.setup()
        