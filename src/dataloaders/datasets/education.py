import os
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset


@register_dataset('education')
class EducationDataset(BaseDataset):

    def __init__(self,**kwargs):
        super(EducationDataset, self).__init__(**kwargs)
        self.root = kwargs.get('root', 'data/Education')
        data_name = kwargs['filename']

        self.data_path = os.path.join(self.root, data_name)
        
        self.scaler_name = kwargs.get('scaler_name', 
                                      'education_scalers.p')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                                [{}])
       
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        
        self.target = 'SCHL'
        self.cat_cols = ['Gender','Race','Marital',
                         'ESR', 'RELP', 'DIS', 'ESP', 
                         'CIT', 'MIG', 'MIL', 'ANC', 
                         'NATIVITY', 'DEAR', 'DEYE', 
                         'DREM','COW']
        self.num_cols = ['AGEP','PINCP','WKHP','OCCP','POBP']
        self.labels = [0,1,2]
        self.clean_data_path = os.path.join(self.root,'education_clean.csv')
        self.setup()
        