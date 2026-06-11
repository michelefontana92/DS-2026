from ..base_run import BaseRun
from architectures import ArchitectureFactory

class CentralizedCompasRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(CentralizedCompasRun, self).__init__(**kwargs)
        self.input = 14
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        self.num_classes=2
        self.output = self.num_classes
        self.model = ArchitectureFactory.create_architecture('mlp2hidden',model_params={
                                                'input': self.input,
                                                'hidden1': self.hidden1,
                                                'hidden2': self.hidden2,
                                                'dropout': self.dropout,
                                                'output': self.output})
        self.dataset = 'compas'
        self.data_root = self.data_path('Centralized_Compas')
        self.clean_data_path = self.data_path('Centralized_Compas', 'compas_clean.csv')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                               [
                                                ('Race',
                                                    {'race':['Other', 'African-American', 
                                                             'Caucasian'
                                                             ]}
                                                ),
                                                ('Gender',{'sex':['Male','Female']}),
                                                ('Age',{'age_cat':['Greater than 45','25 - 45','Less than 25']}),
                                                ('GenderRace',{
                                                    'race':['Other', 'African-American', 
                                                             'Caucasian'  
                                                             ],
                                                    'sex':['Male','Female']
                                                }),
                                                ('GenderAge',{
                                                    'age_cat':['Greater than 45','25 - 45','Less than 25'],
                                                    'sex':['Male','Female']
                                                }),
                                                ('RaceAge',{
                                                    'age_cat':['Greater than 45','25 - 45','Less than 25'],
                                                    'race':['Other', 'African-American', 
                                                             'Caucasian' ]
                                                }),
                                                ('GenderRaceAge',{
                                                    'age_cat':['Greater than 45','25 - 45','Less than 25'],
                                                    'race':['Other', 'African-American', 
                                                             'Caucasian'  
                                                             ],
                                                    'sex':['Male','Female']
                                                }),
                                                ])
        
        

    def setUp(self):
        pass
    def run(self):
        pass
    def tearDown(self):
        pass
