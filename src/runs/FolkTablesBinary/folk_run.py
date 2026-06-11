from ..base_run import BaseRun
from architectures import ArchitectureFactory

class FolkTablesBinaryRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(FolkTablesBinaryRun, self).__init__(**kwargs)
        self.num_classes=2
        self.input = 10
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        self.output = self.num_classes
        self.diff=0.04
        self.model = ArchitectureFactory.create_architecture('mlp2hidden',model_params={
                                                'input': self.input,
                                                'hidden1': self.hidden1,
                                                'hidden2': self.hidden2,
                                                'dropout': self.dropout,
                                                'output': self.output})
        self.dataset = 'folktables_binary'
        self.data_root = self.data_path('Folktables_binary')
        self.clean_data_path = self.data_path('Folktables_binary', 'folktables_CA_binary_clean.csv')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',[
                                                
                                                ('Marital',{
                                                     'Marital':['Married','Never Married','Divorced','Other']}),   
                                                
                                                ('Job',{'Job':['Public Employee','Self Employed','Private Employee']}),
                                                 ('Race',{'Race':['White','Black','Asian','Other','Indigenous']}),
                                                 ('Gender',{'Gender':['Male','Female']}),
                                                 
                                                ('JobMarital',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),
                                                
                                                
                                                ('JobRace',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Race':['White','Black','Asian','Other','Indigenous']
                                                    }),
                                                
                                                ('GenderRace',{
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Gender':['Male','Female'],
                                                    }),
                                                ('RaceMarital',{
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),
                                                
                                                ('GenderMarital',{
                                                    'Gender':['Male','Female'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),
                                                
                                                ('GenderRaceMarital',{
                                                    'Gender':['Male','Female'],
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),

                                                ('JobRaceMarital',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),
                                                
                                                
                                                
                                                ('JobMaritalGender',{
                                                     'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    'Gender':['Male','Female']
                                                     }),
                                                ('JobRaceMaritalGender',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    'Gender':['Male','Female']
                                                    }),
                                                    
                                                    ]
                                                )
                                             
    def setUp(self):
        pass
    def run(self):
        pass
    def tearDown(self):
        pass
