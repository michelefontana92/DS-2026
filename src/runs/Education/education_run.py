from ..base_run import BaseRun
from architectures import ArchitectureFactory

class EducationRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(EducationRun, self).__init__(**kwargs)
        self.num_classes=3
        self.input = 21#85#21
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        self.output = self.num_classes
    
        self.model = ArchitectureFactory.create_architecture('mlp2hidden',model_params={
                                                'input': self.input,
                                                'hidden1': self.hidden1,
                                                'hidden2': self.hidden2,
                                                'dropout': self.dropout,
                                                'output': self.output})
        self.dataset = 'education'
        self.data_root = self.data_path('Education')
        self.clean_data_path = self.data_path('Education', 'education_clean.csv')
        
        self.sensitive_attributes = kwargs.get('sensitive_attributes',[
                                                
                                                ('Marital',{
                                                     'Marital':['Married','Never Married','Divorced','Other']}),   
                                                
                                                 ('Race',{'Race':['White','Black','Asian','Other','Indigenous']}),
                                                 ('Gender',{'Gender':['Male','Female']}),
                                                 
                                               
                                            
                                                
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

                                               
                                                    
                                                    ]
                                                )
                                             
    def setUp(self):
        pass
    def run(self):
        pass
    def tearDown(self):
        pass
    def eval(self):
        pass
