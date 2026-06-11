from ..base_run import BaseRun
from architectures import ArchitectureFactory

class CentralizedMEPRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(CentralizedMEPRun, self).__init__(**kwargs)
        self.input = 43
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
        self.dataset = 'mep'
        self.data_root = self.data_path('Centralized_MEP')
        self.clean_data_path = self.data_path('Centralized_MEP', 'mep1_clean.csv')
        race_values = ['Hispanic','Black','White','Other']
        marry_values = ['Married','Never Married','Other']
        gender_values = ['Male','Female']
        race_var = 'RACE'
        gender_var = 'SEX'
        marry_var = 'MARRY'
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                               [
                                                ('Race',
                                                    {race_var:race_values}
                                                ),
                                                ('Gender',
                                                 {gender_var:gender_values}
                                                 ),
                                                 ('Marital',
                                                  {marry_var:marry_values}),
                                                ('GenderRace',
                                                 {gender_var:gender_values,
                                                  race_var:race_values}),
                                                ('GenderMarital',
                                                 {gender_var:gender_values,
                                                  marry_var:marry_values}),
                                                ('RaceMarital',
                                                 {marry_var:marry_values,
                                                  race_var:race_values}),
                                                ('GenderRaceMarital',
                                                 {gender_var:gender_values,
                                                  marry_var:marry_values,
                                                  race_var:race_values})
                                                ])
        
        

    def setUp(self):
        pass
    def run(self):
        pass
    def tearDown(self):
        pass
