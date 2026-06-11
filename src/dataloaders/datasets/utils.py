import numpy as np
def find_bucket(n,thresholds):
    for i in range(len(thresholds)-1):
        if n>=thresholds[i] and n<=thresholds[i+1]:
            return i
    if n>thresholds[-1]:
        return len(thresholds)-1
    return -1

def _assign(x,sensitive_dict,key):
    
    if x in sensitive_dict[key]:
        return x
    else:
        #print(f'Assigned NoneGroup to {x}')
        return 'NoneGroup'

import itertools

def assign_group_id(df, sensitive_attributes):
    """
    Federated-safe assignment: assegna ID consistenti tra client.
    Ritorna:
        - df con colonne group_id_<name>
        - dizionario globale id_to_combination per ciascun gruppo
    """
    id_to_combination = {}

    for (name, group_dict) in sensitive_attributes:
        columns = list(group_dict.keys())
        values = [group_dict[col] for col in columns]
        combinations = list(itertools.product(*values))  # deterministico e ordinato
        combination_ids = {tuple(comb): idx for idx, comb in enumerate(combinations)}

        id_to_combination[name] = {
            idx: dict(zip(columns, comb)) for comb, idx in combination_ids.items()
        }

        def get_combination_id(row):
            row_tuple = tuple(row[col] for col in columns)
            return combination_ids.get(row_tuple, -1)  # -1 se non trovata

        df[f'group_id_{name}'] = df.apply(get_combination_id, axis=1)

    #print(f'Group IDs: {df[[f"group_id_{name}" for name in sensitive_attributes.keys()]].head()}')
    #print(f'ID to combination: {id_to_combination}')
    return df, id_to_combination



def assign_group_id_old(data,sensitive_dict,group_name):
    if len(sensitive_dict.keys()) == 0:
        data[f'group_id_{group_name}'] = data.apply(lambda x: -1, axis=1)
        return data
    for key in sensitive_dict.keys():
        if type(data[key].iloc[0]) == str:
            #print(f'Assigning group id for {key}')
            #print('Sensitive dict:',sensitive_dict[key])
            data[key+'_new'] = data[key].apply(lambda x:_assign(x,sensitive_dict,key))#lambda x: x if x in sensitive_dict[key] else 'NoneGroup')
        elif str(data[key].iloc[0]).isnumeric():
            thresholds = sensitive_dict[key]
            #sort the thresholds
            if np.array(thresholds).min() > data[key].min():
                thresholds.append(data[key].min())
            if np.array(thresholds).max() < data[key].max():
                thresholds.append(data[key].max())
            
            thresholds.sort()
           
            data[key+'_new'] = data[key].apply(lambda x: find_bucket(x,thresholds))
            data[key+'_new'] = data[key+'_new'].apply(str)
        else:
            raise ValueError('Data type not supported')
        
    sensitive = [x+'_new' for x in sensitive_dict.keys()]
    data['group']= data[sensitive].apply(lambda x: ''.join(x), axis=1)
    group_ids = {}
    current_id = 0
    for group in data['group'].unique():
        if 'NoneGroup' in group:
            group_ids[group] = -1
        else:
            group_ids[group] = current_id
            current_id += 1
    data[f'group_id_{group_name}'] = data['group'].map(group_ids)
    data.drop(sensitive+['group'],axis=1,inplace=True)
    return data