import pandas as pd
import numpy as np

import pandas as pd
import operator

# Map string operators to Python's internal math engine
OPERATOR_MAP = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne
}

DEFAULT_SCHEMA = {
    'vitals': {
        'name_col': 'OBSERVATION_NAME', 
        'code_col': 'OBSERVATION_CODE',
        'value_col': 'OBSERVATION_RESULT_CLEAN'
    },
    'pathology': {
        'name_col': 'ORDER_NAME',         
        'code_col': 'ORDER_CODE',         
        'value_col': 'RESULT_CLEANED' 
    },
    'interventions': {
        'name_col': 'INTERVENTION_NAME',
        'code_col': 'INTERVENTION_CODE',
        'value_col': 'INTERVENTION_STATUS'
    },
    'microbiology': {
        'name_col': 'ORDER_NAME', 
        'code_col': 'ORDER_CODE', 
        'value_col': 'ORGANISM_BUG'
    }
}

def derive_flag_from_yaml(dataset, 
                          output_name: str, 
                          rules: list, 
                          schema=DEFAULT_SCHEMA):
    """
    Evaluates clinical rules passed dynamically from a YAML configuration.
    """
    processed_tables = []
    unique_tables = set(r['table'] for r in rules)
    
    for table_name in unique_tables:
        df = dataset.get_current_stay(table_name)
        if df.empty:
            continue
            
        name_col = schema[table_name].get('name_col')
        code_col = schema[table_name].get('code_col')
        value_col = schema[table_name].get('value_col')
            
        df['calc_flag'] = 0
        val = pd.to_numeric(df[value_col], errors='coerce')
        
        # Filter rules for this specific table
        table_rules = [r for r in rules if r['table'] == table_name]
        
        for rule in table_rules:
            target_names = rule.get('names', [])
            target_codes = rule.get('codes', [])
            
            # Fetch the actual math function (e.g., operator.gt for '>')
            op_func = OPERATOR_MAP[rule['operator']]
            threshold = rule['threshold']
            
            mask = pd.Series(False, index=df.index)
            if target_names and name_col in df.columns:
                mask = mask | df[name_col].isin(target_names)
            if target_codes and code_col in df.columns:
                mask = mask | df[code_col].isin(target_codes)
                
            # op_func(val, threshold) executes the math: val > threshold
            df.loc[mask & op_func(val, threshold), 'calc_flag'] = 1
            
        processed_tables.append(df[['SUBJECT', 'std_admission_time', 'calc_flag']])
        
    if not processed_tables:
        return pd.DataFrame(columns=['SUBJECT', 'std_admission_time', output_name])
        
    combined = pd.concat(processed_tables, ignore_index=True)
    results = combined.groupby(['SUBJECT', 'std_admission_time'])['calc_flag'].max()
    
    return results.reset_index(name=output_name)




def derive_score_from_yaml(dataset, 
                           output_name: str, 
                           rules: list, 
                           schema=DEFAULT_SCHEMA):
    """
    Evaluates clinical rules dynamically, supporting variable point assignments.
    """
    processed_tables = []
    unique_tables = set(r['table'] for r in rules)
    
    for table_name in unique_tables:
        df = dataset.get_current_stay(table_name)
        if df.empty:
            continue
            
        name_col = schema[table_name].get('name_col')
        code_col = schema[table_name].get('code_col')
        value_col = schema[table_name].get('value_col')
            
        df['calc_score'] = 0
        val = pd.to_numeric(df[value_col], errors='coerce')
        
        table_rules = [r for r in rules if r['table'] == table_name]
        
        for rule in table_rules:
            target_names = rule.get('names', [])
            target_codes = rule.get('codes', [])
            
            op_func = OPERATOR_MAP[rule['operator']]
            threshold = rule['threshold']
            
            # Default to 1 point if not specified (perfect for SIRS flags)
            points = rule.get('points', 1) 
            
            mask = pd.Series(False, index=df.index)
            if target_names and name_col in df.columns:
                mask = mask | df[name_col].isin(target_names)
            if target_codes and code_col in df.columns:
                mask = mask | df[code_col].isin(target_codes)
                
            # Safely assign points: Only update if the new points are HIGHER 
            # than what the row currently has (prevents a 1-pt rule from overwriting a 2-pt rule)
            score_mask = mask & op_func(val, threshold)
            df.loc[score_mask & (df['calc_score'] < points), 'calc_score'] = points
            
        processed_tables.append(df[['SUBJECT', 'std_admission_time', 'calc_score']])
        
    if not processed_tables:
        return pd.DataFrame(columns=['SUBJECT', 'std_admission_time', output_name])
        
    combined = pd.concat(processed_tables, ignore_index=True)
    
    # Group by admission and take the max severity score across all rules and tables
    results = combined.groupby(['SUBJECT', 'std_admission_time'])['calc_score'].max()
    
    return results.reset_index(name=output_name)


def derive_historical_condition(dataset, 
                                output_name: str, 
                                sources: list, 
                                **kwargs):
    """
    Evaluates historical conditions (e.g. Charlson comorbidities) from clinical tables.
    Supports exact matching and hierarchical prefix matching (ICD-10/SNOMED).
    """
    processed_tables = []
    
    for source in sources:
        table_name = source['context_name']
        code_col = source['code_col']
        vocab_key = source['vocab_key']
        match_type = source.get('match_type', 'exact')
        
        # Pull the corresponding code list from kwargs (e.g., target_codes, icd10_codes)
        valid_codes = kwargs.get(vocab_key, [])
        if not valid_codes:
            continue
            
        # 1. Fetch the history table for current admissions
        # (Assuming the dataset handles joining problems to the std_admission_time)
        df = dataset.get_history(table_name)
        if df.empty or code_col not in df.columns:
            continue
            
        df['calc_flag'] = 0
        
        # Drop empty rows to prevent string matching errors
        valid_mask = df[code_col].notna()
        
        # 2. Apply Prefix or Exact matching
        if match_type == 'prefix':
            # pandas requires a tuple for multiple prefixes
            prefixes = tuple(valid_codes) 
            mask = valid_mask & df[code_col].astype(str).str.startswith(prefixes)
        else:
            # Exact match
            mask = valid_mask & df[code_col].isin(valid_codes)
            
        df.loc[mask, 'calc_flag'] = 1
        
        processed_tables.append(df[['SUBJECT', 'std_admission_time', 'calc_flag']])
        
    # 3. Group and return
    if not processed_tables:
        return pd.DataFrame(columns=['SUBJECT', 'std_admission_time', output_name])
        
    combined = pd.concat(processed_tables, ignore_index=True)
    results = combined.groupby(['SUBJECT', 'std_admission_time'])['calc_flag'].max()
    
    return results.reset_index(name=output_name)