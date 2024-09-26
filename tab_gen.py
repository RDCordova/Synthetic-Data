import pandas as pd
import numpy as np
import boto3
import json
from botocore.config import Config
from langchain_community.chat_models import BedrockChat
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
import math

#extract schema from csv 
def extract_csv_schema(file_path):
    df = pd.read_csv(file_path)
    schema = dict(df.dtypes)
    return schema

#load model 
def load_model(top_k,top_p,temperature):
    config = Config(read_timeout=1000)

    bedrock_runtime = boto3.client(service_name='bedrock-runtime', 
                          region_name='us-east-1',
                          config=config)

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    
    model_kwargs = { 
        "max_tokens": 200000,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": ["\n\nHuman"],
        
    }
    
 
    
    model = BedrockChat(
        client=bedrock_runtime,
        model_id=model_id,
        model_kwargs=model_kwargs,
    )

    return model

def format_exmaples(df):
    exmaples_str = "" 
    for index, row in df.iterrows():
        row_string = ", ".join([f"{df.columns[i]}: {row[i]}" for i in range(len(row))])
        exmaples_str += row_string + "\n"
    return exmaples_str

def format_rows(row):
    return  ", ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])

def row_len(text):
    return len(text)

def max_examples(csv_data, max_context_tokens, prompt_tokens, response_token_buffer=0.3):
    formatted_exmaples = csv_data.apply(format_rows, axis =1)
    char_per_row = int(formatted_exmaples.apply(row_len).mean())
    max_available = int((max_context_tokens*(1-response_token_buffer)) / char_per_row)
    max_examples_percent = int(0.6 * len(csv_data))
    return (min(max_available, max_examples_percent),char_per_row)

def generate_examples(csv_data, num_examples):
    return csv_data.sample(n=num_examples)

def create_prompt_template():
    template = """
    You are a helpful AI assistant for creating new datasets. 
    You are to generate new observations based on the provided examples and schema. The distibutions of value should be similar to the examples.  
    The formatting of the new obervation should match the formatting of the exmaples. ie  column 1: value 1, column 2: value 2...column n: value n
    Only return the new observations, do not include any explanation. Please genrate a number of new observation equal to the count.

    examples: {examples}

    schema: {schema}
    
    count: {count}

    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["examples", "schema", "count"]
    )

    return prompt


def generate_response(csv_data,num_examples,schema, row_tokens):
    examples = generate_examples(csv_data, num_examples)
    prompt_examples = format_exmaples(examples)
    prompt_template = create_prompt_template()
    filled_prompt = prompt_template.format(
    examples=prompt_examples,
    schema=schema,
    count = math.ceil((5000 / row_tokens))
    )
    message = HumanMessage(content=filled_prompt)

    response = llm([message])

    return response

def create_new_dataframe(text):
    obs = text.split('\n')

    data_dicts = []
    for line in obs:

        pairs = line.split(', ')
        entry_dict = {}
        for pair in pairs:
            if ': ' in pair:
                k, v = pair.split(': ', 1)  
                entry_dict[k.strip()] = v.strip()
        data_dicts.append(entry_dict)
    return pd.DataFrame(data_dicts)


def gen_all_obs(gen_cycles,csv_data,num_examples,schema,row_tokens):
    df_list = []
    for i in range(gen_cycles):
        temp = create_new_dataframe(generate_response(csv_data,num_examples,schema, row_tokens).content)
        df_list.append(temp)
    final_df = pd.concat(df_list, ignore_index=True)
    return final_df


file_path = "tips.csv"
csv_data = pd.read_csv(file_path)


#Step 2 define model 
top_k = 250
top_p = 0.9
temperature = 1
llm = load_model(top_k,top_p,temperature)


#step 3 get schema
schema = extract_csv_schema(file_path)


# Step 4 get vals for functions
max_context_tokens = 100000  # Example context window size (Claude 3's token limit)
prompt_tokens = 1000  # Estimate for tokens used by the prompt
response_token_buffer=0.3
num_examples, row_tokens = max_examples(csv_data, max_context_tokens, prompt_tokens, response_token_buffer)
num_obs = 100
gen_cycles = math.ceil(num_obs / (5000 / row_tokens))

#Step 6 generate obs
obs_df = gen_all_obs(gen_cycles,csv_data,num_examples,schema,row_tokens)
final_df= obs_df[:num_obs]
