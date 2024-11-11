from openai import OpenAI
import pandas as pd
import random as rd
import json
import time

LLM_MODEL = 'llama3.2'

PREDICT_ROWS = 10

OUTPUT_SUFFIX = ""
INPUT_FILE_NAME = '2018'
OUTPUT_FILE_NAME = f'{INPUT_FILE_NAME}_llm_{LLM_MODEL.replace(".", "_")}_{OUTPUT_SUFFIX}'

DATASETS_DIRECTORY = 'datasets'

INPUT_FILE = f'{DATASETS_DIRECTORY}/{INPUT_FILE_NAME}'
OUTPUT_FILE = f'{DATASETS_DIRECTORY}/{OUTPUT_FILE_NAME}'




# Configure the OpenAI client to use http://localhost:11434/v1 as base url 
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

def predict(row:pd.Series)->str:
    context = f'''
    <context>
    You are a helpful assistant, who is trying to help a user with predict fly delay.
    As input you get the following information:
    <input>
    flight_info:
        flight_date: <flight_date>
        OP_CARRIER: <carrier>
        OP_CARRIER_FL_NUM: <flight_number>
        ORIGIN: <origin_airport>
        DEST: <destination_airport>
        CRS_DEP_TIME: <scheduled_departure_time>
        DISTANCE: <distance_between_airports>
        CRS_ARR_TIME: <scheduled_arrival_time>
    </input>
    </context>
    <task>
    - Based on this information, predict <delay_value> (number).
    - Explain <reason> of delay (exacly one sentence).
    - Do not predict based on wather.
    - Predict only based on the information provided.
    - Answer in JSON using <answer_template> and nothing more.
    - Do not return any other information.
    </task>
    <answer_template>
    {{
        "delay_minutes": "<delay_value>",
        "delay_reason": "<reason>"
    }}
    </answer_template>
    <example_answers>
    - prediction:
        delay_minutes: 10
        delay_reason: "From this airport to that airport, the flight is always delayed."
    - prediction:
        delay_minutes: -20
        delay_reason: "In this hour, the flight is always before the time."
    - prediction:
        delay_minutes: 0
        delay_reason: "The flights of this carrier is always on time."
    - prediction:
        delay_minutes: 30
        delay_reason: "The flights of this carrier is sometimes delayed."
    - prediction:
        delay_minutes: -10
        delay_reason: "Flights from this airport are always before the time."
    - prediction:
        delay_minutes: 20
        delay_reason: "Rush hours."
    
    </example_answers>
'''
    flight_info = f'''
    Predict the delay of the following flight:
    <input>
    flight_info:
        flight_date: {row['FL_DATE']}
        OP_CARRIER: {row['OP_CARRIER']}
        OP_CARRIER_FL_NUM: {row['OP_CARRIER_FL_NUM']}
        ORIGIN: {row['ORIGIN']}
        DEST: {row['DEST']}
        CRS_DEP_TIME: {row['CRS_DEP_TIME']}
        DISTANCE: {row['DISTANCE']}
        CRS_ARR_TIME: {row['CRS_ARR_TIME']}
    </input>
'''
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": flight_info}
        ]
    )
    return response.choices[0].message.content

def modify_row(row_data:pd.Series)->pd.Series:
    while True:
        prediction = predict(row_data)
        
        print(f"FL_DATA:\n {row_data[['FL_DATE', 'OP_CARRIER', 'OP_CARRIER_FL_NUM', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'DISTANCE', 'CRS_ARR_TIME']]}\n")
        print(f"PREDICTION:\n {prediction}")
        
        prediction = prediction.replace('\n', '').replace('`','')
        prediction = prediction[prediction.find("{"):]
        
        try:
            json_prediction = json.loads(prediction)
            print(f"DELAY: {json_prediction['delay_minutes']}, REASON: {json_prediction['delay_reason']}\n\n")
            with pd.option_context('mode.chained_assignment', None):
                row_data['llm_delay_minutes'] = json_prediction['delay_minutes']
                row_data['llm_delay_reason'] = json_prediction['delay_reason']
            return row_data
        except:
            print(f"ERROR Running once again!\n")

def main():
    df = pd.read_csv(f'{INPUT_FILE}.csv')
    df['llm_delay_minutes'] = None
    df['llm_delay_reason'] = None
    
    indexes = rd.sample(range(len(df)), PREDICT_ROWS)
    
    start_time = time.time()
    
    with open(f'{OUTPUT_FILE}_indexes.txt', 'w') as f:
        for i in indexes:
            print(f"GENERATING ROW: {indexes.index(i) + 1}")
            print(f"INDEX: {i}")
            row_data = df.iloc[i]
            
            modified_row = modify_row(row_data)
            df.iloc[i] = modified_row
            
            f.write(f"{i}\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    genrate_time_for_row = elapsed_time / PREDICT_ROWS
    print(f"""
{{
    'GENARATED_ROWS': {PREDICT_ROWS},
    'ELAPSED_TIME': '{elapsed_time} seconds',
    'GENERATE_TIME_FOR_ROW':'{genrate_time_for_row}',
    'LLM_MODEL': '{LLM_MODEL}' 
}}
        """)

    df.to_csv(f'{OUTPUT_FILE}.csv', index=False)
    

if __name__ == '__main__':
    main()