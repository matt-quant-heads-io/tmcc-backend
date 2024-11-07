from datetime import datetime, timedelta
from scratch import extract_entities_from_user_query
import pandas as pd

import requests
import os
from dotenv import load_dotenv
import openai
import pandas as pd
import json
import ast

from openai import OpenAI
import cohere
import weaviate
from dateutil import parser
from typing import List, Dict, Any

from s3 import s3Handler
from pydantic import BaseModel
import numpy as np
import random
from datetime import datetime, timedelta

from dateutil import parser

import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

from scratch import determine_query_pattern, get_list_of_companies, map_list_of_companies_and_query_to_list_of_queries, map_competitors_and_query_to_list_of_queries, split_companies, preprocess_user_query, openai_client, cohere_client, weaviate_client, merge_charts, merge_frames, do_global_calculate
import yfinance as yahooFinance

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")


load_dotenv()


def process_queries(query):
    try:
        original_query = query
        pattern = determine_query_pattern(query)
        pattern = pattern.lstrip().rstrip()

        # import pdb; pdb.set_trace()
        
        if pattern.lower() == 'get_list_of_companies':
            print("Calling 'get_list_of_companies'")
            companies = get_list_of_companies(query)
            queries = map_list_of_companies_and_query_to_list_of_queries(query, companies)
        elif pattern.lower() == 'get_competitors':
            print("Calling 'get_competitors'")
            competitors = get_competitors(query)
            queries = map_competitors_and_query_to_list_of_queries(query, competitors) 
        elif pattern.lower() == 'split_companies':
            queries = split_companies(query)
        elif pattern.lower() == 'neither':
            queries = preprocess_user_query(query)
        else: 
            raise Exception(f"Got pattern from query pattern of {pattern.lower()}")
        return queries
    except Exception as e:
        print(f"An error occurred during research execution: {str(e)}")
        # logging.error(, exc_info=True)
    


def extract_entities_from_user_query_for_perform_quantitative_vector_search(question, debug=False):
    system_prompt = """ 
    You are an NLP extraction tool. Your task is to take a user query and extract the following entity types if they appear in the query: 'ticker', 'company', 'dates'.
    Below are the descriptions of the values corresponding to each entity type:
        - 'ticker': a company ticker.
        - 'company': a company name.
        - 'to_date': A date that should be a string in the form 'YYYY-MM-DD'. If there are multiple dates in the list they should be sorted such that the most recent date appears last and is mapped to the 'to_date' entity. If there are multiple dates the most recent date should be 
                    the 'to_date' entity and the oldest date should be the 'from_date' entity.
        - 'from_date': A date that should be a string in the form 'YYYY-MM-DD'. If there are multiple dates in the list they should be sorted such that the most recent date appears last and is mapped to the 'to_date' entity. If there are multiple dates the most recent date should be 
                    the 'to_date' entity and the oldest date should be the 'from_date' entity.
        

    The response should be a JSON list of dictionaries where each dictionary element contains a key called 'entity' whose value is the entity type extracted 
    (i.e. is one of the listed types above) and contains a key called 'value' that is the extracted value from the user query. If the entity type doesn't appear in the user 
    query then it should not appear in the output.

    ## EXAMPLES
    query: "Identify sections discussing 'supply chain disruptions' in the latest 10-Q filings for Apple (AAPL). Summarize the potential impacts on their operations.", answer: answer: [('ticker', 'AAPL'), ('from_date', '2024-09-30')]
    query: "Identify sections discussing 'supply chain disruptions' in the latest 10-Q filings for Boeing (BA). Summarize the potential impacts on their operations.", answer: answer: [('ticker', 'BA'), ('from_date', '2024-09-30')]
    query: "Identify sections discussing 'supply chain disruptions' in the latest 10-Q filings for Home Depot (HD). Summarize the potential impacts on their operations.", answer: answer: [('ticker', 'HD'), ('from_date', '2024-09-30')]
    query: "Compare the discussion of 'revenue growth strategies' in 10-K and 10-Q filings from 2018 to 2023 for Coca-Cola (KO). How has their strategy evolved over time?", answer: [('ticker', 'KO'), ('from_date', '2018-01-01'), ('to_date', '2023-12-31')]
    query: "Compare the discussion of 'revenue growth strategies' in 10-K and 10-Q filings from 2018 to 2023 for PepsiCo (PEP). How has their strategy evolved over time?", answer: [('ticker', 'PEP'), ('from_date', '2018-01-01'), ('to_date', '2023-12-31')] 
    query: "whats the variance between the inventory ratio and return on assets growth for first 3 quarters of 2021?", answer: [('ticker', 'AAPL'), ('from_date', '2021-01-01'), ('to_date', '2021-09-30')]
    query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL'), ('from_date', '2023-01-01'), ('to_date', '2023-12-31')]
    query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT'), ('from_date', '2022-09-30'), ('to_date', '2024-06-30')]
    query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT'), ('from_date', '2022-09-30'), ('to_date', '2024-06-30')]
    query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL'), ('from_date', '2023-01-01'), ('to_date', '2023-12-31')]
    query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT'), ('from_date', '2022-09-30'), ('to_date', '2024-06-30')]
    query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT'), ('from_date', '2022-09-30'), ('to_date', '2024-06-30')]
    query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL')]
    query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT'), ('from_date', '2022-09-30'), ('to_date', '2024-06-30')]
    query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT'), ('from_date', '2022-09-30'), ('to_date', '2024-06-30')]
    query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL'), ('from_date', '2023-01-01'), ('to_date', '2023-12-31')]
    query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT'), ('from_date', '2022-09-30'), ('to_date', '2024-06-30')]
    query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT'), ('from_date', '2022-09-30'), ('to_date', '2024-06-30')]
    query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    query: "what's ebay's revenue growth (%) quarter over quarter for 2022 versus its comps", answer: [('ticker', 'EBAY'), ('from_date', '2023-01-01'), ('to_date', '2023-12-31')]
    query: "compare the cogs between 2021 and 2023 between AMZN and its comps", answer: [('ticker', 'AMZN')]
    query: "compare the cogs between 2019 and 2021 between wmt and its comps", answer: [('ticker', 'WMT')]
    query: "compare the cogs between 2021 and 2022 between costco and its comps", answer: [('ticker', 'COST')]
    query: "Compare Amazon's logistics and fulfillment expenses as a percentage of net sales over the past 2 years. How have changes in these expenses impacted Amazon's operating margin, and what strategies have been implemented to optimize their supply chain efficiency?", answer: [('ticker', 'AMZN'), ('from_date', '2022-09-30'), ('to_date', '2024-06-30')]
    query: "Compare MSFT's cloud sales growth to IBM's", answer: [('ticker', 'MSFT'), ('ticker', 'IBM')]
    query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", answer: [('ticker', 'JNJ'), ('ticker', 'MRK'), ('from_date', '2021-09-30'), ('to_date', '2024-06-30')]
    query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", answer: [('ticker', 'JNJ'), ('ticker', 'MRK'), ('from_date', '2021-09-30'), ('to_date', '2024-06-30')]
    query: "what's amzn's revenue growth during 2023 versus its comps", answer: [('ticker', 'AMZN'), ('from_date', '2023-01-01'), ('to_date', '2023-12-31')]
    query: "How has apple’s gross margins trended compared to Intel and Microsoft’s over the past 3 years?",answer: [('ticker', 'AAPL'), ('ticker', 'INTC'), ('ticker', 'MSFT'), ('from_date', '2021-09-30'), ('to_date', '2024-06-30')]
    query: "How many times did aapl discuss artificial intelligence in their most recent 10K or 10Q?", answer: [('ticker', 'AAPL'), ('from_date', '2024-06-30')]
    """

    prompt = """
    query: {question}
    """

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ],   
    )

    try:
        content = json.loads(response.to_json())["choices"][0]["message"]["content"]
        if "json" in content:
            entities = ast.literal_eval(content[content.find("```json\n")+len("```json\n"):len(content)-3])
        else:
            entities = ast.literal_eval(content)    

        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "extract_entities_from_user_query", "inputs": [question], "outputs": [{"entities": entities}]}, indent=6))
        return entities
    except Exception as e:
        print(f"Error inside extract_entities_from_user_query: {e}")
        return []


def text_to_graphql(question, entities, debug=False):
    # import pdb; pdb.set_trace()
    system_prompt = """Generate a GraphQL query for a Weaviate backend database that retrieves 10Q and 10K company financials filings based on specified filters, a user-provided question, and the specified collection.
    When applying filing_type filter if you see '10-Q' in the Question map it to '10Q'. Likewise, when you see '10-K' in the Question map it to '10K'. In other words, do not include the hyphen for the 'filing_type' filter.
    Also next include the graphql keyword 'limit' in your response since we do not want to limit the number of retrieved documents arbitrarily. If the question contains both '10-K' and '10-Q' then DO NOT use the 'filing_type'
    as a filter in your graphql response.

    # Input Details
    - **Collection**
    - **Filters**: These include:
    - `accession_number`: String
    - `company_name`: String
    - `filing_type`: String (e.g., 10Q, 10K)
    - `page_number`: Float
    - `report_date`: String
    - `filing_url`: String
    - `text`: String (search within text)
    - `ticker`: String

    # Steps

    1. **Interpret the User's Question**: Understand the specific data or insights the user is seeking from the database.
    2. **Identify Relevant Filters**: Determine which filters from the provided list apply based on the user's question and goals.
    3. **Format GraphQL Query**:
    - Use the filters to construct a GraphQL query.
    - Select the relevant fields from the database for the query.

    # Output Format

    - The output should be a GraphQL query string formatted to fetch data from the Weaviate database using the specified filters and question context. Ensure correct syntax and structure of the GraphQL query.

    # Examples

    **Example Input**: 
    Question: "how many times did aapl mention macro demand concerns in their filings since 2023?"
    Collection: "Dow30_10K_10Q"
    Filters: { "ticker": "AAPL",  "from_date":  "2023-01-01T00:00:00.00Z", "to_date": "2024-06-30T00:00:00.00Z"}

    **Example Output**:
    ```graphql
    {
        Get {
            Dow30_10K_10Q(
            nearText: {concepts: ["macro demand concerns"]}
            where: {operator: And, operands: [{path: ["ticker"], operator: Equal, valueText: "AAPL"}, {path: ["report_date"], operator: GreaterThanEqual, valueDate: "2023-01-01T00:00:00.00Z"}, {path: ["report_date"], operator: LessThanEqual, valueDate: "2024-06-30T00:00:00.00Z"}]}
            ) {
            ticker
            text
            report_date
            accession_number
            company_name
            filing_type
            page_number
            filing_url
            }
        }
    }
    ```

    **Example Input**: 
    Question: "how many times did MSFT mention the activision anti-trust case in their filings since 2021?"
    Collection: "Dow30_10K_10Q"
    Filters: { "ticker": "MSFT",  "from_date":  "2021-01-01T00:00:00Z"}

    **Example Output**:
    ```graphql
    {
        Get {
            Dow30_10K_10Q(
            nearText: {concepts: ["activision anti-trust case"]}
            where: {operator: And, operands: [{path: ["ticker"], operator: Equal, valueText: "MSFT"}, {path: ["report_date"], operator: GreaterThanEqual, valueDate: "2021-01-01T00:00:00Z"}]}
            ) {
            ticker
            text
            report_date
            accession_number
            company_name
            filing_type
            page_number
            filing_url
            }
        }
    }
    ```

    **Example Input**: 
    Question: "how many times did MSFT mention the activision anti-trust case in their filings since 2021?"
    Collection: "Dow30_10K_10Q"
    Filters: { "ticker": "MSFT",  "from_date":  "2021-01-01T00:00:00Z"}

    **Example Output**:
    ```graphql
    {
        Get {
            Dow30_10K_10Q(
            nearText: {concepts: ["activision anti-trust case"]}
            where: {operator: And, operands: [{path: ["ticker"],  operator: Equal, valueText: "MSFT"}, {path: ["report_date"], operator: GreaterThanEqual, valueDate: "2021-01-01T00:00:00Z"}]}
            ) {
            ticker
            text
            report_date
            accession_number
            company_name
            filing_type
            page_number
            filing_url
            }
        }
    }
    ```

    **Example Input**: 
    Question: "how many times did aapl raise concerns about supply chain issues in their filings between 2017-06-27 and 2023-06-27?"
    Collection: "Dow30_10K_10Q"
    Filters: { "ticker": "AAPL",  "from_date":  "2017-06-27T00:00:00Z", "to_date": "2023-06-27T00:00:00Z"}

    **Example Output**:
    ```graphql
    {
        Get {
            Dow30_10K_10Q(
            nearText: {concepts: ["supply chain issues"]}
            where: {operator: And, operands: [{path: ["ticker"], operator: Equal, valueText: "AAPL"}, {path: ["report_date"], operator: GreaterThanEqual, valueDate: "2017-06-27T00:00:00Z"}, {path: ["report_date"], operator: LessThanEqual, valueDate: "2023-06-27T00:00:00Z"}]}
            ) {
            ticker
            text
            report_date
            accession_number
            company_name
            filing_type
            page_number
            filing_url
            }
        }
    }
    ```

    **Example Input**: 
    Question: "how many times did msft raise concerns about supply chain issues in their filings in the last 3 years?"
    Collection: "Dow30_10K_10Q"
    Filters: { "ticker": "MSFT",  "from_date":  "2021-06-30T00:00:00Z"}

    **Example Output**:
    ```graphql
    {
        Get {
            Dow30_10K_10Q(
            nearText: {concepts: ["supply chain issues"]}
            where: {operator: And, operands: [{path: ["ticker"], operator: Equal, valueText: "MSFT"}, {path: ["report_date"], operator: GreaterThanEqual, valueDate: "2021-06-30T00:00:00Z"}]}
            ) {
            ticker
            text
            report_date
            accession_number
            company_name
            filing_type
            page_number
            filing_url
            }
        }
    }

    **Example Input**: "how many times did aapl discuss artificial intelligence in their most recent 10K or 10Q?"
    Collection: "Dow30_10K_10Q"
    Filters: { "ticker": "AAPL"}

    **Example Output**:
    ```graphql
    {
        Get {
            Dow30_10K_10Q(
            nearText: {concepts: ["artificial intelligence"]}
            where: {operator: And, operands: [{path: ["ticker"], operator: Equal, valueText: "AAPL"}]}
            ) {
            ticker
            text
            report_date
            accession_number
            company_name
            filing_type
            page_number
            filing_url
            }
        }
    }

    **Example Input**: "how many times has CAT raised international growth concerns in their filings over the past 2 years?"
    Collection: "Dow30_10K_10Q"
    Filters: { "ticker": "AAPL"}

    **Example Output**:
    ```graphql
    {
        Get {
            Dow30_10K_10Q(
            nearText: {concepts: ["international growth concerns"]}
            where: {operator: And, operands: [{path: ["ticker"], operator: Equal, valueText: "CAT"}, {path: ["report_date"], operator: GreaterThanEqual, valueDate: "2022-09-30T00:00:00.00Z"}, {path: ["report_date"], operator: LessThanEqual, valueDate: "2024-06-30T00:00:00.00Z"}]}
            ) {
            ticker
            text
            report_date
            accession_number
            company_name
            filing_type
            page_number
            filing_url
            }
        }
    }
    '''
    """
    user_prompt = """
    Question: {question}
    Collection: {collection}
    Filters: {filters}
    """
    # question = "how many times did msft raise concerns about supply chain issues in their filings in the last 3 years?"
    # filters = { "ticker": "MSFT",  "from_date":  "2021-06-30T00:00:00Z"}
    # import pdb; pdb.set_trace()
    processed_entities = []
    for entity in entities:
        if entity["entity"] in ["from_date", "to_date"]:
            processed_entities.append({entity["entity"]: f'{entity["value"]}T00:00:00.00Z'})
        else:
            processed_entities.append({entity["entity"]: entity["value"]})
    
    collection = "Dow30_10K_10Q"

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=question, collection=collection, filters=processed_entities)}
        ],
    )

    try:  
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        if "```graphql" in response:
            response = response.replace("```graphql", "").replace("```", "")
        # import pdb; pdb.set_trace()
        print(f"graphql:\n{response}")
        return response
    except Exception as e:
        print(f"Error inside text_to_graphql: {e}")
        return []


def run_graphql_query_against_weaviate_instance(query):
    "curl http://localhost/v1/graphql -X POST -H 'Content-type: application/json' -d '{GraphQL query}'"
    url = f"{WEAVIATE_URL}/v1/graphql"
    payload = {
        "operationName": "",
        "query": query,
        "variables": {}
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WEAVIATE_API_KEY}",
        "X-Openai-Api-Key": OPENAI_API_KEY
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        succeeded = True
    else:
        succeeded = False
    # import pdb; pdb.set_trace()

    print(response.json())
    return response.json(), succeeded


def do_calculate_for_qual_and_quant(question, qual_and_quant_df):
    try:
        system_prompt = """ 
        You are a tool used in a rag system to determine an equation that needs to be applied to a pandas dataframe. Your job is to generate 
        the string representation of the pandas calculation such that the python eval function can be called on the string.
        Your response should not contain anything other than the pandas equation to be called. Including anything other than just the equation 
        will cause the functin to fail. 
        
        Given a user query and the supplied json that represents the pandas DataFrame, return the equation to apply to the DataFrame that performs the calculation
        necessary to answer the question. Below are some examples of user query, json data, and response triplets. Assume that the pandas DataFrame is called 'qual_and_quant_df'.
        Use the name qual_and_quant_df to represent that pandas DataFrame in your response. 

        Examples:
        'user query': "how many times did msft raise concerns about supply chain issues in their filings in the last 3 years?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did msft raise concerns about supply chain issues in their filings in the last 3 years?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did msft raise concerns about supply chain issues in their filings in the last 3 years?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did msft raise concerns about supply chain issues in their filings in the last 3 years?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True); qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True); qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True); qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True); qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True); qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True); qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True); qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True); qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True); qual_and_quant_df.drop(['accession_number', 'text', 'ticker'], axis=1, inplace=True); qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?", 'data': ['accession_number', 'company_name', 'filing_type', 'page_number', 'report_date', 'url', 'text', 'ticker'], 'answer': "qual_and_quant_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True); qual_and_quant_df.drop(['accession_number', 'text', 'ticker'], axis=1, inplace=True); qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        """

        prompt = """
        'user query': {question}
        'data': {qual_and_quant_df}
        """

        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.format(question=question, qual_and_quant_df=list(qual_and_quant_df.columns))}
            ],
        )
        
        # import pdb; pdb.set_trace()
        new_response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        if "```python" in new_response:
            new_response = new_response[9: len(new_response)-3]


        print(f"'user query': {question}")
        print(f"'data': {list(qual_and_quant_df.columns)}")
        print(f"'answer': {new_response}")
        # import pdb; pdb.set_trace()
        if "filing_url" in new_response:
            new_resposne = new_response.replace('filing_url', 'url')
        exec(new_response)

        if isinstance(qual_and_quant_df, pd.Series):
            qual_and_quant_df = qual_and_quant_df.to_frame()
            # new_qual_and_quant_df.dropna(inplace=True)
        elif isinstance(qual_and_quant_df, pd.DataFrame):
            pass
        else:
            qual_and_quant_df = pd.DataFrame({"result": qual_and_quant_df})
        
        # import pdb; pdb.set_trace()
        return qual_and_quant_df
    except Exception as e:
        print(f"Error inside do_calculate_for_qual_and_quant: {e}")
        import pdb; pdb.set_trace()
        return qual_and_quant_df



def should_do_calculate_for_qual_and_quant(question, debug=False):
    system_prompt = """ 
    You are a tool used to guide a RAG system. Your job is to determine where a calculator function needs to be called or not.
    Your response should be a string that is either 'True' or 'False'. Your response should not contain anything other than either 
    'True' or 'False'. Given a user query and a supplied json contain data used to answer the user determine wheter a calculation needs
    to be executed or if the current raw json data is sufficient to answer the quesiton and thus doesn't require any further calculations.
    Below are some examples of user query, json data, and response triplets.

    Examples:
    user query: "How has apple's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if i bought the stock each time?", answer: "False"
    user query: "How has apple's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if i bought the stock each time?", answer: "False"
    user query: "How has apple's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if i bought the stock each time?", answer: "False"
    user query: "How has apple's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if i bought the stock each time?", answer: "False"
    user query: "how many times did AAPL mention macro concerns in their filings since 2023? whats been the stock performance during these instances?", answer: "True"
    user query: "how many times did MSFT mention the activision anti-trust case in their filings since 2021?", answer: "True"
    user query: "how many times did AAPL mention macro concerns in their filings since 2023? whats been the stock performance during these instances?", answer: "True"
    user query: "how many times did MSFT mention the activision anti-trust case in their filings since 2021?", answer: "True"
    
    """

    prompt = """
    user query: {question}
    """

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ],
        
    )

    try:
        
        import pdb 
        # pdb.set_trace()
        # print(f"financials inside should_local_calculate: {financials.to_json()}")
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        calculation_required = ast.literal_eval(response)
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "should_local_calculate", "inputs": [question, financials], "outputs": [{"calculation_required": calculation_required}]}, indent=6))
        
        print(f"response from should_do_calculate_for_qual_and_quant: {response}")
        return calculation_required
    except Exception as e:
        print(f"Error inside should_do_calculate_for_qual_and_quant: {e}")
        return False


def perform_vector_search(question, results, debug=False):
    try:
        entities = extract_entities_from_user_query(question, debug)
        filters = []
        to_date = None
        from_date = None
        ticker = None
        for entity in entities:
            if entity["entity"] == "from_date":
                filters.append(
                    {"path": ["report_date"], "operator": "GreaterThanEqual", "valueDate": f'{entity["value"]}T00:00:00.00Z'}
                )
            elif entity["entity"] == "to_date":
                to_date = entity["value"]
                filters.append(
                    {"path": ["report_date"], "operator": "LessThanEqual", "valueDate": f'{entity["value"]}T00:00:00.00Z'}
                )
            elif entity["entity"] == "ticker":
                ticker = entity["value"].upper()
                filters.append(
                    {"path": ["ticker"], "operator": "Equal", "valueText": ticker}
                )
                
        response = (
            weaviate_client.query
            .get("Dow30_10K_10Q", ["filing_type", "logo", "company_name", "ticker", "accession_number", "filing_url", "text", "page_number", "report_date"])
            .with_near_text({
                "concepts": [question]
            })
            .with_where({
                "operator": "And",
                "operands": filters
            })
            .with_additional(["distance"])
            .do()
        )

        if "data" not in response:
            print(f"[FAILED] to get response for weaviate query in perform_vector_search")
            results["Context"].append(
                f"I failed to retrieve documents from my vector search for the query: {question}"
            )
            return results

        results_df = pd.DataFrame(response["data"]['Get'])
        if len(results_df) > 0:
            results_df = pd.DataFrame(response["data"]['Get']["Dow30_10K_10Q"])
            if len(results_df) == 0:
                print(f"[WARNING] Vector search retrieved 0 records!")
                return results_df

            results_df['importance'] = results_df['_additional'].apply(lambda x: 1.0 - x['distance'])
            results_df.drop("_additional", axis=1, inplace=True)
            results_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True)
            results_df["temp_date_sort_key"] =  pd.to_datetime(results_df["report_date"])
            results_df.sort_values(by=["temp_date_sort_key"], ascending=False, inplace=True)
            results_df.drop("temp_date_sort_key", axis=1, inplace=True)
            results_df.rename({"filing_url": "url"}, axis=1, inplace=True)
            results_df.reset_index(drop=True, inplace=True)
            
            results["Context"].append(
                f"Database results for query (ticker={ticker}): {question} \n\n{results_df.to_json()}"
            )
            citations = [{"text": t["text"], "id": "", "logo": "", "page_number": t["page_number"], "url": t["filing_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0} for t in response["data"]["Get"]["Dow30_10K_10Q"]]
            citations_for_backend = [{"report_date": f'{t["report_date"].split("T")[0]}', "text": t["text"], "id": "", "logo": "", "page_number": t["page_number"], "url": t["filing_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0} for t in response["data"]["Get"]["Dow30_10K_10Q"]]
            citations = sorted(citations, key=lambda d: d['importance'], reverse=True)
            
            df_citations_frontend = pd.DataFrame.from_records(citations)
            citations = df_citations_frontend.to_dict(orient='records')
            df_citations = pd.DataFrame.from_records(citations_for_backend)
            
            if "citations" not in results["finalAnalysis"]:
                results["finalAnalysis"]["citations"] = []    
            results["finalAnalysis"]["citations"].extend(citations)

            if "insights" not in results["finalAnalysis"]:
                results["finalAnalysis"]["insights"] = [] 

            if len(results['GetCompanyFinancials']) > 0 and ticker.upper() in results['GetCompanyFinancials']:
                company_financials_df = results['GetCompanyFinancials'][ticker]
                
                merge_key = None
                if 'Q' in company_financials_df['report_date'][0]:
                    merge_key = 'Calendar Date'
                else:
                    merge_key = f'report_date'

                results_df.rename({'report_date': merge_key}, axis=1, inplace=True)
                results_df[merge_key] = pd.to_datetime(results_df[merge_key].str[:10]).dt.strftime('%Y-%m-%d')
                results_df["temp_date_sort_key"] = pd.to_datetime(results_df[merge_key])
                results_df.sort_values(by=["temp_date_sort_key"], ascending=False, inplace=True)
                results_df.drop("temp_date_sort_key", axis=1, inplace=True)
                results_df.reset_index(drop=True, inplace=True)

                results_df = results_df.merge(company_financials_df, left_on=merge_key, right_on=merge_key)

                results['VectorSearch'][ticker.upper()] = results_df
                del results['GetCompanyFinancials'][ticker.upper()]

                results["finalAnalysis"]["citations"].extend(citations)
                results["finalAnalysis"]["tables"][ticker.upper()] = results_df
                return results
            else:
                results["VectorSearch"][ticker.upper()] = results_df
                results["finalAnalysis"]["citations"].extend(citations)
                results["finalAnalysis"]["tables"][ticker.upper()] = results_df
                
                response = '\n\n'.join([c["text"] for c in citations])
                results["Context"].append(
                    f"Response to Query: {question} \n\n{response}"
                ) 

                return results

            if debug:
                with open(DEBUG_ABS_FILE_PATH, "a") as f:
                    f.write(json.dumps({"function": "perform_vector_search", "inputs": [question], "outputs": [{"response": response}]}, indent=6))
        
        else:
            results["Context"].append(
                f"Response to Query (retrieved no results): {question} \n\n{response}"
            )
        
        return results
    except Exception as e:
        print(f"Error inside perform_vector_search: {e}")
        return results



def get_final_analysis(query, results, debug=False):
    try:
            # import pdb; pdb.set_trace()
        final_analysis = results["finalAnalysis"]
        context = results["Context"]

        # import pdb; pdb.set_trace()

        if "charts" in results["finalAnalysis"]:
            results = merge_charts(results)
        # if "tables" in results["finalAnalysis"]:
        #     results = merge_tables(results)
        results = merge_frames(results)
        

        if "workbookData" not in results["finalAnalysis"]:
            results["finalAnalysis"]["workbookData"] = None
        # import pdb; pdb.set_trace()
        if "ResultsFromGetFinancials" in results:
            should_calculate = False #should_global_calculate(query, df_temp, debug)
            print(f"should_global_calculate: {should_calculate}")
            if should_calculate:
                results = do_global_calculate(query, results["finalAnalysis"]["tables"], debug)
                print(f"results after do_global_calculate: {results}")

        # print(f"context: {context}")
        context = '\n'.join([str(c) for c in context])

        # import pdb; pdb.set_trace()

        system_content = """
        Given a user-supplied query and a context that contains the raw data to necessary to answer the query, generate a response that synthesizes the provided context to answer the question.
        """

        prompt = """
        query: {query}

        context: {context}
        """

        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt.format(query=query, context=context)}
            ],
            
        )
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]

        if "finalAnalysis" not in results:
            results["finalAnalysis"] = {

            }
        if "insights" not in results["finalAnalysis"]:
            results["finalAnalysis"]["insights"] = []

        results["finalAnalysis"]["insights"].append({
            "title": f"Response to Query: {query}",
            "content": f"\n\n{response}"
        })

        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "get_final_analysis", "inputs": [query], "outputs": [{"response": response}]}, indent=6))
        # print(f"results: {results}")
        # import pdb; pdb.set_trace()
        return results
    except Exception as e:
        print(f"Error parsing ChatGPT response: {e}")
        return results

vector_search_answers = []
queries = [
    "How have AAPL, BA, and Home Depot (HD) dealt with supply chain constraints? What strategies have they implemented?",
    "What's the status of aapl's anti-trust cases since 2022?",
    "Analyze the sentiment of the forward-looking statements in the latest 10-K filings for JPMorgan Chase (JPM) and Goldman Sachs (GS). Identify whether the tone is predominantly positive, negative, or neutral.",
    "Compare the discussion of 'revenue growth strategies' in 10-K and 10-Q filings from 2018 to 2023 for Coca-Cola (KO) and PepsiCo (PEP). How have these strategies evolved over time?",
    "Search for mentions of 'inflation' and 'interest rates' in 10-Q filings for consumer goods companies like Walmart (WMT) and McDonald's (MCD) from Q1 2022 to Q2 2023. What concerns or strategies are reported in relation to these economic indicators?"
]
for query in queries:
    results = {"Query": query, "Context": [], "Execution": {}, "finalAnalysis": {"tables": {}, "charts": {}}, "MarketData": {}, "MarketDataForBacktest": {}, "QualAndQuant": {}, "GetNews": {}, "GetCompanyFinancials": {}, "GetEstimates": {}, "RunBackTest": {}, "VectorSearch": {}, "Tables": {}}
    processed_queries = process_queries(query)
    for processed_query in processed_queries:
        results = perform_vector_search(processed_query, results, debug=False)

    # import pdb; pdb.set_trace()
    vector_search_answers.append({
        "OriginalQuery": query,
        "ProcessedQueries": processed_queries,
        "Results": {k:v.to_dict(orient="records") for k,v in results['VectorSearch'].items()}
    })
    results = {"Query": query, "Context": [], "Execution": {}, "finalAnalysis": {"tables": {}, "charts": {}}, "MarketData": {}, "MarketDataForBacktest": {}, "QualAndQuant": {}, "GetNews": {}, "GetCompanyFinancials": {}, "GetEstimates": {}, "RunBackTest": {}, "VectorSearch": {}, "Tables": {}}




with open("/home/ubuntu/tmcc-backend/batch_queries_test_vector_search.json", "w") as f:
    f.write(json.dumps(vector_search_answers))
    
