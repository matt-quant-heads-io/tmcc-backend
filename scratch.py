import requests
import os
from dotenv import load_dotenv
import openai
import pandas as pd
import json
import ast

# Load environment variables
load_dotenv()

# Get API keys from environment variables
POLYGON_API_KEY = "1ZIKxd5xw6LXMwO5R57iBuvBSuhBeJJV"
OPENAI_API_KEY = "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"

# Set up OpenAI API client
openai.api_key = OPENAI_API_KEY

def extract_entities_from_user_query(question):
    prompt = """ 
    Below are the descriptions of the values corresponding to each entity type:
        - 'ticker': a company ticker.
        - 'company': a company name.
        - 'dates': a list of dates mentioned in the user query. This should be a string in the form 'YYYY-MM-DD'. If there are multiple dates in the list they should be sorted such that the oldest date appears first and the most recent date appears last.

    The response should be a JSON list of dictionaries where each dictionary element contains a key called 'entitiy' whose value is the entity type extracted 
    (i.e. is one of the listed types above) and contains a key called 'value' that is the extracted value from the user query. If the entity type doesn't appear in the user 
    query then it should not appear in the output. 

    ## EXAMPLES
    user query: What is aapl's revenue for 2023?, answer: [('ticker', 'aapl')]
    user query: How has amzn's income tax trended for the past 10 years?, answer: [('ticker', 'amzn')]

    user query: {question}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an NLP extraction tool. Your task is to take a user query and extract the following entity types if they appear in the query: 'ticker', 'company', 'dates'."},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )

    try:
        response = response["choices"][0]["message"]["content"]
        entities = ast.literal_eval(response[response.find("["):response.find("]")+1])
        print(f'response["choices"][0]: {type(entities)}')
        return entities
    except Exception as e:
        print(f"Error parsing ChatGPT response: {e}")
        return []


def get_stock_financials(symbol, limit=5):
    """
    Fetch financial data for a given stock symbol using Polygon API.
    
    :param symbol: Stock symbol (e.g., 'AAPL' for Apple Inc.)
    :param limit: Number of financial reports to retrieve (default: 5)
    :return: JSON response containing financial data
    """
    base_url = f"https://api.polygon.io/vX/reference/financials?limit=100&apiKey={POLYGON_API_KEY}"
    
    params = {
        "ticker": symbol,
        "limit": limit
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        
        data = response.json()["results"]
        df = pd.DataFrame.from_records(data)

        # Flatten the 'financials' column
        financials_df = pd.json_normalize(df['financials'])
        cols_to_keep = []
        for col in list(financials_df.columns):
            statement, label, col_type = col.split(".")
            if statement in ["balance_sheet", "income_statement", "cash_flow_statement"] and col_type == "value":
                cols_to_keep.append(col)
        
        financials_df = financials_df[cols_to_keep]
        # Combine the main DataFrame with the flattened financials
        result_df = pd.concat([df.drop('financials', axis=1), financials_df], axis=1)
        
        result_df["end_date"] = pd.to_datetime(result_df["end_date"])
        result_df.sort_values(by=["end_date"], ascending=False)
        result_df["report_date"] = result_df["fiscal_period"].astype(str) + " " + result_df["fiscal_year"].astype(str)
        result_df = result_df[~result_df["report_date"].isin(["TTM", "TTM "])]
        result_df = result_df[[c for c in result_df.columns if c not in ['start_date', 'end_date', 'timeframe', 'fiscal_period', 'fiscal_year',
       'cik', 'sic', 'tickers', 'company_name', 'filing_date',
       'acceptance_datetime', 'source_filing_url', 'source_filing_file_url']]]
        result_df.set_index("report_date", inplace=True)
        result_df.rename({c: c.split(".")[1] for c in result_df.columns}, axis=1, inplace=True)

        return result_df.T
    else:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")


def get_relevant_rows(financials, question):
    """
    Prompt ChatGPT to identify the relevant rows from the financial data.

    :param financials: JSON response containing financial data
    :param question: User's question about the financial data
    :return: List of relevant row indices
    """
    # Prepare the prompt for ChatGPT
    prompt = f"""Given the following financial data for a company and the user's Question, return only the relevant values from the financial data table required to answer the question.
    The output should be a python list of string values:

    {financials}

    Question: {question}

    Please identify the rows in the supplied financials table that are necessary to answer the question.
    Return your answer as a Python list that contains only the row names.
    """

    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes financial data."},
            {"role": "user", "content": prompt.format(financials=financials, question=question)}
        ]
    )

    # Extract the relevant row indices from the API response
    try:
        response = response["choices"][0]["message"]["content"]
        relevant_rows = ast.literal_eval(response[response.find("["):response.find("]")+1])
        print(f'response["choices"][0]: {response}')
        # relevant_rows = eval(.strip())
        
        # relevant_rows = eval(response["choices"][0]["message"]["content"])
        # if not isinstance(relevant_rows, list) or not all(isinstance(i, int) for i in relevant_rows):
        #     raise ValueError("Invalid response format")
        return relevant_rows
    except Exception as e:
        print(f"Error parsing ChatGPT response: {e}")
        return []


def get_relevant_columns(financials, question):
    """
    Prompt ChatGPT to identify the relevant rows from the financial data.

    :param financials: JSON response containing financial data
    :param question: User's question about the financial data
    :return: List of relevant row indices
    """
    # Prepare the prompt for ChatGPT
    prompt = f"""Given the following financial data for a company and the user's Question, return only the relevant values from the financial data table required to answer the question.
    The output should be a python list of string values:

    {financials}

    Question: {question}

    Please identify the rows in the supplied financials table that are necessary to answer the question.
    Return your answer as a Python list that contains only the row names.
    """

    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes financial data."},
            {"role": "user", "content": prompt.format(financials=financials, question=question)}
        ]
    )

    # Extract the relevant row indices from the API response
    try:
        response = response["choices"][0]["message"]["content"]
        relevant_rows = ast.literal_eval(response[response.find("["):response.find("]")+1])
        
        return relevant_rows
    except Exception as e:
        print(f"Error parsing ChatGPT response: {e}")
        return []


def get_financials(question):
    entities = extract_entities_from_user_query(question)
    ticker = None
    for ent in entities:
        if ent["entity"] == "ticker":
            ticker = ent["value"]
            break
    financials = get_stock_financials(ticker.upper())
    relevant_rows = get_relevant_rows(list(financials.index.values), question)
    print(f"relevant_rows: {relevant_rows}")
    relevant_columns = get_relevant_columns(list(financials.columns), question)
    financials = financials.loc[relevant_rows, relevant_columns]

    print(f"financials: {financials}")




# Example usage
if __name__ == "__main__":
    try:
        question = "What was aapl's company's revenue growth over the last three years?"
        get_financials(question)

    except Exception as e:
        print(f"An error occurred!")
