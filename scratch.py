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

from s3 import s3Handler


load_dotenv()

# For highlighting
LOCAL_DOWNLOAD_PATH = "/home/ubuntu/tmcc-frontend-reactapp/public/citations" #"/home/ubuntu/tmcc-backend/citations" #"/home/ubuntu/hattan-backend/temp_download_path_for_highlighting"
BUCKET_FOR_HIGHLIGHTED_DOCS = "highlighted-docs"
BUCKET_FOR_UNHIGHLIGHTED_DOCS = "sec-filings2" 
DEBUG_ABS_FILE_PATH = "/home/ubuntu/tmcc-backend/debug.json"

s3_handler = s3Handler()    

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
)
cohere_client = cohere.Client(api_key=cohere_api_key)
weaviate_client = weaviate.Client(
    url=WEAVIATE_URL,                                    # Replace with your Weaviate Cloud URL
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    additional_headers = {
        "X-OpenAI-Api-Key": OPENAI_API_KEY  # Replace with your inference API key
    }
)


def preprocess_user_query(question, debug=False):
    system_prompt = """ 
    You are an NLP tool. Given a user query, your task is to determine if the user query contains a company name. If it does,
    replace the company name with its corresponding ticker and respond with the new query (which should now contain the ticker instead of the company name).
    If the query doesn't contain a company name then respond with the EXACT query that was supplied initially. If the query already contains a ticker then respond with the EXACT query that was supplied initially. 
    Your response should ONLY contain the query and nothing else, no preamble. Below are some example user queries and their corresponding responses.

    Examples:
    query: "whats amzn's quarterly revenue for 2022? compare this to its comps", Answer: "whats amzn's quarterly revenue for 2022? compare this to its comps"
    query: "whats Amazon's quarterly revenue for 2022? compare this to its comps", Answer: "whats amzn's quarterly revenue for 2022? compare this to its comps"
    query: "whats target's cogs between 2021 and 2023?", Answer: "whats TGT's cogs between 2021 and 2023?"

    """

    prompt = """
    user query: {question}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )

    try:
        response = json.loads(response.json())
        processed_query = response["choices"][0]["message"]["content"]
        entities = extract_entities_from_user_query(processed_query, debug)
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "preprocess_user_query", "input": [question], "processed_query": processed_query, "entities": entities}, indent=6))
        return processed_query, entities
    except Exception as e:
        print(f"Error inside processed_query: {e}")
        return []


def get_list_of_companies(question, debug=False):
    system_prompt = """ 
    You are a hedge fund analyst tasked with identifying a list of public companies that fit the criteria given by a user query. Given a user query return a python list of company tickers. 
    Each company ticker in the list should be a string ticker. Your response should only contain the list of tickers that are companies that satisfy the requirement outlined by the user query.
    """

    prompt = """
    user query: {question}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )

    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]
        competitors = ast.literal_eval(response[response.find("["):response.find("]")+1])
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "get_list_of_companies", "inputs": [question], "outputs": [{"competitors": competitors}]}, indent=6))
        return competitors
    except Exception as e:
        print(f"Error inside get_list_of_company: {e}")
        return []


def should_local_calculate(question, financials, debug=False):
    system_prompt = """ 
    You are a tool used to guide a RAG system. Your job is to determine where a calculator function needs to be called or not.
    Your response should be a string that is either 'True' or 'False'. Your response should not contain anything other than either 
    'True' or 'False'. Given a user query and a supplied json contain data used to answer the user determine wheter a calculation needs
    to be executed or if the current raw json data is sufficient to answer the quesiton and thus doesn't require any further calculations.
    Below are some examples of user query, json data, and response triplets.

    Examples:
    user query: "What was microsoft's effect of exchange rate on Cash and Cash Equivalents in Q3 2020 and Q4 2020?", data: {"Q3 2020":{"cash_and_cash_equivalents":10000000000.0,"exchange_rate":1.0},"Q4 2020":{"cash_and_cash_equivalents":10000000000.0,"exchange_rate":1.0}}, response: "False"
    user query: "what's amzn's revenue for 2022 versus its comps", data: {"revenue": {"2022": [1000000000000, 1000000000025, 4000000000000, 3000000000000]}}, response: "False"
    user query: "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenue": {"2022": [1000000000000, 1000000000025, 4000000000000, 3000000000000]}}, response: "True"
    user query: "what's tgt's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenues":{"Q1 2022":24197000000.0,"Q2 2022":25160000000.0,"Q3 2022":25652000000.0,"Q4 2022":30996000000.0}}, response: "True"
    user query: "what's wmt's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenues":{"Q1 2022":116444000000.0,"Q2 2022":121234000000.0,"Q3 2022":127101000000.0,"Q4 2022":149204000000.0}}, response: "True"
    user query: "What was the total amount spent on Acquisitions Net of Cash Acquired and Purchases of Intangible and Other Assets by MSFT in the year 2022?", data: {"Q4 2022":{"cash":104757000000.0,"intangible_assets":null,"fixed_assets":74398000000.0,"net_cash_flow_from_investing_activities":-9729000000.0},"Q3 2022":{"cash":104693000000.0,"intangible_assets":null,"fixed_assets":70298000000.0,"net_cash_flow_from_investing_activities":-16171000000.0},"Q2 2022":{"cash":125369000000.0,"intangible_assets":null,"fixed_assets":67214000000.0,"net_cash_flow_from_investing_activities":-1161000000.0},"Q1 2022":{"cash":130615000000.0,"intangible_assets":null,"fixed_assets":63772000000.0,"net_cash_flow_from_investing_activities":-3250000000.0}}, response: "True"
    user query: "What was the trend in MSFT's Payments for Repurchase of Common Stock from Q3 2020 to Q1 2023?", data: {"Q3 2020":{"payments_for_repurchase_of_common_stock":-10000000000.0},"Q1 2023":{"payments_for_repurchase_of_common_stock":-10000000000.0}}, response: "False"
    """

    prompt = """
    user query: {question} data: {financials}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, financials=financials.to_json())}
        ]
    )

    try:
        response = json.loads(response.json())
        print(f"financials inside should_local_calculate: {financials.to_json()}")
        response = response["choices"][0]["message"]["content"]
        calculation_required = ast.literal_eval(response)
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "should_local_calculate", "inputs": [question, financials], "outputs": [{"calculation_required": calculation_required}]}, indent=6))
        
        print(f"response from should_local_calculate: {response}")
        return calculation_required
    except Exception as e:
        print(f"Error inside should_local_calculate: {e}")
        return []

def do_local_calculate(question, financials, debug=False):
    system_prompt = """ 
    You are a tool used in a rag system to determine an equation that needs to be applied to a pandas dataframe. Your job is to generate 
    the string representation of the pandas calculation such that the python eval function can be called on the string.
    Your response should not contain anything other than the pandas equation to be called. Including anything other than just the equation 
    will cause the functin to fail. 
    
    Given a user query and the supplied json that represents the pandas DataFrame, return the equation to apply to the DataFrame that performs the calculation
    necessary to answer the question. Below are some examples of user query, json data, and response triplets. Assume that the pandas DataFrame is called 'financials'.
    Use the name financials to represent that pandas DataFrame in your response. 

    Examples:
    user query: "what was the total amount spent on acquisitions net of cash acquired and purchases of intangible and other assets by msft in the year 2022?", data: {"Q4 2022":{"cash":104757000000.0,"intangible_assets":null,"fixed_assets":74398000000.0,"net_cash_flow_from_investing_activities":-9729000000.0},"Q3 2022":{"cash":104693000000.0,"intangible_assets":null,"fixed_assets":70298000000.0,"net_cash_flow_from_investing_activities":-16171000000.0},"Q2 2022":{"cash":125369000000.0,"intangible_assets":null,"fixed_assets":67214000000.0,"net_cash_flow_from_investing_activities":-1161000000.0},"Q1 2022":{"cash":130615000000.0,"intangible_assets":null,"fixed_assets":63772000000.0,"net_cash_flow_from_investing_activities":-3250000000.0}}, response: "df['net_cash_flow_from_investing_activities'].sum() - (df['cash'][-1] - df['cash'][0]) - (df['intangible_assets'][-1] - df['intangible_assets'][0])"
    user query: "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenues":{"Q1 2022":116444000000.0,"Q2 2022":121234000000.0,"Q3 2022":127101000000.0,"Q4 2022":149204000000.0}}, response: "financials['revenues'].pct_change().round(2)"
    user query: "what's WMT's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenues":{"Q1 2022":116444000000.0,"Q2 2022":121234000000.0,"Q3 2022":127101000000.0,"Q4 2022":149204000000.0}}, response: "financials['revenues'].pct_change().round(2)"
    user query: "how did msft's net cash provided by operating activities change from q3 2020 to q1 2021?", data: '{"net_cash_flow_from_operating_activities":{"Q3 2020":17504000000.0,"Q1 2021":19335000000.0},"net_cash_flow_from_operating_activities_continuing":{"Q3 2020":17504000000.0,"Q1 2021":19335000000.0}}',response: "df['net_cash_flow_from_operating_activities'].pct_change().round(2)"
    """

    prompt = """
    user query: {question}
    json data: {financials}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, financials=financials[::-1])}
        ]
    )

    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]

        # import pdb; pdb.set_trace()
        new_financials = eval(response)

        if isinstance(new_financials, pd.Series):
            new_financials = new_financials.to_frame()
            new_financials.dropna(inplace=True)
        else:
            new_financials = pd.DataFrame({"result": new_financials})
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "do_local_calculate", "inputs": [question, financials], "outputs": [{"new_financials": new_financials}]}, indent=6))
        
        import pdb; pdb.set_trace()
        return new_financials
    except Exception as e:
        print(f"Error inside do_local_calculate: {e}")
        return []


def get_competitors(question, debug=False):
    system_prompt = """ 
    You are a hedge fund analyst tasked with identifying company competitors. Gien a user query return a python list of company competitors. 
    Each competitor in the list should be a string ticker. Your response should only contain the list of tickers that are competitors to
    the company or ticker mentioned in the user query. Do not include ANYTHING other than the python list of only the ticker values!
    """

    prompt = """
    user query: {question}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )

    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]
        # print(f"response: {response}")
        competitors = ast.literal_eval(response[response.find("["):response.find("]")+1])
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "get_competitors", "inputs": [question], "outputs": [{"competitors": competitors}]}, indent=6))
        return competitors
    except Exception as e:
        print(f"Error inside get_competitors: {e}")
        return []


def get_research_plan(question, debug=False):
    system_content = """
    You are the task planner on an elite team of world-class financial researchers. Your team delivers insightful research  based on any company or industry and produce factual based results for your clients.

    You job is to receive a query from the client, and break it into a series of sub-tasks that are assigned to one of the tools to complete. 

    Think step by step through the process of how your team will get to the answer. When describing the task, just give a quick few words in bullet format.

    Research Assistants:
    'get_competitors': Finds companies or competitors for a given query
    'get_sec_financials': Pulls reported financials from SEC filings
    'get_stock_price_date': Pulls stock market price data
    'get_final_analysis': Produces a final analysis of the research
    'perform_vector_search': Performs (predominantly) qualitative queries against internal market databses

    Use the JSON as shown in the below examples when producing your response.

    Task Breakdown Examples:
    {
    "queries": [
        {
            "query": "how has GOOG framed recent anti trust references in their filings?",
            "tasks": [
                {
                "task": "perform_vector_search",
                "description": "Perform a vector search of the internal market database",
                "status": "pending"
                },
                {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
                },
            ]
        },
        {
            "query": "what's amzn's revenue trend?",
            "tasks": [
                {
                "task": "get_sec_financials",
                "description": "For each identified company, gather recent financial statements, focusing on income statements over the last few quarters or years.",
                "status": "pending"
                },
                {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
                },
            ]
        },
        {
            "query": "what's amzn's revenue trend versus its comps?",
            "tasks": [
                {
                "task": "get_competitors",
                "description": "Find company competitors to run a comprehensive comprative analysis",
                "status": "pending"
                },
                {
                "task": "get_sec_financials",
                "description": "For each identified company, gather recent financial statements, focusing on income statements over the last few quarters or years.",
                "status": "pending"
                },
                {
                "task": "get_final_analysis",
                "description": "Produce a comprehensive final analysis from the retrieved financial statements.",
                "status": "pending"
                },
            ]
        },
        {
        "query": "How does Rivian’s sales growth compare to other EV manufacturers?",
        "tasks": [
            {
                "task": "get_competitors",
                "description": "Find company competitors to run a comprehensive comprative analysis",
                "status": "pending"
            },
            {
            "task": "get_sec_financials",
            "description": "Obtain sales figures for Rivian and the identified competitors over a comparable period, preferably the last few years to analyze growth trends.",
            "status": "pending"
            },
            {
                "task": "get_final_analysis",
                "description": "Produce a comprehensive final analysis from the retrieved information.",
                "status": "pending"
            },
        ]
        },
        {
        "query": "How do the margin profiles of the major US defense companies compare?",
        "tasks": [
            {
                "task": "get_list_of_companies",
                "description": "Identify competitors to use in the analysis",
                "status": "pending"
            },
            {
            "task": "get_sec_financials",
            "description": "For each identified company, gather recent financial statements, focusing on income statements over the last few quarters or years.",
            "status": "pending"
            },
            {
                "task": "get_final_analysis",
                "description": "Tie the retrieved financial information into a comprehensive final analysis.",
                "status": "pending"
            },
        ]
        },
        {
        "query": "What's amzn's revenue?",
        "tasks": [
            {
            "task": "get_sec_financials",
            "description": "For each identified company, gather recent financial statements, focusing on income statements over the last few quarters or years.",
            "status": "pending"
            },
            {
                "task": "get_final_analysis",
                "description": "Tie the retrieved financial information into a comprehensive final analysis.",
                "status": "pending"
            },
        ]
        },
        {
            "query": "what's amzn's revenue trend?",
            "tasks": [
                {
                "task": "get_sec_financials",
                "description": "For each identified company, gather recent financial statements, focusing on income statements over the last few quarters or years.",
                "status": "pending"
                },
                {
                "task": "get_final_analysis",
                "description": "Tie the retrieved financial information into a comprehensive final analysis.",
                "status": "pending"
                },
            ]
        }
    ]
    }
    """
    prompt = """
        user query: {question}
    """


    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )

    
    try:
        response = json.loads(response.json())
        research_plan = json.loads(response["choices"][0]["message"]["content"])["queries"][0]["tasks"]
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "get_research_plan", "inputs": [question], "outputs": [{"research_plan": research_plan}]}, indent=6))
        return research_plan
    except Exception as e:
        print(f"Error inside get_research_plan response: {e}")
        return []   


def extract_entities_from_user_query(question, debug=False):
    system_prompt = """ 
    You are an NLP extraction tool. Your task is to take a user query and extract the following entity types if they appear in the query: 'ticker', 'company', 'dates'.
    Below are the descriptions of the values corresponding to each entity type:
        - 'ticker': a company ticker.
        - 'company': a company name.
        - 'dates': a list of dates mentioned in the user query. This should be a string in the form 'YYYY-MM-DD'. If there are multiple dates in the list they should be sorted such that the oldest date appears first and the most recent date appears last. If there are multiple dates the earlier date should
                   be called 'from_date' and the later date should be called 'to_date'.

    The response should be a JSON list of dictionaries where each dictionary element contains a key called 'entity' whose value is the entity type extracted 
    (i.e. is one of the listed types above) and contains a key called 'value' that is the extracted value from the user query. If the entity type doesn't appear in the user 
    query then it should not appear in the output.

    ## EXAMPLES
    user query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL'), ('from_date', '2023-01-01'), ('to_date', '2023-12-31')]
    user query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    user query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    user query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL'), ('from_date', '2023-01-01'), ('to_date', '2023-12-31')]
    user query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    user query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    user query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL')]
    user query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    user query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    user query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL'), ('from_date', '2023-01-01'), ('to_date', '2023-12-31')
    user query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    user query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    user query: "what's ebay's revenue growth (%) quarter over quarter for 2022 versus its comps", answer: [('ticker', 'EBAY'), ('from_date', '2023-01-01'), ('to_date', '2023-12-31')]
    """

    prompt = """
    user query: {question}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )

    try:
        response = json.loads(response.json())
        if "json" in response["choices"][0]["message"]["content"]:
            trimmed_response = response["choices"][0]["message"]["content"][response["choices"][0]["message"]["content"].index("json") + 4: len(response["choices"][0]["message"]["content"]) - 3]
            entities = json.loads(trimmed_response)
        else:
            entities = ast.literal_eval(response["choices"][0]["message"]["content"])

        # import pdb; pdb.set_trace()

        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "extract_entities_from_user_query", "inputs": [question], "outputs": [{"entities": entities}]}, indent=6))
        return entities
    except Exception as e:
        print(f"Error inside extract_entities_from_user_query: {e}")
        return []


def get_stock_financials(symbol, limit=5, debug=False):
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
        
        # TODO: Uncomment this after getting quarterly stuffs to work (add routing for filtering on cases: just FY, just Q*, mixed)
        # result_df[(~result_df["report_date"].isin(["TTM", "TTM "]))]
        result_df = result_df[(~result_df["report_date"].isin(["TTM", "TTM "])) & (~result_df["report_date"].str.contains("FY"))]
        result_df = result_df[[c for c in result_df.columns if c not in ['start_date', 'end_date', 'timeframe', 'fiscal_period', 'fiscal_year',
       'cik', 'sic', 'tickers', 'company_name', 'filing_date',
       'acceptance_datetime', 'source_filing_url', 'source_filing_file_url']]]
        result_df.set_index("report_date", inplace=True)
        result_df.rename({c: c.split(".")[1] for c in result_df.columns}, axis=1, inplace=True)

        return result_df.T
    else:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")


def get_relevant_rows(financials, question, debug=False):
    """
    Prompt ChatGPT to identify the relevant rows from the financial data.

    :param financials: JSON response containing financial data
    :param question: User's question about the financial data
    :return: List of relevant row indices
    """

    system_prompt = """
    You are a helpful assistant that analyzes financial data. Given a user question and list of financial reporting terms, determine which of the
    financial terms are required to answer the question. The response should be a python list of the values in the supplied rows. Below are some examples:

    Examples:
        {
            "queries": [
                {
                    "query": "what's amzn's cogs from 2022 to 2023?",
                    "rows": ['current_liabilities','liabilities_and_equity','long_term_debt','other_current_assets','equity','noncurrent_assets','other_noncurrent_liabilities','assets','noncurrent_liabilities','equity_attributable_to_noncontrolling_interest','current_assets','equity_attributable_to_parent','liabilities','other_current_liabilities','accounts_payable','inventory','revenues','operating_expenses','cost_of_revenue','net_income_loss_attributable_to_parent','diluted_average_shares','diluted_earnings_per_share','benefits_costs_expenses','basic_average_shares','income_loss_from_continuing_operations_before_tax','basic_earnings_per_share','net_income_loss_attributable_to_noncontrolling_interest','costs_and_expenses','preferred_stock_dividends_and_other_adjustments','net_income_loss','income_tax_expense_benefit','participating_securities_distributed_and_undistributed_earnings_loss_basic','income_tax_expense_benefit_deferred','operating_income_loss','income_loss_from_equity_method_investments','net_income_loss_available_to_common_stockholders_basic','nonoperating_income_loss','income_loss_before_equity_method_investments','income_loss_from_continuing_operations_after_tax','gross_profit','net_cash_flow_from_operating_activities_continuing','net_cash_flow_from_investing_activities','net_cash_flow_continuing','net_cash_flow_from_financing_activities','net_cash_flow','net_cash_flow_from_financing_activities_continuing','net_cash_flow_from_investing_activities_continuing','net_cash_flow_from_operating_activities','interest_expense_operating','intangible_assets','other_noncurrent_assets','wages','exchange_gains_losses','fixed_assets','other_operating_income_expenses','income_tax_expense_benefit_current','cash','depreciation_and_amortization','other_operating_expenses','commitments_and_contingencies'],
                    "response": ['cost_of_revenue']
                },
                {
                    "query": "what was the total amount spent on acquisitions net of cash acquired and purchases of intangible and other assets by msft in the year 2022?",
                    "rows": ["fixed_assets","liabilities","current_liabilities","other_noncurrent_liabilities","assets","wages","long_term_debt","noncurrent_assets","other_current_liabilities","other_noncurrent_assets","current_assets","equity_attributable_to_parent","cash","other_current_assets","liabilities_and_equity","equity","accounts_payable","equity_attributable_to_noncontrolling_interest","noncurrent_liabilities","inventory","net_cash_flow_from_financing_activities","net_cash_flow","net_cash_flow_from_financing_activities_continuing","net_cash_flow_from_operating_activities","net_cash_flow_from_investing_activities_continuing","net_cash_flow_from_operating_activities_continuing","net_cash_flow_continuing","net_cash_flow_from_investing_activities","net_income_loss_attributable_to_noncontrolling_interest","operating_expenses","benefits_costs_expenses","participating_securities_distributed_and_undistributed_earnings","net_income_loss_attributable_to_parent","income_tax_expense_benefit_deferred","income_tax_expense_benefit","operating_income_loss","net_income_loss","diluted_average_shares","costs_and_expenses","basic_earnings_per_share","income_loss_from_continuing_operations_after_tax","cost_of_revenue","net_income_loss_available_to_common_stockholders","revenues","other_operating_expenses","research_and_development","basic_average_shares","nonoperating_income_loss","income_loss_from_continuing_operations_before_tax","diluted_earnings_per_share","preferred_stock_dividends_and_other_adjustments","interest_expense_operating","income_tax_expense_benefit_current","gross_profit","exchange_gains_losses","income_loss_before_equity_method_investments","cost_of_revenue_goods","intangible_assets"],
                    "response": ['cash', 'intangible_assets', 'net_cash_flow_from_investing_activities']
                },
                {
                    "query": "What was the trend in MSFT's Payments for Repurchase of Common Stock from Q3 2020 to Q1 2023?",
                    "rows": ["fixed_assets","liabilities","current_liabilities","other_noncurrent_liabilities","assets","wages","long_term_debt","noncurrent_assets","other_current_liabilities","other_noncurrent_assets","current_assets","equity_attributable_to_parent","cash","other_current_assets","liabilities_and_equity","equity","accounts_payable","equity_attributable_to_noncontrolling_interest","noncurrent_liabilities","inventory","net_cash_flow_from_financing_activities","net_cash_flow","net_cash_flow_from_financing_activities_continuing","net_cash_flow_from_operating_activities","net_cash_flow_from_investing_activities_continuing","net_cash_flow_from_operating_activities_continuing","net_cash_flow_continuing","net_cash_flow_from_investing_activities","net_income_loss_attributable_to_noncontrolling_interest","operating_expenses","benefits_costs_expenses","participating_securities_distributed_and_undistributed_earnings","net_income_loss_attributable_to_parent","income_tax_expense_benefit_deferred","income_tax_expense_benefit","operating_income_loss","net_income_loss","diluted_average_shares","costs_and_expenses","basic_earnings_per_share","income_loss_from_continuing_operations_after_tax","cost_of_revenue","net_income_loss_available_to_common_stockholders","revenues","other_operating_expenses","research_and_development","basic_average_shares","nonoperating_income_loss","income_loss_from_continuing_operations_before_tax","diluted_earnings_per_share","preferred_stock_dividends_and_other_adjustments","interest_expense_operating","income_tax_expense_benefit_current","gross_profit","exchange_gains_losses","income_loss_before_equity_method_investments","cost_of_revenue_goods","intangible_assets"],
                    "response": ["net_cash_flow_from_financing_activities"]
                }
            ]
        }
    """

    prompt = """
    query: {question}
    rows: {financials}

    Please identify the rows in the supplied financials table that are necessary to answer the question.
    Return your answer as a Python list that contains only the row names.
    """

    # Call the OpenAI API
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, financials=financials)}
        ]
    )

    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]
        relevant_rows = ast.literal_eval(response[response.find("["):response.find("]")+1])

        with open("/home/ubuntu/tmcc-backend/get_relevant_rows_example_to_add.json", "w") as f:
                f.write(
                    f"""
                        "query": "{question}",
                        "rows": {financials},
                        "response": {relevant_rows}
                    """
                )

        return relevant_rows
    except Exception as e:
        print(f"Error parsing ChatGPT response: {e}")
        return []


def get_relevant_columns(financials, question, debug=False):
    """
    Prompt ChatGPT to identify the relevant rows from the financial data.

    :param financials: JSON response containing financial data
    :param question: User's question about the financial data
    :return: List of relevant row indices
    """

    system_prompt = """
    You are a helpful assistant that analyzes financial data. Given a user question and list of financial reporting dates, determine which of the
    dates are required to answer the question. The response should be a python list of the values in the supplied rows. Below are some examples:

    Examples:
        {
            "queries": [
                {
                    "query": "What was the trend in MSFT's Payments for Repurchase of Common Stock from Q3 2020 to Q1 2023?",
                    "rows": ['TTM 2024', 'Q4 2024', 'Q3 2024', 'Q2 2024', 'Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021', 'Q4 2020', 'Q3 2020', 'Q2 2020', 'Q1 2020', 'Q4 2019', 'Q3 2019', 'Q2 2019', 'Q1 2019', 'Q4 2018', 'Q3 2018', 'Q2 2018', 'Q1 2018', 'Q4 2017', 'Q3 2017', 'Q2 2017', 'Q1 2017', 'Q4 2016', 'Q3 2016', 'Q2 2016', 'Q1 2016', 'Q4 2015', 'Q3 2015', 'Q2 2015', 'Q1 2015', 'Q4 2014', 'Q3 2014', 'Q2 2014', 'Q1 2014', 'Q4 2013', 'Q3 2013', 'Q2 2013', 'Q1 2013', 'Q4 2012', 'Q3 2012', 'Q2 2012', 'Q1 2012', 'Q4 2011', 'Q3 2011', 'Q2 2011', 'Q1 2011', 'Q4 2010', 'Q2 2010', 'Q1 2010'],
                    "response": ["Q3 2020","Q4 2020","Q1 2021","Q2 2021","Q3 2021","Q4 2021","Q1 2022","Q2 2022","Q3 2022","Q4 2022","Q1 2023"]
                },
                {
                    "query": "What was the total revenue for MSFT in the US for the period Q1 2021 to Q3 2021?",
                    "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['Q1 2021', 'Q2 2021', 'Q3 2021'],

                }, 
                {
                "query": "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps",
                "rows": ['Q2 2024', 'Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021', 'Q4 2020', 'Q3 2020', 'Q2 2020', 'Q1 2020', 'Q4 2019', 'Q3 2019', 'Q2 2019', 'Q1 2019', 'Q4 2018', 'Q3 2018', 'Q2 2018', 'Q1 2018', 'Q3 2017', 'Q2 2017', 'Q4 2017', 'Q4 2016', 'Q3 2016', 'Q2 2016', 'Q1 2016', 'Q4 2015', 'Q3 2015', 'Q2 2015', 'Q1 2015', 'Q4 2014', 'Q3 2014', 'Q2 2014', 'Q1 2014', 'Q4 2013', 'Q3 2013', 'Q2 2013', 'Q1 2013', 'Q4 2012', 'Q3 2012', 'Q2 2012', 'Q1 2012', 'Q4 2011', 'Q3 2011', 'Q2 2011', 'Q1 2011', 'Q4 2010', 'Q3 2010', 'Q2 2010', 'Q1 2010', 'Q4 2009', 'Q3 2009', 'Q2 2009', 'Q1 2009'],
                "response": ['Q4 2022','Q3 2022','Q2 2022','Q1 2022']
                }
                {
                    "query": "what's amzn's revenue for 2022? compare this to its comps",
                    "rows": ['Q2 2024', 'Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021', 'Q4 2020', 'Q3 2020', 'Q2 2020', 'Q1 2020', 'Q4 2019', 'Q3 2019', 'Q2 2019', 'Q1 2019', 'Q4 2018', 'Q3 2018', 'Q2 2018', 'Q1 2018', 'Q3 2017', 'Q2 2017', 'Q4 2017', 'Q4 2016', 'Q3 2016', 'Q2 2016', 'Q1 2016', 'Q4 2015', 'Q3 2015', 'Q2 2015', 'Q1 2015', 'Q4 2014', 'Q3 2014', 'Q2 2014', 'Q1 2014', 'Q4 2013', 'Q3 2013', 'Q2 2013', 'Q1 2013', 'Q4 2012', 'Q3 2012', 'Q2 2012', 'Q1 2012', 'Q4 2011', 'Q3 2011', 'Q2 2011', 'Q1 2011', 'Q4 2010', 'Q3 2010', 'Q2 2010', 'Q1 2010', 'Q4 2009', 'Q3 2009', 'Q2 2009', 'Q1 2009'],
                    "response": ['Q4 2022','Q3 2022','Q2 2022','Q1 2022']

                },
                {
                    "query": "What's WMT's quarterly revenue for 2023? Compare this to its comps",
                    "rows": ['Q1 2025', 'Q4 2024', 'FY 2024', 'Q3 2024', 'Q2 2024', 'Q1 2024', 'Q4 2023', 'FY 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'FY 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'FY 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021', 'Q4 2020', 'FY 2020', 'Q3 2020', 'Q2 2020', 'Q1 2020', 'Q4 2019', 'FY 2019', 'Q3 2019', 'Q2 2019', 'Q1 2019', 'Q4 2018', 'FY 2018', 'Q3 2018', 'Q2 2018', 'Q1 2018', 'Q4 2017', 'FY 2017', 'Q3 2017', 'Q2 2017', 'Q1 2017', 'Q4 2016', 'FY 2016', 'Q3 2016', 'Q2 2016', 'Q1 2016', 'Q4 2015', 'FY 2015', 'Q3 2015', 'Q2 2015', 'Q1 2015', 'Q4 2014', 'FY 2014', 'Q3 2014', 'Q2 2014', 'Q1 2014', 'Q4 2013', 'FY 2013', 'Q3 2013', 'Q2 2013', 'Q1 2013', 'Q4 2012', 'FY 2012', 'Q3 2012', 'Q2 2012', 'Q1 2012', 'Q4 2011', 'FY 2011', 'Q3 2011', 'Q2 2011', 'Q1 2011', 'Q4 2010', 'FY 2010', 'Q3 2010', 'Q2 2010', 'Q1 2010'],
                    "response": ['Q4 2023','Q3 2023','Q2 2023','Q1 2023']

                },
                {
                    "query": "what's TGT's revenue for 2022 versus its comps?",
                    "rows": ['Q2 2025', 'Q1 2025', 'Q4 2024', 'FY 2024', 'Q3 2024', 'Q2 2024', 'Q1 2024', 'Q4 2023', 'FY 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'FY 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'FY 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021', 'Q4 2020', 'FY 2020', 'Q3 2020', 'Q2 2020', 'Q1 2020', 'Q4 2019', 'FY 2019', 'Q3 2019', 'Q2 2019', 'Q1 2019', 'Q4 2018', 'FY 2018', 'Q3 2018', 'Q2 2018', 'Q1 2018', 'Q4 2017', 'FY 2017', 'Q3 2017', 'Q2 2017', 'Q1 2017', 'Q4 2016', 'FY 2016', 'Q3 2016', 'Q2 2016', 'Q1 2016', 'Q4 2015', 'FY 2015', 'Q3 2015', 'Q2 2015', 'Q1 2015', 'Q4 2014', 'FY 2014', 'Q3 2014', 'Q2 2014', 'Q1 2014', 'Q4 2013', 'FY 2013', 'Q3 2013', 'Q2 2013', 'Q1 2013', 'Q4 2012', 'FY 2012', 'Q3 2012', 'Q2 2012', 'Q1 2012', 'Q4 2011', 'FY 2011', 'Q3 2011', 'Q2 2011', 'Q1 2011', 'Q3 2010', 'Q2 2010', 'Q1 2010'],
                    "response": ['Q4 2022','Q3 2022','Q2 2022','Q1 2022']

                },
                {
                    "query": "what's WMT's revenue for 2022 versus its comps?",
                    "rows": ['Q1 2025', 'Q4 2024', 'FY 2024', 'Q3 2024', 'Q2 2024', 'Q1 2024', 'Q4 2023', 'FY 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'FY 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'FY 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021', 'Q4 2020', 'FY 2020', 'Q3 2020', 'Q2 2020', 'Q1 2020', 'Q4 2019', 'FY 2019', 'Q3 2019', 'Q2 2019', 'Q1 2019', 'Q4 2018', 'FY 2018', 'Q3 2018', 'Q2 2018', 'Q1 2018', 'Q4 2017', 'FY 2017', 'Q3 2017', 'Q2 2017', 'Q1 2017', 'Q4 2016', 'FY 2016', 'Q3 2016', 'Q2 2016', 'Q1 2016', 'Q4 2015', 'FY 2015', 'Q3 2015', 'Q2 2015', 'Q1 2015', 'Q4 2014', 'FY 2014', 'Q3 2014', 'Q2 2014', 'Q1 2014', 'Q4 2013', 'FY 2013', 'Q3 2013', 'Q2 2013', 'Q1 2013', 'Q4 2012', 'FY 2012', 'Q3 2012', 'Q2 2012', 'Q1 2012', 'Q4 2011', 'FY 2011', 'Q3 2011', 'Q2 2011', 'Q1 2011', 'Q4 2010', 'FY 2010', 'Q3 2010', 'Q2 2010', 'Q1 2010'],
                    "response": ['Q4 2022','Q3 2022','Q2 2022','Q1 2022']

                },
                {
                    "query": "what's amzn's revenue versus it comps for 2022?",
                    "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['Q4 2022','Q3 2022','Q2 2022','Q1 2022']

                },
                {
                    "query": "what's amzn's quarterly revenue trend versus its comps?",
                    "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['Q4 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','Q3 2021','Q2 2021','Q1 2021']

                },
                {
                    "query": "what's amzn's quarterly revenue trend versus its comps?",
                    "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['Q4 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','Q3 2021','Q2 2021','Q1 2021']

                },
                {
                    "query": "what's amzn's revenue for the first 3 quarters of 2022?",
                    "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['Q3 2022','Q2 2022','Q1 2022']
                },
                {
                    "query": "Get MSFT's quarterly cogs over from 2020 to 2021",
                    "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['Q4 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','Q3 2020','Q2 2020','Q1 2020']
                },
                {
                    "query": "Get jpm cogs (annual) from 2020 to 2021",
                    "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['FY 2021','FY 2020']
                },
                {
                    "query": "what's amzn's cogs from 2022 to 2023?",
                    "rows": "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['FY 2022','FY 2023']
                },
                {
                    "query": "what's amzn's revenue for 2022?",
                    "rows": "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['FY 2022']
                },
                {
                    "query": "what's amzn's total revenue for 2022?",
                    "rows": "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['FY 2022']
                },
                {
                    "query": "what's amzn's quarterly revenue for 2022?",
                    "rows": "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022']
                },
                {
                    "query": "how has amzn's revenue trended across 2022?",
                    "rows": "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022']
                },
                {
                    "query": "how has amzn's revenue trended between 2021 and 2023?",
                    "rows": "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['FY 2023', 'FY 2022', 'FY 2021']
                },
                {
                    "query": "what's amzn's revenue for 2022?",
                    "rows": "rows": ['Q2 2024','Q1 2024','Q4 2023','FY 2023','Q3 2023','Q2 2023','Q1 2023','Q4 2022','FY 2022','Q3 2022','Q2 2022','Q1 2022','Q4 2021','FY 2021','Q3 2021','Q2 2021','Q1 2021','Q4 2020','FY 2020','Q3 2020','Q2 2020','Q1 2020','Q4 2019','FY 2019','Q3 2019','Q2 2019','Q1 2019','Q4 2018','FY 2018','Q3 2018','Q2 2018','Q1 2018','FY 2017','Q3 2017','Q2 2017','Q4 2017','Q4 2016','FY 2016','Q3 2016','Q2 2016','Q1 2016','Q4 2015','FY 2015','Q3 2015','Q2 2015','Q1 2015','Q4 2014','FY 2014','Q3 2014','Q2 2014','Q1 2014','Q4 2013','FY 2013','Q3 2013','Q2 2013','Q1 2013','Q4 2012','FY 2012','Q3 2012','Q2 2012','Q1 2012','Q4 2011','FY 2011','Q3 2011','Q2 2011','Q1 2011','Q4 2010','FY 2010','Q3 2010','Q2 2010','Q1 2010','Q4 2009','FY 2009','Q3 2009','Q2 2009','Q1 2009'],
                    "response": ['FY 2022']
                },
            ]
        }
    """

    prompt = """
    query: {question}
    rows: {financials}

    Please identify the rows in the supplied financials table that are necessary to answer the question.
    Return your answer as a Python list that contains only the row names.
    """

    # Call the OpenAI API
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, financials=financials)}
        ]
    )

    try:
        # import pdb; pdb.set_trace()
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]
        relevant_columns = ast.literal_eval(response[response.find("["):response.find("]")+1])
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "get_relevant_columns", "inputs": [question, financials], "outputs": [{"relevant_columns": relevant_columns}]}, indent=6))
        return relevant_columns
    except Exception as e:
        print(f"Error parsing ChatGPT response: {e}")
        return []


source_dict = {
  'title': 'Annual 10-K 2022',
  'subtitle': 'Highlights the most recent annual latest sales breakdown by product line.',
  'importance_score': 0.63954395,
  'extracted_text': "\nupdates",
  'date': '2022-12-31',
  'company': 'google',
  'page_number': 9,
  'doc_download_url': 'https://sec-filings2.s3.us-east-1.amazonaws.com/0001652044-23-000016.pdf',
  'icon': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-google" viewBox="0 0 16 16"><path d="M15.545 6.558a9.4 9.4 0 0 1 .139 1.626c0 2.434-.87 4.492-2.384 5.885h.002C11.978 15.292 10.158 16 8 16A8 8 0 1 1 8 0a7.7 7.7 0 0 1 5.352 2.082l-2.284 2.284A4.35 4.35 0 0 0 8 3.166c-2.087 0-3.86 1.408-4.492 3.304a4.8 4.8 0 0 0 0 3.063h.003c.635 1.893 2.405 3.301 4.492 3.301 1.078 0 2.004-.276 2.722-.764h-.003a3.7 3.7 0 0 0 1.599-2.431H8v-3.08z"/></svg>'
}

import boto3
import os
import pandas as pd
from botocore.exceptions import ClientError
import pymupdf  
import uuid




class s3Handler:
    def __init__(self):
        self._s3_resource_handler = boto3.resource("s3")
        self._s3_client = boto3.client("s3")

    def list_buckets(self):
        """
        Get the buckets in all Regions for the current account.

        :param s3_resource: A Boto3 S3 resource. This is a high-level resource in Boto3
                            that contains collections and factory methods to create
                            other high-level S3 sub-resources.
        :return: The list of buckets.
        """
        try:
            buckets = list(self._s3_resource_handler.buckets.all())
        except ClientError:
            raise
        else:
            return buckets

    def download_file(
        self, bucket_name, object_path, download_path, local_filename
    ):
        try:
            if not os.path.exists(download_path):
                os.makedirs(download_path)

            self._s3_client.download_file(
                bucket_name,
                f"{object_path}",
                f"{download_path}/{local_filename}",
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print("The object does not exist.")
            else:
                raise

    def upload_file(self, local_file, bucket_name, s3_filename):
        try:
            self._s3_client.upload_file(local_file, bucket_name, s3_filename, {"Metadata": {"Content-Type": "application/pdf"}})
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print("The object does not exist.")
            else:
                raise


def add_highlighting_to_citations_pdfs(citations):
    if not os.path.exists(LOCAL_DOWNLOAD_PATH):
        os.makedirs(LOCAL_DOWNLOAD_PATH)

    new_citations = []
    page_number_set_for_dedupe = set()
    citation_text_set_for_dedupe = set()
    for citation in citations:
        # download pdf to local path
        unhighlighted_pdf_filename = citation["url"]
        s3_handler.download_file(
                BUCKET_FOR_UNHIGHLIGHTED_DOCS,
                f"dow30/{unhighlighted_pdf_filename}",
                LOCAL_DOWNLOAD_PATH,
                unhighlighted_pdf_filename
            )
        local_path_to_unhighlighted_pdf = f"{LOCAL_DOWNLOAD_PATH}/{unhighlighted_pdf_filename}"
        os.system(f"chmod 777 {local_path_to_unhighlighted_pdf}")

        print(f"downloaded {unhighlighted_pdf_filename} to {LOCAL_DOWNLOAD_PATH}")

        # perform highlighting according to the page number and citations
        highlighted_pdf_filename_prefix = unhighlighted_pdf_filename.split('.pdf')[0]
        highlighted_pdf_filename = f"{highlighted_pdf_filename_prefix}_{uuid.uuid4()}.pdf"
        doc = pymupdf.open(local_path_to_unhighlighted_pdf)
        # # load desired page (0-based page number)
        page_number = int(citation['page_number'])
        page = doc[page_number]
        text_to_highlight = citation["text"][:int(len(citation["text"])*0.9)]

        if page_number in page_number_set_for_dedupe and text_to_highlight[:len(text_to_highlight)//2] in citation_text_set_for_dedupe:
            continue

        page_number_set_for_dedupe.add(page_number)
        citation_text_set_for_dedupe.add(text_to_highlight[:len(text_to_highlight)//2])
        
        # import pdb; pdb.set_trace()
        rects = page.search_for(text_to_highlight)
        if len(rects) == 0:
            continue
        p1 = rects[0].tl
        p2 = rects[-1].br
        page.add_highlight_annot(start=p1, stop=p2)
        pdf_to_save_pdf_before_upload_to_s3 = f"{LOCAL_DOWNLOAD_PATH}/{highlighted_pdf_filename}"

        print(f"Saved pdf with highlighting to {pdf_to_save_pdf_before_upload_to_s3}")
        
        # save pdf to local path again (overwriting it)
        doc.save(pdf_to_save_pdf_before_upload_to_s3)
        
        # upload to s3 ("highlighting" bucket)
        s3_handler.upload_file(
            pdf_to_save_pdf_before_upload_to_s3, BUCKET_FOR_UNHIGHLIGHTED_DOCS, highlighted_pdf_filename        
        )
        
        # remove local pdf from local download path
        os.system(f"chmod 777 {pdf_to_save_pdf_before_upload_to_s3}")
        os.remove(local_path_to_unhighlighted_pdf)
        # os.remove(pdf_to_save_pdf_before_upload_to_s3)

         # Update citation with new url
        new_citation = {
            "id": str(uuid.uuid4()),
            "logo": citation["logo"],
            "page_number": citation["page_number"]+1,
            "url": f"citations/{highlighted_pdf_filename}",#f"https://{BUCKET_FOR_HIGHLIGHTED_DOCS}.s3.amazonaws.com/{highlighted_pdf_filename}", #pdf_to_save_pdf_before_upload_to_s3,#
            "title": citation["title"],
            "company": citation["company"],
            "importance": citation["importance"],
            "text": citation["text"]
        }

        print(f"new_citation['url']: {new_citation['url']}")

        new_citations.append(new_citation)

    return new_citations            


def perform_vector_search(question, results, debug=False):
    entities = extract_entities_from_user_query(question, debug)
    ticker = None
    filters = []
    for ent in entities:
        if ent["entity"] == "ticker":
            ticker = ent["value"]
            filters.append({
                "path": ["ticker"],
                "operator": "Equal",
                "valueText": ticker.upper()
            })
        elif ent["entity"] == "from_date":
            filters.append({
                "path": ["report_date"],
                "operator": "GreaterThanEqual",
                "valueDate": f'{str(parser.parse(ent["value"])).split(" ")[0]}T00:00:00.000Z'
            })
        elif ent["entity"] == "to_date":
            filters.append({
                "path": ["report_date"],
                "operator": "LessThanEqual",
                "valueDate": f'{str(parser.parse(ent["value"])).split(" ")[0]}T00:00:00.000Z'
            })
            

    response = (
        weaviate_client.query
        .get("Dow30_10K_10Q", ["filing_type", "company_name", "ticker", "accession_number", "s3_doc_url", "text", "page_number", "report_date"])
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

    if "finalAnalysis" not in results:
        results["finalAnalysis"] = {

        }

    citations = [{"text": t["text"], "id": "", "logo": "", "page_number": t["page_number"], "url": t["s3_doc_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0 - float(t["_additional"]["distance"])} for t in response["data"]["Get"]["Dow30_10K_10Q"]]
    citations = sorted(citations, key=lambda d: d['importance'], reverse=True)
    
    citations = add_highlighting_to_citations_pdfs(citations[:10])
    
    if "citations" not in results["finalAnalysis"]:
        results["finalAnalysis"]["citations"] = []    
    results["finalAnalysis"]["citations"].extend(citations)

    if "insights" not in results["finalAnalysis"]:
        results["finalAnalysis"]["insights"] = [] 
    
    response = '\n\n'.join([c["text"] for c in citations])
    results["Context"].append(
        f"Response to Query: {question} \n\n{response}"
    ) 

    if debug:
        with open(DEBUG_ABS_FILE_PATH, "a") as f:
            f.write(json.dumps({"function": "perform_vector_search", "inputs": [question], "outputs": [{"response": response}]}, indent=6))
    return results


def get_financials(question, results, debug=False):
    entities = extract_entities_from_user_query(question, debug)
    ticker = None
    for ent in entities:
        if ent["entity"] == "ticker":
            ticker = ent["value"]
            break
            
    financials = get_stock_financials(ticker.upper())
    relevant_rows = get_relevant_rows(list(financials.index.values), question)
    print(f"relevant_rows: {relevant_rows}")
    relevant_columns = get_relevant_columns(list(financials.columns), question)
    print(f"relevant_columns: {relevant_columns}")

    financials = financials.loc[relevant_rows, relevant_columns]

    calculation_required = should_local_calculate(question, financials)
    print(f"calculation_required: {calculation_required}")
    if calculation_required:
        financials = do_local_calculate(question, financials.T)
        print(f"financials after do_local_calculate: {financials}")

    financials = financials.rename(columns={c: f"{ticker.upper()}.{c}" for c in financials.columns})
    chart_data = financials.to_dict('records')

    results["Context"].append(
        f"{financials.T.columns[0]} results for {ticker.upper()}: {financials.T.to_dict('records')}"
    )

    if "ResultsFromGetFinancials" not in results:
        results["ResultsFromGetFinancials"] = {

        }

    results["ResultsFromGetFinancials"][ticker.upper()] = financials.to_json()

    if "finalAnalysis" not in results:
        results["finalAnalysis"] = {

        }

    if "charts" not in results["finalAnalysis"]:
        results["finalAnalysis"]["charts"] = {

        }

    if "tables" not in results["finalAnalysis"]:
        results["finalAnalysis"]["tables"] = {

        }

    financials[f"Date_{ticker}"] = financials.index.values
    table =  {
            "headers": list(financials.columns),
            "rows": financials.values.tolist()
    }
    results["finalAnalysis"]["charts"][ticker.upper()] = chart_data
    results["finalAnalysis"]["tables"][ticker.upper()]= table
    if "workbookData" not in results:
        results["workbookData"] = {
        }

    results["workbookData"][ticker.upper()] = table

    if debug:
        with open(DEBUG_ABS_FILE_PATH, "a") as f:
            f.write(json.dumps({"function": "get_financials", "inputs": [question], "outputs": [{"financials": financials.to_json()}]}, indent=6))
    return financials.to_json()



def merge_charts(results, debug=False):
    charts_dicts = results["finalAnalysis"]["charts"]
    datas = []
    for ticker, chart_dict in charts_dicts.items():
        datas.append(chart_dict)

    processed_data = [{} for i in range(len(datas[0]))]
    datapoints = []
    for i in range(len(datas[0])):
        for j in range(len(datas)):
            processed_data[i].update(datas[j][i])
            # print(f"processed_data[{i}]: {processed_data[i]}")

    # print(f"processed_data: {[k for k in processed_data[0].keys()]}")
    # print(f"processed_data: {processed_data}")
    chart = {
            "type": "line",
            "data": processed_data,
            "dataKeys": [k for k in processed_data[0].keys()]
    }
    results["finalAnalysis"]["charts"] = {f'Chart for Query: {results["Query"]}': chart}
    return results


def merge_tables(results, debug=False):
    tables_dicts = results["finalAnalysis"]["tables"]
    datas = []
    for ticker, table_dict in tables_dicts.items():
        datas.append(table_dict)

    processed_data = [[] for i in range(len(datas[0]['rows']))]
    datapoints = []
    headers = []
    for i in range(len(datas)):
        data_dict = datas[i]
        headers += data_dict["headers"]
        rows = data_dict["rows"]
        for j, val in enumerate(rows):
            processed_data[j].extend(val)

    df_temp = pd.DataFrame(data=processed_data, columns = headers)
    other_cols = [c for c in df_temp.columns if "Date" not in c]
    date_col = [c for c in df_temp.columns if "Date" in c][0]

    df_temp = df_temp[[date_col] + other_cols]
    df_temp.rename(columns={date_col: "Date"}, inplace=True)

    table = {
        f'Table for Query: {results["Query"]}': {
            "headers": list(df_temp.columns),
            "rows": df_temp.values.tolist()
        }
    }
    results["finalAnalysis"]["tables"] = table
    results["finalAnalysis"]["workbookData"] = {"headers": table[f'Table for Query: {results["Query"]}']["headers"], "rows": table[f'Table for Query: {results["Query"]}']["rows"]}
    return results, df_temp
    

def get_final_analysis(query, results, debug=False):
    final_analysis = results["finalAnalysis"]
    context = results["Context"]

    if "charts" in results["finalAnalysis"]:
        results = merge_charts(results)
    if "tables" in results["finalAnalysis"]:
        results, df_temp = merge_tables(results)

    if "workbookData" not in results["finalAnalysis"]:
        results["finalAnalysis"]["workbookData"] = None

    if "ResultsFromGetFinancials" in results:
        should_calculate = should_global_calculate(query, results["ResultsFromGetFinancials"], debug)
        print(f"should_global_calculate: {should_calculate}")
        if should_calculate:
            results = do_global_calculate(query, results["ResultsFromGetFinancials"], debug)
            print(f"results after do_global_calculate: {results}")

    # print(f"context: {context}")
    context = '\n'.join([str(c) for c in context])

    system_content = """
    Given a user-supplied query and a context that contains the raw data to necessary to answer the query, generate a response that synthesizes the provided context to answer the question.
    """

    prompt = """
    query: {query}

    context: {context}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt.format(query=query, context=context)}
        ]
    )
    
    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]

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
        return results
    except Exception as e:
        print(f"Error parsing ChatGPT response: {e}")
        return []


def should_global_calculate(question, financials, debug=False):
    system_prompt = """ 
    You are a tool used to guide a RAG system. Your job is to determine where a calculator function needs to be called or not.
    Your response should be a string that is either 'True' or 'False'. Your response should not contain anything other than either 
    'True' or 'False'. Given a user query and a supplied json contain data used to answer the user determine wheter a calculation needs
    to be executed or if the current raw json data is sufficient to answer the quesiton and thus doesn't require any further calculations.
    Below are some examples of user query, json data, and response triplets.

    Examples:
    user query: "What was the trend in MSFT's Payments for Repurchase of Common Stock from Q3 2020 to Q1 2023?", data: '{"Date":{"0":"net_cash_flow_from_financing_activities"},"MSFT.Q3 2020":{"0":-14645000000.0},"MSFT.Q4 2020":{"0":-12262000000.0},"MSFT.Q1 2021":{"0":-10289000000.0},"MSFT.Q2 2021":{"0":-13634000000.0},"MSFT.Q3 2021":{"0":-13192000000.0},"MSFT.Q4 2021":{"0":-11371000000.0},"MSFT.Q1 2022":{"0":-16276000000.0},"MSFT.Q2 2022":{"0":-11986000000.0},"MSFT.Q3 2022":{"0":-17345000000.0},"MSFT.Q4 2022":{"0":-13269000000.0},"MSFT.Q1 2023":{"0":-10883000000.0}}', response: "False"
    user query: "what's amzn's revenue for 2022 versus its comps", data: {"revenue": {"2022": [1000000000000, 1000000000025, 4000000000000, 3000000000000]}}, response: "False"
    user query: "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenue": {"2022": [1000000000000, 1000000000025, 4000000000000, 3000000000000]}}, response: "True"
    user query: "what's tgt's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenues":{"Q1 2022":24197000000.0,"Q2 2022":25160000000.0,"Q3 2022":25652000000.0,"Q4 2022":30996000000.0}}, response: "True"
    user query: "what's wmt's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenues":{"Q1 2022":116444000000.0,"Q2 2022":121234000000.0,"Q3 2022":127101000000.0,"Q4 2022":149204000000.0}}, response: "True"
    """

    prompt = """
    user query: {question}
    json data: {financials}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, financials=financials)}
        ]
    )

    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]
        import pdb
        # pdb.set_trace()
        # print(f"response: {response}")
        calculation_required = ast.literal_eval(response)
        # print(f"calculation_required: {calculation_required}")
        return calculation_required
    except Exception as e:
        print(f"Error inside should_local_calculate: {e}")
        return []


"""
GOOD EXAMPLE OF BAR CHART:
==========================

- what's aapl's revenue for the first 3 quarters of 2022? compare this to comps


"""


def do_global_calculate(question, financials, debug=False):
    system_prompt = """ 
    You are a tool used in a rag system to determine an equation that needs to be applied to a pandas dataframe. Your job is to generate 
    the string representation of the pandas calculation such that the python eval function can be called on the string.
    Your response should not contain anything other than the pandas equation to be called. Including anything other than just the equation 
    will cause the functin to fail. 
    
    Given a user query and the supplied json that represents the pandas DataFrame, return the equation to apply to the DataFrame that performs the calculation
    necessary to answer the question. Below are some examples of user query, json data, and response triplets. Assume that the pandas DataFrame is called 'financials'.
    Use the name financials to represent that pandas DataFrame in your response. 

    Examples:
    user query: "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenues":{"Q1 2022":116444000000.0,"Q2 2022":121234000000.0,"Q3 2022":127101000000.0,"Q4 2022":149204000000.0}}, response: "financials['revenues'].pct_change().round(2)"
    user query: "what's WMT's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenues":{"Q1 2022":116444000000.0,"Q2 2022":121234000000.0,"Q3 2022":127101000000.0,"Q4 2022":149204000000.0}}, response: "financials['revenues'].pct_change().round(2)"
    """

    prompt = """
    user query: {question}
    json data: {financials}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, financials=financials)}
        ]
    )

    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]
        print(f"equation from inside do_global_calculate: {response}")
        # pdb.set_trace()
        # print(f"response: {response}")
        new_financials = eval(response)
        # print(f"new_financials: {new_financials}")
        if isinstance(new_financials, pd.Series):
            new_financials = new_financials.to_frame()
            new_financials.dropna(inplace=True)
        else:
            new_financials = pd.DataFrame({"result": new_financials})
        return new_financials
    except Exception as e:
        print(f"Error inside do_local_calculate: {e}")
        return []

def response_formatter(final_response, task_json):
    # print(f"task_json: {task_json}, {type(task_json)}")
    system_content = """
    You are a financial analyst tasked with presenting a response to your manager's query in the best possible way.
    You are provided with the correct answer to your boss's query already and your job is to decide how best
    to present the information to your manager. Your options are to present the information in a chart or in an excel workbook.
    Given the response to your managers query and the raw data used to generate the response, provide the best way to present the data.
    Your output should be a list containing any combination of the following: ['text', 'chart', 'workbook'].
    """

    prompt = """
    response: {final_response}

    data: {data}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt.format(final_response=final_response, data=task_json['get_sec_financials']['data_for_final_analysis'])}
        ]
    )
    
    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]
        return response
    except Exception as e:
        # print(f"Error parsing ChatGPT response: {e}")
        return []