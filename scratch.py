import requests
import os
from dotenv import load_dotenv
import openai
import pandas as pd
import json
import ast

from openai import OpenAI
# Load environment variables
load_dotenv()

# Get API keys from environment variables
POLYGON_API_KEY = "1ZIKxd5xw6LXMwO5R57iBuvBSuhBeJJV"
OPENAI_API_KEY = "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"
client = OpenAI(
    api_key=OPENAI_API_KEY,
)


def preprocess_user_query(question):
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )

    try:
        response = json.loads(response.json())
        processed_query = response["choices"][0]["message"]["content"]
        # processed_query = ast.literal_eval(response[response.find("["):response.find("]")+1])
        print(f"processed_query: {processed_query}")
        entities = extract_entities_from_user_query(processed_query)
        return processed_query, entities
    except Exception as e:
        print(f"Error inside processed_query: {e}")
        return []


def get_list_of_companies(question):
    system_prompt = """ 
    You are a hedge fund analyst tasked with identifying a list of public companies that fit the criteria given by a user query. Given a user query return a python list of company tickers. 
    Each company ticker in the list should be a string ticker. Your response should only contain the list of tickers that are companies that satisfy the requirement outlined by the user query.
    """

    prompt = """
    user query: {question}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )

    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]
        companies = ast.literal_eval(response[response.find("["):response.find("]")+1])
        print(f"list of companies: {companies}")
        return competitors
    except Exception as e:
        print(f"Error inside get_list_of_company: {e}")
        return []


# def should_calculate(question, financials):
#     system_prompt = """ 
#     You are a tool used to guide a RAG system. Your job is to determine where a calculator function needs to be called or not.
#     Your response should be a string that is either 'True' or 'False'. Your response should not contain anything other than either 
#     'True' or 'False'. Given a user query determine wheter a calculation needs to be executed or there is no requirement for any further calculations.
#     Below are some examples of user query, and response pairs.

#     Examples:
#     user query: "what's amzn's revenue for 2022 versus its comps", response: "False"
#     user query: "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps", response: "True"
#     user query: "what's tgt's revenue growth (%) quarter over quarter for 2022 versus its comps", response: "True"
#     user query: "what's wmt's revenue growth (%) quarter over quarter for 2022 versus its comps", response: "True"
#     """

#     prompt = """
#     user query: {question}
#     json data: {financials}
#     """

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt.format(question=question, financials=financials)}
#         ]
#     )

#     try:
#         response = json.loads(response.json())
#         response = response["choices"][0]["message"]["content"]
#         import pdb
#         # pdb.set_trace()
#         print(f"response: {response}")
#         calculation_required = ast.literal_eval(response)
#         print(f"calculation_required: {calculation_required}")
#         return calculation_required
#     except Exception as e:
#         print(f"Error inside should_calculate: {e}")
#         return []

def should_calculate(question, financials):
    system_prompt = """ 
    You are a tool used to guide a RAG system. Your job is to determine where a calculator function needs to be called or not.
    Your response should be a string that is either 'True' or 'False'. Your response should not contain anything other than either 
    'True' or 'False'. Given a user query and a supplied json contain data used to answer the user determine wheter a calculation needs
    to be executed or if the current raw json data is sufficient to answer the quesiton and thus doesn't require any further calculations.
    Below are some examples of user query, json data, and response triplets.

    Examples:
    user query: "what's amzn's revenue for 2022 versus its comps", data: {"revenue": {"2022": [1000000000000, 1000000000025, 4000000000000, 3000000000000]}}, response: "False"
    user query: "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenue": {"2022": [1000000000000, 1000000000025, 4000000000000, 3000000000000]}}, response: "True"
    user query: "what's tgt's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenues":{"Q1 2022":24197000000.0,"Q2 2022":25160000000.0,"Q3 2022":25652000000.0,"Q4 2022":30996000000.0}}, response: "True"
    user query: "what's wmt's revenue growth (%) quarter over quarter for 2022 versus its comps", data: {"revenues":{"Q1 2022":116444000000.0,"Q2 2022":121234000000.0,"Q3 2022":127101000000.0,"Q4 2022":149204000000.0}}, response: "True"
    """

    prompt = """
    user query: {question}
    json data: {financials}
    """

    response = client.chat.completions.create(
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
        print(f"response: {response}")
        calculation_required = ast.literal_eval(response)
        print(f"calculation_required: {calculation_required}")
        return calculation_required
    except Exception as e:
        print(f"Error inside should_calculate: {e}")
        return []

def do_local_calculate(question, financials):
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, financials=financials[::-1])}
        ]
    )

    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]
        import pdb
        pdb.set_trace()
        print(f"response: {response}")
        new_financials = eval(response)
        print(f"new_financials: {new_financials}")
        if isinstance(new_financials, pd.Series):
            new_financials = new_financials.to_frame()
            new_financials.dropna(inplace=True)
        else:
            new_financials = pd.DataFrame({"result": new_financials})
        return new_financials
    except Exception as e:
        print(f"Error inside do_local_calculate: {e}")
        return []


def get_competitors(question):
    system_prompt = """ 
    You are a hedge fund analyst tasked with identifying company competitors. Gien a user query return a python list of company competitors. 
    Each competitor in the list should be a string ticker. Your response should only contain the list of tickers that are competitors to
    the company or ticker mentioned in the user query. Do not include ANYTHING other than the python list of only the ticker values!
    """

    prompt = """
    user query: {question}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )

    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]
        print(f"response: {response}")
        competitors = ast.literal_eval(response[response.find("["):response.find("]")+1])
        print(f"competitors: {competitors}")
        return competitors
    except Exception as e:
        print(f"Error inside get_competitors: {e}")
        return []


def get_research_plan(question):
    system_content = """
    You are the task planner on an elite team of world-class financial researchers. Your team delivers insightful research  based on any company or industry and produce factual based results for your clients.

    You job is to receive a query from the client, and break it into a series of sub-tasks that are assigned to one of the tools to complete. 

    Think step by step through the process of how your team will get to the answer. When describing the task, just give a quick few words in bullet format.

    Research Assistants:
    'get_competitors': Finds companies or competitors for a given query
    'get_sec_financials': Pulls reported financials from SEC filings
    'get_stock_price_date': Pulls stock market price data
    'get_final_analysis': Produces a final analysis of the research

    Use the JSON as shown in the below examples when producing your response.

    Task Breakdown Examples:
    {
    "queries": [
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


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )

    
    try:
        response = json.loads(response.json())
        research_plan = json.loads(response["choices"][0]["message"]["content"])["queries"][0]["tasks"]
        # research_plan = [t.update({"status": "pending"}) for t in research_plan]
        # print(f"research_plan: {research_plan}")
        # research_plan = ast.literal_eval(response[response.find("["):response.find("]")+1])
        print(f"research_plan (type: {type(research_plan)}):\n{research_plan}")
        return research_plan
    except Exception as e:
        print(f"Error inside get_research_plan response: {e}")
        return []


def extract_entities_from_user_query(question):
    system_prompt = """ 
    You are an NLP extraction tool. Your task is to take a user query and extract the following entity types if they appear in the query: 'ticker', 'company', 'dates'.
    Below are the descriptions of the values corresponding to each entity type:
        - 'ticker': a company ticker.
        - 'company': a company name.
        - 'dates': a list of dates mentioned in the user query. This should be a string in the form 'YYYY-MM-DD'. If there are multiple dates in the list they should be sorted such that the oldest date appears first and the most recent date appears last.

    The response should be a JSON list of dictionaries where each dictionary element contains a key called 'entity' whose value is the entity type extracted 
    (i.e. is one of the listed types above) and contains a key called 'value' that is the extracted value from the user query. If the entity type doesn't appear in the user 
    query then it should not appear in the output.

    ## EXAMPLES
    user query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL')]
    user query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    user query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    user query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL')]
    user query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    user query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    user query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL')]
    user query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    user query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    user query: "What is aapl's revenue for 2023?", answer: [('ticker', 'AAPL')]
    user query: "How has amzn's income tax trended for the past 10 years?", answer: [('ticker', 'AMZN')]
    user query: "how has msft's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "how has MSFT's revenue trended in the past 2 years?", answer: [('ticker', 'MSFT')]
    user query: "How has amzn's revenue trended?", answer: [('ticker', 'AMZN')]
    user query: "what's ebay's revenue growth (%) quarter over quarter for 2022 versus its comps", answer: [('ticker', 'EBAY')]
    """

    prompt = """
    user query: {question}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )
    print(f"question: {question}")
    print(f"response: {response}")

    try:
        response = json.loads(response.json())
        print(f"response2: {response}")
        if "json" in response["choices"][0]["message"]["content"]:
            import pdb
            # pdb.set_trace()
            trimmed_response = response["choices"][0]["message"]["content"][response["choices"][0]["message"]["content"].index("json") + 4: len(response["choices"][0]["message"]["content"]) - 3]
            entities = json.loads(trimmed_response)
        else:
            entities = ast.literal_eval(response["choices"][0]["message"]["content"])

        print(f"entities: {entities}")
        return entities
    except Exception as e:
        print(f"Error inside extract_entities_from_user_query: {e}")
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


def get_relevant_rows(financials, question):
    """
    Prompt ChatGPT to identify the relevant rows from the financial data.

    :param financials: JSON response containing financial data
    :param question: User's question about the financial data
    :return: List of relevant row indices
    """
    # Prepare the prompt for ChatGPT
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
    response = client.chat.completions.create(
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
    system_prompt = """
    You are a helpful assistant that analyzes financial data. Given a user question and list of financial reporting dates, determine which of the
    dates are required to answer the question. The response should be a python list of the values in the supplied rows. Below are some examples:

    Examples:
        {
            "queries": [ 
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
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, financials=financials)}
        ]
    )

    # Extract the relevant row indices from the API response
    try:
        response = json.loads(response.json())
        response = response["choices"][0]["message"]["content"]
        relevant_rows = ast.literal_eval(response[response.find("["):response.find("]")+1])
        import pdb
        # pdb.set_trace()
        
        return relevant_rows
    except Exception as e:
        print(f"Error parsing ChatGPT response: {e}")
        return []


def get_financials(question, results):
    entities = extract_entities_from_user_query(question)
    # import pdb
    # pdb.set_trace()
    ticker = None
    for ent in entities:
        if ent["entity"] == "ticker":
            ticker = ent["value"]
            break
            
    financials = get_stock_financials(ticker.upper())
    relevant_rows = get_relevant_rows(list(financials.index.values), question)
    relevant_columns = get_relevant_columns(list(financials.columns), question)
    financials = financials.loc[relevant_rows, relevant_columns]

    calculation_required = should_calculate(question, financials)
    if calculation_required:
        financials = do_local_calculate(question, financials.T[::-1])
    print(f"calculation_required: {calculation_required}")


    financials = financials.rename(columns={financials.columns[0]: f"{ticker.upper()}.{financials.columns[0]}"})
    chart_data = financials.to_dict('records')

    results["Context"].append(
        f"{financials.T.columns[0]} results for {ticker.upper()}: {financials.T.to_dict('records')}"
    )

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

    return financials.to_json()



def merge_charts(results):
    charts_dicts = results["finalAnalysis"]["charts"]
    datas = []
    for ticker, chart_dict in charts_dicts.items():
        datas.append(chart_dict)

    processed_data = [{} for i in range(len(datas[0]))]
    datapoints = []
    for i in range(len(datas[0])):
        for j in range(len(datas)):
            processed_data[i].update(datas[j][i])
            print(f"processed_data[{i}]: {processed_data[i]}")

    chart = {
            "type": "line",
            "data": processed_data,
            "dataKeys": [k for k in processed_data[0].keys()]
    }
    results["finalAnalysis"]["charts"] = {f'Chart for Query: {results["Query"]}': chart}
    return results


"""

chart = {
    ticker: {
        "type": "line",
        "data": financials.to_dict('records'),
        "dataKeys": [ticker]
    }
}

table = {
    table_name: {
        "headers": list(financials.columns),
        "rows": financials.values.tolist()
    }
}
"""
def merge_tables(results):
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
    print(f"table: {table}")
    results["finalAnalysis"]["tables"] = table
    results["finalAnalysis"]["workbookData"] = {"headers": table[f'Table for Query: {results["Query"]}']["headers"], "rows": table[f'Table for Query: {results["Query"]}']["rows"]}
    return results
    

def get_final_analysis(query, results):
    final_analysis = results["finalAnalysis"]
    context = results["Context"]
    results = merge_charts(results)
    results = merge_tables(results)

    print(f"context: {context}")
    context = '\n'.join([str(c) for c in context])

    system_content = """
    Given a user-supplied query and a context that contains the raw data to necessary to answer the query, generate a response that synthesizes the provided context to answer the question.
    """

    prompt = """
    query: {query}

    context: {context}
    """

    response = client.chat.completions.create(
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
        return results
    except Exception as e:
        print(f"Error parsing ChatGPT response: {e}")
        return []


def response_formatter(final_response, task_json):
    print(f"task_json: {task_json}, {type(task_json)}")
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

    response = client.chat.completions.create(
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
        print(f"Error parsing ChatGPT response: {e}")
        return []