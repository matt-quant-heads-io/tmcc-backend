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

import yfinance as yahooFinance
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch as th

SENTIMENT_TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
SENTIMENT_MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


load_dotenv()


class GPTResponseFormat(BaseModel):
    answer: str

class NLPExtractionResponseFormat(BaseModel):
    answer: dict


class Task(BaseModel):
    task: str
    description: str
    status: str

class ResearchPlanResponseFormat(BaseModel):
    tasks: list[Task]

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

browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")


openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
)
cohere_client = cohere.Client(api_key=cohere_api_key)
auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)

weaviate_client = client = weaviate.Client(
  url="https://meyofexrqzye9frlt6ugg.c0.us-east1.gcp.weaviate.cloud",
  auth_client_secret=auth_config,
  additional_headers = {
        "X-OpenAI-Api-Key": OPENAI_API_KEY  # Replace with your inference API key
    }
)

llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo-preview", api_key=OPENAI_API_KEY)

# weaviate_client = weaviate.Client(
#     url=WEAVIATE_URL,                                    # Replace with your Weaviate Cloud URL
#     auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
#     additional_headers = {
#         "X-OpenAI-Api-Key": OPENAI_API_KEY  # Replace with your inference API key
#     }
# )

def google_search(search_keyword, time_frame):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": f"site:bloomberg.com {search_keyword}",
        "tbs":
        f"qdr:{time_frame}"  # time_frame can be 'd' for day, 'w' for week, 'm' for month, 'y' for year
    })
    headers = {'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=payload)
    return response.json()


def web_scraping(url):
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    data = {"url": url}
    response = requests.post(
        f"https://chrome.browserless.io/content?token={browserless_api_key}",
        headers=headers,
        json=data)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    else:
        return f"Failed to scrape: HTTP {response.status_code}"


def summarize_content(content, objective):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106", api_key=OPENAI_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])

    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt,
                                         input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(llm=llm,
                                         chain_type='map_reduce',
                                         map_prompt=map_prompt_template,
                                         combine_prompt=map_prompt_template,
                                         verbose=False)

    return summary_chain.run(input_documents=docs, objective=objective)


from datetime import datetime

date_string = "Feb 16, 2024"
date_object = datetime.strptime(date_string, "%b %d, %Y")


def perform_news_search_via_google(question, results, time_frame='m', debug=False):
    # Step 1: Perform Google search
    search_results = google_search(question, time_frame)
    entities = extract_entities_from_user_query(question)
    tickers = [e["value"] for e in entities if e["entity"] == "ticker"]
    ticker = tickers[0]
    if len(tickers) == 0:
        tickers = ""
    elif len(tickers) == 1:
        tickers = tickers[0]
    else:
        tickers = ','.join(tickers)

    # Step 2: Scrape and summarize top articles
    summaries = []
    citations = []
    for result in search_results.get('organic', [])[:3]:  # Process top 3 results
        url = result.get('link')
        if url:
            report_date = str(datetime.strptime(result.get('date'), "%b %d, %Y")).split(' ')[0]
            content = web_scraping(url)
            summary = summarize_content(
                content, f"Summarize the key points related to: {question}"
            )
            summaries.append({
                'title': result.get('title'),
                'url': url,
                'summary': summary,
                'report_date': report_date
            })

            new_citation = {
                "report_date": report_date,
                "id": str(uuid.uuid4()),
                "logo": "",
                "page_number": "",
                "url": url,#f"https://{BUCKET_FOR_HIGHLIGHTED_DOCS}.s3.amazonaws.com/{highlighted_pdf_filename}", #pdf_to_save_pdf_before_upload_to_s3,#
                "title": result.get('title'),
                "company": tickers,
                "importance": 1,
                "text": summary
            }
            citations.append(new_citation)

    # Step 3: Synthesize findings
    synthesis_prompt = f"""
    Based on the following summaries of Bloomberg articles about "{question}":

    {json.dumps(summaries, indent=2)}

    Please provide a comprehensive analysis that:
    1. Identifies the main trends or themes across the articles
    2. Highlights any conflicting viewpoints or debates
    3. Summarizes the overall impact or implications discussed
    4. Suggests potential future developments or areas for further research

    Your analysis should be well-structured, insightful, and about 500 words long.
    """

    synthesis = llm.predict(synthesis_prompt)
    results["Context"].append({f"Response from query {question}": {
        'query': question,
        'time_frame': time_frame,
        'summaries': summaries,
    }})

    if "finalAnalysis" not in results:
        results["finalAnalysis"] = {}
    
    if "citations" not in results["finalAnalysis"]:
        results["finalAnalysis"]["citations"] = []    

    results["finalAnalysis"]["citations"].extend(citations)
    df_citations = pd.DataFrame.from_records(citations)
    df_citations["temp_date_sort_key"] = pd.to_datetime(df_citations["report_date"])
    df_citations.sort_values(by=["temp_date_sort_key"], ascending=False, inplace=True)
    df_citations.drop("temp_date_sort_key", axis=1, inplace=True)
    df_citations.reset_index(drop=True, inplace=True)
    results["GetNews"][ticker] = df_citations
    results["finalAnalysis"]["tables"][ticker] = df_citations

    import pdb; pdb.set_trace()

    return results


# def perform_news_search_via_google(question, results, time_frame='m', debug=False):
#     # Step 1: Perform Google search
#     search_results = google_search(question, time_frame)
#     entities = extract_entities_from_user_query(question)
#     tickers = [e["value"] for e in entities if e["entity"] == "ticker"]
#     if len(tickers) == 0:
#         tickers = ""
#     elif len(tickers) == 1:
#         tickers = tickers[0]
#     else:
#         tickers = ','.join(tickers)

#     # Step 2: Scrape and summarize top articles
#     summaries = []
#     citations = []
#     import pdb; pdb.set_trace()
#     for result in search_results.get('organic', [])[:3]:  # Process top 3 results
#         url = result.get('link')
#         if url:
#             content = web_scraping(url)
#             summary = summarize_content(
#                 content, f"Summarize the key points related to: {question}"
#             )
#             summaries.append({
#                 'title': result.get('title'),
#                 'url': url,
#                 'summary': summary
#             })

#             new_citation = {
#                 "id": str(uuid.uuid4()),
#                 "logo": "",
#                 "page_number": "",
#                 "url": url,#f"https://{BUCKET_FOR_HIGHLIGHTED_DOCS}.s3.amazonaws.com/{highlighted_pdf_filename}", #pdf_to_save_pdf_before_upload_to_s3,#
#                 "title": result.get('title'),
#                 "company": tickers,
#                 "importance": 1,
#                 "text": summary
#             }
#             citations.append(new_citation)

#     # Step 3: Synthesize findings
#     synthesis_prompt = f"""
#     Based on the following summaries of Bloomberg articles about "{question}":

#     {json.dumps(summaries, indent=2)}

#     Please provide a comprehensive analysis that:
#     1. Identifies the main trends or themes across the articles
#     2. Highlights any conflicting viewpoints or debates
#     3. Summarizes the overall impact or implications discussed
#     4. Suggests potential future developments or areas for further research

#     Your analysis should be well-structured, insightful, and about 500 words long.
#     """

#     synthesis = llm.predict(synthesis_prompt)
#     results["Context"].append({f"Response from query {question}": {
#         'query': question,
#         'time_frame': time_frame,
#         'summaries': summaries,
#     }})

#     if "finalAnalysis" not in results:
#         results["finalAnalysis"] = {}
    
#     if "citations" not in results["finalAnalysis"]:
#         results["finalAnalysis"]["citations"] = []    
#     results["finalAnalysis"]["citations"].extend(citations)

#     return results


# def preprocess_user_query(question, debug=False):
#     system_prompt = """ 
#     You are an NLP tool. Given a user query, your task is to determine if the user query contains a company name. If it does,
#     replace the company name with its corresponding ticker and respond with the new query (which should now contain the ticker instead of the company name).
#     If the query doesn't contain a company name then respond with the EXACT query that was supplied initially. If the query already contains a ticker then respond with the EXACT query that was supplied initially. 
#     Your response should ONLY contain the query and nothing else, no preamble. Below are some example user queries and their corresponding responses.

#     Examples:
#     query: "whats amzn's quarterly revenue for 2022? compare this to its comps", answer: ["whats amzn's quarterly revenue for 2022?", "whats aapl's quarterly revenue for 2022?", "whats msft's quarterly revenue for 2022?"]
#     query: "whats Amazon's quarterly revenue for 2022?", answer: ["whats amzn's quarterly revenue for 2022?"]
#     query: "whats target's cogs between 2021 and 2023?", answer: ["whats TGT's cogs between 2021 and 2023?"]
#     query: "how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after? compare this to AAPL", answer: ["how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?", how many times did AAPL mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?]


#     """

#     prompt = """
#     user query: {question}
#     """

#     response = openai_client.beta.chat.completions.parse(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt.format(question=question)}
#         ],
        
#     )

#     try:
#         # import pdb; pdb.set_trace()
#         response = json.loads(response.to_json())["choices"][0]["message"]["content"]
#         processed_queries = ast.literal_eval(response[response.find("["):response.find("]")+1])
#         entities = extract_entities_from_user_query(processed_queries, debug)
#         if debug:
#             with open(DEBUG_ABS_FILE_PATH, "a") as f:
#                 f.write(json.dumps({"function": "preprocess_user_query", "input": [question], "processed_query": processed_query, "entities": entities}, indent=6))
#         return processed_queries, entities
#     except Exception as e:
#         print(f"Error inside processed_query: {e}")
#         return []

def map_list_of_companies_and_query_to_list_of_queries(query, companies, debug=False):
    system_prompt = """ 
    You are an query transformer tool. Given a user query, and a list of companies your job is create a list queries where each query in the list contains only one company from the list.
    Each query in your response should preserve the essence of user query but include 1 company from the list of companies. If the original query contains any company names then the corresponding answer should contain the company's ticker NOT the company name.
    Below are examples. Your response should be a list of strings where each
    string contains 1 and ONLY 1 company from the companies list and each query preserves the original intent of the user supplied query.

    Examples:
    query: "Of the largest technology stocks in the dow 30, which have revenue growth exceeding 10%?", companies: ['BA', 'LMT'], answer: ["What's BA's revenue growth? Does it exceed 10%?", "What's LMT's revenue growth? Does it exceed 10%?"]
    query: "Of the largest technology stocks how many times was macro concerns mentioned in their filings since 2023? What has been their coinciding price performance?", companies: ['TSLA', 'MSFT', 'AAPL', 'CRM'], answer: ["How many times did TSLA mention macro concerns in their filings since 2023? How has this coincided with their stock performance?", "How many times did MSFT mention macro concerns in their filings since 2023? How has this coincided with their stock performance?", "How many times did AAPL mention macro concerns in their filings since 2023? How has this coincided with their stock performance?", "How many times did CRM mention macro concerns in their filings since 2023? How has this coincided with their stock performance?"]
    query: "run a backtest on the tech stocks of the dow 30 assuming i buy a company every time they mention macro concerns in a filings", companies: ['AAPL', 'AMGN', 'CRM', 'CSCO', 'INTC', 'MSFT'], answer: ["Run a backtest on AAPL, buying shares every time they mention macro concerns in their filings.", "Run a backtest on AMGN, buying shares every time they mention macro concerns in their filings.", "Run a backtest on CRM, buying shares every time they mention macro concerns in their filings.", "Run a backtest on CSCO, buying shares every time they mention macro concerns in their filings.", "Run a backtest on INTC, buying shares every time they mention macro concerns in their filings.", "Run a backtest on MSFT, buying shares every time they mention macro concerns in their filings."]
    """

    prompt = """
    user query: {question}
    companies: {companies}
    """

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=query, companies=companies)}
        ],   
    )

    try:
        # import pdb; pdb.set_trace()
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        processed_queries = ast.literal_eval(response)
        processed_queries = [p.lstrip().rstrip() for p in processed_queries]
        return processed_queries
    except Exception as e:
        print(f"Error inside map_list_of_companies_and_query_to_list_of_queries: {e}")
        return [query]



def map_competitors_and_query_to_list_of_queries(query, competitors, debug=False):
    system_prompt = """ 
    You are an query transformer tool. Given a user query, and a list of companies your job is create a list queries where each query in the list contains only one company from the list.
    The user-supplied query should already contain a company ticker. Include this query in your final response. In addition, replace this ticker in each subsequent 
    query you provide in your response with one of the companies from the supplied list. Your response should be a list of query strings, 1 for the original supplied query and one for each
    company ticker in the supplied list with the given ticker replacing the ticker from the original supplied query. If the original query contains any company names then the corresponding answer should contain the company's ticker NOT the company name. Below are examples. Your response should be a list of strings where each
    string contains 1 and ONLY 1 company from the companies list plus the original supplied query and each query preserves the original intent of the user supplied query.

    Examples:
    query: "Of the largest technology stocks in the dow 30, which have revenue growth exceeding 10%?", companies: ['BA', 'LMT'], answer: ["What's BA's revenue growth? Does it exceed 10%?", "What's LMT's revenue growth? Does it exceed 10%?"]
    query: "Of the largest technology stocks how many times was macro concerns mentioned in their filings since 2023? What has been their coinciding price performance?", companies: ['TSLA', 'MSFT', 'AAPL', 'CRM'], answer: ["How many times did TSLA mention macro concerns in their filings since 2023? How has this coincided with their stock performance?", "How many times did MSFT mention macro concerns in their filings since 2023? How has this coincided with their stock performance?", "How many times did AAPL mention macro concerns in their filings since 2023? How has this coincided with their stock performance?", "How many times did CRM mention macro concerns in their filings since 2023? How has this coincided with their stock performance?"]
    """

    prompt = """
    user query: {question}
    companies: {competitors}
    """

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=query, competitors=competitors)}
        ],   
    )

    try:
        # import pdb; pdb.set_trace()
        # response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        # processed_queries = ast.literal_eval(response[response.find("["):response.find("]")+1])
        # entities = extract_entities_from_user_query(processed_queries, debug)
        # if debug:
        #     with open(DEBUG_ABS_FILE_PATH, "a") as f:
        #         f.write(json.dumps({"function": "preprocess_user_query", "input": [question], "processed_query": processed_query, "entities": entities}, indent=6))
        # return processed_queries, entities
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        processed_queries = ast.literal_eval(response)
        return processed_queries
    except Exception as e:
        print(f"Error inside map_competitors_and_query_to_list_of_queries: {e}")
        return [response]


def split_companies(query, debug=False):
    system_prompt = """ 
    You are an NLP tool. Given a user query, your task is given a user query, if the query contains more than 1 company or ticker in the query then split the query into multiple answer queries such that each query contains 1 and ONLY 1 company or ticker in it.
    The resultant list of query strings should preserve the original mean of the supplied query. If the original query contains any company names then the corresponding answer should contain the company's ticker NOT the company name. Below are some example queries and the corresponding correct answer. Your response should be a list of query strings
    ONLY and each query string in the list should only contain one company or ticker. Below are some examples.

    Examples:
    query: "how has Ford's revenue trended since 2020? what has been the stock performance over that time period? Compare to TSLA", answer: ["how has F's revenue trended since 2020? what has been the stock performance over that time period?", "how has TSLA's revenue trended since 2020? what has been the stock performance over that time period?"]
    query: "how has Ford's revenue trended since 2020? what has been the stock performance over that time period? Compare to TESLA", answer: ["how has F's revenue trended since 2020? what has been the stock performance over that time period?", "how has TSLA's revenue trended since 2020? what has been the stock performance over that time period?"]
    query: "how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after? compare this to AAPL", answer: ["how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?", "how many times did AAPL mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?"]
    query: "Who has the largest revenue growth out of AAPL, MSFT, and LMT?", answer: ["Whats the revenue growth for AAPL?", "Whats the revenue growth for MSFT?", "Whats the revenue growth for LMT?"]
    query: "how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after? compare this to AAPL", answer: ["how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?", "how many times did AAPL mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?"]
    query: "Who has the largest revenue growth out of AAPL, MSFT, and LMT?", answer: ["Whats the revenue growth for AAPL?", "Whats the revenue growth for MSFT?", "Whats the revenue growth for LMT?"]
    query: "how has Ford's revenue trended since 2020? what has been the stock performance over that time period? Compare to TSLA", answer: ["how has F's revenue trended since 2020? what has been the stock performance over that time period?", "how has TSLA's revenue trended since 2020? what has been the stock performance over that time period?"]
    query: "how has Ford's revenue trended since 2020? what has been the stock performance over that time period? Compare to TESLA", answer: ["how has F's revenue trended since 2020? what has been the stock performance over that time period?", "how has TSLA's revenue trended since 2020? what has been the stock performance over that time period?"]
    query: "Compare Ford versus Tesla's revenue growth and stock performance since 2020? Which has appreciated more and by how much?", answer: ["What's F's revenue growth and stock performance since 2020?", "What's TSLA's revenue growth and stock performance since 2020?"]
    query: "How do the debt levels of Apple and MSFT compare?", answer: ["What are the debt levels of AAPL?", "What are the debt levels of MSFT?"]
    query: "Identify sections discussing 'supply chain disruptions' in the latest 10-Q filings for Apple (AAPL), Boeing (BA), and Home Depot (HD). Summarize the potential impacts on their operations.", answer: ["Identify sections discussing 'supply chain disruptions' in the latest 10-Q filings for Apple (AAPL)", "Identify sections discussing 'supply chain disruptions' in the latest 10-Q filings for Boeing (BA)", "Identify sections discussing 'supply chain disruptions' in the latest 10-Q filings for Home Depot (HD)"]
    query: "Count the number of times 'non-GAAP financial measures' are mentioned in the last three years of 10-K filings for all technology companies in the Dow 30, including Microsoft (MSFT), Intel (INTC), and IBM.", answer: ["Count the number of times 'non-GAAP financial measures' are mentioned in the last three years of 10-K filings for Microsoft (MSFT)", "Count the number of times 'non-GAAP financial measures' are mentioned in the last three years of 10-K filings for Intel (INTC)", "Count the number of times 'non-GAAP financial measures' are mentioned in the last three years of 10-K filings for IBM"]
    query: "Analyze the sentiment of the forward-looking statements in the latest 10-K filings for JPMorgan Chase (JPM) and Goldman Sachs (GS). Identify whether the tone is predominantly positive, negative, or neutral.", answer: ["Analyze the sentiment of the forward-looking statements in the latest 10-K filings for JPMorgan Chase (JPM). Identify whether the tone is predominantly positive, negative, or neutral.", "Analyze the sentiment of the forward-looking statements in the latest 10-K filings for Goldman Sachs (GS). Identify whether the tone is predominantly positive, negative, or neutral."]
    query: "Extract discussions on 'climate change risk' within 10-K filings for all energy and industrial companies in the Dow 30, including Chevron (CVX) and Caterpillar (CAT). What are the common risk mitigation strategies noted?", answer: ["Extract discussions on 'climate change risk' within 10-K filings for Chevron (CVX). What are the common risk mitigation strategies noted?", "Extract discussions on 'climate change risk' within 10-K filings for Caterpillar (CAT). What are the common risk mitigation strategies noted?"]
    query: "Compare the discussion of 'revenue growth strategies' in 10-K and 10-Q filings from 2018 to 2023 for Coca-Cola (KO) and PepsiCo (PEP). How have these strategies evolved over time?", answer: ["Summarize the revenue growth strategies in 10-K and 10-Q filings from 2018 to 2023 for Coca-Cola (KO). How have their strategies evolved over time?","Summarize the revenue growth strategies in 10-K and 10-Q filings from 2018 to 2023 for PepsiCo (PEP). How have their strategies evolved over time?"]
    """

    prompt = """
    user query: {question}
    """

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=query)}
        ],   
    )

    try:
        # import pdb; pdb.set_trace()
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        if "json" in response:
            processed_queries = ast.literal_eval(response[response.find("```json\n")+len("```json\n"):len(response)-3])
        else:
            processed_queries = ast.literal_eval(response)
        # entities = extract_entities_from_user_query(processed_queries, debug)
        # import pdb; pdb.set_trace()
        return processed_queries
    except Exception as e:
        print(f"Error inside split_companies: {e}")
        return [response]



def determine_query_pattern(question, debug=False):
    system_prompt = """ 
    You are an NLP tool. Given a user query, your task is to determine if either a 'get_list_of_companies' function, a 'get_competitors' function, a 'split_companies' function, or 'neither' should be called for the given user query. If neither need to get called 
    then just return the value 'neither'. 'split_companies' should only be called if the query contains more than one company or company ticker in it. Below are some example queries and the corresponding correct answer. Your response should be a string and ONLY contain one of the values 'get_list_of_companies', 'get_competitors', or 
    'neither' and NOTHING else.

    Examples:
    query: "How have the inventory ratios of the largest dow 30 tech stocks trended since 2022? How do they differ in how they discuss inventory management strategies in their filings?", answer: 'get_list_of_companies'
    query: "whats amzn's quarterly revenue for 2022? compare this to its comps", answer: "get_competitors"
    query: "whats Amazon's quarterly revenue for 2022?", answer: 'neither'
    query: "whats target's cogs between 2021 and 2023?", answer: 'neither'
    query: "how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after? compare this to AAPL", answer: 'split_companies'
    query: "how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after? compare this their comps", answer: 'get_competitors'
    query: "Of the largest technology stocks in the dow 30, which have revenue growth exceeding 10%?", answer: 'get_list_of_companies'
    query: "Who has the largest revenue growth out of AAPL, MSFT, and LMT?", answer: 'split_companies'
    query: "Identify sections discussing 'supply chain disruptions' in the latest 10-Q filings for Apple (AAPL), Boeing (BA), and Home Depot (HD). Summarize the potential impacts on their operations.", answer: 'split_companies'
    query: "Count the number of times 'non-GAAP financial measures' are mentioned in the last three years of 10-K filings for all technology companies in the Dow 30, including Microsoft (MSFT), Intel (INTC), and IBM.", answer: 'split_companies' 
    query: "Count the number of times 'non-GAAP financial measures' are mentioned in the last three years of 10-K filings for all technology companies in the Dow 30", answer: 'get_list_of_companies' 
    query: "Analyze the sentiment of the forward-looking statements in the latest 10-K filings for JPMorgan Chase (JPM) and Goldman Sachs (GS). Identify whether the tone is predominantly positive, negative, or neutral.", answer: 'split_companies' 
    query: "Extract discussions on 'climate change risk' within 10-K filings for all energy and industrial companies in the Dow 30, including Chevron (CVX) and Caterpillar (CAT). What are the common risk mitigation strategies noted?", answer: 'split_companies'
    query: "Compare the discussion of 'revenue growth strategies' in 10-K and 10-Q filings from 2018 to 2023 for Coca-Cola (KO) and PepsiCo (PEP). How have these strategies evolved over time?", answer: 'split_companies'   
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
        # import pdb; pdb.set_trace()
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        # processed_queries = ast.literal_eval(response[response.find("["):response.find("]")+1])
        # entities = extract_entities_from_user_query(processed_queries, debug)
        # import pdb; pdb.set_trace()
        return response
    except Exception as e:
        print(f"Error inside processed_query: {e}")
        return []


def preprocess_user_query(question, debug=False):
    system_prompt = """ 
    You are an NLP tool. Given a user query, your task is to determine if the user query contains a company name. If it does,
    replace the company name with its corresponding ticker and respond with the new query (which should now contain the ticker instead of the company name).
    If the query doesn't contain a company name then respond with the EXACT query that was supplied initially. If the query already contains a ticker then respond with the EXACT query that was supplied initially. 
    Your response should ONLY contain the list of queries and nothing else, no preamble. Below are some example user queries and their corresponding responses.

    Examples:
    query: "whats amzn's quarterly revenue for 2022? compare this to its comps", answer: ["whats amzn's quarterly revenue for 2022?", "whats aapl's quarterly revenue for 2022?", "whats msft's quarterly revenue for 2022?"]
    query: "whats Amazon's quarterly revenue for 2022?", answer: ["whats amzn's quarterly revenue for 2022?"]
    query: "whats target's cogs between 2021 and 2023?", answer: ["whats TGT's cogs between 2021 and 2023?"]
    query: "how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after? compare this to AAPL", answer: ["how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?", "how many times did AAPL mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?"]


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
        # import pdb; pdb.set_trace()
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        processed_queries = ast.literal_eval(response)
        # entities = extract_entities_from_user_query(processed_queries, debug)
        # if debug:
        #     with open(DEBUG_ABS_FILE_PATH, "a") as f:
        #         f.write(json.dumps({"function": "preprocess_user_query", "input": [question], "processed_query": processed_query, "entities": entities}, indent=6))
        # return processed_queries, entities
        return processed_queries
    except Exception as e:
        print(f"Error inside processed_query: {e}")
        return []


# def preprocess_user_query(question, debug=False):
#     system_prompt = """ 
#     You are an NLP tool. Given a user query, your task is to determine if the user query contains a company name. If it does,
#     replace the company name with its corresponding ticker and respond with the new query (which should now contain the ticker instead of the company name).
#     If the query doesn't contain a company name then respond with the EXACT query that was supplied initially. If the query already contains a ticker then respond with the EXACT query that was supplied initially. 
#     Your response should ONLY contain the query and nothing else, no preamble. Below are some example user queries and their corresponding responses.

#     Examples:
#     query: "whats amzn's quarterly revenue for 2022? compare this to its comps", answer: ["whats amzn's quarterly revenue for 2022?", "whats aapl's quarterly revenue for 2022?", "whats msft's quarterly revenue for 2022?"]
#     query: "whats Amazon's quarterly revenue for 2022?", answer: ["whats amzn's quarterly revenue for 2022?"]
#     query: "whats target's cogs between 2021 and 2023?", answer: ["whats TGT's cogs between 2021 and 2023?"]
#     query: "how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after? compare this to AAPL", answer: ["how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?", how many times did AAPL mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?]


#     """

#     prompt = """
#     user query: {question}
#     """

#     response = openai_client.beta.chat.completions.parse(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt.format(question=question)}
#         ],
        
#     )

#     try:
#         # import pdb; pdb.set_trace()
#         response = json.loads(response.to_json())["choices"][0]["message"]["content"]
#         processed_queries = ast.literal_eval(response[response.find("["):response.find("]")+1])
#         entities = extract_entities_from_user_query(processed_queries, debug)
#         if debug:
#             with open(DEBUG_ABS_FILE_PATH, "a") as f:
#                 f.write(json.dumps({"function": "preprocess_user_query", "input": [question], "processed_query": processed_query, "entities": entities}, indent=6))
#         return processed_queries, entities
#     except Exception as e:
#         print(f"Error inside processed_query: {e}")
#         return []


def get_list_of_companies(question, debug=False):
    system_prompt = """ 
    You are a hedge fund analyst tasked with identifying a list of public companies that fit the criteria given by a user query. Given a user query return a JSON list of company tickers. 
    Each company ticker in the list should be a string ticker. Your response should only contain the list of tickers that are companies that satisfy the requirement outlined by the user query.
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
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        competitors = ast.literal_eval(response[response.find("["):response.find("]")+1])[:3]
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "get_list_of_companies", "inputs": [question], "outputs": [{"competitors": competitors}]}, indent=6))
        return competitors
    except Exception as e:
        print(f"Error inside get_list_of_company: {e}")
        import pdb; pdb.set_trace()
        return []


def should_local_calculate(question, financials, debug=False):
    system_prompt = """ 
    You are a tool used to guide a RAG system. Your job is to determine where a calculator function needs to be called or not.
    Your response should be a string that is either 'True' or 'False'. Your response should not contain anything other than either 
    'True' or 'False'. Given a user query and a supplied json contain data used to answer the user determine wheter a calculation needs
    to be executed or if the current raw json data is sufficient to answer the quesiton and thus doesn't require any further calculations.
    Below are some examples of user query, json data, and response triplets.

    Examples:
    user query: "What was microsoft's effect of exchange rate on Cash and Cash Equivalents in Q3 2020 and Q4 2020?", data: ['exchange_rate'], answer: "False"
    user query: "what's amzn's revenue for 2022 versus its comps", data: ['revenue'], answer: "False"
    user query: "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "True"
    user query: "what's tgt's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "True"
    user query: "what's wmt's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "True"
    user query: "What was the total amount spent on Acquisitions Net of Cash Acquired and Purchases of Intangible and Other Assets by MSFT in the year 2022?", data: ['cash', 'intangible_assets', 'fixed_assets', 'net_cash_flow_from_investing_activities'], answer: "True"
    user query: "What was the trend in MSFT's Payments for Repurchase of Common Stock from Q3 2020 to Q1 2023?", data: ['revenue'], answer: "False"
    user query: "Examine the trends in American Express's credit card delinquency rates and provisions for credit losses over the past five years.", data: ['provision_for_loan_lease_and_other_losses', 'net_income_loss', 'noninterest_expense', 'interest_income_expense_after_provision_for_losses', 'current_assets', 'liabilities', 'long_term_debt'], answer: "True"
    user query: "Compare MSFT's cloud sales growth to IBM's", data: ['revenues'], answer: "True"
    user query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", data: ['revenues', 'research_and_development'], answer: "True"
    user query: "How has apple’s gross margins trended compared to Intel and Microsoft’s over the past 3 years?", data: ['gross_profit', 'revenues'], answer: "True"
    """

    prompt = """
    user query: {question} data: {financials}
    """

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, financials=financials)}
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
        
        print(f"response from should_local_calculate: {response}")
        return calculation_required
    except Exception as e:
        print(f"Error inside should_local_calculate: {e}")
        return False



"""
user query: "examine the changes in lmt's debt-to-equity ratio over the last 2 years. what are the implications for its financial stability compared to lmt and air in the aerospace industry?", 'data': ['equity', 'liabilities'], 'answer': "financials['debt_to_equity_ratio'] = financials['liabilities'] / financials['equity']; financials.drop(['liabilities', 'equity'], axis=1, inplace=True)"
user query: "examine the changes in air's debt-to-equity ratio over the last 2 years. what are the implications for its financial stability compared to competitors in the aerospace industry?", 'data': ['equity', 'liabilities'], 'answer': "financials['debt_to_equity_ratio'] = financials['liabilities'] / financials['equity']; financials.drop(['liabilities', 'equity'], axis=1, inplace=True)"
user query: "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "financials['revenue_growth'] = financials['revenue'].pct_change(periods=-1).round(2).dropna(); financials.drop(['revenue'], axis=1, inplace=True)"
user query: "what's WMT's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "financials['revenue_growth'] = financials['revenue'].pct_change(periods=-1).round(2).dropna(); financials.drop(['revenue'], axis=1, inplace=True)"
user query: "how did msft's net cash provided by operating activities change from q3 2020 to q1 2021?", data: ['net_cash_flow_from_operating_activities'], answer: "financials['net_cash_flow_from_operating_activities_change'] = financials['net_cash_flow_from_operating_activities'].pct_change(periods=-1).round(2).dropna(); financials.drop(['net_cash_flow_from_operating_activities'], axis=1, inplace=True)"        
user query: "what's amzn's revenue growth during 2023 versus its comps", data: ['revenue'], answer: "financials['revenue_growth'] = financials['revenue'].pct_change(periods=-1).round(2).dropna(); financials.drop(['revenue'], axis=1, inplace=True)"
user query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", data: ['revenues', 'research_and_development'], answer: "financials['research_and_development_to_revenue_pct'] = financials['research_and_development'] / financials['revenues']; financials.drop(['revenues', 'research_and_development'], axis=1, inplace=True)"   
user query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", data: ['revenues', 'research_and_development'], answer: "financials['research_and_development_to_revenue_pct'] = financials['research_and_development'] / financials['revenues']; financials.drop(['revenues', 'research_and_development'], axis=1, inplace=True)"
user query: "How has apple’s gross margins trended compared to Intel and Microsoft’s over the past 3 years?", data: ['gross_profit', 'revenues'], answer: "financials['gross_margin_change'] = financials['gross_margin'] = financials['gross_profit'] / financials['revenues'];  financials.drop(['revenues', 'gross_profit'], axis=1, inplace=True)"
user query: "Examine the changes in Boeing's debt-to-equity ratio over the last five years. What are the implications for its financial stability compared to competitors in the aerospace industry?", data: ['long_term_debt', 'equity'], answer: "financials['debt_to_equity_ratio'] = financials['long_term_debt'] / financials['equity']; financials.drop(['long_term_debt', 'equity'], axis=1, inplace=True)"
user query: "Show how amzn's cogs have grown for the past 4 quarters?", data: ['cost_of_revenue'], answer: "financials['cogs_growth'] = financials['cost_of_revenue'].pct_change(periods=-1).round(2).dropna(); financials.drop(['cost_of_revenue'], axis=1, inplace=True); financials.dropna(inplace=True)"
user query: "How has amzn's cogs grown annually quarter over quarter over 2023 vs 2022?", data: ['cost_of_revenue'], answer: "financials['cogs_growth'] = financials['cost_of_revenue'].pct_change(periods=-4).round(2).dropna(); financials.drop(['cost_of_revenue'], axis=1, inplace=True)"
user query: "How has amzn's cogs grown annually quarter over quarter over 2023?", , data: ['cost_of_revenue'], , answer: "financials['cogs_growth'] = financials['cost_of_revenue'].pct_change(periods=-4).round(2).dropna(); financials.drop(['cost_of_revenue'], axis=1, inplace=True)"
user query: "how has aapl’s cogs trended compared to intc and msft’s over the past 3 years?", data: ['cost_of_revenue'], answer: "financials['cogs_growth'] = financials['cost_of_revenue'].pct_change(periods=-1).round(2).dropna(); financials.drop(['cost_of_revenue'], axis=1, inplace=True)"
user query: "How has apple’s gross margins trended compared to Intel and Microsoft’s over the past 3 years?", data: ['gross_profit', 'revenues'], answer: "financials['gross_margin'] = financials['gross_profit'] / financials['revenues']; financials.drop(['revenues', 'gross_profit'], axis=1, inplace=True)"
user query: "Show me the percentage change of aapl's gross margins versus Intel and Microsoft over the past 3 years?", data: ['gross_profit', 'revenues'], answer: "financials['gross_margin'] = financials['gross_profit'] / financials['revenues']; financials['gross_margin_change'] = financials['gross_margin'].pct_change(periods=-1).round(2).dropna(); financials.drop(['revenues', 'gross_profit'], axis=1, inplace=True)"

"""


def do_local_calculate(question, financials, debug=False):
    try:
        system_prompt = """ 
        You are a tool used in a rag system to determine an equation that needs to be applied to a pandas dataframe. Your job is to generate 
        the string representation of the pandas calculation such that the python eval function can be called on the string.
        Your response should not contain anything other than the pandas equation to be called. Including anything other than just the equation 
        will cause the functin to fail. 
        
        Given a user query and the supplied json that represents the pandas DataFrame, return the equation to apply to the DataFrame that performs the calculation
        necessary to answer the question. Below are some examples of user query, json data, and response triplets. Assume that the pandas DataFrame is called 'financials'.
        Use the name financials to represent that pandas DataFrame in your response. 

        Examples:
        user query: "examine the changes in lmt's debt-to-equity ratio over the last 2 years. what are the implications for its financial stability compared to lmt and air in the aerospace industry?", 'data': ['equity', 'liabilities'], 'answer': "financials['debt_to_equity_ratio'] = financials['liabilities'] / financials['equity']"
        user query: "examine the changes in air's debt-to-equity ratio over the last 2 years. what are the implications for its financial stability compared to competitors in the aerospace industry?", 'data': ['equity', 'liabilities'], 'answer': "financials['debt_to_equity_ratio'] = financials['liabilities'] / financials['equity']"
        user query: "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "financials['revenue_growth'] = financials['revenue'].pct_change(periods=-1).round(2).dropna(); financials.drop(['revenue'], axis=1, inplace=True)"
        user query: "what's WMT's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "financials['revenue_growth'] = financials['revenue'].pct_change(periods=-1).round(2).dropna(); financials.drop(['revenue'], axis=1, inplace=True)"
        user query: "how did msft's net cash provided by operating activities change from q3 2020 to q1 2021?", data: ['net_cash_flow_from_operating_activities'], answer: "financials['net_cash_flow_from_operating_activities_change'] = financials['net_cash_flow_from_operating_activities'].pct_change(periods=-1).round(2).dropna()"        
        user query: "what's amzn's revenue growth during 2023 versus its comps", data: ['revenue'], answer: "financials['revenue_growth'] = financials['revenue'].pct_change(periods=-1).round(2).dropna()"
        user query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", data: ['revenues', 'research_and_development'], answer: "financials['research_and_development_to_revenue_pct'] = financials['research_and_development'] / financials['revenues']"   
        user query: "How has AAPL's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if I bought the stock each time?", data: ['revenue', 'Fiscal Date'], answer: financials['revenue_growth'] = financials['revenue'].pct_change(periods=-1).round(2).dropna()"
        user query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", data: ['revenues', 'research_and_development'], answer: "financials['research_and_development_to_revenue_pct'] = financials['research_and_development'] / financials['revenues']"
        user query: "How has apple’s gross margins trended compared to Intel and Microsoft’s over the past 3 years?", data: ['gross_profit', 'revenues'], answer: "financials['gross_margin_change'] = financials['gross_margin'] = financials['gross_profit'] / financials['revenues'];  financials.drop(['revenues', 'gross_profit'], axis=1, inplace=True)"
        user query: "Examine the changes in Boeing's debt-to-equity ratio over the last five years. What are the implications for its financial stability compared to competitors in the aerospace industry?", data: ['long_term_debt', 'equity'], answer: "financials['debt_to_equity_ratio'] = financials['long_term_debt'] / financials['equity']"
        user query: "Show how amzn's cogs have grown for the past 4 quarters?", data: ['cost_of_revenue'], answer: "financials['cogs_growth'] = financials['cost_of_revenue'].pct_change(periods=-1).round(2).dropna(); financials.drop(['cost_of_revenue'], axis=1, inplace=True)"
        user query: "How has amzn's cogs grown annually quarter over quarter over 2023 vs 2022?", data: ['cost_of_revenue'], answer: "financials['cogs_growth'] = financials['cost_of_revenue'].pct_change(periods=-4).round(2).dropna()"
        user query: "How has amzn's cogs grown annually quarter over quarter over 2023?", data: ['cost_of_revenue'], answer: "financials['cogs_growth'] = financials['cost_of_revenue'].pct_change(periods=-4).round(2).dropna()"
        user query: "how has aapl’s cogs trended compared to intc and msft’s over the past 3 years?", data: ['cost_of_revenue'], answer: "financials['cogs_growth'] = financials['cost_of_revenue'].pct_change(periods=-1).round(2).dropna()"
        user query: "How has apple’s gross margins trended compared to Intel and Microsoft’s over the past 3 years?", data: ['gross_profit', 'revenues'], answer: "financials['gross_margin'] = financials['gross_profit'] / financials['revenues']"
        user query: "Show me the percentage change of aapl's gross margins versus Intel and Microsoft over the past 3 years?", data: ['gross_profit', 'revenues'], answer: "financials['gross_margin'] = financials['gross_profit'] / financials['revenues']; financials['gross_margin_change'] = financials['gross_margin'].pct_change(periods=-1).round(2).dropna()"
        user query: "Calculate the correlation between the % change of aapl’s gross margins and its inventory ratio for 2022.", data: ['gross_profit', 'revenues', 'cost_of_revenue', 'inventory'], answer: "financials['gross_margin'] = financials['gross_profit'] / financials['revenues']; financials['gross_margin_change'] = financials['gross_margin'].pct_change(periods=-1).round(2).dropna(); financials['inventory_ratio'] = financials['inventory'] / financials['revenues']; financials['inventory_ratio_change'] = financials['inventory_ratio'].pct_change(periods=-1).round(2).dropna(); financials['correlation'] = financials['gross_margin_change'].corr(financials['inventory_ratio_change'])
        user query: "whats the variance between the inventory ratio and return on assets growth for first 3 quarters of 2021 for cat?", data: ['inventory', 'totalAssets', 'netIncome', 'Fiscal Date'], answer: "financials['inventory_ratio'] = financials['inventory'] / financials['totalAssets']; financials['return_on_assets'] = financials['netIncome'] / financials['totalAssets']; financials['return_on_assets_growth'] = financials['return_on_assets'].pct_change(periods=-1).round(2).dropna(); financials['variance'] = financials['inventory_ratio'].var() - financials['return_on_assets_growth'].var()"
        user query: "If we exclude the impact of M&A, which segment has dragged down MMM's overall growth in 2022?", data: ['revenue'], answer: financials['revenue_growth'] = financials['revenue'].pct_change(periods=-1).round(2).dropna()"
        user query: "How has AAPL's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if I bought the stock each time?", data: ['revenue', 'Fiscal Date'], answer: financials['revenue_growth'] = financials['revenue'].pct_change(periods=-1).round(2).dropna()"
        user query: "What is ADBE's year-over-year change in unadjusted operating income from FY2015 to FY2016 (in units of percents and round to one decimal place)? Give a solution to the question by using the income statement.", data: ['operatingIncome', 'Calendar Date'], answer: "financials['operating_income_yoy_change'] = financials['operatingIncome'].pct_change(periods=-4).round(3).mul(100)"
        user query: "How much has MSFT spent on research and development as a percentage of R&D for each of the past 5 years? And what products have been launched in that time frame and what has been their contribution to revenue?", data: ['researchAndDevelopmentExpenses', 'revenue', 'Fiscal Date'], answer: "financials['research_and_development_pct'] = financials['researchAndDevelopmentExpenses'] / financials['revenue']"    
        user query: "Calculate AAPL’s profit margin and inventory ratio according to its 10K’s/Q filings. What’s the historical correlation of these values? How has the stock reacted in the month preceding and following earnings when the prior quarter’s correlations were low vs when they were high?", data: ['totalCurrentAssets', 'inventory', 'revenue', 'netIncome', 'costOfRevenue', 'Calendar Date', 'report_date'], answer: "financials['profit_margin'] = financials['netIncome'] / financials['revenue']; financials['inventory_ratio'] = financials['inventory'] / financials['revenue']; financials['correlation'] = financials['profit_margin'].corr(financials['inventory_ratio']);"
        user query: Of the instances in which AAPL referenced its supply chain, how many were positive versus negative? Show me the correlation to revenue growth, data: ['revenue', 'Fiscal Date', 'report_date'], answer: "financials['revenue_growth'] = financials['revenue'].pct_change(periods=-1).round(2).dropna()"
        """

        prompt = """
        user query: {question}
        json data: {financials}
        """

        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.format(question=question, financials=list(financials.columns))}
            ],
        )

        new_response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        if new_response.startswith('```'):
            new_response = new_response.replace('```', '"')

        print(f"user query: {question}, data: {list(financials.columns)}")
        print(f"answer: {new_response}")
        # import pdb; pdb.set_trace()

        exec(new_response)

        if isinstance(financials, pd.Series):
            financials = financials.to_frame()
            # new_financials.dropna(inplace=True)
        elif isinstance(financials, pd.DataFrame):
            pass
        else:
            financials = pd.DataFrame({"result": financials})
        
        # import pdb; pdb.set_trace()
        return financials
    except Exception as e:
        print(f"Error inside do_local_calculate: {e}")
        return financials

"""
query: 
relevant_rows: ['equity', 'liabilities']
relevant_columns: ['Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021', 'Q4 2020', 'Q3 2020', 'Q2 2020', 'Q1 2020', 'Q4 2019', 'Q3 2019', 'Q2 2019', 'Q1 2019']
"""
def get_competitors(question, debug=False):
    system_prompt = """ 
    You are a hedge fund analyst tasked with identifying company competitors. Gien a user query return a JSON list of company competitors. 
    Each competitor in the list should be a string ticker. Your response should only contain the list of tickers that are competitors to
    the company or ticker mentioned in the user query. Do not include ANYTHING other than the JSON list of only the ticker values!
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
        # import pdb; pdb.set_trace()
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        # print(f"response: {response}")
        competitors = ast.literal_eval(response[response.find("["):response.find("]")+1])
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "get_competitors", "inputs": [question], "outputs": [{"competitors": competitors}]}, indent=6))
        return competitors[:3]
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
    'get_market_data': Pulls stock market price data
    'get_final_analysis': Produces a final analysis of the research
    'perform_vector_search': Performs (predominantly) qualitative queries against internal market databses
    'get_market_data': Pull market data for relevant symbol(s)
    'perform_quantitative_vector_search': Performs quantitative queries against internal market databases
    'perform_news_search_via_google': Perform a news search on a company or event
    'run_backtest': Run a backtest on the data

    Use the JSON as shown in the below examples when producing your response. Your response should be a valid JSON object.

    Task Breakdown Examples:

    {
        "query": "Of the instances in which AAPL referenced its supply chain, how many were positive versus negative? Show me the revenue growth and the price performance tied to each instance",
        "tasks": [
            {
                "task": "get_sec_financials",
                "description": "Gather recent financial statements focusing on income statements over the last few quarters or years.",
                "status": "pending
            },
            {
                "task": "perform_quantitative_vector_search",
                "description": "Perform computations atop vector search",
                "status": "pending
            },
            {
                "task": "get_market_data",
                "description": "Retrieve market data",
                "status": "pending
            },
            {
                "task": "run_backtest",
                "description": "Perform a backtest und the supplied parameters",
                "status": "pending"
            },
            {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
            }
        ]
    }
    {
        "query": "Run a backtest on CSCO, buying shares every time they mention macro concerns in their filings.",
        "tasks": [
            {
                "task": "perform_quantitative_vector_search",
                "description": "Perform computations atop vector search",
                "status": "pending
            },
            {
                "task": "get_market_data",
                "description": "Retrieve market data",
                "status": "pending
            },
            {
                "task": "run_backtest",
                "description": "Perform a backtest und the supplied parameters",
                "status": "pending"
            },
            {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
            }
        ]
    }
    {
        "query": "Run a backtest on MSFT, buying shares every time they mention macro concerns in their filings.",
        "tasks": [
            {
                "task": "perform_quantitative_vector_search",
                "description": "Perform computations atop vector search",
                "status": "pending
            },
            {
                "task": "get_market_data",
                "description": "Retrieve market data",
                "status": "pending
            },
            {
                "task": "run_backtest",
                "description": "Perform a backtest und the supplied parameters",
                "status": "pending"
            },
            {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
            }
        ]
    }
    {
        "query": "How much has MSFT spent on research and development as a percentage of R&D for each of the past 5 years? And what products have been launched in that time frame and what has been their contribution to revenue?",
        "tasks": [
            {
                "task": "get_sec_financials",
                "description": "Gather recent financial statements focusing on income statements over the last few quarters or years.",
                "status": "pending"
            },
            {
                "task": "perform_vector_search",
                "description": "Perform a vector search of the internal market database",
                "status": "pending"
            },
            {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
            }
        ]
    }
    {
        "query": What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places. Base your judgments on the information provided primarily in the statement of income and the statement of financial position.,
        "tasks": [
                {
                    "task": "get_sec_financials",
                    "description": "Gather recent financial statements focusing on income statements over the last few quarters or years.",
                    "status": "pending"
                },
                {
                    "task": "get_final_analysis",
                    "description": "Mold the information into a final analysis",
                    "status": "pending"
                }
        ]
    
    }
    {
        "query": "Does 3M maintain a stable trend of dividend distribution?",
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
            }
        ]

    
    }
    {
        "query": "Which debt securities are registered to trade on a national securities exchange under 3M's name as of Q2 of 2023?",
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
            }
        ]
    
    }
    {
        "query": "How has apple's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if i bought the stock each time?",
        "tasks": [
            {
                    "task": "get_sec_financials",
                    "description": "For each identified company, gather recent financial statements, focusing on income statements over the last few quarters or years.",
                    "status": "pending"
            },
            {
                    "task": "perform_quantitative_vector_search",
                    "description": "Perform computations atop vector search",
                    "status": "pending
            },
            {
                "task": "get_market_data",
                "description": "Retrieve market data",
                "status": "pending
            },
            {
                "task": "run_backtest",
                "description": "Perform a backtest under the supplied parameters",
                "status": "pending"
            },
            {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
            }
        ]
    
    }
    {
        "query": "How has apple's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if i bought the stock each time?",
        "tasks": [
            {
                "task": "get_sec_financials",
                "description": "For each identified company, gather recent financial statements, focusing on income statements over the last few quarters or years.",
                "status": "pending"
            },
            {
                "task": "perform_quantitative_vector_search",
                "description": "Perform computations atop vector search",
                "status": "pending
            },
            {
                "task": "get_market_data",
                "description": "Retrieve market data",
                "status": "pending
            },
            {
                "task": "run_backtest",
                "description": "Perform a backtest under the supplied parameters",
                "status": "pending"
            },
            {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
            }
        ]
    
    }
    {
        "query": "what's the status of AAPL’s anti-trust cases since 2022? How has the stock performed since then?",
        "tasks": [
            {
                "task": "perform_vector_search",
                "description": "Perform a vector search of the internal market database",
                "status": "pending"
            },
            {
                "task": "get_market_data",
                "description": "Retrieve market data",
                "status": "pending
            },
            {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
            }
        ]
    }
    {
        "query": What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places. Base your judgments on the information provided primarily in the statement of income and the statement of financial position.,
        "tasks": [
                {
                    "task": "get_sec_financials",
                    "description": "Gather recent financial statements focusing on income statements over the last few quarters or years.",
                    "status": "pending"
                },
                {
                    "task": "get_final_analysis",
                    "description": "Mold the information into a final analysis",
                    "status": "pending"
                }
            ]
    
    }
    {
        "query": "what's the status of AAPL’s anti-trust cases since 2022? How has the stock performed since then?",
        "tasks": [
            {
                "task": "perform_vector_search",
                "description": "Perform a vector search of the internal market database",
                "status": "pending"
            },
            {
                "task": "get_market_data",
                "description": "Retrieve market data",
                "status": "pending
            },
            {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
            }
        ]
    }
    {
        "query": "Calculate apple’s profit margin and inventory ratio according to its 10K’s/Q filings. What’s the historical correlation of these values? How has the stock reacted in the month preceding and following earnings when the prior quarter’s correlations were low vs when they were high?",
        "tasks": [
                {
                    "task": "get_sec_financials",
                    "description": "For each identified company, gather recent financial statements, focusing on income statements over the last few quarters or years.",
                    "status": "pending"
                },
                {
                    "task": "get_market_data",
                    "description": "Retrieve market data",
                    "status": "pending
                },
                {
                    "task": "get_final_analysis",
                    "description": "Mold the information into a final analysis",
                    "status": "pending"
                }
            ]
    }
    {
        "query": "Calculate apple’s profit margin and inventory ratio according to its 10K’s/Q filings. What’s the historical correlation of these values? How has the stock reacted in the month preceding and following earnings when the prior quarter’s correlations were low vs when they were high?",
        "tasks": [
                {
                    "task": "get_sec_financials",
                    "description": "For each identified company, gather recent financial statements, focusing on income statements over the last few quarters or years.",
                    "status": "pending"
                },
                {
                    "task": "get_market_data",
                    "description": "Retrieve market data",
                    "status": "pending
                },
                {
                    "task": "get_final_analysis",
                    "description": "Mold the information into a final analysis",
                    "status": "pending"
                }
            ]
    }
    {
        "query": "why did walmart acquire vizio?",
        "tasks": [
            {
                "task": "perform_news_search_via_google",
                "description": "Search through relevant news & articles",
                "status": "pending"
            },
            {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
            }
        ]
    }

        {
            "query": "What’s apple’s revenue growth for the past 2 years? Compute the correlation between revenue growth and its margin growth. Compare it to msft who had the higher average gross margin?",
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
            "query": "What’s apple’s revenue growth for the past 2 years? Compute the correlation between revenue growth and its margin growth. Compare it to msft who had the higher gross margin?",
            "tasks": [
                {
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
            "query": "how many times did MSFT mention the activision anti-trust case in their filings since 2021? Would I have made money if I bought the stock after those filings?",
            "tasks": [
                {
                    "task": "perform_quantitative_vector_search",
                    "description": "Perform computations atop vector search",
                    "status": "pending
                },
                {
                    "task": "get_market_data",
                    "description": "Retrieve market data",
                    "status": "pending
                },
                {
                    "task": "run_backtest",
                    "description": "Perform a backtest under the supplied parameters",
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
            "query": "how many times did msft mention artificial intelligence in their filings",
            "tasks": [
                {
                "task": "perform_quantitative_vector_search",
                "description": "Perform computations atop vector search",
                "status": "pending"
                },
                {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
                },
            ]

        }
        {
            "query": "how many times has CAT raised international growth concerns in their filings over the past 2 years?",
            "tasks": [
                {
                "task": "perform_quantitative_vector_search",
                "description": "Perform computations atop vector search",
                "status": "pending"
                },
                {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
                },
            ]
        }
        {
            "query": "how many times did msft raise concerns about supply chain issues in their filings in the last 3 years?",
            "tasks": [
                {
                "task": "perform_quantitative_vector_search",
                "description": "Perform computations atop vector search",
                "status": "pending"
                },
                {
                "task": "get_final_analysis",
                "description": "Mold the information into a final analysis",
                "status": "pending"
                },
            ]
        }
        {
            "query": "investigate the trends in JP Morgan's net interest margin and loan growth over the past five years. How have changes in interest rates and economic conditions influenced their profitability?",
            "tasks": [
                {
                "task": "perform_vector_search",
                "description": "Perform a vector search of the internal market database",
                "status": "pending"
                },
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
        },
        {
            "query": "pull up a daily chart for aapl from 2023-01-09 to 2023-02-10",
            "tasks": [
                {
                "task": "build_market_data_chart",
                "description": "Build market data chart",
                "status": "pending"
                },
                
            ]
        },
        {
            "query": "pull up a daily chart for aapl from 2023-01-09 to 2023-02-10",
            "tasks": [
                {
                "task": "build_market_data_chart",
                "description": "Build market data chart",
                "status": "pending"
                },
                
            ]
        }
        {
            "query": "how many times did aapl mention macro concerns in their filings since 2023? What has been the affect on stock performance?",
            "tasks": [
                {
                    "task": "perform_quantitative_vector_search",
                    "description": "Perform computations atop vector search to determine how many times AAPL mentioned macro concerns in their filings since 2023.",
                    "status": "pending"
                },
                {
                    "task": "get_market_data",
                    "description": "Retrieve AAPL stock market data from 2023 to analyze stock performance.",
                    "status": "pending"
                },
                {
                    "task": "get_final_analysis",
                    "description": "Produce a final analysis",
                    "status": "pending"
                }
            ]
        }
         {
            "query": "how many times did aapl mention macro concerns in their filings since 2023? What has been the affect on stock performance?",
            "tasks": [
                {
                    "task": "perform_quantitative_vector_search",
                    "description": "Perform computations atop vector search to determine how many times AAPL mentioned macro concerns in their filings since 2023.",
                    "status": "pending"
                },
                {
                    "task": "get_market_data",
                    "description": "Retrieve AAPL stock market data from 2023 to analyze stock performance.",
                    "status": "pending"
                },
                {
                    "task": "get_final_analysis",
                    "description": "Produce a final analysis",
                    "status": "pending"
                }
            ]
        }
        {
            "query": "get me recent news for amzn",
            "tasks": [
                {
                "task": "get_news_by_symbol",
                "description": "Get news by symbol",
                "status": "pending"
                },
            ]
        },
        {
            "query": "Compare Amazon's logistics and fulfillment expenses as a percentage of net sales over the past 2 years. How have changes in these expenses impacted Amazon's operating margin, and what strategies have been implemented to optimize their supply chain efficiency?",
            "tasks": [
                {
                "task": "get_sec_financials",
                "description": "Tie the retrieved financial information into a comprehensive final analysis.",
                "status": "pending"
                },
                {
                "task": "perform_vector_search",
                "description": "Perform a vector search of the internal market database",
                "status": "pending"
                },
            ]
        },
        {
            "query": "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?",
            "tasks": [
                {
                "task": "get_sec_financials",
                "description": "Tie the retrieved financial information into a comprehensive final analysis.",
                "status": "pending"
                },
                {
                "task": "perform_vector_search",
                "description": "Perform a vector search of the internal market database",
                "status": "pending"
                },
            ]
        },
        {
            "query": "Examine the changes in Boeing's debt-to-equity ratio over the last five years. What are the implications for its financial stability compared to competitors in the aerospace industry?".
            "tasks": [
                {
                    "task": "get_competitors",
                    "description": "Find company competitors to run a comprehensive comprative analysis",
                    "status": "pending"
                },
                {
                "task": "get_sec_financials",
                "description": "Tie the retrieved financial information into a comprehensive final analysis.",
                "status": "pending"
                }
            ]
        },
        
        {
            "query": "get me latest legal headlines for AAPL",
            "tasks": [
                {
                "task": "get_news_by_symbol",
                "description": "Get news by symbol",
                "status": "pending"
                },
            ]
        },
        {
            "query": "run a backtest on my tech basket",
            "tasks": [
                {
                "task": "run_backtest",
                "description": "Perform a backtest und the supplied parameters",
                "status": "pending"
                },
            ]
        },
        {
            "query": "run a backtest on my tech basket whereby I buy the stock every time the management team guides up in the earnings release. would i have made money?",
            "tasks": [
                {
                "task": "run_backtest",
                "description": "Perform a backtest under the supplied parameters",
                "status": "pending"
                },
            ]
        }
        {
            "query": "Calculate apple’s profit margin and inventory ratio according to its 10K’s/Q filings. What’s the historical correlation of these values? How has the stock reacted in the month preceding and following earnings when the prior quarter’s correlations were low vs when they were high?",
            "tasks": [
                    {
                        "task": "get_sec_financials",
                        "description": "For each identified company, gather recent financial statements, focusing on income statements over the last few quarters or years.",
                        "status": "pending"
                    },
                    {
                        "task": "get_market_data",
                        "description": "Retrieve market data",
                        "status": "pending
                    },
                    {
                        "task": "get_final_analysis",
                        "description": "Mold the information into a final analysis",
                        "status": "pending"
                    }
                ]
        }
        {
            "query": "Calculate apple’s profit margin and inventory ratio according to its 10K’s/Q filings. What’s the historical correlation of these values? How has the stock reacted in the month preceding and following earnings when the prior quarter’s correlations were low vs when they were high?",
            "tasks": [
                    {
                        "task": "get_sec_financials",
                        "description": "For each identified company, gather recent financial statements, focusing on income statements over the last few quarters or years.",
                        "status": "pending"
                    },
                    {
                        "task": "get_market_data",
                        "description": "Retrieve market data",
                        "status": "pending
                    },
                    {
                        "task": "get_final_analysis",
                        "description": "Mold the information into a final analysis",
                        "status": "pending"
                    }
                ]
        }
    """
    prompt = """
        user query: {question}
    """


    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt.format(question=question)}
        ]
    )

    
    try:
        # import pdb; pdb.set_trace()
        content = json.loads(response.to_json())['choices'][0]['message']['content']
        if "json" in content:
            research_plan = json.loads(content[content.find("```json\n")+len("```json\n"):len(content)-3])["tasks"]
        else:
            research_plan = json.loads(content)["tasks"]

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
    query: "whats the variance between the inventory ratio and return on assets growth for first 3 quarters of 2021?", [('ticker', 'AAPL'), ('from_date', '2021-01-01'), ('to_date', '2021-09-30')]
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
        import pdb; pdb.set_trace()
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


# timespan can be minute, hour, day, week, month
# def get_market_data(question, results, debug=False):
#     # symbol, multiplier=1, timespan="day", from_date="", to_date="",
#     if False:
#         pass
#     else:
#         entities = extract_entities_from_user_query(question)
#         params = {}
#         from_date = "2020-01-01"
#         # import pdb; pdb.set_trace()
#         now = (datetime.now() + timedelta(days=-1)).date()
#         to_date = f"{now.year}-{now.month}-{now.day}"
#         tickers = []
#         for entity in entities:
#             if entity["entity"] == "from_date":
#                 from_date = entity["value"]
#             elif entity["entity"] == "to_date":
#                 to_date = entity["value"]
#             elif entity["entity"] == "ticker":
#                 tickers.append(entity["value"].upper())

#         to_date = (parser.parse(to_date) + timedelta(days=30)).date()
#         if to_date > now:
#             to_date = now

#     # import pdb; pdb.set_trace()
#     try:
#         for ticker in tickers:
#             # base_url = f"https://api.polygon.io/v2/aggs/ticker/{[','.join(tickers)]}/range/1/day/{from_date}/{to_date}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
#             base_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"

#             response = requests.get(base_url)
#             # import pdb; pdb.set_trace()
#             # import pdb; pdb.set_trace()
#             if response.status_code == 200:
#                 data = response.json()["results"]
#                 df = pd.DataFrame.from_records(data)
#                 # import pdb; pdb.set_trace()
#                 # df['t'] = pd.to_datetime(df['t'], unit='ns')
                
#                 df['report_date'] = pd.bdate_range(start=from_date, end=to_date)[:len(df)].strftime("%Y-%m-%d")
#                 df['report_date'] = pd.to_datetime(df['report_date'])
#                 df.sort_values(by=['report_date'], inplace=True)
                
                
#                 # df['ToDate'] = df['report_date'] + pd.offsets.BusinessDay(n=30)
#                 # df['ToDate'] = df['ToDate'].dt.strftime('%Y-%m-%d')
#                 # import pdb; pdb.set_trace()
#                 df['60 Day Performance'] = df['c'].pct_change(-60)
#                 df['30 Day Performance'] = df['c'].pct_change(-30)
#                 df['report_date'] = df['report_date'].dt.strftime('%Y-%m-%d')

#                 df_qual_and_quant = None
#                 if ticker in results['QualAndQuant']:
#                     df_qual_and_quant = results['QualAndQuant'][ticker] #[v for v in results['QualAndQuant'].values()][0]
#                     df_qual_and_quant['report_date'] = pd.to_datetime(df_qual_and_quant['report_date']).dt.strftime('%Y-%m-%d')
#                     df_qual_and_quant = df_qual_and_quant.merge(df, left_on='report_date', right_on='report_date')
#                     results["Context"].append(
#                         {f"{question} (ticker={ticker})" : df_qual_and_quant}
#                     )
#                     df_qual_and_quant.set_index("report_date", inplace=True)
#                     results["finalAnalysis"]["tables"][ticker.upper()] = df_qual_and_quant
#                     results['MarketData'][ticker] = df_qual_and_quant
                
#                 elif ticker in results['GetCompanyFinancials']:
#                     df_company_financials = results['GetCompanyFinancials'][ticker].reset_index() #[v for v in results['QualAndQuant'].values()][0]
#                     import pdb; pdb.set_trace()
#                     df_company_financials['report_date'] = pd.to_datetime(df_company_financials['report_date']).dt.strftime('%Y-%m-%d')
                    
#                     df_company_financials = df_company_financials.merge(df, left_on='report_date', right_on='report_date')
#                     results["Context"].append(
#                         {f"{question} (ticker={ticker})" : df_company_financials}
#                     )
#                     df_company_financials.set_index("report_date", inplace=True)
#                     results["finalAnalysis"]["tables"][ticker.upper()] = df_company_financials
#                     results['MarketData'][ticker] = df_company_financials


#         return results
#     except Exception as e:
#         print(f"Error in get_market_data {e}")
#         return results



def do_calculate_for_get_market_data(question, df_market_data):
    try:
        system_prompt = """ 
        You are a tool used in a rag system to determine an equation that needs to be applied to a pandas dataframe. Your job is to generate 
        the string representation of the pandas calculation such that the python eval function can be called on the string.
        Your response should not contain anything other than the pandas equation to be called. Including anything other than just the equation 
        will cause the functin to fail. 
        
        Given a user query and the supplied json that represents the pandas DataFrame, return the equation to apply to the DataFrame that performs the calculation
        necessary to answer the question. Below are some examples of user query, json data, and response triplets. Assume that the pandas DataFrame is called 'df_market_data'.
        Use the name df_market_data to represent that pandas DataFrame in your response. The 'data' field contains the list of starting columns in the provided dataframe 'df_market_data'.
        Also, it is important to note that 'df_market_data' is order in terms of most recent dates first. Take this into account when calculating row-wise metrics such as price change etc. (see the examples below).

        Examples:
        'user query': "Compare F's revenue growth since 2020 relative to its stock? Which stock has appreciated more and by how much?", 'data': ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 'answer': "df_market_data['Stock Returns'] = df_market_data['Close'].pct_change(periods=-1)[::-1].cumsum()[::-1].round(2)"
        'user query': "Compare TSLA's revenue growth since 2020 relative to its stock? Which stock has appreciated more and by how much?", 'data': ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 'answer': "df_market_data['Stock Returns'] = df_market_data['Close'].pct_change(periods=-1)[::-1].cumsum()[::-1].round(2)"
        'user query': "Compare F's revenue growth since 2020 relative to its stock? Which stock has appreciated more and by how much?", data: ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 'answer': "df_market_data['Stock Returns'] = df_market_data['Close'].pct_change(periods=-1)[::-1].cumsum()[::-1].round(2)"
        'user query': "Compare TSLA's revenue growth since 2020 relative to its stock? Which stock has appreciated more and by how much?", data: ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 'answer': "df_market_data['Stock Returns'] = df_market_data['Close'].pct_change(periods=-1)[::-1].cumsum()[::-1].round(2)"        
        'user query': "Calculate apple’s profit margin and inventory ratio according to its 10K’s/Q filings. What’s the historical correlation of these values? How has the stock reacted in the month preceding and following earnings when the prior quarter’s correlations were low vs when they were high?", data: ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 'answer': "df_market_data['1 month after'] = df_market_data['Close'].pct_change(periods=20).round(2); df_market_data['1 month prior'] = df_market_data['Close'].pct_change(periods=-20).round(2);" 
        'user query': "Calculate apple’s profit margin and inventory ratio according to its 10K’s/Q filings. What’s the historical correlation of these values? How has the stock reacted in the month preceding and following earnings when the prior quarter’s correlations were low vs when they were high?", data: ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 'answer': "df_market_data['1 month after'] = df_market_data['Close'].pct_change(periods=20).round(2); df_market_data['1 month prior'] = df_market_data['Close'].pct_change(periods=-20).round(2);"
        'user query': "How has apple’s stock performed post 2021 in the following quarters after management has referenced macro concerns?", 'data': ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 'answer': "df_market_data['Stock Returns'] = df_market_data['Close'].pct_change(periods=-1)[::-1].cumsum()[::-1].round(2)"
        'user query': How many times since 2020 has MSFT mentioned global macro concerns? Would I have made money if I bought the stock at these times, 'data': ['report_date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 'answer': "df_market_data['Returns'] = df_market_data['Close'].pct_change(periods=-1)[::-1].cumsum()[::-1].round(2)
        'user query': "Calculate AAPL’s profit margin and inventory ratio according to its 10K’s/Q filings. What’s the historical correlation of these values? How has the stock reacted in the month preceding and following earnings when the prior quarter’s correlations were low vs when they were high?", 'data': ['Calendar Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 'answer': "df_market_data['1 month after'] = df_market_data['Close'].pct_change(periods=20).round(2); df_market_data['1 month prior'] = df_market_data['Close'].pct_change(periods=-20).round(2)"
        'user query': "Of the instances in which AAPL referenced its supply chain, how many were positive versus negative? Show me the revenue growth and the price performance tied to each instance", 'data': ['report_date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 'answer': "df_market_data['Price Performance'] = df_market_data['Close'].pct_change(periods=-1)[::-1].cumsum()[::-1].round(2)"
        'user query': Of the instances in which AAPL referenced its supply chain, how many were positive versus negative? Show me the revenue growth and the price performance tied to each instance, 'data': ['report_date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], df_market_data['Price Performance'] = df_market_data['Close'].pct_change(periods=-1)[::-1].cumsum()[::-1].round(2)
        """

        prompt = """
        user query: {question}
        data: {data_columns}
        """

        data_columns = list(df_market_data.columns)
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.format(question=question, data_columns=data_columns)}
            ],
        )

        new_response = json.loads(response.to_json())["choices"][0]["message"]["content"]

        print(f"'user query': {question}, 'data': {list(df_market_data.columns)}")
        print(f"'answer': {new_response}")
        # import pdb; pdb.set_trace();

        exec(new_response)

        if isinstance(df_market_data, pd.Series):
            df_market_data = df_market_data.to_frame()
            # new_financials.dropna(inplace=True)
        elif isinstance(df_market_data, pd.DataFrame):
            pass
        else:
            df_market_data = pd.DataFrame({"result": df_market_data})
        
        # import pdb; pdb.set_trace()
        return df_market_data
    except Exception as e:
        print(f"Error inside do_calculate_for_get_market_data: {e}")
        return df_market_data


def realign_market_df_to_closest_dt_from_upstream_df(df, company_financials_df, merge_key):
    temp_company_financials_df = company_financials_df.copy()
    temp_df = df.copy()
    df_closest_date_index_to_other_index = []
    actual_date_from_other_index = []
    temp_company_financials_df["RefDate"] = pd.to_datetime(temp_company_financials_df[merge_key])
    temp_df["RefDate"] = pd.to_datetime(temp_df[merge_key])
    for date_ref in temp_company_financials_df["RefDate"].values:
        df_closest_date_index_to_other_index.append(temp_df["RefDate"][(temp_df["RefDate"] - date_ref).abs().idxmin()].strftime('%Y-%m-%d'))
        actual_date_from_other_index.append(str(date_ref).split('T')[0])

    # import pdb; pdb.set_trace()
    df = df[df[merge_key].isin(df_closest_date_index_to_other_index)] 
    df[merge_key] = list(set(actual_date_from_other_index))
    
    return df, company_financials_df


def should_calculate_for_get_market_data(question, df_columns):
    try:
        system_prompt = """ 
        You are a tool used to guide a RAG system. Your job is to determine whether a calculator function needs to be called or not.
        Your response should be a string that is either 'True' or 'False'. Your response should not contain anything other than 
        'True' or 'False'. Given a user query and a list of columns names corresponding to available data used to answer the user's question,
        determine wheter a calculation needs to be done or if the available data is sufficient to answer the quesiton already and thus no further calculations
        are requird. Below are some examples of user query, available colummns, and response triplets.

        Examples:    
        user query: "What's TSLA's revenue growth and stock growth since 2020?", data: ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], answer: "True"
        user query: "What's F's revenue growth and stock growth since 2020?", data: ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], answer: "True"
        user query: "What's TSLA's revenue growth and stock growth since 2020?", data: ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], answer: "True"
        """

        prompt = """
        user query: {question} data: {df_columns}
        """
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.format(question=question, df_columns=df_columns)}
            ],   
        )


        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        calculation_required = ast.literal_eval(response)
        print(f"response from should_calculate_for_get_market_data: {response}")
        return calculation_required
    except Exception as e:
        print(f"Error inside should_calculate_for_get_market_data: {e}")
        return False




def get_market_data(question, results, debug=False):
    try:
        # symbol, multiplier=1, timespan="day", from_date="", to_date="",
        # import pdb; pdb.set_trace()
        entities = extract_entities_from_user_query(question)
        ticker = [ent["value"] for ent in entities if ent["entity"] == "ticker"][0]
        solo_call = False
        if len(results["RunBackTest"]) > 0:
            if ticker in results["RunBackTest"]:
                backtest_df = results["RunBackTest"][ticker]
                merge_key = None
                if 'Q' in backtest_df['report_date'][0]:
                    merge_key = 'Calendar Date'
                else:
                    merge_key = f'report_date'

                backtest_df[merge_key] = pd.to_datetime(backtest_df[merge_key]).dt.strftime('%Y-%m-%d')
                to_date = backtest_df[merge_key][0]
                from_date = backtest_df[merge_key][-1]
                df = yahooFinance.Ticker(ticker).history(period="max")
                if len(df) > 0:
                    df.reset_index(inplace=True)
                    df.rename({'Date': merge_key}, axis=1, inplace=True)
                    df[merge_key] = df[merge_key].dt.strftime("%Y-%m-%d")
                    df = df[::-1]
                    df['60 Day Performance'] = df['Close'].pct_change(-60)
                    df['30 Day Performance'] = df['Close'].pct_change(-30)
                    df = df[df[merge_key].isin(backtest_df[merge_key])]
                    df = df[[c for c in df.columns if c not in ['Open', 'High', 'Low', 'Dividends', 'Stock Splits']]]
                    backtest_df = backtest_df.merge(df, left_on=merge_key, right_on=merge_key)
                    # backtest_df.set_index("report_date", inplace=True)
                    results["finalAnalysis"]["tables"][ticker.upper()] = backtest_df
                    results['MarketData'][ticker] = backtest_df
                    results["Context"].append(
                        {f"{question} (ticker={ticker})" : backtest_df}
                    )
                    del results["RunBackTest"][ticker]
            else:
                solo_call = True
        elif len(results["GetNews"]) > 0:
            if ticker.upper() in results["GetNews"]:
                news_df = results["GetNews"][ticker.upper()] 
                # company_financials_df.set_index(pd.RangeIndex(len(company_financials_df)), inplace=True)
                merge_key = None
                if 'Q' in news_df['report_date'][0]:
                    merge_key = 'Calendar Date'
                else:
                    merge_key = f'report_date'


                news_df[merge_key] = pd.to_datetime(news_df[merge_key]).dt.strftime('%Y-%m-%d')
                to_date = news_df[merge_key][0]
                from_date = news_df[merge_key][len(news_df)-1]
                df = yahooFinance.Ticker(ticker).history(period="max")
                import pdb; pdb.set_trace()
                if len(df) > 0:
                    df.reset_index(inplace=True)
                    df.rename({'Date': merge_key}, axis=1, inplace=True)
                    df[merge_key] = df[merge_key].dt.strftime("%Y-%m-%d")
                    df = df[::-1]
                    all_market_data_columns = list(df.columns)
                    calculation_required = should_calculate_for_get_market_data(question, all_market_data_columns)
                    print(f"calculation_required: {calculation_required}")
                    if calculation_required:
                        df = do_calculate_for_get_market_data(question, df)
                        
                    
                    backtest_df = df.copy()
                    backtest_news_df = news_df.copy()
                    # NOTE: Add in highest high (lowest low) over 1 QTR & 2 QTR
                    backtest_df['1 QTR'] = backtest_df['Close'].pct_change(-90)
                    # backtest_df['2 QTR'] = backtest_df['Close'].pct_change(-180)
                    backtest_df, backtest_news_df = realign_market_df_to_closest_dt_from_upstream_df(backtest_df, backtest_news_df, merge_key)
                    backtest_df = backtest_df[[c for c in backtest_df.columns if c not in ['Volume', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits']]]
                    backtest_news_df = backtest_news_df.merge(backtest_df, left_on=merge_key, right_on=merge_key)
                    results['MarketDataForBacktest'][ticker.upper()] = backtest_news_df


                    df, news_df = realign_market_df_to_closest_dt_from_upstream_df(df, news_df, merge_key)
                    df = df[[c for c in df.columns if c not in ['Volume', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits']]]
                    news_df = news_df.merge(df, left_on=merge_key, right_on=merge_key)
                    results["Context"].append(
                        {f"{question} (ticker={ticker})" : news_df}
                    )
                    results["finalAnalysis"]["tables"][ticker.upper()] = news_df
                    results['MarketData'][ticker.upper()] = news_df
                    del results["GetNews"][ticker]
            else:
                solo_call = True
        elif len(results["QualAndQuant"]) > 0:
            if ticker.upper() in results["QualAndQuant"]:
                qual_and_quant_vector_search_df = results["QualAndQuant"][ticker.upper()] 
                # company_financials_df.set_index(pd.RangeIndex(len(company_financials_df)), inplace=True)
                merge_key = None
                if 'Q' in qual_and_quant_vector_search_df['report_date'][0]:
                    merge_key = 'Calendar Date'
                else:
                    merge_key = f'report_date'

                # import pdb; pdb.set_trace()
                qual_and_quant_vector_search_df[merge_key] = pd.to_datetime(qual_and_quant_vector_search_df[merge_key]).dt.strftime('%Y-%m-%d')
                to_date = qual_and_quant_vector_search_df[merge_key][0]
                from_date = qual_and_quant_vector_search_df[merge_key][len(qual_and_quant_vector_search_df)-1]
                df = yahooFinance.Ticker(ticker).history(period="max")
                
                if len(df) > 0:
                    df.reset_index(inplace=True)
                    df.rename({'Date': merge_key}, axis=1, inplace=True)
                    df[merge_key] = df[merge_key].dt.strftime("%Y-%m-%d")
                    df = df[::-1]
                    all_market_data_columns = list(df.columns)
                    calculation_required = should_calculate_for_get_market_data(question, all_market_data_columns)
                    print(f"calculation_required: {calculation_required}")
                    if calculation_required:
                        df = do_calculate_for_get_market_data(question, df)

                    backtest_df = df.copy()
                    backtest_qual_and_quant_vector_search_df = qual_and_quant_vector_search_df.copy()
                    # NOTE: Add in highest high (lowest low) over 1 QTR & 2 QTR
                    backtest_df['1 QTR'] = backtest_df['Close'].pct_change(-90)
                    # backtest_df['2 QTR'] = backtest_df['Close'].pct_change(-180)
                    backtest_df, backtest_qual_and_quant_vector_search_df = realign_market_df_to_closest_dt_from_upstream_df(backtest_df, backtest_qual_and_quant_vector_search_df, merge_key)
                    backtest_df = backtest_df[[c for c in backtest_df.columns if c not in ['Volume', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits']]]
                    backtest_qual_and_quant_vector_search_df = backtest_qual_and_quant_vector_search_df.merge(backtest_df, left_on=merge_key, right_on=merge_key)
                    backtest_qual_and_quant_vector_search_df.drop_duplicates(inplace=True)
                    results['MarketDataForBacktest'][ticker.upper()] = backtest_qual_and_quant_vector_search_df
                        
                    df, qual_and_quant_vector_search_df = realign_market_df_to_closest_dt_from_upstream_df(df, qual_and_quant_vector_search_df, merge_key)
                    df = df[[c for c in df.columns if c not in ['Volume', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits']]]
                    qual_and_quant_vector_search_df = qual_and_quant_vector_search_df.merge(df, left_on=merge_key, right_on=merge_key)
                    qual_and_quant_vector_search_df.drop_duplicates(inplace=True)
                    results["Context"].append(
                        {f"{question} (ticker={ticker})" : qual_and_quant_vector_search_df}
                    )
                    results["finalAnalysis"]["tables"][ticker.upper()] = qual_and_quant_vector_search_df
                    results['MarketData'][ticker.upper()] = qual_and_quant_vector_search_df
                    del results["QualAndQuant"][ticker]
            else:
                solo_call = True
        elif len(results["VectorSearch"]) > 0:
            if ticker.upper() in results["VectorSearch"]:
                vector_search_df = results["VectorSearch"][ticker.upper()] 
                # company_financials_df.set_index(pd.RangeIndex(len(company_financials_df)), inplace=True)
                merge_key = None
                if 'Q' in vector_search_df['report_date'][0]:
                    merge_key = 'Calendar Date'
                else:
                    merge_key = f'report_date'


                vector_search_df[merge_key] = pd.to_datetime(vector_search_df[merge_key]).dt.strftime('%Y-%m-%d')
                to_date = vector_search_df[merge_key][0]
                from_date = vector_search_df[merge_key][len(vector_search_df)-1]
                df = yahooFinance.Ticker(ticker).history(period="max")
                import pdb; pdb.set_trace()
                if len(df) > 0:
                    df.reset_index(inplace=True)
                    df.rename({'Date': merge_key}, axis=1, inplace=True)
                    df[merge_key] = df[merge_key].dt.strftime("%Y-%m-%d")
                    df = df[::-1]
                    all_market_data_columns = list(df.columns)
                    calculation_required = should_calculate_for_get_market_data(question, all_market_data_columns)
                    print(f"calculation_required: {calculation_required}")
                    if calculation_required:
                        df = do_calculate_for_get_market_data(question, df)

                    backtest_df = df.copy()
                    backtest_vector_search_df = vector_search_df.copy()
                    # NOTE: Add in highest high (lowest low) over 1 QTR & 2 QTR
                    backtest_df['1 QTR'] = backtest_df['Close'].pct_change(-90)
                    # backtest_df['2 QTR'] = backtest_df['Close'].pct_change(-180)
                    backtest_df, backtest_vector_search_df = realign_market_df_to_closest_dt_from_upstream_df(backtest_df, backtest_vector_search_df, merge_key)
                    backtest_df = backtest_df[[c for c in backtest_df.columns if c not in ['Volume', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits']]]
                    backtest_vector_search_df = backtest_vector_search_df.merge(backtest_df, left_on=merge_key, right_on=merge_key)
                    backtest_vector_search_df.drop_duplicates(inplace=True)
                    results['MarketDataForBacktest'][ticker.upper()] = backtest_vector_search_df
                        
                    df, vector_search_df = realign_market_df_to_closest_dt_from_upstream_df(df, vector_search_df, merge_key)
                    df = df[[c for c in df.columns if c not in ['Volume', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits']]]
                    vector_search_df = vector_search_df.merge(df, left_on=merge_key, right_on=merge_key)
                    vector_search_df.drop_duplicates(inplace=True)
                    results["Context"].append(
                        {f"{question} (ticker={ticker})" : vector_search_df}
                    )
                    results["finalAnalysis"]["tables"][ticker.upper()] = vector_search_df
                    results['MarketData'][ticker.upper()] = vector_search_df
                    del results["VectorSearch"][ticker]
            else:
                solo_call = True
        elif len(results["GetEstimates"]) > 0:
            if ticker in results["GetEstimates"]:
                estimates_df = results["GetEstimates"][ticker] 
                merge_key = None
                if 'Q' in estimates_df['report_date'][0]:
                    merge_key = 'Calendar Date'
                else:
                    merge_key = f'report_date'
                estimates_df[merge_key] = pd.to_datetime(estimates_df[merge_key]).dt.strftime('%Y-%m-%d')
                to_date = estimates_df[merge_key][0]
                from_date = estimates_df[merge_key][-1]
                df = yahooFinance.Ticker(ticker).history(period="max")
                if len(df) > 0:
                    df.reset_index(inplace=True)
                    df.rename({'Date': merge_key}, axis=1, inplace=True)
                    df[merge_key] = df[merge_key].dt.strftime("%Y-%m-%d")
                    df = df[::-1]
                    df['60 Day Performance'] = df['Close'].pct_change(-60)
                    df['30 Day Performance'] = df['Close'].pct_change(-30)
                    df = df[df[merge_key].isin(estimates_df[merge_key])]
                    df = df[[c for c in df.columns if c not in ['Open', 'High', 'Low', 'Dividends', 'Stock Splits']]]
                    estimates_df = estimates_df.merge(df, left_on=merge_key, right_on=merge_key)
                    # estimates_df.set_index("report_date", inplace=True)
                    results["Context"].append(
                        {f"{question} (ticker={ticker})" : estimates_df}
                    )
                    
                    results["finalAnalysis"]["tables"][ticker.upper()] = estimates_df
                    results['MarketData'][ticker.upper()] = estimates_df
                    del results["GetEstimates"][ticker]
            else:
                solo_call = True
        elif len(results["GetCompanyFinancials"]) > 0:
            if ticker.upper() in results["GetCompanyFinancials"]:
                # import pdb; pdb.set_trace()
                company_financials_df = results["GetCompanyFinancials"][ticker.upper()] 
                # company_financials_df.set_index(pd.RangeIndex(len(company_financials_df)), inplace=True)
                merge_key = None
                if 'Q' in company_financials_df['report_date'][0]:
                    merge_key = 'Calendar Date'
                else:
                    merge_key = f'report_date'


                company_financials_df[merge_key] = pd.to_datetime(company_financials_df[merge_key]).dt.strftime('%Y-%m-%d')
                to_date = company_financials_df[merge_key][0]
                from_date = company_financials_df[merge_key][len(company_financials_df)-1]
                df = yahooFinance.Ticker(ticker).history(period="max")
                
                if len(df) > 0:
                    df.reset_index(inplace=True)
                    df.rename({'Date': merge_key}, axis=1, inplace=True)
                    df[merge_key] = df[merge_key].dt.strftime("%Y-%m-%d")
                    df = df[::-1]
                    all_market_data_columns = list(df.columns)
                    calculation_required = should_calculate_for_get_market_data(question, all_market_data_columns)
                    print(f"calculation_required: {calculation_required}")
                    if calculation_required:
                        df = do_calculate_for_get_market_data(question, df)


                    backtest_df = df.copy()
                    backtest_company_financials_df = company_financials_df.copy()
                    # NOTE: Add in highest high (lowest low) over 1 QTR & 2 QTR
                    backtest_df['1 QTR'] = backtest_df['Close'].pct_change(-90)
                    # backtest_df['2 QTR'] = backtest_df['Close'].pct_change(-180)
                    backtest_df, backtest_company_financials_df = realign_market_df_to_closest_dt_from_upstream_df(backtest_df, backtest_company_financials_df, merge_key)
                    backtest_df = backtest_df[[c for c in backtest_df.columns if c not in ['Volume', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits']]]
                    backtest_company_financials_df = backtest_company_financials_df.merge(backtest_df, left_on=merge_key, right_on=merge_key)
                    backtest_company_financials_df.drop_duplicates(inplace=True)
                    results['MarketDataForBacktest'][ticker.upper()] = backtest_company_financials_df
                        
                    df, company_financials_df = realign_market_df_to_closest_dt_from_upstream_df(df, company_financials_df, merge_key)
                    df = df[[c for c in df.columns if c not in ['Volume', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits']]]
                    company_financials_df = company_financials_df.merge(df, left_on=merge_key, right_on=merge_key)
                    company_financials_df.drop_duplicates(inplace=True)
                    results["Context"].append(
                        {f"{question} (ticker={ticker})" : company_financials_df}
                    )
                    results["finalAnalysis"]["tables"][ticker.upper()] = company_financials_df
                    results['MarketData'][ticker.upper()] = company_financials_df
                    del results["GetCompanyFinancials"][ticker]
            else:
                solo_call = True
        
        if solo_call:
            params = {}
            from_date = "2020-01-01"
            # import pdb; pdb.set_trace()
            now = (datetime.now() + timedelta(days=-1)).date()
            to_date = f"{now.year}-{now.month}-{now.day}"
            tickers = []
            for entity in entities:
                if entity["entity"] == "from_date":
                    from_date = entity["value"]
                elif entity["entity"] == "to_date":
                    to_date = entity["value"]
                elif entity["entity"] == "ticker":
                    tickers.append(entity["value"].upper())

            to_date = (parser.parse(to_date) + timedelta(days=30)).date()
            if to_date > now:
                to_date = now
            df = yahooFinance.Ticker(ticker).history(period="max")
            if len(df) > 0:
                df.reset_index(inplace=True)
                df.rename({'Date': 'report_date'}, axis=1, inplace=True)
                df['report_date'] = df['report_date'].dt.strftime("%Y-%m-%d")
                df = df[::-1]
                df['60 Day Performance'] = df['Close'].pct_change(-60)
                df['30 Day Performance'] = df['Close'].pct_change(-30)
                results["finalAnalysis"]["tables"][ticker.upper()] = df
                results['MarketData'][ticker] = df
                results["Context"].append(
                    {f"{question} (ticker={ticker})" : df}
                )

        return results
    except Exception as e:
        print(f"Error in get_market_data {e}")
        return results



# def build_stock_screener(question, results, symbol, multiplier=1, timespan="day", from_date="", to_date="", debug=False):
#     entities = extract_entities_from_user_query(question, debug=debug)
#     symbol = None
#     from_date = None
#     to_date = None
#     timespan = None
#     multiplier = None
#     for entity in entities:
#         if entity["entity"] == "ticker":
#             symbol = entity["value"]

def get_news_by_symbol(question, results, symbol, multiplier=1, limit=10, from_date="", to_date="", debug=False):
    try:
        
        entities = extract_entities_from_user_query(question, debug=debug)


        symbol = None
        from_date = None
        to_date = None
        timespan = None
        multiplier = None


        # curl --request GET \
        # --url 'https://api.benzinga.com/api/v2/news?token=8ff1fcc8246e4804b71dd00458b2496b&tickers=AAPL' \
        # --header 'accept: application/json'



        for entity in entities:
            if entity["entity"] == "ticker":
                symbol = entity["value"]
            if entity["entity"] == "from_date":
                from_date = entity["value"]
            if entity["entity"] == "to_date":
                to_date = entity["value"]
            if entity["entity"] == "timespan":
                timespan = entity["value"]
            if entity["entity"] == "multiplier":
                multiplier = entity["value"]

        params = {
            "ticker": symbol
        }

        #if multiplier:
        #     params["multiplier"] = multiplier
        # if timespan:
        #     params["timespan"] = timespan
        # if from_date:
        #     params["from_date"] = from_date
        # if to_date:
        #    params["to_date"] = to_date 

        # response = requests.get(base_url, params=params)
        # base_url = f"https://api.polygon.io/v2/reference/news?ticker=aapl&published_utc=2024-06-13&order=desc&limit=10&sort=published_utc&apiKey={POLYGON_API_KEY}"
        headers = { 
            "accept": "application/json"
        }
        base_url = f"https://api.benzinga.com/api/v2/news?token=8ff1fcc8246e4804b71dd00458b2496b&tickers=AAPL&topics=legal" # {symbol}
        response = requests.get(base_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
                
        if "finalAnalysis" not in results:
            results["finalAnalysis"] = {}
        if "headlines" not in results["finalAnalysis"]:
            results["finalAnalysis"]["headlines"] = {
            }

        results["finalAnalysis"]["headlines"][symbol.upper()] = data

        results["Context"].append(
                f"News headlines for {symbol.upper()}: {data}"
            )

        return results
    except Exception as e:
        print(f"Error in get_news_by_symbol {e}")
        return results


def build_market_data_chart(question, results, symbol, multiplier=1, timespan="day", from_date="", to_date="", debug=False):
    try:
        entities = extract_entities_from_user_query(question, debug=debug)
        symbol = None
        from_date = None
        to_date = None
        timespan = None
        multiplier = None
        for entity in entities:
            if entity["entity"] == "ticker":
                symbol = entity["value"]
            if entity["entity"] == "from_date":
                from_date = entity["value"]
            if entity["entity"] == "to_date":
                to_date = entity["value"]
            if entity["entity"] == "timespan":
                timespan = entity["value"]
            if entity["entity"] == "multiplier":
                multiplier = entity["value"]
        
        # Expected:{'AMZN': [{'AMZN.revenues': -0.16}, {'AMZN.revenues': -0.06}, {'AMZN.revenues': -0.05}]}
        market_data = get_market_data(symbol, multiplier=1, timespan="day", from_date="", to_date="", debug=False)
        if "finalAnalysis" not in results:
            results["finalAnalysis"] = {}
        if "charts" not in results["finalAnalysis"]:
            results["finalAnalysis"]["charts"] = {

            }
        import pdb; pdb.set_trace()
        results["finalAnalysis"]["charts"][symbol.upper()] =  market_data[['c']].to_dict(orient="records")

        return results
    except Exception as e:
        print(f"Error at build_market_data_chart: {e}")
        return results




# def get_stock_financials_old(symbol, limit=5, debug=False):
#     """
#     Fetch financial data for a given stock symbol using Polygon API.
    
#     :param symbol: Stock symbol (e.g., 'AAPL' for Apple Inc.)
#     :param limit: Number of financial reports to retrieve (default: 5)
#     :return: JSON response containing financial data
#     """

#     # "balance-sheet-statement/AAPL?apikey=tSJPBoMv79Baig8DXj50Oky1p4oQbyhU"
#     # tSJPBoMv79Baig8DXj50Oky1p4oQbyhU
    
#     statement_types = ["balance-sheet-statement", "cash-flow-statement", "income-statement"]
#     dfs = []
    
#     for statement_type in statement_types:
#         base_url = f"https://financialmodelingprep.com/api/v3/{statement_type}/{symbol.upper()}?apikey={FMP_API_KEY}" #f"https://api.polygon.io/vX/reference/financials?limit=100&apiKey={POLYGON_API_KEY}"
        
#         params = {
#             "ticker": symbol,
#             "limit": limit
#         }
        
#         # response = requests.get(base_url, params=params)
#         response = requests.get(base_url)
#         if response.status_code == 200:
#             try:
#                 data = response.json()
#                 df = pd.DataFrame.from_records(data)
                

#                 dfs.append(df)

#             #     # Flatten the 'financials' column
#             #     financials_df = pd.json_normalize(df['financials'])
                
#             #     cols_to_keep = []
#             #     for col in list(financials_df.columns):
#             #         statement, label, col_type = col.split(".")
#             #         # if statement in ["balance_sheet", "income_statement", "cash_flow_statement"] and col_type == "value":
#             #         if (col_type == "value"):
#             #             cols_to_keep.append(col)
                
#             #     # import pdb; pdb.set_trace()
#             #     financials_df = financials_df[cols_to_keep]
#             #     # Combine the main DataFrame with the flattened financials
#             #     result_df = pd.concat([df.drop('financials', axis=1), financials_df], axis=1)
                
#             #     result_df["end_date"] = pd.to_datetime(result_df["end_date"])
#             #     result_df.sort_values(by=["end_date"], ascending=True)
#             #     result_df["report_date"] = result_df["fiscal_period"].astype(str) + " " + result_df["fiscal_year"].astype(str)
#             #     # import pdb; pdb.set_trace()
                
#             #     # TODO: Uncomment this after getting quarterly stuffs to work (add routing for filtering on cases: just FY, just Q*, mixed)
#             #     # result_df[(~result_df["report_date"].isin(["TTM", "TTM "]))]
#             #     result_df = result_df[(~result_df["report_date"].isin(["TTM", "TTM "])) & (~result_df["report_date"].str.contains("FY")) & (result_df["timeframe"] == "quarterly")]
#             #     # import pdb; pdb.set_trace()
#             #     result_df = result_df[[c for c in result_df.columns if c not in ['start_date', 'end_date', 'timeframe', 'fiscal_period', 'fiscal_year',
#             # 'cik', 'sic', 'tickers', 'company_name', 'filing_date',
#             # 'acceptance_datetime', 'source_filing_url', 'source_filing_file_url']]]
#             #     result_df.set_index("report_date", inplace=True)
#             #     result_df.rename({c: c.split(".")[1] for c in result_df.columns}, axis=1, inplace=True)

#             #     return result_df.T
#             except Exception as e:
#                 print(f"Error in get_stock_financials: {e}")
#                 # import pdb; pdb.set_trace()
#                 continue
#         else:
#             print(f"Error in get_stock_financials")
#             # import pdb; pdb.set_trace()
#             raise Exception(f"Error fetching data: {response.status_code} - {response.text}")

    

#     result_df = pd.concat(dfs)
#     result_df["end_date"] = pd.to_datetime(result_df["date"])
#     result_df.sort_values(by=["date"], ascending=True)
# #     result_df["report_date"] = result_df["period"].astype(str) + " " + result_df["fiscal_year"].astype(str)
# #     return df

def get_stock_financials_old(symbol, mode='calendar', limit=5, debug=False):
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
        try:
            data = response.json()["results"]
            df = pd.DataFrame.from_records(data)
            # import pdb; pdb.set_trace()

            # Flatten the 'financials' column
            financials_df = pd.json_normalize(df['financials'])
            
            cols_to_keep = []
            for col in list(financials_df.columns):
                statement, label, col_type = col.split(".")
                # if statement in ["balance_sheet", "income_statement", "cash_flow_statement"] and col_type == "value":
                if (col_type == "value"):
                    cols_to_keep.append(col)
            
            
            financials_df = financials_df[cols_to_keep]
            # Combine the main DataFrame with the flattened financials
            result_df = pd.concat([df.drop('financials', axis=1), financials_df], axis=1)
            
            result_df["end_date"] = pd.to_datetime(result_df["end_date"])
            result_df.sort_values(by=["end_date"], ascending=True)

            result_df["report_date"] = result_df["fiscal_period"].astype(str) + " " + result_df["fiscal_year"].astype(str)
            # import pdb; pdb.set_trace()
            
            # TODO: Uncomment this after getting quarterly stuffs to work (add routing for filtering on cases: just FY, just Q*, mixed)
            # result_df[(~result_df["report_date"].isin(["TTM", "TTM "]))]
            result_df = result_df[(~result_df["report_date"].isin(["TTM", "TTM "])) & (~result_df["report_date"].str.contains("FY")) & (result_df["timeframe"] == "quarterly")]
            # import pdb; pdb.set_trace()
            result_df = result_df[[c for c in result_df.columns if c not in ['start_date', 'timeframe', 'fiscal_period', 'fiscal_year',
        'cik', 'sic', 'tickers', 'company_name', 'filing_date',
        'acceptance_datetime', 'source_filing_url', 'source_filing_file_url']]]

            df_index = None            
            if mode.lower() == "fiscal":
                df_index = "report_date"
                result_df.drop(f"end_date", inplace=True)
                result_df.rename({"end_date": "Calendar Date"}, axis=1, inplace=True)
                result_df.set_index("report_date", inplace=True)
            else: #NOTE i.e. elif mode.lower() == "calendar":
                df_index = "end_date"
                result_df.rename({"report_date": "Fiscal Date"}, axis=1, inplace=True)
                result_df.rename({"end_date": "report_date"}, axis=1, inplace=True)
                result_df["report_date"] = result_df["report_date"].dt.strftime('%Y-%m-%d')
                result_df.set_index("report_date", inplace=True)

            result_df.rename({c: c.split(".")[1] for c in result_df.columns if c not in ["end_date", "report_date", "Fiscal Date", "Calendar Date"]}, axis=1, inplace=True)

            # import pdb; pdb.set_trace()

            return result_df
        except Exception as e:
            print(f"Error in get_stock_financials: {e}")
            
            return None
    else:
        print(f"Error in get_stock_financials")
        # import pdb; pdb.set_trace()
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")


def get_trade_stats(backtest_df, long_short_multiplier):
    trades_df = backtest_df.copy()
    trades_dict = {

    }

    # import pdb; pdb.set_trace()
    # trades_df['1 QTR'] = (trades_df['Close'] - trades_df['1 QTR']) / trades_df['Close']
    trades_df['Cumul. Return'] = trades_df['1 QTR'][::-1].cumsum()
    total_num_trades = len(trades_df)
    num_winning_trades = float(total_num_trades) - float((len(trades_df[trades_df['1 QTR']<0.0])*long_short_multiplier)) #+ (len(trades_df[trades_df['1 QTR']<0.0])*long_short_multiplier)
    num_losing_trades = total_num_trades - int(num_winning_trades)
    winning_pct = float(num_winning_trades) / (float(num_winning_trades)+float(num_losing_trades))
    avg_winning_trade = None
    avg_losing_trade = None
    largest_winning_trade = None
    largest_losing_trade = None
    profit_factor = None
    expectation = None
    # import pdb; pdb.set_trace()
    if long_short_multiplier > 0:
        avg_winning_trade = trades_df[trades_df['1 QTR']>0.0]['1 QTR'].mean()
        avg_losing_trade = trades_df[trades_df['1 QTR']<0.0]['1 QTR'].mean()
        largest_winning_trade = trades_df[trades_df['1 QTR']>0.0]['1 QTR'].max()
        largest_losing_trade = trades_df[trades_df['1 QTR']<0.0]['1 QTR'].max()
        profit_factor = abs(trades_df[trades_df['1 QTR']>0.0]['1 QTR'].sum() / trades_df[trades_df['1 QTR']<0.0]['1 QTR'].sum())
        expectation = avg_winning_trade*winning_pct + avg_losing_trade*(1-winning_pct)
    else:
        avg_winning_trade = trades_df[trades_df['1 QTR']<0.0]['1 QTR'].mean()
        avg_losing_trade = trades_df[trades_df['1 QTR']>0.0]['1 QTR'].mean()
        largest_winning_trade = trades_df[trades_df['1 QTR']<0.0]['1 QTR'].max()
        largest_losing_trade = trades_df[trades_df['1 QTR']>0.0]['1 QTR'].max()
        profit_factor = trades_df[trades_df['1 QTR']<0.0]['1 QTR'].sum() / trades_df[trades_df['1 QTR']>0.0]['1 QTR'].sum()
        expectation = avg_winning_trade*winning_pct + avg_losing_trade*(1-winning_pct)

    stats = [
        {"title": "Number of trades", "value": total_num_trades},
        {"title": "expectation", "value": expectation},
        {"title": "Profit factor", "value": profit_factor},
        {"title": "Largest winning trade (%)", "value": largest_winning_trade},
        {"title": "Largest losing trade (%)", "value": largest_losing_trade},
        {"title": "win %", "value": winning_pct},
        {"title": "avg. win (%)", "value": avg_winning_trade},
        {"title": "avg. loss (%)", "value": avg_losing_trade}
    ]
    stats_df = pd.DataFrame.from_records(stats)
    stats_df.replace(np.inf, None, inplace=True)
    stats_df.replace(-np.inf, None, inplace=True)
    stats_df.replace(np.nan, None, inplace=True)
    stats = stats_df.to_dict(orient="records")


    print(f"stats: {stats}")

    return stats, trades_df


def run_backtest(query, results, debug=False):
    print(f"Inside run_backtest: {query}")

    # import pdb; pdb.set_trace()
    entities = extract_entities_from_user_query(query, debug=debug)
    # import pdb; pdb.set_trace()
    ticker = [ent["value"] for ent in entities if ent["entity"]=="ticker"][0]
    # import pdb; pdb.set_trace()
    if ticker in results["MarketDataForBacktest"]:
        market_data_df = results["MarketDataForBacktest"][ticker]
    elif ticker in results["MarketData"]:
        market_data_df = results["MarketData"][ticker]
    # NOTE: implement get_long_short_multiplier
    long_short_multiplier = 1 #get_long_short_multiplier(query)
    backtest_df = market_data_df.copy()
    

    stats, trades_df = get_trade_stats(backtest_df, long_short_multiplier)
    trades_df.replace(np.nan, None, inplace=True)
    trades_df.replace(np.inf, None, inplace=True)
    trades_df.replace(-np.inf, None, inplace=True)

    # trade_dict = {"Date": [], "Instrument": [], "Entry": [], "Exit 1M": [], "Exit 6M": [], "Exit 1Y": [], "Cumul. Return": []}
    # filled_trade_dict = fill_trade_dict(trade_dict)
    # trade_df = pd.DataFrame(filled_trade_dict)

    # import pdb; pdb.set_trace()
    trades_table = {
        "headers": list(trades_df.columns),
        "rows": trades_df.values.tolist()
    }
    equity_curve_df = trades_df[['report_date', 'Cumul. Return']]
    equity_curve_df.set_index('report_date', inplace=True)


    equity_curve_chart = {
        "data": trades_df[::-1].to_dict("records"),
        "dataKeys": ['Cumul. Return'],
        "type": "line"
    }
    # results["RunBackTest"] = {"title": f"Backtest Report for Query: {query}", "stats": stats, "trades": trades_table, "equity_curve": equity_curve_chart}
    results["Context"].append({"Backtest Results": results["RunBackTest"]})
    results["finalAnalysis"]["backTest"] = {"stats": stats, "trades": trades_table, "equity_curve": equity_curve_chart}

    print(f"Inside run_backtest: {results}")
    return results

"""
'query': Calculate AAPL’s profit margin and inventory ratio according to its 10K’s/Q filings. What’s the historical correlation of these values? How has the stock reacted in the month preceding and following earnings when the prior quarter’s correlations were low vs when they were high?
'avalailable rows': ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses', 'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'link', 'finalLink', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt', 'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow', 'Calendar Date']
'correctly selected rows': ['revenue', 'netIncome', 'inventory']
"""
def get_relevant_rows(financials, question, debug=False):
    """
    Prompt ChatGPT to identify the relevant rows from the financial data.

    :param financials: JSON response containing financial data
    :param question: User's question about the financial data
    :return: List of relevant row indices
    """

    system_prompt = """
    You are a helpful assistant that analyzes financial data. Given a user question and list of financial reporting terms, determine which of the
    financial terms are required to answer the question. The response should be a JSON list of the values in the supplied rows. Below are some examples:

    Examples:
       "query": "Compare F's revenue growth and stock performance since 2020? Which has appreciated more and by how much?", "available rows": ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses', 'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'link', 'finalLink', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt', 'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow', 'Fiscal Date'], "correctly selected rows": ['revenue']
       "query": "Compare TSLA's revenue growth and stock performance since 2020? Which has appreciated more and by how much?", "available rows": ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses', 'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'link', 'finalLink', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt', 'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow', 'Fiscal Date'], "correctly selected rows": ['revenue']   
       "query": "How has apple's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if i bought the stock each time?", 'avalailable rows': ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses', 'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'link', 'finalLink', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt', 'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow', 'Fiscal Date'], "correctly selected rows": ['revenue']
       "query": "How has apple's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if i bought the stock each time?", 'avalailable rows': ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses', 'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'link', 'finalLink', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt', 'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow', 'Fiscal Date'], "correctly selected rows": ['revenue']
       "query": "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places. Base your judgments on the information provided primarily in the statement of income and the statement of financial position.", "available rows": ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses', 'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'link', 'finalLink', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt', 'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow', 'Fiscal Date'], "correctly selected rows": ['revenue', 'totalNonCurrentAssets']
       "query": "Does 3M maintain a stable trend of dividend distribution?", "available rows": ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses', 'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'link', 'finalLink', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt', 'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow', 'Fiscal Date'], "correctly selected rows": ['dividendsPaid']
       "query": "You are an investment banker and your only resource(s) to answer the following question is (are): the statement of financial position and the cash flow statement. Here's the question: what is the FY2015 operating cash flow ratio for Adobe? Operating cash flow ratio is defined as: cash from operations / total current liabilities. Round your answer to two decimal places.", ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses', 'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'link', 'finalLink', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt', 'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow', 'Fiscal Date'], "correctly selected rows": ['netCashProvidedByOperatingActivities', 'totalCurrentLiabilities']
       "query": "What are the debt levels of AAPL?", 'avalailable rows': ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses', 'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'link', 'finalLink', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt', 'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow', 'Fiscal Date'], "correctly selected rows": ['longTermDebt', 'shortTermDebt', 'totalDebt', 'netDebt']
       "query": "What are the debt levels of MSFT?", 'avalailable rows': ['revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses', 'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'link', 'finalLink', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt', 'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow', 'Fiscal Date'], "correctly selected rows": ['longTermDebt', 'shortTermDebt', 'totalDebt', 'netDebt']
    """

    prompt = """
    query: {question}
    rows: {financials}

    Please identify the rows in the supplied financials table that are necessary to answer the question.
    Return your answer as a Python list that contains only the row names.
    """

    # Call the OpenAI API
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, financials=sorted(financials))}
        ],
    )

    try:
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        relevant_rows = ast.literal_eval(response[response.find("["):response.find("]")+1])
        print(f"'query': {question}")
        print(f"'avalailable rows': {financials}")
        print(f"'correctly selected rows': {relevant_rows}")
        # import pdb; pdb.set_trace()
        if len(relevant_rows) == 0:
            import pdb; pdb.set_trace()

        return relevant_rows
    except Exception as e:
        print(f"Error get_relevant_rows: {e}")
        return []


def get_relevant_fiscal_columns(financials, question, debug=False):
    """
    Prompt ChatGPT to identify the relevant rows from the financial data.

    :param financials: JSON response containing financial data
    :param question: User's question about the financial data
    :return: List of relevant row indices
    """

    system_prompt = """
    You are a helpful assistant that analyzes financial data. Given a user question, determine which of the
    dates are required to answer the question. The response should be a JSON list of the values  as shown in the examples below:

    Examples:     
        query: "compare the cogs between 2021 and 2023 between AMZN and its comps", answer: ['Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021'] 
        query: "investigate the trends in JP Morgan's net interest margin and loan growth over the past 2 years. How have changes in interest rates and economic conditions influenced their profitability?", answer: ['Q2 2024', 'Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q4 2021', 'Q3 2021', 'Q2 2021'] 
        query: "what's amzn's cogs from 2022 to 2023?", answer: ['Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022']
        query: "what was the total amount spent on acquisitions net of cash acquired and purchases of intangible and other assets by msft in the year 2022?", answer: ['Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022']
        query: "What was the trend in MSFT's Payments for Repurchase of Common Stock from Q3 2020 to Q1 2023?", answer: ['Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021', 'Q4 2020', 'Q3 2020']
        query: "Compare MSFT's cloud sales growth to IBM's", answer: ['Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022']
        query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", answer: [Q2 2024, 'Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'Q3 2021', 'Q2 2021']
        query: "Examine the changes in lmt's debt-to-equity ratio over the last 2 years. What are the implications for its financial stability compared to competitors in the aerospace industry?", answer: ['Q2 2024', 'Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022']
        query: "Examine the changes in Boeing's debt-to-equity ratio over the last 2 years. What are the implications for its financial stability compared to competitors in the aerospace industry?", answer: ['Q2 2024', 'Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022']
        query: "Show how amzn's cogs have grown quarterly for the past 4 quarters?", answer: ['Q2 2024', 'Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023'] 
        query: "How has amzn's cogs grown annually quarter over quarter over 2023 vs 2022?", answer: ['Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021']
        query: "How has amzn's cogs grown annually quarter over quarter over 2023?", answer: ['Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023']
        query: "Show me the percentage change of aapl's gross margins versus Intel and Microsoft over the past 3 years?", answer: ['Q2 2024', 'Q1 2024','Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021']
        query: "calculate the correlation between the % change of aapl’s gross margins and its inventory ratio over the past 2 years. how does it compare to msft?", answer: ['Q2 2024', 'Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023', 'Q1 2023', 'Q4 2022', 'Q3 2022', 'Q2 2022', 'Q1 2022', 'Q4 2021', 'Q3 2021', 'Q2 2021', 'Q1 2021'] 
        query: "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places. Base your judgments on the information provided primarily in the statement of income and the statement of financial position.", answer: ['Q4 2019', 'Q3 2019', 'Q2 2019', 'Q1 2019', 'Q4 2018', 'Q3 2018', 'Q2 2018', 'Q1 2018']
        query: "What is Adobe's year-over-year change in unadjusted operating income from FY2015 to FY2016 (in units of percents and round to one decimal place)? Give a solution to the question by using the income statement.", answer: ['Q4 2016', 'Q3 2016', 'Q2 2016', 'Q1 2016', 'Q4 2015', 'Q3 2015', 'Q2 2015', 'Q1 2015']
    
    """

    prompt = """
    query: {question}

    Please identify the rows that are necessary to answer the question.
    Return your answer as a JSON list that contains only the row names.
    """

    # Call the OpenAI API
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ],
        
    )

    try:
        # import pdb; pdb.set_trace()
        
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        relevant_columns = ast.literal_eval(response[response.find("["):response.find("]")+1])
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "get_relevant_fiscal_columns", "inputs": [question, financials], "outputs": [{"relevant_columns": relevant_columns}]}, indent=6))
        return relevant_columns
    except Exception as e:
        print(f"Error inside get_relevant_fiscal_columns: {e}")
        return []



"""NOTE: OLD GET_RELEVANT COLUMNS 
query: "compare the cogs between 2021 and 2023 between AMZN and its comps", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-31','2011-09-30','2011-06-30','2011-03-31','2010-12-31','2010-09-30','2010-06-30','2010-03-31','2009-12-31','2009-09-30','2009-06-30','2009-03-30'], answer: ['2023-12-31', '2023-09-30', '2023-06-30', '2023-03-31', '2022-12-31', '2022-09-30', '2022-06-30', '2022-03-31'] 
        query: "investigate the trends in JP Morgan's net interest margin and loan growth over the past 2 years. How have changes in interest rates and economic conditions influenced their profitability?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-12-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-31','2011-09-30','2011-06-30','2011-03-31','2010-09-30','2010-06-30','2010-03-31','2009-12-31','2009-09-30','2009-06-30','2009-03-30'] , answer: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30']
        query: "what's amzn's cogs from 2022 to 2023?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-31','2011-09-30','2011-06-30','2011-03-31','2010-12-31','2010-09-30','2010-06-30','2010-03-31','2009-12-31','2009-09-30','2009-06-30','2009-03-30'], answer: ['2023-12-31', '2023-09-30', '2023-06-30', '2023-03-31', '2022-12-31', '2022-09-30', '2022-06-30', '2022-03-31'] 
        query: "what was the total amount spent on acquisitions net of cash acquired and purchases of intangible and other assets by msft in the year 2022?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-12-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-30','2011-09-30','2011-06-30','2011-03-31','2010-12-31','2010-09-30','2010-06-30','2009-12-31','2009-09-30'], answer: ['2022-12-31','2022-09-30','2022-06-30','2022-03-31']
        query: "What was the trend in MSFT's Payments for Repurchase of Common Stock from Q3 2020 to Q1 2023?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-12-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-30','2011-09-30','2011-06-30','2011-03-31','2010-12-31','2010-09-30','2010-06-30','2009-12-31','2009-09-30'], answer: ['2022-12-31','2022-09-30','2022-06-30','2022-03-31'], answer: ['2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30']
        query: "Compare MSFT's cloud sales growth to IBM's", available columns: , answer: []
        query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-12-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-31','2011-09-30','2011-06-30','2011-03-31','2010-09-30','2010-06-30','2010-03-31','2009-12-31','2009-09-30','2009-06-30','2009-03-30'] , answer: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30', '2021-03-31', '2020-12-31','2020-09-30','2020-06-30', '2020-03-31']
        query: "Examine the changes in lmt's debt-to-equity ratio over the last 2 years. What are the implications for its financial stability compared to competitors in the aerospace industry?", available columns: , answer: []
        query: "Examine the changes in Boeing's debt-to-equity ratio over the last 2 years. What are the implications for its financial stability compared to competitors in the aerospace industry?", available columns: , answer: []
        query: "Show how amzn's cogs have grown quarterly for the past 4 quarters?", available columns: , answer: ['Q2 2024', 'Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023'] 
        query: "How has amzn's cogs grown annually quarter over quarter over 2023 vs 2022?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-31','2011-09-30','2011-06-30','2011-03-31','2010-12-31','2010-09-30','2010-06-30','2010-03-31','2009-12-31','2009-09-30','2009-06-30','2009-03-30'] , answer: ['2023-12-31', '2023-09-30', '2023-06-30', '2023-03-31', '2022-12-31', '2022-09-30', '2022-06-30', '2022-03-31']
        query: "How has amzn's cogs grown annually quarter over quarter over 2023?", available columns: , answer: []
        query: "Show me the percentage change of aapl's gross margins versus Intel and Microsoft over the past 3 years?", available columns: , answer: []
        query: "calculate the correlation between the % change of aapl’s gross margins and its inventory ratio over the past 2 years. how does it compare to msft?", available columns: , answer: []
"""
def get_relevant_calendar_columns(financials, question, debug=False):
    """
    Prompt ChatGPT to identify the relevant rows from the financial data.

    :param financials: JSON response containing financial data
    :param question: User's question about the financial data
    :return: List of relevant row indices
    """

    system_prompt = """
    You are a helpful assistant that analyzes financial data. Given a user question, determine which of the
    dates are required to answer the question. The response should be a JSON list of the values  as shown in the examples below:

    Examples:     
        query: "How has AAPL's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if I bought the stock each time?", available columns: ['2024-09-28', '2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26', '2021-03-27', '2020-12-26', '2020-09-26', '2020-06-27', '2020-03-28', '2019-12-28', '2019-09-28', '2019-06-29', '2019-03-30', '2018-12-29', '2018-09-29', '2018-06-30', '2018-03-31', '2017-12-30', '2017-09-30', '2017-07-01', '2017-04-01', '2016-12-31', '2016-09-24', '2016-06-25', '2016-03-26', '2015-12-26', '2015-09-26', '2015-06-27', '2015-03-28', '2014-12-27', '2014-09-27', '2014-06-28', '2014-03-29', '2013-12-28', '2013-09-28', '2013-06-29', '2013-03-30', '2012-12-29', '2012-09-29', '2012-06-30', '2012-03-31', '2011-12-31', '2011-09-24', '2011-06-25', '2011-03-26', '2010-12-25', '2010-09-25', '2010-06-26', '2010-03-27', '2009-12-26', '2009-09-26', '2009-06-27', '2009-03-28', '2008-12-27', '2008-09-27', '2008-06-28', '2008-03-29', '2007-12-29', '2007-09-29', '2007-06-30', '2007-03-31', '2006-12-30', '2006-09-30', '2006-07-01', '2006-04-01'], answer: ['2024-09-28', '2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26', '2021-03-27', '2020-12-26', '2020-09-26', '2020-06-27', '2020-03-28', '2019-12-28', '2019-09-28', '2019-06-29', '2019-03-30', '2018-12-29', '2018-09-29', '2018-06-30', '2018-03-31', '2017-12-30']
        query: "compare the cogs between 2021 and 2023 between AMZN and its comps", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-31','2011-09-30','2011-06-30','2011-03-31','2010-12-31','2010-09-30','2010-06-30','2010-03-31','2009-12-31','2009-09-30','2009-06-30','2009-03-30'], answer: ['2023-12-31', '2023-09-30', '2023-06-30', '2023-03-31', '2022-12-31', '2022-09-30', '2022-06-30', '2022-03-31'] 
        query: "investigate the trends in JP Morgan's net interest margin and loan growth over the past 2 years. How have changes in interest rates and economic conditions influenced their profitability?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-12-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-31','2011-09-30','2011-06-30','2011-03-31','2010-09-30','2010-06-30','2010-03-31','2009-12-31','2009-09-30','2009-06-30','2009-03-30'] , answer: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30']
        query: "what's amzn's cogs from 2022 to 2023?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-31','2011-09-30','2011-06-30','2011-03-31','2010-12-31','2010-09-30','2010-06-30','2010-03-31','2009-12-31','2009-09-30','2009-06-30','2009-03-30'], answer: ['2023-12-31', '2023-09-30', '2023-06-30', '2023-03-31', '2022-12-31', '2022-09-30', '2022-06-30', '2022-03-31'] 
        query: "what was the total amount spent on acquisitions net of cash acquired and purchases of intangible and other assets by msft in the year 2022?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-12-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-30','2011-09-30','2011-06-30','2011-03-31','2010-12-31','2010-09-30','2010-06-30','2009-12-31','2009-09-30'], answer: ['2022-12-31','2022-09-30','2022-06-30','2022-03-31']
        query: "What was the trend in MSFT's Payments for Repurchase of Common Stock from Q3 2020 to Q1 2023?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-12-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-30','2011-09-30','2011-06-30','2011-03-31','2010-12-31','2010-09-30','2010-06-30','2009-12-31','2009-09-30'], answer: ['2022-12-31','2022-09-30','2022-06-30','2022-03-31'], answer: ['2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30']
        query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-12-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-31','2011-09-30','2011-06-30','2011-03-31','2010-09-30','2010-06-30','2010-03-31','2009-12-31','2009-09-30','2009-06-30','2009-03-30'] , answer: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30', '2021-03-31', '2020-12-31','2020-09-30','2020-06-30', '2020-03-31']
        query: "How has amzn's cogs grown annually quarter over quarter over 2023 vs 2022?", available columns: ['2024-06-30','2024-03-31','2023-12-31','2023-09-30','2023-06-30','2023-03-31','2022-12-31','2022-09-30','2022-06-30','2022-03-31','2021-12-31','2021-09-30','2021-06-30','2021-03-31','2020-12-31','2020-09-30','2020-06-30','2020-03-31','2019-12-31','2019-09-30','2019-06-30','2019-03-31','2018-12-31','2018-09-30','2018-06-30','2018-03-31','2017-09-30','2017-06-30','2017-03-31','2016-12-31','2016-09-30','2016-06-30','2016-03-31','2015-12-31','2015-09-30','2015-06-30','2015-03-31','2014-12-31','2014-09-30','2014-06-30','2014-03-31','2013-12-31','2013-09-30','2013-06-30','2013-03-31','2012-12-31','2012-09-30','2012-06-30','2012-03-31','2011-12-31','2011-09-30','2011-06-30','2011-03-31','2010-12-31','2010-09-30','2010-06-30','2010-03-31','2009-12-31','2009-09-30','2009-06-30','2009-03-30'], answer: ['2023-12-31', '2023-09-30', '2023-06-30', '2023-03-31', '2022-12-31', '2022-09-30', '2022-06-30', '2022-03-31']
        query: "What’s apple’s revenue growth for the past 2 years? Compute the correlation between revenue growth and its margin growth. Compare it to msft who had the higher average gross margin?", available columns: ['2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26', '2021-03-27', '2020-12-26', '2020-09-26', '2020-06-27', '2020-03-28', '2019-12-28', '2019-09-28', '2019-06-29', '2019-03-30', '2018-12-29', '2018-09-29', '2018-06-30', '2018-03-31', '2017-12-30', '2017-09-30', '2017-07-01', '2017-04-01', '2016-12-31', '2016-09-24', '2016-06-25', '2016-03-26', '2015-12-26', '2015-09-26', '2015-06-27', '2015-03-28', '2014-12-27', '2014-09-27', '2014-06-28', '2014-03-29', '2013-12-28', '2013-09-28', '2013-06-29', '2013-03-30', '2012-12-29', '2012-09-29', '2012-06-30', '2012-03-31', '2011-12-31', '2011-09-24', '2011-06-25', '2011-03-26', '2010-12-25', '2010-09-25', '2010-06-26', '2010-03-27', '2009-12-26', '2009-09-26', '2009-06-27'], answer: ['2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26']
        query: "What’s apple’s revenue growth for the past 2 years? Compute the correlation between revenue growth and its margin growth", available columns: ['2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26', '2021-03-27', '2020-12-26', '2020-09-26', '2020-06-27', '2020-03-28', '2019-12-28', '2019-09-28', '2019-06-29', '2019-03-30', '2018-12-29', '2018-09-29', '2018-06-30', '2018-03-31', '2017-12-30', '2017-09-30', '2017-07-01', '2017-04-01', '2016-12-31', '2016-09-24', '2016-06-25', '2016-03-26', '2015-12-26', '2015-09-26', '2015-06-27', '2015-03-28', '2014-12-27', '2014-09-27', '2014-06-28', '2014-03-29', '2013-12-28', '2013-09-28', '2013-06-29', '2013-03-30', '2012-12-29', '2012-09-29', '2012-06-30', '2012-03-31', '2011-12-31', '2011-09-24', '2011-06-25', '2011-03-26', '2010-12-25', '2010-09-25', '2010-06-26', '2010-03-27', '2009-12-26', '2009-09-26', '2009-06-27'], answer: ['2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26']
        query: "What’s apple’s revenue growth for the past 2 years?", available columns: ['2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26', '2021-03-27', '2020-12-26', '2020-09-26', '2020-06-27', '2020-03-28', '2019-12-28', '2019-09-28', '2019-06-29', '2019-03-30', '2018-12-29', '2018-09-29', '2018-06-30', '2018-03-31', '2017-12-30', '2017-09-30', '2017-07-01', '2017-04-01', '2016-12-31', '2016-09-24', '2016-06-25', '2016-03-26', '2015-12-26', '2015-09-26', '2015-06-27', '2015-03-28', '2014-12-27', '2014-09-27', '2014-06-28', '2014-03-29', '2013-12-28', '2013-09-28', '2013-06-29', '2013-03-30', '2012-12-29', '2012-09-29', '2012-06-30', '2012-03-31', '2011-12-31', '2011-09-24', '2011-06-25', '2011-03-26', '2010-12-25', '2010-09-25', '2010-06-26', '2010-03-27', '2009-12-26', '2009-09-26', '2009-06-27'], answer: ['2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26']
        query: "How has AAPL's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if I bought the stock each time?", available columns: ['2024-09-28', '2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26', '2021-03-27', '2020-12-26', '2020-09-26', '2020-06-27', '2020-03-28', '2019-12-28', '2019-09-28', '2019-06-29', '2019-03-30', '2018-12-29', '2018-09-29', '2018-06-30', '2018-03-31', '2017-12-30', '2017-09-30', '2017-07-01', '2017-04-01', '2016-12-31', '2016-09-24', '2016-06-25', '2016-03-26', '2015-12-26', '2015-09-26', '2015-06-27', '2015-03-28', '2014-12-27', '2014-09-27', '2014-06-28', '2014-03-29', '2013-12-28', '2013-09-28', '2013-06-29', '2013-03-30', '2012-12-29', '2012-09-29', '2012-06-30', '2012-03-31', '2011-12-31', '2011-09-24', '2011-06-25', '2011-03-26', '2010-12-25', '2010-09-25', '2010-06-26', '2010-03-27', '2009-12-26', '2009-09-26', '2009-06-27', '2009-03-28', '2008-12-27', '2008-09-27', '2008-06-28', '2008-03-29', '2007-12-29', '2007-09-29', '2007-06-30', '2007-03-31', '2006-12-30', '2006-09-30', '2006-07-01', '2006-04-01'], answer: ['2024-09-28', '2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26', '2021-03-27', '2020-12-26', '2020-09-26', '2020-06-27', '2020-03-28', '2019-12-28', '2019-09-28', '2019-06-29', '2019-03-30', '2018-12-29', '2018-09-29', '2018-06-30', '2018-03-31', '2017-12-30']
        query: "How has AAPL's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if I bought the stock each time?", available columns: ['2024-09-28', '2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26', '2021-03-27', '2020-12-26', '2020-09-26', '2020-06-27', '2020-03-28', '2019-12-28', '2019-09-28', '2019-06-29', '2019-03-30', '2018-12-29', '2018-09-29', '2018-06-30', '2018-03-31', '2017-12-30', '2017-09-30', '2017-07-01', '2017-04-01', '2016-12-31', '2016-09-24', '2016-06-25', '2016-03-26', '2015-12-26', '2015-09-26', '2015-06-27', '2015-03-28', '2014-12-27', '2014-09-27', '2014-06-28', '2014-03-29', '2013-12-28', '2013-09-28', '2013-06-29', '2013-03-30', '2012-12-29', '2012-09-29', '2012-06-30', '2012-03-31', '2011-12-31', '2011-09-24', '2011-06-25', '2011-03-26', '2010-12-25', '2010-09-25', '2010-06-26', '2010-03-27', '2009-12-26', '2009-09-26', '2009-06-27', '2009-03-28', '2008-12-27', '2008-09-27', '2008-06-28', '2008-03-29', '2007-12-29', '2007-09-29', '2007-06-30', '2007-03-31', '2006-12-30', '2006-09-30', '2006-07-01', '2006-04-01'], answer: ['2024-09-28', '2024-06-29', '2024-03-30', '2023-12-30', '2023-09-30', '2023-07-01', '2023-04-01', '2022-12-31', '2022-09-24', '2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25', '2021-06-26', '2021-03-27', '2020-12-26', '2020-09-26', '2020-06-27', '2020-03-28', '2019-12-28', '2019-09-28', '2019-06-29', '2019-03-30', '2018-12-29', '2018-09-29', '2018-06-30', '2018-03-31', '2017-12-30']
    
    """

    prompt = """
    query: {question}
    available columns: {available_columns}

    Please identify the rows that are necessary to answer the question.
    Return your answer as a JSON list that contains only the row names.
    """

    # Call the OpenAI API
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, available_columns=financials)}
        ],
        
    )

    try:
        
        
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        # import pdb; pdb.set_trace()
        relevant_columns = ast.literal_eval(response[response.find("["):response.find("]")+1])
        # import pdb; pdb.set_trace()
        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "get_relevant_fiscal_columns", "inputs": [question, financials], "outputs": [{"relevant_columns": relevant_columns}]}, indent=6))
        return relevant_columns
    except Exception as e:
        print(f"Error inside get_relevant_fiscal_columns: {e}")
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


def text_to_graphql(question, entities, debug=False):
    # import pdb; pdb.set_trace()
    system_prompt = """Generate a GraphQL query for a Weaviate backend database that retrieves 10Q and 10K company financials filings based on specified filters, a user-provided question, and the specified collection.
    When applying filing_type filter if you see '10-Q' in the Question map it to '10Q'. Likewise, when you see '10-K' in the Question map it to '10K'. In other words, do not include the hyphen for the 'filing_type' filter.
    Also next include the graphql keyword 'limit' in your response since we do not want to limit the number of retrieved documents arbitrarily. If the question contains both '10-K' and '10-Q' then DO NOT use the 'filing_type'
    as a filter in your graphql response. Do not include any column references in your response that are not included in the Input Details section below (even if they may make logical sense in the question).

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
        'user query': "If I had bought AAPL stock every time they mentioned supply chain concerns in their filing if I had held it over the following quarter?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], answer: ""qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))""
        'user query': "how many times did msft raise concerns about supply chain issues in their filings in the last 3 years?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did msft raise concerns about supply chain issues in their filings in the last 3 years?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did msft raise concerns about supply chain issues in their filings in the last 3 years?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did msft raise concerns about supply chain issues in their filings in the last 3 years?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did MSFT mention the activision anti-trust case in their filings?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user query': "how many times did CRM mention macro concerns in their filings since 2023? what has been the stock's performance 60 days after?", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df['Cumulative Instances'] = list(range(1, len(qual_and_quant_df)+1))"
        'user_query': "Of the instances in which AAPL referenced its supply chain, how many were positive versus negative? Show me the revenue growth and the price performance tied to each instance", 'data': ['accession_number', 'company_name', 'filing_type', 'url', 'page_number', 'report_date', 'text', 'ticker', 'Sentiment'], 'answer': "qual_and_quant_df.loc[:, 'Total Positive'] = len(qual_and_quant_df[qual_and_quant_df['Sentiment']>0]); qual_and_quant_df.loc[:, 'Total Negative'] = len(qual_and_quant_df[qual_and_quant_df['Sentiment']<0]);"
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
        import pdb; pdb.set_trace()
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


def realign_qual_and_quant_df_to_closest_dt_from_upstream_df(qual_and_quant_df, company_financials_df, merge_key):
    import pdb; pdb.set_trace()
    temp_company_financials_df = company_financials_df.copy()
    temp_df = qual_and_quant_df.copy()
    company_financials_df = temp_df.merge(company_financials_df, left_on=merge_key, right_on=merge_key)
    
    return qual_and_quant_df, company_financials_df


def extract_entities_from_user_query(question, debug=False):
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
        import pdb; pdb.set_trace()
        return []


def get_sentiment(df):
    for i in tqdm(df.index):
        try:
            headline = df.loc[i, 'text']
        except Exception as e:
            print(f"Error in get_sentiment: {e}")
            print(' \'text\' column might be missing from dataframe')

        input = SENTIMENT_TOKENIZER(headline, padding = True, truncation = True, return_tensors='pt')
        # Estimate output
        output = SENTIMENT_MODEL(**input)
        # Pass model output logits through a softmax layer.
        predictions = th.softmax(output.logits, dim=-1)
        sentiment_index = th.argmax(predictions).item()
        sentiment_scores = predictions.tolist()[0]
        if sentiment_scores[0] > sentiment_scores[1]:
            sentiment_score = predictions[0][0].tolist()
        else:
            sentiment_score = (-1)*predictions[0][1].tolist()

        df.loc[i, 'Sentiment'] = sentiment_score

    return df


def perform_quantitative_vector_search(question, results, debug=False):
    try:    
        entities = extract_entities_from_user_query(question, debug)
        filters = []

        to_date = None
        from_date = None
        ticker = None
        for entity in entities:
            if entity["entity"] == "from_date":
                filters.append(entity)
            elif entity["entity"] == "to_date":
                to_date = entity["value"]
                filters.append(entity)
            elif entity["entity"] == "ticker":
                ticker = entity["value"].upper()
                entity["value"] = ticker
                filters.append(entity)

        graphql_query = text_to_graphql(question, filters, debug)
        response, succeeded = run_graphql_query_against_weaviate_instance(graphql_query)
        if not succeeded:
            print(f"[FAILED] run_graphql_query_against_weaviate_instance")
            results["Context"].append(
                {
                    "Failed to retrieve records for quantiative vector search for the user query: {question}"
                }
            )
            return results
        
        qual_and_quant_df = pd.DataFrame(response["data"]['Get'])
        if len(qual_and_quant_df) > 0:
            qual_and_quant_df = pd.DataFrame(response["data"]['Get']["Dow30_10K_10Q"])
            qual_and_quant_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True)
            qual_and_quant_df["temp_date_sort_key"] =  pd.to_datetime(qual_and_quant_df["report_date"])
            qual_and_quant_df.sort_values(by=["temp_date_sort_key"], ascending=False, inplace=True)
            qual_and_quant_df.drop("temp_date_sort_key", axis=1, inplace=True)
            qual_and_quant_df.rename({"filing_url": "url"}, axis=1, inplace=True)
            qual_and_quant_df.reset_index(drop=True, inplace=True)

            qual_and_quant_df = get_sentiment(qual_and_quant_df)
            print(f"qual_and_quant_df:\n{qual_and_quant_df.head()}\n\n")

            if_should_do_calculate_for_qual_and_quant = should_do_calculate_for_qual_and_quant(question)
            
            if if_should_do_calculate_for_qual_and_quant:
                qual_and_quant_df = do_calculate_for_qual_and_quant(question, qual_and_quant_df)
            
            # if "text" in qual_and_quant_df.columns:
            #     qual_and_quant_df.drop(columns=["text"], inplace=True)

            results["Context"].append(
                f"Qualitative and Quantitative results for query (ticker={ticker}): {question} \n\n{qual_and_quant_df.to_json()}"
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

                qual_and_quant_df.rename({'report_date': merge_key}, axis=1, inplace=True)
                qual_and_quant_df[merge_key] = pd.to_datetime(qual_and_quant_df[merge_key].str[:10]).dt.strftime('%Y-%m-%d')
                qual_and_quant_df["temp_date_sort_key"] = pd.to_datetime(qual_and_quant_df[merge_key])
                qual_and_quant_df.sort_values(by=["temp_date_sort_key"], ascending=False, inplace=True)
                qual_and_quant_df.drop("temp_date_sort_key", axis=1, inplace=True)
                qual_and_quant_df.reset_index(drop=True, inplace=True)

                qual_and_quant_df = qual_and_quant_df.merge(company_financials_df, left_on=merge_key, right_on=merge_key)
                results['QualAndQuant'][ticker.upper()] = qual_and_quant_df
                del results['GetCompanyFinancials'][ticker.upper()]

                results["finalAnalysis"]["citations"].extend(citations)
                results["finalAnalysis"]["tables"][ticker.upper()] = qual_and_quant_df
                return results

            else:
                results["QualAndQuant"][ticker.upper()] = df_citations
                results["finalAnalysis"]["citations"].extend(citations)
                results["finalAnalysis"]["tables"][ticker.upper()] = qual_and_quant_df
                
                response = '\n\n'.join([c["text"] for c in citations])
                results["Context"].append(
                    f"Response to Query: {question} \n\n{response}"
                ) 
                return results

            if debug:
                with open(DEBUG_ABS_FILE_PATH, "a") as f:
                    f.write(json.dumps({"function": "perform_quantitative_vector_search", "inputs": [question], "outputs": [{"response": response}]}, indent=6))
        
        else:
            results["Context"].append(
                f"Response to Query (retrieved no results): {question} \n\n{response}"
            )
        
        return results
    except Exception as e:
        print(f"Error inside perform_quantitative_vector_search: {e}")
        return results


    


# def perform_quantitative_vector_search(question, results, debug=False):
    
#     # NOTE: BELOW IS THE CASE WHERE perform_quantitative_vector_search IS CALLED FIRST! (THEREFORE CHECK FOR GET_FINANCIALS (BY TICKER) IN RESULTS ABOVE)
#     try:    
#         entities = extract_entities_from_user_query(question, debug)
#         # import pdb; pdb.set_trace()
#         now = (datetime.now() + timedelta(days=-1)).date()
#         to_date = f"{now.year}-{now.month}-{now.day}"
#         tickers = []
#         for entity in entities:
#             if entity["entity"] == "from_date":
#                 from_date = entity["value"]
#             elif entity["entity"] == "to_date":
#                 to_date = entity["value"]
#             elif entity["entity"] == "ticker":
#                 tickers.append(entity["value"].upper())

#         # import pdb; pdb.set_trace()
        
#         ticker = tickers[0]
#         graphql_query = text_to_graphql(question, entities, debug)
#         response = run_graphql_query_against_weaviate_instance(graphql_query)
        

#         qual_and_quant_df = pd.DataFrame(response["data"]['Get'])

#         if len(qual_and_quant_df) > 0:
#             qual_and_quant_df = pd.DataFrame(response["data"]['Get']["Dow30_10K_10Q"])
#             if_should_do_calculate_for_qual_and_quant = should_do_calculate_for_qual_and_quant(question)
            
#             if if_should_do_calculate_for_qual_and_quant:
#                 qual_and_quant_df = do_calculate_for_qual_and_quant(question, qual_and_quant_df)
            
#             if "text" in qual_and_quant_df.columns:
#                 qual_and_quant_df.drop(columns=["text"], inplace=True)
#             # qual_and_quant_df.set_index("report_date", inplace=True)
#             # qual_and_quant_df.sort_index(inplace=True, ascending=False)

            
#             # import pdb; pdb.set_trace()
#             results["Context"].append(
#                 f"Qualitative and Quantitative results for query (ticker={ticker}): {question} \n\n{qual_and_quant_df.to_json()}"
#             )

#             # citations = [{"text": t["text"], "id": "", "logo": "", "page_number": t["page_number"], "url": t["filing_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0 - float(t["_additional"]["distance"])} for t in results["data"]["Get"]["Dow30_10K_10Q"]]
#             citations = [{"text": t["text"], "id": "", "logo": "", "page_number": t["page_number"], "url": t["filing_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0} for t in response["data"]["Get"]["Dow30_10K_10Q"]]
#             citations_for_backend = [{"report_date": f'{t["report_date"].split("T")[0]}', "text": t["text"], "id": "", "logo": "", "page_number": t["page_number"], "url": t["filing_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0} for t in response["data"]["Get"]["Dow30_10K_10Q"]]
#             citations = sorted(citations, key=lambda d: d['importance'], reverse=True)


#             qual_and_quant_df.drop_duplicates(subset=['filing_url', 'page_number'], inplace=True)
#             df_citations_frontend = pd.DataFrame.from_records(citations)
#             df_citations_frontend.drop_duplicates(subset=['url', 'page_number'], inplace=True)
#             df_citations_frontend.reset_index(drop=True, inplace=True)
#             citations = df_citations_frontend.to_dict(orient='records')

#             df_citations = pd.DataFrame.from_records(citations_for_backend)
#             df_citations.drop_duplicates(subset=['url', 'page_number'], inplace=True)

#             # import pdb; pdb.set_trace()            
#             # citations = add_highlighting_to_citations_pdfs(citations[:10])
            
#             if "citations" not in results["finalAnalysis"]:
#                 results["finalAnalysis"]["citations"] = []    
#             results["finalAnalysis"]["citations"].extend([citations])

#             if "insights" not in results["finalAnalysis"]:
#                 results["finalAnalysis"]["insights"] = [] 

            
#             df_citations["temp_date_sort_key"] =  pd.to_datetime(df_citations["report_date"])
#             df_citations.sort_values(by=["temp_date_sort_key"], ascending=False, inplace=True)
#             df_citations.drop("temp_date_sort_key", axis=1, inplace=True)
#             df_citations.reset_index(drop=True, inplace=True)
#             print(f"df_citations:\n{df_citations.head()}")
#             if len(results['GetCompanyFinancials']) > 0 and ticker.upper() in results['GetCompanyFinancials']:
#                 company_financials_df = results['GetCompanyFinancials'][ticker]
                
#                 merge_key = None
#                 if 'Q' in company_financials_df['report_date'][0]:
#                     merge_key = 'Calendar Date'
#                 else:
#                     merge_key = f'report_date'

#                 qual_and_quant_df.rename({'report_date': merge_key}, axis=1, inplace=True)
#                 qual_and_quant_df[merge_key] = pd.to_datetime(qual_and_quant_df[merge_key].str[:10]).dt.strftime('%Y-%m-%d')
#                 qual_and_quant_df["temp_date_sort_key"] = pd.to_datetime(qual_and_quant_df[merge_key])
#                 qual_and_quant_df.sort_values(by=["temp_date_sort_key"], ascending=False, inplace=True)
#                 qual_and_quant_df.drop("temp_date_sort_key", axis=1, inplace=True)
#                 qual_and_quant_df.reset_index(drop=True, inplace=True)

#                 qual_and_quant_df = qual_and_quant_df.merge(company_financials_df, left_on=merge_key, right_on=merge_key)

#                 # qual_and_quant_df, company_financials_df = realign_qual_and_quant_df_to_closest_dt_from_upstream_df(qual_and_quant_df, company_financials_df, merge_key)
                
#                 # company_financials_df = company_financials_df.merge(qual_and_quant_df, left_on=merge_key, right_on=merge_key)
#                 results['QualAndQuant'][ticker.upper()] = qual_and_quant_df
#                 del results['GetCompanyFinancials'][ticker.upper()]

#                 results["finalAnalysis"]["citations"].extend(citations)
#                 results["finalAnalysis"]["tables"][ticker.upper()] = qual_and_quant_df
#                 return results

#             else:
#                 results["QualAndQuant"][ticker.upper()] = df_citations
                
#                 # citations = add_highlighting_to_citations_pdfs(citations[:10])

#                 results["finalAnalysis"]["citations"].extend(citations)

                
#                 # NOTE: fix the ticker to get from the entities
#                 # ticker  = [e for e in entities if e["entity"] == "ticker"][0]["value"]
#                 results["finalAnalysis"]["tables"][ticker.upper()] = qual_and_quant_df
#                 # results["finalAnalysis"]["tables"]["MSFT"]= qual_and_quant_df
                
#                 response = '\n\n'.join([c["text"] for c in citations])
#                 results["Context"].append(
#                     f"Response to Query: {question} \n\n{response}"
#                 ) 
#                 # print(f"results from vector search: {results}")
#                 return results

#             if debug:
#                 with open(DEBUG_ABS_FILE_PATH, "a") as f:
#                     f.write(json.dumps({"function": "perform_quantitative_vector_search", "inputs": [question], "outputs": [{"response": response}]}, indent=6))
        
#         else:
#             results["Context"].append(
#                 f"Response to Query (retrieved no results): {question} \n\n{response}"
#             ) 

        
#         return results
#     except Exception as e:
#         print(f"Error inside perform_quantitative_vector_search: {e}")
#         return results
    

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

# def perform_vector_search(question, results, debug=False):
#     try:
#         entities = extract_entities_from_user_query(question, debug)
#         filters = []
#         to_date = None
#         from_date = None
#         ticker = None
#         for entity in entities:
#             if entity["entity"] == "from_date":
#                 filters.append(entity)
#             elif entity["entity"] == "to_date":
#                 to_date = entity["value"]
#                 filters.append(entity)
#             elif entity["entity"] == "ticker":
#                 ticker = entity["value"].upper()
#                 entity["value"] = ticker
#                 filters.append(entity)

#         import pdb; pdb.set_trace()
                
#         response = (
#             weaviate_client.query
#             .get("Dow30_10K_10Q", ["filing_type", "logo", "company_name", "ticker", "accession_number", "filing_url", "text", "page_number", "report_date"])
#             .with_near_text({
#                 "concepts": [question]
#             })
#             .with_where({
#                 "operator": "And",
#                 "operands": filters
#             })
#             .with_additional(["distance"])
#             .do()
#         )

#         if response.status_code != 200:
#             print(f"[FAILED] to get response for weaviate query in perform_vector_search")
#             results["Context"].append(
#                 f"I failed to retrieve documents from my vector search for the query: {question}"
#             )
#             return results

#         results_df = pd.DataFrame(response["data"]['Get'])

#         if len(qual_and_quant_df) > 0:
#             results_df = pd.DataFrame(response["data"]['Get']["Dow30_10K_10Q"])
#             results_df.drop_duplicates(subset=['accession_number', 'page_number'], inplace=True)
#             results_df["temp_date_sort_key"] =  pd.to_datetime(results_df["report_date"])
#             results_df.sort_values(by=["temp_date_sort_key"], ascending=False, inplace=True)
#             results_df.drop("temp_date_sort_key", axis=1, inplace=True)
#             results_df.rename({"filing_url": "url"}, axis=1, inplace=True)
#             results_df.reset_index(drop=True, inplace=True)

#             if_should_do_calculate_for_qual_and_quant = should_do_calculate_for_qual_and_quant(question)
            
#             if if_should_do_calculate_for_qual_and_quant:
#                 results_df = do_calculate_for_qual_and_quant(question, results_df)
            
#             if "text" in qual_and_quant_df.columns:
#                 results_df.drop(columns=["text"], inplace=True)
#             # qual_and_quant_df.set_index("report_date", inplace=True)
#             # qual_and_quant_df.sort_index(inplace=True, ascending=False)

            
#             # import pdb; pdb.set_trace()
#             results["Context"].append(
#                 f"Database results for query (ticker={ticker}): {question} \n\n{results_df.to_json()}"
#             )

#             # citations = [{"text": t["text"], "id": "", "logo": "", "page_number": t["page_number"], "url": t["filing_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0 - float(t["_additional"]["distance"])} for t in results["data"]["Get"]["Dow30_10K_10Q"]]
#             citations = [{"text": t["text"], "id": "", "logo": "", "page_number": t["page_number"], "url": t["filing_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0} for t in response["data"]["Get"]["Dow30_10K_10Q"]]
#             citations_for_backend = [{"report_date": f'{t["report_date"].split("T")[0]}', "text": t["text"], "id": "", "logo": "", "page_number": t["page_number"], "url": t["filing_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0} for t in response["data"]["Get"]["Dow30_10K_10Q"]]
#             citations = sorted(citations, key=lambda d: d['importance'], reverse=True)


            
#             df_citations_frontend = pd.DataFrame.from_records(citations)
#             citations = df_citations_frontend.to_dict(orient='records')

#             df_citations = pd.DataFrame.from_records(citations_for_backend)
#             # df_citations.drop_duplicates(subset=['url', 'page_number'], inplace=True)

#             # import pdb; pdb.set_trace()            
#             # citations = add_highlighting_to_citations_pdfs(citations[:10])
            
#             if "citations" not in results["finalAnalysis"]:
#                 results["finalAnalysis"]["citations"] = []    
#             results["finalAnalysis"]["citations"].extend(citations)

#             if "insights" not in results["finalAnalysis"]:
#                 results["finalAnalysis"]["insights"] = [] 

#             if len(results['GetCompanyFinancials']) > 0 and ticker.upper() in results['GetCompanyFinancials']:
#                 company_financials_df = results['GetCompanyFinancials'][ticker]
                
#                 merge_key = None
#                 if 'Q' in company_financials_df['report_date'][0]:
#                     merge_key = 'Calendar Date'
#                 else:
#                     merge_key = f'report_date'

#                 results_df.rename({'report_date': merge_key}, axis=1, inplace=True)
#                 results_df[merge_key] = pd.to_datetime(results_df[merge_key].str[:10]).dt.strftime('%Y-%m-%d')
#                 results_df["temp_date_sort_key"] = pd.to_datetime(results_df[merge_key])
#                 results_df.sort_values(by=["temp_date_sort_key"], ascending=False, inplace=True)
#                 results_df.drop("temp_date_sort_key", axis=1, inplace=True)
#                 results_df.reset_index(drop=True, inplace=True)

#                 results_df = results_df.merge(company_financials_df, left_on=merge_key, right_on=merge_key)

#                 results['VectorSearch'][ticker.upper()] = results_df
#                 del results['GetCompanyFinancials'][ticker.upper()]

#                 results["finalAnalysis"]["citations"].extend(citations)
#                 results["finalAnalysis"]["tables"][ticker.upper()] = results_df
#                 return results

#             else:
#                 results["VectorSearch"][ticker.upper()] = results_df
                
#                 # citations = add_highlighting_to_citations_pdfs(citations[:10])

#                 results["finalAnalysis"]["citations"].extend(citations)

                
#                 # NOTE: fix the ticker to get from the entities
#                 # ticker  = [e for e in entities if e["entity"] == "ticker"][0]["value"]
#                 results["finalAnalysis"]["tables"][ticker.upper()] = results_df
#                 # results["finalAnalysis"]["tables"]["MSFT"]= qual_and_quant_df
                
#                 response = '\n\n'.join([c["text"] for c in citations])
#                 results["Context"].append(
#                     f"Response to Query: {question} \n\n{response}"
#                 ) 
#                 # print(f"results from vector search: {results}")
#                 return results

#             if debug:
#                 with open(DEBUG_ABS_FILE_PATH, "a") as f:
#                     f.write(json.dumps({"function": "perform_vector_search", "inputs": [question], "outputs": [{"response": response}]}, indent=6))
        
#         else:
#             results["Context"].append(
#                 f"Response to Query (retrieved no results): {question} \n\n{response}"
#             )
        
#         return results
#     except Exception as e:
#         print(f"Error inside perform_vector_search: {e}")
#         return results


# def perform_vector_search(question, results, debug=False):
#     # import pdb; pdb.set_trace()
#     try:
#         entities = extract_entities_from_user_query(question, debug)
#         filters = []
#         # now = (datetime.now() + timedelta(days=-1)).date()
#         to_date = None
#         from_date = None
#         ticker = None
#         for entity in entities:
#             if entity["entity"] == "from_date":
#                 filters.append(entity)
#             elif entity["entity"] == "to_date":
#                 to_date = entity["value"]
#                 filters.append(entity)
#             elif entity["entity"] == "ticker":
#                 ticker = entity["value"].upper()
#                 entity["value"] = ticker
#                 filters.append(entity)
                
#         response = (
#             weaviate_client.query
#             .get("Dow30_10K_10Q", ["filing_type", "logo", "company_name", "ticker", "accession_number", "filing_url", "text", "page_number", "report_date"])
#             .with_near_text({
#                 "concepts": [question]
#             })
#             .with_where({
#                 "operator": "And",
#                 "operands": filters
#             })
#             .with_additional(["distance"])
#             .do()
#         )

#         if response.status_code != 200:
#             print(f"[FAILED] to get response for weaviate query in perform_vector_search")
#             results["Context"].append(
#                 f"I failed to retrieve documents from my vector search for the query: {question}"
#             )
#             return results

#         results_df = pd.DataFrame(response["data"]['Get'])

#         citations = [{"text": t["text"], "id": "", "logo": t["logo"], "page_number": t["page_number"], "url": t["filing_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0 - float(t["_additional"]["distance"])} for t in response["data"]["Get"]["Dow30_10K_10Q"]]
#         citations_for_backend = [{"report_date": f'{t["report_date"].split("T")[0]}', "text": t["text"], "id": "", "logo": "", "page_number": t["page_number"], "url": t["filing_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0 - float(t["_additional"]["distance"])} for t in response["data"]["Get"]["Dow30_10K_10Q"]]
#         if len(citations_for_backend) == 0:
#             return results
#         citations = sorted(citations, key=lambda d: d['importance'], reverse=True)
#         df_citations_frontend = pd.DataFrame.from_records(citations)
#         df_citations_frontend.drop_duplicates(subset=['title', 'page_number'], inplace=True)
#         df_citations_frontend.reset_index(drop=True, inplace=True)
#         citations = df_citations_frontend.to_dict(orient='records')

#         df_citations = pd.DataFrame.from_records(citations_for_backend)
#         df_citations.drop_duplicates(subset=['title', 'page_number'], inplace=True)

       
        
#         # citations = add_highlighting_to_citations_pdfs(citations[:10])
        
#         if "citations" not in results["finalAnalysis"]:
#             results["finalAnalysis"]["citations"] = []    
#         results["finalAnalysis"]["citations"].extend(citations)

#         if "insights" not in results["finalAnalysis"]:
#             results["finalAnalysis"]["insights"] = [] 

#         # import pdb; pdb.set_trace()
#         df_citations["temp_date_sort_key"] = pd.to_datetime(df_citations["report_date"])
#         df_citations.sort_values(by=["temp_date_sort_key"], ascending=False, inplace=True)
#         df_citations.drop("temp_date_sort_key", axis=1, inplace=True)
#         df_citations.reset_index(drop=True, inplace=True)
#         print(f"df_citations:\n{df_citations.head()}")
        
#         if len(results['GetCompanyFinancials']) > 0 and ticker.upper() in results['GetCompanyFinancials']:
#             company_financials_df = results['GetCompanyFinancials'][ticker]
            
#             merge_key = None
#             if 'Q' in company_financials_df['report_date'][0]:
#                 merge_key = 'Calendar Date'
#             else:
#                 merge_key = f'report_date'

#             df_citations.rename({'report_date': merge_key}, axis=1, inplace=True)
#             df_citations[merge_key] = pd.to_datetime(df_citations[merge_key].str[:10]).dt.strftime('%Y-%m-%d')
#             df_citations["temp_date_sort_key"] = pd.to_datetime(df_citations[merge_key])
#             df_citations.sort_values(by=["temp_date_sort_key"], ascending=False, inplace=True)
#             df_citations.drop("temp_date_sort_key", axis=1, inplace=True)
#             df_citations.reset_index(drop=True, inplace=True)

#             df_citations = df_citations.merge(company_financials_df, left_on=merge_key, right_on=merge_key)

#             # qual_and_quant_df, company_financials_df = realign_qual_and_quant_df_to_closest_dt_from_upstream_df(qual_and_quant_df, company_financials_df, merge_key)
#             # import pdb; pdb.set_trace()
#             # company_financials_df = company_financials_df.merge(qual_and_quant_df, left_on=merge_key, right_on=merge_key)
#             results['VectorSearch'][ticker.upper()] = df_citations
#             del results['GetCompanyFinancials'][ticker.upper()]

#             results["finalAnalysis"]["citations"].extend(citations)
#             results["finalAnalysis"]["tables"][ticker.upper()] = df_citations
#             return results
        
#         else:
#             results["VectorSearch"][ticker.upper()] = df_citations
            
            
#             response = '\n\n'.join([c["text"] for c in citations])
#             results["Context"].append(
#                 f"Response to Query: {question} \n\n{response}"
#             ) 
#             # print(f"results from vector search: {results}")
#             return results

#         if debug:
#             with open(DEBUG_ABS_FILE_PATH, "a") as f:
#                 f.write(json.dumps({"function": "perform_vector_search", "inputs": [question], "outputs": [{"response": response}]}, indent=6))
#         return results
#     except Exception as e:
#         print(f"Error inside perform_vector_search: {e}")
#         return results

# def perform_vector_search(question, results, debug=False):
#     entities = extract_entities_from_user_query(question, debug)

#     graphql_query = text_to_graphql(question, entities, debug)
#     results = run_graphql_query_against_weaviate_instance(graphql_query)

#     import pdb; pdb.set_trace()
#     ticker = None
#     filters = []
#     for ent in entities:
#         if ent["entity"] == "ticker":
#             ticker = ent["value"]
#             filters.append({
#                 "path": ["ticker"],
#                 "operator": "Equal",
#                 "valueText": ticker.upper()
#             })
#         elif ent["entity"] == "from_date":
#             filters.append({
#                 "path": ["report_date"],
#                 "operator": "GreaterThanEqual",
#                 "valueDate": f'{str(parser.parse(ent["value"])).split(" ")[0]}T00:00:00.000Z'
#             })
#         elif ent["entity"] == "to_date":
#             filters.append({
#                 "path": ["report_date"],
#                 "operator": "LessThanEqual",
#                 "valueDate": f'{str(parser.parse(ent["value"])).split(" ")[0]}T00:00:00.000Z'
#             })
            

#     response = (
#         weaviate_client.query
#         .get("Dow30_10K_10Q", ["filing_type", "company_name", "ticker", "accession_number", "filing_url", "text", "page_number", "report_date"])
#         .with_near_text({
#             "concepts": [question]
#         })
#         .with_where({
#             "operator": "And",
#             "operands": filters
#         })
#         .with_additional(["distance"])
#         .do()
#     )

#     if "finalAnalysis" not in results:
#         results["finalAnalysis"] = {

#         }

#     # import pdb; pdb.set_trace()

#     if 'errors' in response:
#         results["Context"].append(
#             f"I failed to retrieve documents from my vector search for the query: {question}"
#         )
#         return results

#     citations = [{"text": t["text"], "id": "", "logo": "", "page_number": t["page_number"], "url": t["filing_url"], "title": f'{t["company_name"]} {t["filing_type"]} {t["report_date"].split("T")[0]}' , "company": t["company_name"], "importance": 1.0 - float(t["_additional"]["distance"])} for t in response["data"]["Get"]["Dow30_10K_10Q"]]
#     citations = sorted(citations, key=lambda d: d['importance'], reverse=True)
    
#     citations = add_highlighting_to_citations_pdfs(citations[:10])
    
#     if "citations" not in results["finalAnalysis"]:
#         results["finalAnalysis"]["citations"] = []    
#     results["finalAnalysis"]["citations"].extend(citations)

#     if "insights" not in results["finalAnalysis"]:
#         results["finalAnalysis"]["insights"] = [] 
    
#     response = '\n\n'.join([c["text"] for c in citations])
#     results["Context"].append(
#         f"Response to Query: {question} \n\n{response}"
#     ) 
#     print(f"results from vector search: {results}")

#     if debug:
#         with open(DEBUG_ABS_FILE_PATH, "a") as f:
#             f.write(json.dumps({"function": "perform_vector_search", "inputs": [question], "outputs": [{"response": response}]}, indent=6))
#     return results


# def get_financials(question, results, debug=False):
#     try:
#         entities = extract_entities_from_user_query(question, debug)
#         tickers = []
#         for ent in entities:
#             if ent["entity"] == "ticker":
#                 tickers.append(ent["value"])

#         for ticker in tickers:
#             if "processed_tickers" not in results:
#                 results["processed_tickers"] = []

#             if ticker.upper() in results["processed_tickers"]:
#                 continue
#             financials = get_stock_financials(ticker.upper())
#             if financials is None or len(financials) == 0:
#                 continue
#             temp_financials = financials.copy()
#             # import pdb; pdb.set_trace()
#             relevant_rows = get_relevant_rows(list(temp_financials.index.values), question)
#             print(f"relevant_rows: {relevant_rows}")
#             relevant_columns = get_relevant_columns(list(temp_financials.columns), question)
#             print(f"relevant_columns: {relevant_columns}")

            
#             filtered_columns = []
#             filtered_rows = []
#             for col in relevant_columns:
#                 if col in temp_financials.columns:
#                     filtered_columns.append(col)

#             for row in relevant_rows:
#                 if row in temp_financials.index.values:
#                     filtered_rows.append(row)

#             temp_financials = temp_financials.loc[filtered_rows, filtered_columns]
#             # temp_financials = temp_financials.dropna()

#             calculation_required = should_local_calculate(question, list(financials.index.values))
#             print(f"calculation_required: {calculation_required}")
#             if calculation_required:
#                 temp_financials = do_local_calculate(question, temp_financials.T)
#                 print(f"temp_financials after do_local_calculate: {temp_financials}")
#                 if len(temp_financials) == 0:
#                     continue


#             # import pdb; pdb.set_trace()

#             # temp_financials = temp_financials.dropna()
#             temp_financials = temp_financials.rename(columns={c: f"{ticker.upper()}.{c}" for c in temp_financials.columns})
#             # chart_data = temp_financials.to_dict('records')
#             chart_data = temp_financials.T.to_json()

#             results["Context"].append(
#                 f"{temp_financials.T.columns[0]} results for {ticker.upper()}: {temp_financials.T.to_json()}"
#             )

#             if "ResultsFromGetFinancials" not in results:
#                 results["ResultsFromGetFinancials"] = {

#                 }

#             results["ResultsFromGetFinancials"][ticker.upper()] = temp_financials.to_json()
#             results["processed_tickers"].append(ticker.upper())

#             if "finalAnalysis" not in results:
#                 results["finalAnalysis"] = {

#                 }

#             if "charts" not in results["finalAnalysis"]:
#                 results["finalAnalysis"]["charts"] = {

#                 }

#             if "tables" not in results["finalAnalysis"]:
#                 results["finalAnalysis"]["tables"] = {

#                 }

#             temp_financials[f"Date_{ticker}"] = temp_financials.index.values
#             table =  {
#                     "headers": list(temp_financials.columns),
#                     "rows": temp_financials.values.tolist()
#             }

#             results["finalAnalysis"]["charts"][ticker.upper()] = temp_financials
#             results["finalAnalysis"]["tables"][ticker.upper()]= temp_financials
#             if "workbookData" not in results:
#                 results["workbookData"] = {
#                 }

#             # import pdb; pdb.set_trace()
#             results["workbookData"][ticker.upper()] = table

#         if debug:
#             with open(DEBUG_ABS_FILE_PATH, "a") as f:
#                 f.write(json.dumps({"function": "get_financials", "inputs": [question], "outputs": [{"financials": temp_financials.to_json()}]}, indent=6))
#         return results
#     except Exception as e:
#         print(f"Error inside get_financials: {e}")
#         return results


def get_fiscal_or_calendar_by_user_query(question):
    """
    Prompt ChatGPT to identify the relevant rows from the financial data.

    :param financials: JSON response containing financial data
    :param question: User's question about the financial data
    :return: List of relevant row indices
    """

    system_prompt = """
    You are a helpful assistant that analyzes financial data. Given a user question, determine whether 'fiscal' or 'calendar' dates should be used to answer the question. Unless other explicitly states via reference to 'fiscal year', 'FY', etc. 
    the response should default to 'calendar'. Note that your response should be a JSON list containing only one element. The one element in your JSON list response should be either the string value 'fiscal' or 'calendar'. Your response should contain nothing else
    other than the JSON list. See the examples below:

    Examples: 
        query: "How has apple's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if i bought the stock each time?", answer: ['calendar']    
        query: "compare the cogs between 2021 and 2023 between AMZN and its comps", answer: ['calendar']
        query: "investigate the trends in JP Morgan's net interest margin and loan growth over the past 2 years. How have changes in interest rates and economic conditions influenced their profitability?", answer: ['calendar']
        query: "what's amzn's cogs from 2022 to 2023?", answer: ['calendar']
        query: "what was the total amount spent on acquisitions net of cash acquired and purchases of intangible and other assets by msft in the year 2022?", answer: ['calendar']
        query: "What was the trend in MSFT's Payments for Repurchase of Common Stock from Q3 2020 to Q1 2023?", answer: ['fiscal']
        query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", answer: ['calendar']
        query: "How has amzn's cogs grown annually quarter over quarter over 2023 vs 2022?", answer: ['calendar']
        query: "What’s apple’s revenue growth for the past 2 years? Compute the correlation between revenue growth and its margin growth", answer: ['calendar']
        query: "What’s apple’s revenue growth for the past 2 years?", answer: ['calendar']
        query: "How has apple's reported revenue growth trended when they mentioned macro concerns in the same filing? Would I have made money if i bought the stock each time?", answer: ['calendar']
        query: "You are an investment banker and your only resource(s) to answer the following question is (are): the statement of financial position and the cash flow statement. Here's the question: what is the FY2015 operating cash flow ratio for Adobe? Operating cash flow ratio is defined as: cash from operations / total current liabilities. Round your answer to two decimal places.", answer: ['fiscal']
        query: "What are the debt levels of MSFT?", answer: ['calendar']
        query: "What are the debt levels of AAPL?", answer: ['calendar']
    """

    prompt = """
    query: {question}

    Please identify the rows that are necessary to answer the question.
    Return your answer as a JSON list that contains only the row names.
    """

    # Call the OpenAI API
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question)}
        ],
    )

    try:
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        fiscal_or_calendar = ast.literal_eval(response[response.find("["):response.find("]")+1])
        return fiscal_or_calendar[0]
    
    except Exception as e:
        print(f"Error parsing ChatGPT response: {e}")
        return []


def get_stock_financials(ticker, mode='calendar', limit=5, debug=False):
    statements=["income-statement", "balance-sheet-statement", "cash-flow-statement"]
    financial_datas = []
    for statement in statements:
        try:
            response = requests.get(f"https://financialmodelingprep.com/api/v3/{statement}/{ticker}?period=quarterly&apikey={FMP_API_KEY}")
            financial_datas.append(pd.DataFrame.from_records(response.json()[:75]))
        except RuntimeError as e:
            print(f"Exception caught in get_financials: {e}")

    result_df = pd.concat([financial_datas[0], financial_datas[1], financial_datas[2]], axis=1)
    result_df = result_df.loc[:,~result_df.columns.duplicated()]
    result_df.rename({"date": "report_date"}, axis=1, inplace=True)
    result_df["report_date"] = pd.to_datetime(result_df["report_date"])
    result_df.sort_values(by=["report_date"], inplace=True, ascending=False)
    result_df['report_date'] = result_df['report_date'].dt.strftime('%Y-%m-%d')
    result_df["Fiscal Date"] = result_df["period"].astype(str) + " " + result_df["calendarYear"].astype(str)
    result_df["Calendar Date"] = result_df["report_date"]
    result_df.drop("report_date", axis=1, inplace=True)
    print(f"mode: {mode}")
    if mode == "calendar":
        result_df.rename({"Calendar Date": "report_date"}, axis=1, inplace=True)
        result_df.set_index("report_date", inplace=True)
    else:
        result_df.rename({"Fiscal Date": "report_date"}, axis=1, inplace=True)
        result_df.set_index("report_date", inplace=True)



    
    result_df.drop(['symbol', 'calendarYear', 'reportedCurrency', 'cik', 'fillingDate', 'acceptedDate', 'period'], axis=1, inplace=True)
    return result_df



def should_do_calculate_for_qual_and_quant(question, debug=False):
    system_prompt = """ 
    You are a tool used to guide a RAG system. Your job is to determine where a calculator function needs to be called or not.
    Your response should be a string that is either 'True' or 'False'. Your response should not contain anything other than either 
    'True' or 'False'. Given a user query and a supplied json contain data used to answer the user determine wheter a calculation needs
    to be executed or if the current raw json data is sufficient to answer the quesiton and thus doesn't require any further calculations.
    Below are some examples of user query, json data, and response triplets.

    Examples:
    user query: "If I had bought AAPL stock every time they mentioned supply chain concerns in their filing if I had held it over the following quarter?", answer: "False"
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


# def get_financials(question, results, debug=False):
#     try:
#         entities = extract_entities_from_user_query(question, debug)
#         ticker = None
#         for ent in entities:
#             if ent["entity"] == "ticker":
#                 ticker = ent["value"]
#                 break

#         fiscal_or_calendar = get_fiscal_or_calendar_by_user_query(question)
#         financials = get_stock_financials(ticker.upper(), mode=fiscal_or_calendar)
#         if financials is None or len(financials) == 0:
#             print(f"[ERROR] Empty financials df retrieved from get_stock_financials!")
#             return results

#         temp_financials = financials.copy()
#         temp_financials.reset_index(inplace=True)
#         if "Q" in temp_financials['report_date'][0] or "FY" in temp_financials['report_date'][0]:
#             sort_key = "Calendar Date"
#         else:
#             sort_key = "report_date"


#         temp_financials["temp_sort_key"] = pd.to_datetime(temp_financials[sort_key])
#         temp_financials.sort_values(by=["temp_sort_key"], ascending=True)
#         temp_financials.drop("temp_sort_key", axis=1, inplace=True)
        
#         # import pdb; pdb.set_trace()
#         financial_field_columns = [c for c in temp_financials.columns if c not in ['Calendar Date', 'Fiscal Date', 'report_date']]
#         report_date_values = list(temp_financials['report_date'].values)
#         relevant_rows = get_relevant_rows(financial_field_columns, question)
#         relevant_rows = list(set([r for r in relevant_rows if r in  financial_field_columns]))

#         if "Q" in temp_financials['report_date'][0] or "FY" in temp_financials['report_date'][0]:
#             other_date_column_to_keep = "Calendar Date"
#             relevant_columns = get_relevant_fiscal_columns(report_date_values, question)
#             relevant_columns = [c for c in relevant_columns if c in report_date_values]
#             relevant_columns = list(set(relevant_columns))
#         else:
#             other_date_column_to_keep = "Fiscal Date"
#             relevant_columns = get_relevant_calendar_columns(report_date_values, question)
#             relevant_columns = [c for c in relevant_columns if c in report_date_values]
#             relevant_columns = list(set(relevant_columns))
        
#         temp_financials = temp_financials[temp_financials['report_date'].isin(relevant_columns)][relevant_rows + [other_date_column_to_keep, "report_date"]]
#         calculation_required = should_local_calculate(question, financial_field_columns)
#         print(f"calculation_required: {calculation_required}")
#         called_calc = False
#         if calculation_required:
#             temp_financials = do_local_calculate(question, temp_financials)
#             print(f"temp_financials after do_local_calculate: {temp_financials}")
#             if len(temp_financials) == 0:
#                 print(f"[ERROR] Got back Empty temp_financials df from do_local_calculate!")
#                 return results

#         temp_financials.reset_index(drop=True,inplace=True)
#         temp_financials['Company'] = [ticker.upper()]*len(temp_financials)
#         # temp_financials.reset_index(drop=True, inplace=True)
#         # temp_financials.set_index(pd.RangeIndex(len(temp_financials)), inplace=True)
#         results["GetCompanyFinancials"][ticker.upper()] = temp_financials
#         chart_data = temp_financials.to_json()

#         results["Context"].append(
#             f"{temp_financials.T.columns[0]} results for {ticker.upper()}: {temp_financials.to_json()}"
#         )

#         return results
#     except Exception as e:
#         import pdb; pdb.set_trace()
#         print(f"Error inside get_financials: {e}")
#         return results

def get_financials(question, results, debug=False):
    try:
        entities = extract_entities_from_user_query(question, debug)
        ticker = None
        for ent in entities:
            if ent["entity"] == "ticker":
                ticker = ent["value"]
                break

        fiscal_or_calendar = get_fiscal_or_calendar_by_user_query(question)
        financials = get_stock_financials(ticker.upper(), mode=fiscal_or_calendar)
        if financials is None or len(financials) == 0:
            print(f"[ERROR] Empty financials df retrieved from get_stock_financials!")
            return results

        temp_financials = financials.copy()
        temp_financials.reset_index(inplace=True)
        if "Q" in temp_financials['report_date'][0] or "FY" in temp_financials['report_date'][0]:
            sort_key = "Calendar Date"
        else:
            sort_key = "report_date"


        temp_financials["temp_sort_key"] = pd.to_datetime(temp_financials[sort_key])
        temp_financials.sort_values(by=["temp_sort_key"], ascending=True)
        temp_financials.drop("temp_sort_key", axis=1, inplace=True)
        
        
        # import pdb; pdb.set_trace()
        financial_field_columns = [c for c in temp_financials.columns if c not in ['Calendar Date', 'Fiscal Date', 'report_date']]
        report_date_values = list(temp_financials['report_date'].values)
        relevant_rows = get_relevant_rows(financial_field_columns, question)
        relevant_rows = list(set([r for r in relevant_rows if r in  financial_field_columns]))

        if "Q" in temp_financials['report_date'][0] or "FY" in temp_financials['report_date'][0]:
            other_date_column_to_keep = "Calendar Date"
            relevant_columns = get_relevant_fiscal_columns(report_date_values, question)
            relevant_columns = [c for c in relevant_columns if c in report_date_values]
            relevant_columns = list(set(relevant_columns))
        else:
            other_date_column_to_keep = "Fiscal Date"
            relevant_columns = get_relevant_calendar_columns(report_date_values, question)
            relevant_columns = [c for c in relevant_columns if c in report_date_values]
            relevant_columns = list(set(relevant_columns))
        
        temp_financials = temp_financials[temp_financials['report_date'].isin(relevant_columns)][relevant_rows + [other_date_column_to_keep, "report_date"]]

        calculation_required = should_local_calculate(question, financial_field_columns)
        print(f"calculation_required: {calculation_required}")
        called_calc = False
        if calculation_required:
            temp_financials = do_local_calculate(question, temp_financials)
            print(f"temp_financials after do_local_calculate: {temp_financials}")
            if len(temp_financials) == 0:
                print(f"[ERROR] Got back Empty temp_financials df from do_local_calculate!")
                return results

        temp_financials.reset_index(drop=True,inplace=True)
        temp_financials['Company'] = [ticker.upper()]*len(temp_financials)
        results["GetCompanyFinancials"][ticker.upper()] = temp_financials
        chart_data = temp_financials.to_json()

        results["Context"].append(
            f"{temp_financials.T.columns[0]} results for {ticker.upper()}: {temp_financials.to_json()}"
        )

        return results
    except Exception as e:
        # import pdb; pdb.set_trace()
        print(f"Error inside get_financials: {e}")
        return results


def merge_charts(results, debug=False):
    try:
        if len(results["finalAnalysis"]["charts"]) == 0:
            return results
        dfs = [t for k, t in results["finalAnalysis"]["charts"].items()]
        master_chart = pd.concat(dfs, axis=1)
        master_chart = master_chart.T
        # import pdb; pdb.set_trace()
        # cols_to_drop = [c for c in master_chart.columns if "Date" in c]
        # master_chart.drop(columns=cols_to_drop, inplace=True)
        master_chart.reset_index(inplace=True)
        # master_chart.rename(columns={"report_date": "Date"}, inplace=True)
        master_chart.replace(np.nan, None, inplace=True)
        chart = {
                "type": "line",
                "data": master_chart.to_dict("records"),
                "dataKeys": [c for c in list(master_chart.columns) if "date" not in c.lower()]
        }
        results["finalAnalysis"]["charts"] = {f'Chart for Query: {results["Query"]}': chart}
        return results
    except Exception as e:
        print(f"Error inside merge_charts: {e}")
        return results

# def merge_tables(results, debug=False):
#     try:
#         if len(results["finalAnalysis"]["tables"]) == 0:
#             return results

#         import pdb; pdb.set_trace()
#         # dfs = [t.reset_index().rename({"report_date": f"{k}_report_date"}, axis=1) for k, t in results["finalAnalysis"]["tables"].items()]
#         dfs = [t.reset_index().rename({"report_date": f"{k}_report_date"}, axis=1) for k, t in results["finalAnalysis"]["tables"].items()]
#         # import pdb; pdb.set_trace()
#         master_table = pd.concat(dfs)
        
#         cols_to_drop = [c for c in master_table.columns if "Date_" in c]
#         master_table.drop(columns=cols_to_drop, inplace=True)

#         # import pdb; pdb.set_trace()
#         master_table.replace(np.nan, None, inplace=True)
#         master_table.reset_index(inplace=True)
#         table = {
#             f'Table for Query: {results["Query"]}': {
#                 "headers": list(master_table.columns),
#                 "rows": master_table.values.tolist()
#             }
#         }
#         results["finalAnalysis"]["tables"] = table
#         results["finalAnalysis"]["workbookData"] = {"headers": table[f'Table for Query: {results["Query"]}']["headers"], "rows": table[f'Table for Query: {results["Query"]}']["rows"]}
#         return results
#     except Exception as e:
#         print(f"Error inside merge_tables: {e}")
#         return results

def merge_tables(results, debug=False):
    try:
        import pdb; pdb.set_trace()
        if len(results["finalAnalysis"]["tables"]) == 0:
            return results

        
        # dfs = [t.reset_index().rename({"report_date": f"{k}_report_date"}, axis=1) for k, t in results["finalAnalysis"]["tables"].items()]
        dfs = [t.reset_index().rename({"report_date": f"{k}_report_date"}, axis=1) for k, t in results["finalAnalysis"]["tables"].items()]
        # import pdb; pdb.set_trace()
        master_table = pd.concat(dfs)
        
        cols_to_drop = [c for c in master_table.columns if "Date_" in c]
        master_table.drop(columns=cols_to_drop, inplace=True)

        # import pdb; pdb.set_trace()
        master_table.replace(np.nan, None, inplace=True)
        master_table.reset_index(inplace=True)
        table = {
            f'Table for Query: {results["Query"]}': {
                "headers": list(master_table.columns),
                "rows": master_table.values.tolist()
            }
        }
        results["finalAnalysis"]["tables"] = table
        results["finalAnalysis"]["workbookData"] = {"headers": table[f'Table for Query: {results["Query"]}']["headers"], "rows": table[f'Table for Query: {results["Query"]}']["rows"]}
        return results
    except Exception as e:
        print(f"Error inside merge_tables: {e}")
        return results


def merge_frames(results, debug=False):
    try:
        # if len(results["finalAnalysis"]["tables"]) == 0:
        #     return results
        all_tickers = set()
        # if len(results["RunBackTest"]) > 0:
        #     results_sections_of_interest = ["RunBackTest"]
        # else:
        #     results_sections_of_interest = ["GetNews", "MarketData", "QualAndQuant", "GetCompanyFinancials"]

        results_sections_of_interest = ["VectorSearch", "GetNews", "MarketData", "QualAndQuant", "GetCompanyFinancials"]
        # import pdb; pdb.set_trace()
        for result_section_of_interest in  results_sections_of_interest:
            data_per_section = results[result_section_of_interest]
            if len(data_per_section) == 0:
                continue
            for ticker, data in data_per_section.items():
                all_tickers.add(ticker)


        ticker_frames_dicts = {ticker: None for ticker in all_tickers}
        for ticker in all_tickers:
            frames_by_ticker = []
            for result_section_of_interest in results_sections_of_interest:
                data_per_section = results[result_section_of_interest]
                if len(data_per_section) == 0:
                    continue
                if ticker in data_per_section:
                    frames_by_ticker.append(data_per_section[ticker])

            # import pdb; pdb.set_trace()
            ticker_frames_dicts[ticker] = pd.concat(frames_by_ticker, axis=1)

        merged_frames = []
        for ticker, frame in ticker_frames_dicts.items():
            merged_frames.append(frame)
       
        # import pdb; pdb.set_trace()
        merged_frames = pd.concat(merged_frames)
        merged_frames.replace(np.nan, None, inplace=True)
        merged_frames.replace(np.inf, None, inplace=True)
        merged_frames.replace(-np.inf, None, inplace=True)


        # import pdb; pdb.set_trace()
        # dfs = [t.reset_index().rename({"report_date": f"{k}_report_date"}, axis=1) for k, t in results["finalAnalysis"]["tables"].items()]
        # dfs = [t.reset_index().rename({"report_date": f"{k}_report_date"}, axis=1) for k, t in results["finalAnalysis"]["tables"].items()]
        # # import pdb; pdb.set_trace()
        # master_table = pd.concat(dfs)
        
        # cols_to_drop = [c for c in master_table.columns if "Date_" in c]
        # master_table.drop(columns=cols_to_drop, inplace=True)

        # # import pdb; pdb.set_trace()
        # master_table.replace(np.nan, None, inplace=True)
        merged_frames.reset_index(inplace=True)
        table = {
            'Tabular results': {
                "headers": list(merged_frames.columns),
                "rows": merged_frames.values.tolist()
            }
        }
        results["finalAnalysis"]["tables"] = table
        results["finalAnalysis"]["workbookData"] = table['Tabular results']
        return results
    except Exception as e:
        print(f"Error inside merge_frames: {e}")
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
        del results["finalAnalysis"]["charts"]
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



def should_global_calculate(question, financials, debug=False):
    system_prompt = """
    You are a tool used in a rag system to determine an equation that needs to be applied to a pandas dataframe. Your job is to generate 
    the string representation of the pandas calculation such that the python eval function can be called on the string.
    Your response should not contain anything other than the pandas equation to be called. Including anything other than just the equation 
    will cause the functin to fail. 
    
    Given a user query and the supplied json that represents the pandas DataFrame, return the equation to apply to the DataFrame that performs the calculation
    necessary to answer the question. Below are some examples of user query, json data, and response triplets. Assume that the pandas DataFrame is called 'financials'.
    Use the name financials to represent that pandas DataFrame in your response. 

    Examples:
    user query: "What was microsoft's effect of exchange rate on Cash and Cash Equivalents in Q3 2020 and Q4 2020?", data: ['exchange_rate'], answer: "False"
    user query: "what's amzn's revenue for 2022 versus its comps", data: ['revenue'], answer: "False"
    user query: "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "True"
    user query: "what's tgt's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "True"
    user query: "what's wmt's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "True"
    user query: "What was the total amount spent on Acquisitions Net of Cash Acquired and Purchases of Intangible and Other Assets by MSFT in the year 2022?", data: ['cash', 'intangible_assets', 'fixed_assets', 'net_cash_flow_from_investing_activities'], answer: "True"
    user query: "What was the trend in MSFT's Payments for Repurchase of Common Stock from Q3 2020 to Q1 2023?", data: ['revenue'], answer: "False"
    user query: "what's amzn's revenue growth during 2023 versus its comps", data: ['revenues'], answer: "True"
    user query: "what's amzn's revenue growth during 2023 versus its comps", data: ['AMZN.revenues'], answer: "False"
    user query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", data: ['revenues', 'research_and_development'], answer: "True"
    user query: "Compare the research and development investments as a percentage of revenue for J&J and Merck over the past three years. Historically how have their drug launches impacted revenue growth?", data: ['research_and_development_to_revenue_pct', 'revenues', 'research_and_development'], answer: "False"
    """

    prompt = """
    user query: {question}
    json data: {financials}
    """



    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.format(question=question, financials=list(financials.columns))}
        ],
        
    )

    try:
        
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        import pdb
        # pdb.set_trace()
        # print(f"response: {response}")
        calculation_required = ast.literal_eval(response)
        # print(f"calculation_required: {calculation_required}")
        return calculation_required
    except Exception as e:
        print(f"Error inside should_global_calculate. Got response from GPT-4: {response}")
        return False


"""
GOOD EXAMPLE OF BAR CHART:
==========================

- what's aapl's revenue for the first 3 quarters of 2022? compare this to comps


"""


def do_global_calculate(question, financials, debug=False):
    try: 
        system_prompt = """ 
        You are a tool used in a rag system to determine an equation that needs to be applied to a pandas dataframe. Your job is to generate 
        the string representation of the pandas calculation such that the python eval function can be called on the string.
        Your response should not contain anything other than the pandas equation to be called. Including anything other than just the equation 
        will cause the functin to fail. 
        
        Given a user query and the supplied json that represents the pandas DataFrame, return the equation to apply to the DataFrame that performs the calculation
        necessary to answer the question. Below are some examples of user query, json data, and response triplets. Assume that the pandas DataFrame is called 'financials'.
        Use the name financials to represent that pandas DataFrame in your response. 

        Examples:
        user query: "what's amzn's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "financials['revenue'].pct_change(periods=-1).round(2).dropna()"
        user query: "what's WMT's revenue growth (%) quarter over quarter for 2022 versus its comps", data: ['revenue'], answer: "financials['revenue'].pct_change(periods=-1).round(2).dropna()"
        user query: "how did msft's net cash provided by operating activities change from q3 2020 to q1 2021?", data: ['net_cash_flow_from_operating_activities'], answer: "df['net_cash_flow_from_operating_activities'].pct_change(periods=-1).round(2).dropna()"
        user query: "what's amzn's revenue growth during 2023 versus its comps", data: ['revenues'], answer: "financials['revenues'].pct_change(periods=-1).round(2).dropna()"
        ""
        
        """

        prompt = """
        user query: {question}
        json data: {financials}
        """

        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.format(question=question, financials=list(financials.columns))}
            ],
            
        )

        
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        print(f"equation from inside do_global_calculate: {response}")
        # pdb.set_trace()
        # print(f"response: {response}")
        exec(response)
        # print(f"financials: {financials}")
        if isinstance(financials, pd.Series):
            financials = financials.to_frame()
            # financials.dropna(inplace=True)
        elif isinstance(financials, pd.DataFrame):
            pass
        else:
            financials = pd.DataFrame({"result": financials})
        return financials
    except Exception as e:
        print(f"Error inside do_global_calculate: {e}")
        return financials


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

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt.format(final_response=final_response, data=task_json['get_sec_financials']['data_for_final_analysis'])}
        ],
        
    )
    
    try:
        
        response = json.loads(response.to_json())["choices"][0]["message"]["content"]
        return response
    except Exception as e:
        # print(f"Error parsing ChatGPT response: {e}")
        return []