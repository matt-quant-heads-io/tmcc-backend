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

from scratch import extract_entities_from_user_query


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


"""
['date', 'symbol', 'reportedCurrency', 'cik', 'fillingDate', 'acceptedDate', 'calendarYear', 'period', 'revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses', 'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization', 'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'link', 'finalLink', 'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock', 'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt', 'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow']
"""

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
    result_df["Fiscal Date"] = result_df["period"].astype(str) + " " + result_df["calendarYear"].astype(str)
    result_df["Calendar Date"] = result_df["report_date"]
    result_df.drop("report_date", axis=1, inplace=True)
    if mode == "calendar":
        result_df.rename({"Calendar Date": "report_date"}, axis=1, inplace=True)
        result_df.set_index("report_date", inplace=True)
    else:
        result_df.rename({"Fiscal Date": "report_date"}, axis=1, inplace=True)
        result_df.set_index("report_date", inplace=True)

    
    result_df.drop(['reportedCurrency', 'cik', 'fillingDate', 'acceptedDate'], axis=1, inplace=True)
    return result_df.T

question="What's apple's revenue for 2022?"
results={}
debug=False

get_stock_financials("AAPL", "calendar", 5, debug)
def get_financials(question, results, debug=False):
    try:
        entities = extract_entities_from_user_query(question, debug)
        tickers = []
        for ent in entities:
            if ent["entity"] == "ticker":
                tickers.append(ent["value"])

        for ticker in tickers:
            if "processed_tickers" not in results:
                results["processed_tickers"] = []

            if ticker.upper() in results["processed_tickers"]:
                continue
            financials = get_stock_financials(ticker.upper())
            if financials is None or len(financials) == 0:
                continue
            temp_financials = financials.copy()
            relevant_rows = get_relevant_rows(list(temp_financials.index.values), question)
            print(f"relevant_rows: {relevant_rows}")
            fiscal_or_calendar = get_fiscal_or_calendar_by_user_query(question)
            if fiscal_or_calendar == "fiscal":
                other_date_column_to_keep = "Calendar Date"
                relevant_columns = get_relevant_fiscal_columns(list(temp_financials.columns), question)
            else:
                other_date_column_to_keep = "Fiscal Date"
                relevant_columns = get_relevant_calendar_columns(list(temp_financials.columns), question)

            print(f"relevant_columns: {relevant_columns}")

            
            filtered_columns = []
            filtered_rows = []
            for col in relevant_columns:
                if col in temp_financials.columns:
                    filtered_columns.append(col)

            for row in relevant_rows:
                if row in temp_financials.index.values:
                    filtered_rows.append(row)

            import pdb; pdb.set_trace()
            temp_financials = temp_financials.loc[filtered_rows+[other_date_column_to_keep], filtered_columns]
            # temp_financials = temp_financials.dropna()
            # import pdb; pdb.set_trace()

            calculation_required = should_local_calculate(question, list(financials.index.values))
            print(f"calculation_required: {calculation_required}")
            if calculation_required:
                temp_financials = do_local_calculate(question, temp_financials.T)
                print(f"temp_financials after do_local_calculate: {temp_financials}")
                if len(temp_financials) == 0:
                    continue


            # import pdb; pdb.set_trace()

            # temp_financials = temp_financials.dropna()
            temp_financials = temp_financials.rename(columns={c: f"{ticker.upper()}.{c}" for c in temp_financials.columns})
            # chart_data = temp_financials.to_dict('records')
            chart_data = temp_financials.T.to_json()

            results["Context"].append(
                f"{temp_financials.T.columns[0]} results for {ticker.upper()}: {temp_financials.T.to_json()}"
            )

            if "ResultsFromGetFinancials" not in results:
                results["ResultsFromGetFinancials"] = {

                }

            results["ResultsFromGetFinancials"][ticker.upper()] = temp_financials.to_json()
            results["processed_tickers"].append(ticker.upper())

            if "finalAnalysis" not in results:
                results["finalAnalysis"] = {

                }

            if "charts" not in results["finalAnalysis"]:
                results["finalAnalysis"]["charts"] = {

                }

            if "tables" not in results["finalAnalysis"]:
                results["finalAnalysis"]["tables"] = {

                }

            temp_financials[f"Date_{ticker}"] = temp_financials.index.values
            table =  {
                    "headers": list(temp_financials.columns),
                    "rows": temp_financials.values.tolist()
            }

            results["finalAnalysis"]["charts"][ticker.upper()] = temp_financials
            results["finalAnalysis"]["tables"][ticker.upper()]= temp_financials
            if "workbookData" not in results:
                results["workbookData"] = {
                }

            # import pdb; pdb.set_trace()
            results["workbookData"][ticker.upper()] = table

        if debug:
            with open(DEBUG_ABS_FILE_PATH, "a") as f:
                f.write(json.dumps({"function": "get_financials", "inputs": [question], "outputs": [{"financials": temp_financials.to_json()}]}, indent=6))
        return results
    except Exception as e:
        print(f"Error inside get_financials: {e}")
        return results