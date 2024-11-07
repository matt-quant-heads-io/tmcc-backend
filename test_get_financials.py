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

from scratch import should_local_calculate, do_local_calculate, get_relevant_fiscal_columns, get_relevant_calendar_columns, get_relevant_rows, get_stock_financials, get_fiscal_or_calendar_by_user_query, get_stock_financials, determine_query_pattern, get_list_of_companies, map_list_of_companies_and_query_to_list_of_queries, map_competitors_and_query_to_list_of_queries, split_companies, preprocess_user_query, openai_client, cohere_client, weaviate_client, merge_charts, merge_frames, do_global_calculate
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
        # temp_financials.reset_index(drop=True, inplace=True)
        # temp_financials.set_index(pd.RangeIndex(len(temp_financials)), inplace=True)
        results["GetCompanyFinancials"][ticker.upper()] = temp_financials
        chart_data = temp_financials.to_json()

        results["Context"].append(
            f"{temp_financials.T.columns[0]} results for {ticker.upper()}: {temp_financials.to_json()}"
        )

        return results
    except Exception as e:
        import pdb; pdb.set_trace()
        print(f"Error inside get_financials: {e}")
        return results



def get_final_analysis(query, results, debug=False):
    try:
        import pdb; pdb.set_trace()
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

get_financials_answers = []
queries = [
    "Analyze and compare the free cash flow for Verizon (VZ), AT&T (T), and Cisco (CSCO) over the last five years. Discuss implications for dividend sustainability and potential reinvestment strategies.",
    "Determine and compare the times interest earned (TIE) ratios for energy companies like ExxonMobil (XOM), Chevron (CVX), and Dow Inc. (DOW) over the past four quarters. How well can these companies cover their interest obligations?",
    "Examine the EPS growth trend for healthcare companies such as Merck (MRK), Johnson & Johnson (JNJ), and Pfizer (PFE) from 2019 to 2023. Discuss any significant shifts or trends in their earnings performance.",
    "Compare the net profit margins and gross profit margins for companies like Disney (DIS), McDonald's (MCD), and Nike (NKE) over the last three years. Identify which company showcases the highest profitability based on these metrics.",
    "Analyze changes in capital expenditures over the last five years for Dow 30 industrial companies, including General Electric (GE), Honeywell (HON), and Caterpillar (CAT). What investment patterns can be observed?",
    "Evaluate the current ratio and quick ratio for consumer goods companies like Coca-Cola (KO), Procter & Gamble (PG), and McDonald's (MCD) from 2020 to 2023. Determine which company maintains the highest liquidity and discuss any potential implications.",
    "Compare the return on equity (ROE) for financial firms like JPMorgan Chase (JPM), Goldman Sachs (GS), and American Express (AXP) over the last eight quarters. How does ROE performance differ among these companies, and what factors contribute to any disparity?",
    "Calculate the debt-to-equity ratios for industrial companies like Boeing (BA), Caterpillar (CAT), and 3M (MMM) over the last five years. Discuss how these ratios reflect each company's financial leverage.",
    "Analyze the revenue growth rate and operating margin growth for technology companies such as Apple (AAPL), Microsoft (MSFT), and Intel (INTC) from 2018 to 2023. How do their financial performances compare over this period?",
    "Calculate and compare the inventory turnover ratios for Home Depot (HD), Walmart (WMT), and Nike (NKE) over the last four quarters. Identify any trends or anomalies.",
    "Evaluate the price-to-earnings (P/E) ratio trends for consumer discretionary companies such as Disney (DIS), Nike (NKE), and Home Depot (HD) over the last decade.",
    "Calculate the dividend yield for Verizon (VZ), Coca-Cola (KO), and Procter & Gamble (PG) over the last five years and analyze their dividend consistency.",
    "Compare the accounts receivable turnover ratios for Dow 30 tech firms like Microsoft (MSFT), Apple (AAPL), and Intel (INTC) over the past four quarters.",
    "Assess the historical beta for industrials like Boeing (BA), 3M (MMM), and Caterpillar (CAT) and discuss the volatility relative to the market over the last 10 years.",
    "Determine the annual net income growth for retailers Walmart (WMT) and Home Depot (HD) from 2010 to 2023 and identify any significant growth peaks or declines.",
    "Perform a trend analysis of cash flow from operating activities for healthcare giants Johnson & Johnson (JNJ), Pfizer (PFE), and Merck (MRK) from 2018 to 2023.",
    "Compare the interest coverage ratios for financial services firms like American Express (AXP) and Visa (V), focusing on their ability to cover interest expenses from earnings over the past five years.",
    "Analyze long-term debt trends for telecommunications companies such as AT&T (T) and Verizon (VZ), examining their debt management strategies over the last decade.",
    "Investigate the effect of share buyback programs on earnings per share (EPS) and share price for technology companies including Intel (INTC), Cisco (CSCO), and IBM over the past five years.",
    "Evaluate the efficiency of inventory management for consumer product companies like PepsiCo (PEP), Coca-Cola (KO), and Procter & Gamble (PG) by analyzing days sales of inventory (DSI).",
    "Predict future earnings growth for ExxonMobil (XOM), Chevron (CVX), and Dow Inc. (DOW) based on historical performance and industry trends over the past 5 years.",
    "Compare the return on assets (ROA) ratios for financial corporations such as JPMorgan Chase (JPM) and Goldman Sachs (GS) over a five-year span.",
    "Analyze the cost of goods sold (COGS) as a percentage of revenue for McDonald's (MCD), Procter & Gamble (PG), and Walt Disney (DIS) over the past three financial years.",
    "Investigate the intangible assets as a proportion of total assets for high-tech companies like Apple (AAPL), Microsoft (MSFT), and Cisco (CSCO) from 2016 to 2023.",
    "Determine the impact of foreign exchange rates on reported earnings for multinational corporations such as IBM, Johnson & Johnson (JNJ), and 3M (MMM) over the last 5 years.",
    "Compare the operating cash flow to sales ratio for Disney (DIS), Nike (NKE), and Coca-Cola (KO) as a measure of cash generation efficiency.", 
    "Explore the historical growth in marketing expenditures for consumer brands such as Coca-Cola (KO), McDonald's (MCD), and Nike (NKE) over the past six years.",
    "Evaluate competitive advantage through gross profit margin comparison for industrial corporations Caterpillar (CAT), 3M (MMM), and Honeywell (HON) over a three-year period.",
    "Analyze historical data on profit margins during economic recessions for major banks like JPMorgan Chase (JPM) and Goldman Sachs (GS).",
    "Compare the quarterly stock price and revenue correlation for Dow 30 technology stocks Apple (AAPL), Intel (INTC), and Microsoft (MSFT).",
    "Assess free cash flow growth in the last decade for consumer discretionary businesses such as Home Depot (HD) and McDonald's (MCD).",
    "Extract trends in research and development (R&D) spending for major healthcare companies Pfizer (PFE), Merck (MRK), and Johnson & Johnson (JNJ).",
    "Analyze correlation between capital investments and revenue growth for General Electric (GE), Caterpillar (CAT), and Boeing (BA).",
    "Compare changes in the total shareholder equity for Intel (INTC), Cisco (CSCO), and IBM over the last 5 years.",
    "Determine how shifts in oil prices have impacted the revenue streams of ExxonMobil (XOM) and Chevron (CVX).",
    "Evaluate changes in pension liabilities for major corporations like Boeing (BA) and General Electric (GE) over the past decade.",
    "Critique strategies around dividend payouts and capital retention for Johnson & Johnson (JNJ) and Procter & Gamble (PG).",
    "Compare annual changes in net profit and administrative expenses for Microsoft (MSFT), Apple (AAPL), and Intel (INTC).",
    "Analyze trends in employee headcount relative to revenue for McDonald's (MCD), Walmart (WMT), and Home Depot (HD).",
    "Identify patterns in quarterly EPS surprises among Dow 30 tech giants Apple (AAPL), Microsoft (MSFT), and Intel (INTC).",
    "Study working capital management efficiency for consumer companies Coca-Cola (KO), Procter & Gamble (PG), and McDonald's (MCD).",
    "Perform a cyclic trend analysis on stock performance vs. economic cycles for diversified stocks like JPMorgan Chase (JPM), and Walt Disney (DIS).",
    "Examine the impact of sector-specific regulations on CapEx trends for energy firms ExxonMobil (XOM) and Chevron (CVX).",
    "Quantify percent change in quarterly sales vs. market share for Cisco (CSCO), IBM, and Intel (INTC) over the last two years.",
    "Investigate historical dividend reinvestment impacts on stock valuations for VISA (V) and American Express (AXP).",
    "Compare post-merger integration costs and outcomes for Dow 30 companies like UnitedHealth Group (UNH) over the last decade.",
    "Quantify correlation between CEO tenure and ROE among Dow 30 companies with long-serving leaders over the last 20 years.",
    "Analyze the shock resilience by examining asset write-downs across industrial companies during the 2008 financial recession.",
    "Study the predictive capability of advertising spend on boosted quarterly revenues for Nike (NKE) and Coca-Cola (KO).",
    "Evaluate portfolio diversification strategies through international operations for IBM, Apple (AAPL), and Microsoft (MSFT).",
    "Compare valuations and stock splits ratios vs. retained earnings impact for Procter & Gamble (PG), and Johnson & Johnson (JNJ).",
    "Analyze quarterly fluctuations in deferred taxes and their eventual realization for Dow 30 firms.",
    "Critically appraise the efficiency gains post-automation in logistics for Walmart (WMT) vs. Home Depot (HD).",
    "Discover effects of patent expiration on revenue decline for healthcare companies Pfizer (PFE) and Merck (MRK).",
    "Benchmark recent mergers' success by sales and profit improvement metrics for notable companies like Johnson & Johnson (JNJ).",
    "Identify patterns in quarterly dividend announcements and market reactions for major corporations ExxonMobil (XOM) and Chevron (CVX).",
    "Examine how shifts in regulatory frameworks affected compliance costs for financial institutions JPMorgan Chase (JPM) and Goldman Sachs (GS)."
]
for query in queries:
    results = {"Query": query, "Context": [], "Execution": {}, "finalAnalysis": {"tables": {}, "charts": {}}, "MarketData": {}, "MarketDataForBacktest": {}, "QualAndQuant": {}, "GetNews": {}, "GetCompanyFinancials": {}, "GetEstimates": {}, "RunBackTest": {}, "VectorSearch": {}, "Tables": {}}
    processed_queries = process_queries(query)
    for processed_query in processed_queries:
        results = get_financials(processed_query, results, debug=False)

    # import pdb; pdb.set_trace()
    get_financials_answers.append({
        "OriginalQuery": query,
        "ProcessedQueries": processed_queries,
        "Results": {k:v.to_dict(orient="records") for k,v in results['GetCompanyFinancials'].items()}
    })
    results = {"Query": query, "Context": [], "Execution": {}, "finalAnalysis": {"tables": {}, "charts": {}}, "MarketData": {}, "MarketDataForBacktest": {}, "QualAndQuant": {}, "GetNews": {}, "GetCompanyFinancials": {}, "GetEstimates": {}, "RunBackTest": {}, "VectorSearch": {}, "Tables": {}}




with open("/home/ubuntu/tmcc-backend/batch_queries_test_get_financials_answers.json", "w") as f:
    f.write(json.dumps(get_financials_answers))
    
"""
GOT THIS WRONG:
===============
"Evaluate the current ratio and quick ratio for McDonald's (MCD) from 2020 to 2023. Determine if MCD maintains the highest liquidity and discuss any potential implications."
"""


# TODO: Check for empty 'correctly selected rows' (i.e. in get_relevant_rows() method. So rerun the 50 examples but put a check to see when the 'correctly selected rows is empty'


# NOTE BELOW LEADS TO EMPTY get_relevant_rows()
# NOTE: Handle this in get_market_data??: 'Assess the historical beta for Boeing (BA) and discuss the volatility relative to the market over the last 10 years.'
# Assess the historical beta for 3M (MMM) and discuss the volatility relative to the market over the last 10 years.