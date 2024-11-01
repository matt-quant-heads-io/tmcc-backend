import os
import ast
import json
from pathlib import Path

import pandas as pd

import requests
from sec_cik_mapper import StockMapper


TICKER_TO_COMPANY = {
    "AAPL": "Apple Inc.",
    "AMGN": "Amgen Inc.",
    "AXP": "American Express Co.",
    "BA": "Boeing Co.",
    "CAT": "Caterpillar Inc.",
    "CRM": "Salesforce.com Inc.",
    "CSCO": "Cisco Systems Inc.",
    "CVX": "Chevron Corp.",
    "DIS": "Walt Disney Co.",
    "DOW": "Dow Inc.",
    "GS": "Goldman Sachs Group Inc.",
    "HD": "Home Depot Inc.",
    "HON": "Honeywell International Inc.",
    "IBM": "International Business Machines Corp.",
    "INTC": "Intel Corp.",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co.",
    "KO": "Coca-Cola Co.",
    "MCD": "McDonald's Corp.",
    "MMM": "3M Co.",
    "MRK": "Merck & Co. Inc.",
    "MSFT": "Microsoft Corp.",
    "NKE": "Nike Inc.",
    "PG": "Procter & Gamble Co.",
    "TRV": "Travelers Companies Inc.",
    "UNH": "UnitedHealth Group Inc.",
    "V": "Visa Inc.",
    "VZ": "Verizon Communications Inc.",
    "WBA": "Walgreens Boots Alliance Inc.",
    "WMT": "Walmart Inc."
}
NUM_10K_TO_DOWNLOAD = 11
NUM_10Q_TO_DOWNLOAD = 37


# Initialize a stock mapper instance
mapper = StockMapper()


def get_cik_from_ticker(ticker):
    return mapper.ticker_to_cik[ticker]


def get_filings_by_ticker(ticker):
    """
    Retrieves the accession numbers and CIKs for a given ticker symbol from the SEC EDGAR API.

    Args:
        ticker (str): The ticker symbol of the company.

    Returns:
        list: A list of dictionaries containing the accession number, CIK, and filing type for each filing.
    """
    submissions_url = "https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {
        "User-Agent": "hello3224@gmail.com"
    }

    cik = get_cik_from_ticker(ticker)
    response = requests.get(submissions_url.format(cik=cik), headers=headers)
    data = response.json()
    # import pdb; pdb.set_trace()
    df = pd.DataFrame.from_dict(data['filings']['recent'])
    
    df = df[df['primaryDocDescription'].str.contains("10")]
    # df['reportDate'] = pd.to_datetime(df['reportDate'])
    # df = df.sort_values(by='reportDate', ascending=False)

    return df, cik


def build_filing_urls_from_dataframe(ticker, cik, filings_df):
    filing_urls = []
    # import pdb; pdb.set_trace()
    for index, row in filings_df.iterrows():
        accession_number = row['accessionNumber']
        primary_doc = row['primaryDocument']
        report_date = row['reportDate']
        # if filings_df["isInlineXBRL"] == 1:
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/Financial_Report.xlsx"
        filing_urls.append(filing_url)
    return filing_urls


import requests
import shutil
import os

def download_file(url, download_path, filename, headers):
    # Create the download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(download_path, filename)

    # Send the GET request and download the file
    response = requests.get(url, stream=True, headers=headers)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, file)
        print(f"Successfully downloaded {url} to {file_path}")
    else:
        print(f"Failed to download {url}")


download_path = "/home/ubuntu/tmcc-backend/company_financials"

from edgar import *

headers = {
        "User-Agent": "hello3224@gmail.com"
    }

# Tell the SEC who you are
set_identity("Michael Mccallum mike.mccalum@indigo.com")
# filings = Company(ticker).get_filings(form=["10-K", "10-Q"]).latest(50)

# import pdb; pdb.set_trace()
for ticker in ["GS"]:#DOW_30_TICKERS:
    cik = get_cik_from_ticker(ticker)
    filings = Company(ticker).get_filings(form=["10-K", "10-Q"]).latest(50)
    for filing in filings.to_pandas().to_dict(orient='records'):
        print(f"Downloading {filing['accession_number']}")
        download_file(f"https://www.sec.gov/Archives/edgar/data/{cik}/{filing['accession_number'].replace('-', '')}/Financial_Report.xlsx", f"{download_path}/{ticker}", f"{filing['accession_number']}.xlsx", headers)



