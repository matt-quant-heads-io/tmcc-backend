import sys
import pathlib
import json
import os
from datetime import datetime
import re
import requests
# from datasets import load_dataset
from typing import List, Dict

import hydra
from omegaconf import OmegaConf

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance
from langchain_openai import OpenAIEmbeddings
from langchain import OpenAI
import openai
from langchain_text_splitters import TokenTextSplitter
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
import pypdf
from sec_api import XbrlApi

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


from langchain import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os
import ast


import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from lib import SEC_APIS_MAP, ADAPTERS_MAP
from sec_edgar_api import EdgarClient

from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings
import base64
from io import BytesIO
from xhtml2pdf import pisa
import weasyprint
import pdfkit
import os


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or "neo4j"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HATTAN_BACKEND_ROOT = os.getenv("HATTAN_BACKEND_ROOT")
if not HATTAN_BACKEND_ROOT:
    raise RuntimeError("Run ../set_proj_root.sh to set env var HATTAN_BACKEND_ROOT")
DATA_DIR = f"{HATTAN_BACKEND_ROOT}/data/finance_bench"
QDRANT_BASE_URL = "http://3.140.46.146:6333" #"http://localhost:6333"
QDRANT_COLLECTION_NAME = "finance_bench"
QDRANT_EMBEDDING_SIZE = 1536
EDGAR_DATA_PREFIX_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_IO_XBRL_BASE_URL = "https://api.sec-api.io/xbrl-to-json?htm-url"
SEC_IO_API_TOKEN = "58f62cb318c7d951b2efa816a1314cee7019a093c357cca02bc7033c8f907d56"
xbrlApi = XbrlApi(SEC_IO_API_TOKEN)
COMPANY_NAME_TO_TICKER_MAP = {
    "3M": "MMM",
    "ACTIVISION BLIZZARD": "ATVI",
    "ADOBE": "ADBE",
    "AES": "AES",
    "AMAZON": "AMZN",
    "APPLE": "AAPL",
    "GOOGLE": "GOOG",
    "MICROSOFT": "MSFT",
    "AMCOR": "AMCR",
    "AMD": "AMD",
    "AMERICANEXPRESS": "AXP",
    "BESTBUY": "BBY",
    "AMERICAN WATERWORKS": "AWK",
    "BLOCK": "SQ",
    "BOEING": "BA",
    "COCACOLA": "KO",
    "CORNING": "GLW",
    "COSTCO": "COST",
    "CVSHEALTH": "CVS",
    "FOOTLOCKER": "FL",
    "GENERALMILLS": "GIS",
    "JPMORGAN": "JPM",
    "KRAFTHEINZ": "KHC",
    "LOCKHEEDMARTIN": "LMT",
    "MGMRESORTS": "MGM",
    "MICROSOFT": "MSFT",
    "NETFLIX": "NFLX",
    "NIKE": "NKE",
    "PAYPAL": "PYPL",
    "PEPSICO": "PEP",
    "PFIZER": "PFE",
    "Pfizer": "PFE",
    "ULTABEAUTY": "ULTA",
    "VERIZON": "VZ",
    "WALMART": "WMT",
}
TICKER_TO_COMPANY_NAME = {v:k for k,v in COMPANY_NAME_TO_TICKER_MAP.items()} 
COMPANY_NAME_PDF_PREFIX_TO_DB_COMPANY_NAME = {
    "3M": "3M",
    "ACTIVISIONBLIZZARD": "activision blizzard",
    "ADOBE": "adobe",
    "AES": "aes",
    "AMAZON": "amazon",
    "AMCOR": "amcor",
    "AMD": "advanced micro devices",
    "AMERICANEXPRESS": "american express",
    "BESTBUY": "best buy",
    "AMERICANWATERWORKS": "american waterworks",
    "BLOCK": "block",
    "BOEING": "boeing",
    "COCACOLA": "coca cola",
    "CORNING": "corning",
    "COSTCO": "costco",
    "CVSHEALTH": "cvs health",
    "FOOTLOCKER": "foot locker",
    "GENERALMILLS": "general mills",
    "JPMORGAN": "jp morgan",
    "KRAFTHEINZ": "kraft heinz",
    "LOCKHEEDMARTIN": "lockheed martin",
    "MGMRESORTS": "mgm resorts",
    "MICROSOFT": "microsoft",
    "NETFLIX": "netflix",
    "NIKE": "nike",
    "PAYPAL": "paypal",
    "PEPSICO": "pepsi",
    "PFIZER": "pfizer",
    "Pfizer": "pfizer",
    "ULTABEAUTY": "ulta beauty",
    "VERIZON": "verizon",
    "WALMART": "walmart",
}
token_text_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=40)
kg_adapter = ADAPTERS_MAP["Neo4jGraphAdapter"]()

# embeddings = OpenAIEmbeddings()
openai_client = openai.Client(
    api_key="sk-proj-Tb7SWc46QhFrvgU6pPY4T3BlbkFJRyGTP45AunM8ULAu9XLA"
)

oai_client = OpenAI()

# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(2))
def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    return oai_client.embeddings.create(input=[text], model=model).data[0].embedding


def read_docs_from_json(json_path: str):
    data = None
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def read_doc_from_pdf(path_to_pdf: str):
    data = None
    with open(path_to_pdf, "rb") as f:
        data = f.read()
    return data.decode("utf-8")


COLLECTION_TYPE_TO_COLLECTION_NAME_MAP = {"10-K": "10K", "10-Q": "10Q", "8-K": "8K"}




FILING_TO_QDRANT_MAP = {
    "10K": "sec_filings2", "8-K": "sec_filings2", "10Q":"sec_filings2", '10-K/A': "sec_filings2"
}


hypothetical_embeddings_template = """
    Given a supplied text below generate a list of 10 questions and 10 answers from the supplied text. 
    The answers you generate should be in complete sentences. Your response should be a JSON with the keys 'hypothetical_questions' and 'hypothetical_answers'.
    The value corresponding to the key 'hypothetical_questions' should be the list of questions you generated.
    The value corresponding to the key 'hypothetical_answers' should be the list of answers you generated.
    Do not include any other information in your reponse besides the JSON.

    text: {text}
"""

hypothetical_embeddings_prompt = PromptTemplate(
    template=hypothetical_embeddings_template,
    input_variables=["text"],
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

def get_hypothetical_questions_and_answers_from_gpt4(text):
    try:
        hypothetical_questions_and_answers_chain = (
                RunnablePassthrough.assign(text=lambda x: x)
                | hypothetical_embeddings_prompt
                | llm
                | StrOutputParser()
            )

        hypothetical_questions_and_answers_response = hypothetical_questions_and_answers_chain.invoke({"text": text})
        if '```json' in hypothetical_questions_and_answers_response:
            hypothetical_questions_and_answers_response = hypothetical_questions_and_answers_response[7:len(hypothetical_questions_and_answers_response)-3]
        print(f"hypothetical_questions_and_answers_response:\n{hypothetical_questions_and_answers_response}")
        print(f"type:\n{type(hypothetical_questions_and_answers_response)}")
        hypothetical_questions_and_answers_response = ast.literal_eval(hypothetical_questions_and_answers_response)
        return hypothetical_questions_and_answers_response
    except:
        return None


def get_hypothetical_embeddings_from_text_chunk(text_chunk):
    # Feed text_chunk to gpt4 to get back json of questions and answers per chunk
    hypothetical_questions_answers_json = get_hypothetical_questions_and_answers_from_gpt4(text_chunk)
    if not hypothetical_questions_answers_json:
        return None
    hypothetical_questions_list, hypothetial_answers_list = hypothetical_questions_answers_json["hypothetical_questions"], hypothetical_questions_answers_json["hypothetical_answers"]
    raw_hypothetical_questions = '\n'.join(hypothetical_questions_list)
    raw_hypothetical_answers = '\n'.join(hypothetial_answers_list)
    
    # Take list of questions and list of answers and create an embedding of each list
    hypothetical_questions_embedding = (
                    oai_client.embeddings.create(
                        input=raw_hypothetical_questions,
                        model="text-embedding-3-small",
                    )
                    .data[0]
                    .embedding
                )
    hypothetical_answers_embedding = (
                    oai_client.embeddings.create(
                        input=raw_hypothetical_answers,
                        model="text-embedding-3-small",
                    )
                    .data[0]
                    .embedding
                )
                
    # return as json {"hypothetical_answers": hypothetical_answers_embedding, "hypothetical_questions": hypothetical_questions_embedding}
    return {"raw_hypothetical_answers": raw_hypothetical_answers, "raw_hypothetical_questions": raw_hypothetical_questions, "hypothetical_answers": hypothetical_answers_embedding, "hypothetical_questions": hypothetical_questions_embedding}


# def load_data_into_qdrant():
#     df = pd.read_csv("/home/ubuntu/hattan-backend/filing_info_for_qdrant.csv")
#     print(f"df:\n{df.head()}")
#     collection_types = list(df["formType"].unique())
#     print(f"collection_types: {collection_types}")
#     print(f"type collection_types: {type(collection_types)}")
#     print(f"cols: {df.columns}")
#     client = QdrantClient(url=QDRANT_BASE_URL, port=None, prefer_grpc=False)


#     for collection_type in collection_types:
#         collection_name = FILING_TO_QDRANT_MAP[collection_type]
#         if client.collection_exists(collection_name=collection_name):
#             client.delete_collection(collection_name=collection_name)

#     for collection_type in collection_types:
#         collection_name = FILING_TO_QDRANT_MAP[collection_type]
#         if not client.collection_exists(collection_name=collection_name):
#             client.create_collection(
#                 collection_name=collection_name,
#                 vectors_config={
#                     "text": VectorParams(
#                         size=QDRANT_EMBEDDING_SIZE,
#                         distance=Distance.EUCLID,
#                     ),
#                     "hypothetical_questions": VectorParams(
#                         size=QDRANT_EMBEDDING_SIZE,
#                         distance=Distance.COSINE,
#                     ),
#                     "hypothetical_answers": VectorParams(
#                         size=QDRANT_EMBEDDING_SIZE,
#                         distance=Distance.COSINE,
#                     ),
#                 }
#             )
#     save_path = "/home/ubuntu/hattan-backend/sec_filings"

#     parent_counter = 0
#     child_counter = 0
#     points = []
#     for idx in range(len(df)):
#         ticker, filing_type, cik, accession_no, filed_date, link_to_filing, link_to_html, period_of_report =  df.iloc[idx, :]
        
#         loader = PyMuPDFLoader(f"{save_path}/{str(accession_no)}.pdf")
#         # TODO: modify this to instead use section chunker
#         data_pages = loader.load()
#         page_number = 0
#         for id, data_page in enumerate(data_pages):
#             parent_counter += 1

#             chunk_metadata = {}
#             chunk_metadata["company_name"] = TICKER_TO_COMPANY_NAME[ticker].lower()
#             chunk_metadata["period_of_report"] = datetime.strptime(
#                 period_of_report, "%Y-%m-%d"
#             ).date()
#             chunk_metadata["date_filed"] = datetime.strptime(
#                 filed_date[:19], "%Y-%m-%dT%H:%M:%S"
#             ).date()
#             chunk_metadata["source"] = link_to_html
#             chunk_metadata["accession_no"] = accession_no
#             chunk_metadata["cik"] = str(cik)
#             chunk_metadata["year_of_report"] = int(period_of_report[:4])
#             chunk_metadata["filing_type"] = filing_type
#             chunk_metadata["ticker"] = ticker.lower()
#             chunk_metadata["parent_id"] = parent_counter
#             chunk_metadata["page_number"] = id
            

#             page_content = (
#                 data_page.page_content[17:]
#                 if data_page.page_content.lower().startswith(
#                     "table of contents"
#                 )
#                 else data_page.page_content
#             )
#             # TODO: modify this to instead use section chunker
#             page_chunks = token_text_splitter.split_text(page_content)
#             for page_chunk in page_chunks:
#                 child_counter += 1
#                 chunk_metadata["text"] = page_chunk

#                 text_embedding = (
#                     oai_client.embeddings.create(
#                         input=page_chunk,
#                         model="text-embedding-3-small",
#                     )
#                     .data[0]
#                     .embedding
#                 )
                
#                 hypoethical_embeddings = get_hypothetical_embeddings_from_text_chunk(page_chunk)
#                 if not hypoethical_embeddings:
#                     continue
#                 chunk_metadata["hypothetical_questions_text"] = hypoethical_embeddings["raw_hypothetical_questions"]
#                 chunk_metadata["hypothetical_answers_text"] = hypoethical_embeddings["raw_hypothetical_answers"]
#                 points.append(
#                     models.PointStruct(
#                         id=child_counter,
#                         vector={
#                             "text": text_embedding,
#                             "hypothetical_questions": hypoethical_embeddings["hypothetical_questions"],
#                             "hypothetical_answers": hypoethical_embeddings["hypothetical_answers"]

#                         },
#                         payload=chunk_metadata,
#                     )
#                 )
#                 print(f"Added Chunk id: {child_counter} to points (page number {page_number}")

#             page_number += 1
#         client.upsert(FILING_TO_QDRANT_MAP[filing_type], points)
#         points = []


def get_accession_number_from_doc_link(doc_link):
    acc_no = None
    if "static-files" in doc_link:
        acc_no = doc_link.split("/")[-1]
    elif "CIK-" in doc_link:
        acc_no = doc_link.split("/")[-1]
        if acc_no.endswith(".pdf"):
            acc_no = acc_no.split(".pdf")[0]
    elif ".pdf" in doc_link:
        acc_no = doc_link.split(".pdf")[0]

    return acc_no


FB_TO_SECIO_FILING_TYPE_FILTER_MAP = {
    "10K": "10-K",
    "10Q": "10-Q",
    "8K": "8-K",
    "EARNINGS": "EARNINGS",
}


def download_submissions(ticker, cik):
    import requests
    import pandas as pd
    import numpy as np

    start_date = "2024-01-01"
    end_date = "2020-12-31"



    sec_url = "https://www.sec.gov/cgi-bin/browse-edgar"
    sec_params = {
        "action": "getcompany",
        "CIK": cik,
        "type": "10-k",
        "dateb": start_date,
        "owner": "exclude",
        "count": 12
    }
    headers = {
    'User-Agent': 'tset12@gmail.com'
    }

    sec_response = requests.get(sec_url, params=sec_params, headers=headers)
    sec_data = sec_response.json()
    print(sec_data)


def load_data_into_qdrant():
    client = QdrantClient(url=QDRANT_BASE_URL, port=None, prefer_grpc=False)


    # for collection_type in collection_types:
    #     collection_name = FILING_TO_QDRANT_MAP[collection_type]
    #     if client.collection_exists(collection_name=collection_name):
    #         client.delete_collection(collection_name=collection_name)

    collection_types = ["sec_filings2"]
    for collection_type in collection_types:
        collection_name = collection_type #FILING_TO_QDRANT_MAP[collection_type]
        if not client.collection_exists(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text": VectorParams(
                        size=QDRANT_EMBEDDING_SIZE,
                        distance=Distance.EUCLID,
                    )
                }
                # vectors_config={
                #     "text": VectorParams(
                #         size=QDRANT_EMBEDDING_SIZE,
                #         distance=Distance.EUCLID,
                #     ),
                #     "hypothetical_questions": VectorParams(
                #         size=QDRANT_EMBEDDING_SIZE,
                #         distance=Distance.COSINE,
                #     ),
                #     "hypothetical_answers": VectorParams(
                #         size=QDRANT_EMBEDDING_SIZE,
                #         distance=Distance.COSINE,
                #     ),
                # }
            )
    ticker_to_cik_json = None
    with open("/home/ubuntu/hattan-backend/bin/ticker_to_cik.json", "r") as f:
        ticker_to_cik_json = json.loads(f.read())

    ticker_to_company_name = None
    with open("/home/ubuntu/hattan-backend/bin/portfolio_holdings_dict.json", "r") as f:
        ticker_to_company_name = json.loads(f.read())

    PDF_PATH = "/home/ubuntu/hattan-backend/bin/filings_data"
    local_pdfs = [f for f in os.listdir(PDF_PATH) if '.pdf' in f]

    num_tickers_loaded = 0
    parent_counter = 0
    child_counter = 0
    points = []
    for ticker, cik in ticker_to_cik_json.items():
        print(f"Starting to load {ticker}")
        local_pdfs_to_load = [f for f in local_pdfs if ticker in f]

            
        for local_pdf in local_pdfs_to_load:
            local_pdf_without_extension = local_pdf.split(".pdf")[0]
            ticker, filing_type, accession_number, period_of_report = local_pdf_without_extension.split('_')
            year_of_report = int(period_of_report.split("-")[0])
            ticker_to_company_name_dict = [m for m in ticker_to_company_name if ticker == m["ticker"]][0]
            print(f"ticker_to_company_name_dict: {ticker_to_company_name_dict}")
            company_name = ticker_to_company_name_dict["company_name"]

            
            loader = PyMuPDFLoader(f"{PDF_PATH}/{local_pdf}")
            # TODO: modify this to instead use section chunker
            data_pages = loader.load()
            page_number = 0
            for page_num, data_page in enumerate(data_pages):
                parent_counter += 1

                chunk_metadata = {}
                
                # NOTE: LEFT OFFF HEREREEEEE!!!
                chunk_metadata["company_name"] = company_name
                chunk_metadata["period_of_report"] = period_of_report
                chunk_metadata["source"] = ""
                chunk_metadata["accession_no"] = accession_number
                chunk_metadata["cik"] = str(cik)
                chunk_metadata["year_of_report"] = int(year_of_report)
                chunk_metadata["filing_type"] = filing_type
                chunk_metadata["ticker"] = ticker.lower()
                chunk_metadata["parent_id"] = parent_counter
                chunk_metadata["page_number"] = page_num
                

                page_content = (
                    data_page.page_content[17:]
                    if data_page.page_content.lower().startswith(
                        "table of contents"
                    )
                    else data_page.page_content
                )
                # TODO: modify this to instead use section chunker
                page_chunks = token_text_splitter.split_text(page_content)
                for page_chunk in page_chunks:
                    child_counter += 1
                    chunk_metadata["text"] = page_chunk

                    text_embedding = (
                        oai_client.embeddings.create(
                            input=page_chunk,
                            model="text-embedding-3-small",
                        )
                        .data[0]
                        .embedding
                    )
                    
                    # hypoethical_embeddings = get_hypothetical_embeddings_from_text_chunk(page_chunk)
                    # if not hypoethical_embeddings:
                    #     continue
                    # chunk_metadata["hypothetical_questions_text"] = hypoethical_embeddings["raw_hypothetical_questions"]
                    # chunk_metadata["hypothetical_answers_text"] = hypoethical_embeddings["raw_hypothetical_answers"]
                    points.append(
                        models.PointStruct(
                            id=child_counter,
                            vector={
                                "text": text_embedding,
                                # "hypothetical_questions": hypoethical_embeddings["hypothetical_questions"],
                                # "hypothetical_answers": hypoethical_embeddings["hypothetical_answers"]

                            },
                            payload=chunk_metadata,
                        )
                    )
                    print(f"Added Chunk id: {child_counter} to points (page number {page_number}")

                page_number += 1
            client.upsert(FILING_TO_QDRANT_MAP[filing_type], points)
            points = []
        num_tickers_loaded += 1
        print(f"Finished loading {ticker}")
        print(f"num_tickers_loaded: {num_tickers_loaded}")


def download_sec_filings_as_local_pdfs():
    dl = Downloader("MyCompanyName", "email223@example.com")
    ticker_to_cik_json = None
    with open("/home/ubuntu/hattan-backend/bin/ticker_to_cik.json", "r") as f:
        ticker_to_cik_json = json.loads(f.read())

    ticker_to_filings_map = {}
    for ticker, cik in ticker_to_cik_json.items():
        
            print(ticker, cik)
            
            try:
                metadatas = dl.get_filing_metadatas(
                    RequestedFilings(ticker_or_cik=ticker, form_type="10-K", limit=3)
                )
            except Exception as e:
                    print(f"Error getting metadatas for ticker {ticker}")
                    continue

            for metadata in metadatas:
                try:
                    html = dl.download_filing(url=metadata.primary_doc_url).decode()
                    print(f"html: {html}")

                    with open(f"/home/ubuntu/hattan-backend/bin/filings_data/{ticker}_10K_{metadata.accession_number}_{metadata.report_date}.html", "w") as f:
                        f.write(html)

                    pdfkit.from_file(f"/home/ubuntu/hattan-backend/bin/filings_data/{ticker}_10K_{metadata.accession_number}_{metadata.report_date}.html", f"/home/ubuntu/hattan-backend/bin/filings_data/{ticker}_10K_{metadata.accession_number}_{metadata.report_date}.pdf")
                    os.remove(f"/home/ubuntu/hattan-backend/bin/filings_data/{ticker}_10K_{metadata.accession_number}_{metadata.report_date}.html")
                except Exception as e:
                    print(f"Error for ticker {ticker}")
                    continue   
                
                
            
            metadatas = dl.get_filing_metadatas(
                RequestedFilings(ticker_or_cik=ticker, form_type="10-Q", limit=8)
            )

            for metadata in metadatas:
                try:  
                    html = dl.download_filing(url=metadata.primary_doc_url).decode()
                    print(f"html: {html}")

                    with open(f"/home/ubuntu/hattan-backend/bin/filings_data/{ticker}_10Q_{metadata.accession_number}_{metadata.report_date}.html", "w") as f:
                        f.write(html)

                    pdfkit.from_file(f"/home/ubuntu/hattan-backend/bin/filings_data/{ticker}_10Q_{metadata.accession_number}_{metadata.report_date}.html", f"/home/ubuntu/hattan-backend/bin/filings_data/{ticker}_10Q_{metadata.accession_number}_{metadata.report_date}.pdf")    # html_content = None
                    os.remove(f"/home/ubuntu/hattan-backend/bin/filings_data/{ticker}_10Q_{metadata.accession_number}_{metadata.report_date}.html")
                except Exception as e:
                    print(f"Error for ticker {ticker}")
                    continue
           


def main():
    # download_sec_filings_as_local_pdfs()
    load_data_into_qdrant()

if __name__ == "__main__":
    TICKERS_TO_LOAD_FINANCIALS_FOR = [
    "META",
    "AMZN",
    "TFIN",
    "KSPI",
    "LBTYA",
    "FYBR",
    "DASH",
    "ALLY",
    "PEGA",
    "NRG",
    "FIP",
    "FPH",
    "SKIN",
    "AER",
    "GRND",
    "GFR",
    "GNW",
    "FTAI",
    "AUR",
    "OTLY",
    "CNK",
    "TKO",
    "WOW",
    "NCMI",
    "UHGWW",
    "MAPS",
    "LLAP",
    "CTV",
    "HIPO",
    "ONON",
    "GOOGL",
]
    # add_relations_and_indexes()
    main()
    # download_10Q_financials()