import os

# from sec_downloader import Downloader
# from sec_downloader.types import RequestedFilings
# from sec_cik_mapper import StockMapper
# import pdfkit
import json
import time

import weaviate
# import weaviate.classes as wvc

# import weaviate.classes.config as wc
# import boto3
# from botocore.exceptions import NoCredentialsError, ClientError
from dateutil import parser
# from openai import OpenAI
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_text_splitters import TokenTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv
import cohere

from constants import DOW_30_TICKERS, NUM_10K_TO_DOWNLOAD, NUM_10Q_TO_DOWNLOAD, TICKER_TO_COMPANY

load_dotenv()

# import weaviate
# from weaviate.classes.init import Auth

# Best practice: store your credentials in environment variables
weaviate_url = os.getenv("WEAVIATE_URL")
if not weaviate_url:
    raise ValueError("WEAVIATE_URL not found. Please set the `WEAVIATE_URL` environment variable.")

weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
if not weaviate_api_key:
    raise ValueError("WEAVIATE_API_KEY not found. Please set the `WEAVIATE_API_KEY` environment variable.")


cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY not found. Please set the `COHERE_API_KEY` environment variable.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("API key not found. Please set the `OPENAI_API_KEY` environment variable.")


co = cohere.Client(api_key=cohere_api_key)
client = weaviate.Client(
    # embedded_options=weaviate.EmbeddedOptions(
    # ),
    url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
    auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key),
    additional_headers = {
        "X-OpenAI-Api-Key": OPENAI_API_KEY  # Replace with your inference API key
    }
)




"""
properties = {
    "filing_type": filing_type,
    "company_name": TICKER_TO_COMPANY[ticker],
    "ticker": ticker,
    "accession_number": accession_number,
    "s3_doc_url": file,
    "text": "",
    "page_number": page_num,
    "report_date": parser.parse(report_date)
}
"""
class_obj = {
    "class": "Dow30_10K_10Q",
    "vectorizer": "text2vec-openai",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
    "vectorIndexConfig":{
        "distance": "cosine"
    },
    "moduleConfig": {
        "text2vec-openai": {
            "vectorizeClassName": True
        }
    },
    "properties": [
        {
            "name": "filing_type",
            "dataType": ["text"],
        },
        {
            "name": "company_name",
            "dataType": ["text"],
        },
        {
            "name": "ticker",
            "dataType": ["text"],
        },
        {
            "name": "accession_number",
            "dataType": ["text"],
        },
        {
            "name": "filing_url",
            "dataType": ["text"],
        },
        {
            "name": "sector",
            "dataType": ["text"],
        },
        {
            "name": "industry",
            "dataType": ["text"],
        },
        {
            "name": "city",
            "dataType": ["text"],
        },
        {
            "name": "state",
            "dataType": ["text"],
        },
        {
            "name": "zip_code",
            "dataType": ["text"],
        },
        {
            "name": "text",
            "dataType": ["text"],
        },
        {
            "name": "logo",
            "dataType": ["text"],
        },
        {
            "name": "country",
            "dataType": ["text"],
        },
        {
            "name": "description",
            "dataType": ["text"],
        },
        {
            "name": "cusip",
            "dataType": ["text"],
        },
        {
            "name": "isin",
            "dataType": ["text"],
        },
        {
            "name": "page_number",
            "dataType": ["number"],
        },
        {
            "name": "report_date",
            "dataType": ["date"],
        },
    ],
}


# if client.schema.exists("Dow30_10K_10Q"):
#     client.schema.delete_class("Dow30_10K_10Q")
# client.schema.create_class(class_obj)

print(client.is_ready())
# import pdb; pdb.set_trace()



def upload_data_to_weaviate(client):
    """ citation attributes form the frontend

    type Citation = {
        id: number
        company: string
        title: string
        importance: number
        url: string
        logo: string
        page_number: number
        report_date: string
    }
    """

    # try:
        
        
        # )
        
        # dow30_collection = client.collections.get("Dow30_10K_10Q")
    # data = None
    # with open("/home/ubuntu/tmcc-backend/data/chunked_filings/chunking_1.json", "r") as f:
    #     data = json.load(f)

    # Configure a batch process
    # client.batch.configure(batch_size=100)  # Configure batch
    # with client.batch as batch:
        # Batch import all Questions
    filings_universe = set(sorted(os.listdir("/home/ubuntu/tmcc-backend/data/chunked_filings")))
    if not os.path.exists("/home/ubuntu/tmcc-backend/to_delete/loading.log"):
        open("/home/ubuntu/tmcc-backend/to_delete/loading.log", "w").write("")
    loaded_filings = set([f.replace("\n", "") for f in open("/home/ubuntu/tmcc-backend/to_delete/loading.log", "r").readlines()])
    filings_to_load = filings_universe - loaded_filings
    print(f"Filings to load: {len(filings_to_load)}")
    

    for file in tqdm(filings_to_load):
        try:
            if not file.endswith(".json") or file in loaded_filings:
                continue

            data = None
            with open(f"/home/ubuntu/tmcc-backend/data/chunked_filings/{file}", "r") as f:
                data = json.load(f)
            
            client.batch.configure(batch_size=300)  # Configure batch
            with client.batch as batch:
                for i, d in enumerate(data):
                    

                    properties = {
                        "filing_type": d["filing_type"],
                        "company_name": d["company_name"],
                        "ticker": d["ticker"],
                        "accession_number": d["accession_number"],
                        "filing_url": d["filing_url"],
                        "text": d["text"],
                        "industry": d["industry"],
                        "city": d["city"],
                        "state": d["state"],
                        "zip_code": d["zip_code"],
                        "logo":  d["logo"],
                        "country":  d["country"],
                        "description": d["description"],
                        "cusip":  d["cusip"],
                        "isin": d["isin"],
                        "page_number": d["page_number"],
                        "report_date": f'{str(parser.parse(d["report_date"])).split(" ")[0]}T00:00:00.000Z'
                }

                    batch.add_data_object(properties, "Dow30_10K_10Q")

            with open(f"/home/ubuntu/tmcc-backend/to_delete/loading.log", "a") as f:
                f.write(file + "\n")

            print(f"Finished {file}")
        except Exception as e:
            print(f"Error {e}")
            raise(e)

        


    # print(data)

        # for file in tqdm(os.listdir(DATA_DIR)):
        #     try:
        #         if not file.endswith(".pdf"):
        #             continue
        #         ticker, filing_type, accession_number, report_date = file.split(".pdf")[0].split("_")
        #         report_year = int(report_date[:4])
        #         loader = PyMuPDFLoader(f"{DATA_DIR}/{file}")
        #         data_pages = loader.load()
        #         page_number = 0
        #         for page_num, data_page in enumerate(data_pages):
        #             properties = {
        #                 "filing_type": filing_type,
        #                 "company_name": TICKER_TO_COMPANY[ticker],
        #                 "ticker": ticker,
        #                 "accession_number": accession_number,
        #                 "s3_doc_url": file,
        #                 "text": "",
        #                 "page_number": page_num,
        #                 "report_date": parser.parse(report_date)
        #             }
        #             page_content = (
        #                 data_page.page_content[17:]
        #                 if data_page.page_content.lower().startswith(
        #                     "table of contents"
        #                 )
        #                 else data_page.page_content
        #             )
        #             page_chunks = token_text_splitter.split_text(page_content)
        #             for page_chunk in page_chunks:
        #                 properties["text"] = page_chunk
        #                 embedding = get_embedding(page_chunk)
        #                 dow30_collection.data.insert(properties, vector={"text": embedding})
        #     except Exception as e:
        #         print(f"Error {e}")
        #         continue
            

        #     print(f"Finished loading {ticker}")

    # finally:
    #     client.close()


upload_data_to_weaviate(client)