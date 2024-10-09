import os

from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings
from sec_cik_mapper import StockMapper
import pdfkit
import json

import weaviate
# import weaviate.classes.config as wc
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dateutil import parser
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import TokenTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv

from constants import DOW_30_TICKERS, NUM_10K_TO_DOWNLOAD, NUM_10Q_TO_DOWNLOAD, TICKER_TO_COMPANY


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("API key not found. Please set the `OPENAI_API_KEY` environment variable.")


DATA_DIR = "/home/ubuntu/tmcc-backend/data"
token_text_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=40)
oai_client = OpenAI(api_key=OPENAI_API_KEY)

# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(2))
def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    return oai_client.embeddings.create(input=[text], model=model).data[0].embedding


def get_cik_from_ticker(ticker):
    mapper = StockMapper()
    cik = mapper.ticker_to_cik[ticker.upper()]
    return cik


def upload_pdf_to_s3(local_file_path, bucket_name, s3_file_path):
    """
    Uploads a PDF file from a local file path to an S3 bucket.
    
    :param local_file_path: str, path to the local PDF file
    :param bucket_name: str, name of the S3 bucket
    :param s3_file_path: str, S3 key (path where the file will be saved in the bucket)
    :return: bool, True if file was uploaded, else False
    """
    # Create an S3 client
    s3_client = boto3.client('s3')
    
    try:
        # Upload the file to S3
        s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
        print(f"File {local_file_path} uploaded to {bucket_name}/{s3_file_path}")
        return True
    except FileNotFoundError:
        print(f"The file {local_file_path} was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except ClientError as e:
        print(f"Client error: {e}")
        return False


def download_sec_filings_as_local_pdfs():
    dl = Downloader("MyCompanyName", "email223@example.com")
    cmd_convert_to_pdf = 'wkhtmltopdf "{html_filename}" "{pdf_filename}"'

    ticker_to_filings_map = {}
    for ticker in DOW_30_TICKERS:
        cik = get_cik_from_ticker(ticker)
        try:
            metadatas = dl.get_filing_metadatas(
                RequestedFilings(ticker_or_cik=ticker, form_type="10-K", limit=11)
            )
        except Exception as e:
                print(f"Error getting metadatas for ticker {ticker}")
                continue

        for metadata in metadatas:
            try:
                html = dl.download_filing(url=metadata.primary_doc_url).decode()

                html_filename = f"/home/ubuntu/tmcc-backend/data/{ticker}_10K_{metadata.accession_number}_{metadata.report_date}.html"
                pdf_filename = f"/home/ubuntu/tmcc-backend/data/{ticker}_10K_{metadata.accession_number}_{metadata.report_date}.pdf"
                with open(html_filename, "w") as f:
                    f.write(html)

                os.system(cmd_convert_to_pdf.format(html_filename=html_filename, pdf_filename=pdf_filename))
                os.remove(html_filename)
            except Exception as e:
                # print(f"Error for ticker {ticker}")
                input(f"Error {e} ticker {ticker}, Press Enter to continue...")
                continue         
        
        try:
            metadatas = dl.get_filing_metadatas(
                RequestedFilings(ticker_or_cik=ticker, form_type="10-Q", limit=33)
            )
        except Exception as e:
                print(f"Error getting metadatas for ticker {ticker}")
                continue

        for metadata in metadatas:
            try:
                html = dl.download_filing(url=metadata.primary_doc_url).decode() 
                html_filename = f"/home/ubuntu/tmcc-backend/data/{ticker}_10Q_{metadata.accession_number}_{metadata.report_date}.html"
                pdf_filename = f"/home/ubuntu/tmcc-backend/data/{ticker}_10Q_{metadata.accession_number}_{metadata.report_date}.pdf"
                with open(html_filename, "w") as f:
                    f.write(html)

                # pdfkit.from_string(str(html), f"/home/ubuntu/tmcc-backend/data/{ticker}_10Q_{metadata.accession_number}_{metadata.report_date}.pdf", options={"enable-local-file-access": ""})    # html_content = None
                os.system(cmd_convert_to_pdf.format(html_filename=html_filename, pdf_filename=pdf_filename))
                os.remove(html_filename)
            except Exception as e:
                # print(f"Error for ticker {ticker}")
                input(f"Error {e} for ticker {ticker}, Press Enter to continue...")
                continue



# def upload_data_to_weaviate():
#     client = weaviate.connect_to_local(
#         host="localhost",
#         port=8080
#     )

#     """ citation attributes form the frontend

#     type Citation = {
#         id: number
#         company: string
#         title: string
#         importance: number
#         url: string
#         logo: string
#         page_number: number
#         report_date: string
#     }
#     """

#     try:
#         assert client.is_live()
#         client.collections.delete("Dow30_10K_10Q")
        
#         ## CREATE COLLLECTION
#         client.collections.create(
#             name="Dow30_10K_10Q",
#             properties=[
#                 wc.Property(name="filing_type", data_type=wc.DataType.TEXT), 
#                 wc.Property(name="company_name", data_type=wc.DataType.TEXT),
#                 wc.Property(name="ticker", data_type=wc.DataType.TEXT),
#                 wc.Property(name="accession_number", data_type=wc.DataType.TEXT),
#                 wc.Property(name="s3_doc_url", data_type=wc.DataType.TEXT),
#                 # wc.Property(name="text_embedding", data_type=wc.DataType.INT_ARRAY),
#                 wc.Property(name="text", data_type=wc.DataType.TEXT),
#                 wc.Property(name="page_number", data_type=wc.DataType.NUMBER),
#                 wc.Property(name="report_date", data_type=wc.DataType.DATE),
#             ],
#             # Define the vectorizer module (none, as we will add our own vectors)
#             vectorizer_config=wc.Configure.Vectorizer.none(),
#             # Define the generative module
#             generative_config=wc.Configure.Generative.cohere()
#         )
#         dow30_collection = client.collections.get("Dow30_10K_10Q")

#         for file in tqdm(os.listdir(DATA_DIR)):
#             try:
#                 if not file.endswith(".pdf"):
#                     continue
#                 ticker, filing_type, accession_number, report_date = file.split(".pdf")[0].split("_")
#                 report_year = int(report_date[:4])
#                 loader = PyMuPDFLoader(f"{DATA_DIR}/{file}")
#                 data_pages = loader.load()
#                 page_number = 0
#                 for page_num, data_page in enumerate(data_pages):
#                     properties = {
#                         "filing_type": filing_type,
#                         "company_name": TICKER_TO_COMPANY[ticker],
#                         "ticker": ticker,
#                         "accession_number": accession_number,
#                         "s3_doc_url": file,
#                         "text": "",
#                         "page_number": page_num,
#                         "report_date": parser.parse(report_date)
#                     }
#                     page_content = (
#                         data_page.page_content[17:]
#                         if data_page.page_content.lower().startswith(
#                             "table of contents"
#                         )
#                         else data_page.page_content
#                     )
#                     page_chunks = token_text_splitter.split_text(page_content)
#                     for page_chunk in page_chunks:
#                         properties["text"] = page_chunk
#                         embedding = get_embedding(page_chunk)
#                         dow30_collection.data.insert(properties, vector={"text": embedding})
#             except Exception as e:
#                 print(f"Error {e}")
#                 continue
            

#             print(f"Finished loading {ticker}")

#     finally:
#         client.close()


def chunk_data():
    # client = weaviate.connect_to_local(
    #     host="localhost",
    #     port=8080
    # )

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
        # assert client.is_live()
    # client.collections.delete("Dow30_10K_10Q")
    
    ## CREATE COLLLECTION
    # client.collections.create(
    #     name="Dow30_10K_10Q",
    #     properties=[
    #         wc.Property(name="filing_type", data_type=wc.DataType.TEXT), 
    #         wc.Property(name="company_name", data_type=wc.DataType.TEXT),
    #         wc.Property(name="ticker", data_type=wc.DataType.TEXT),
    #         wc.Property(name="accession_number", data_type=wc.DataType.TEXT),
    #         wc.Property(name="s3_doc_url", data_type=wc.DataType.TEXT),
    #         # wc.Property(name="text_embedding", data_type=wc.DataType.INT_ARRAY),
    #         wc.Property(name="text", data_type=wc.DataType.TEXT),
    #         wc.Property(name="page_number", data_type=wc.DataType.NUMBER),
    #         wc.Property(name="report_date", data_type=wc.DataType.DATE),
    #     ],
    #     # Define the vectorizer module (none, as we will add our own vectors)
    #     vectorizer_config=wc.Configure.Vectorizer.none(),
    #     # Define the generative module
    #     generative_config=wc.Configure.Generative.cohere()
    # )
    # dow30_collection = client.collections.get("Dow30_10K_10Q")
    chunk_id = 0

    # with open("/home/ubuntu/tmcc-backend/chunking.log", "w") as f:
    #     f.write("")


    chunks = []
    for file in tqdm(sorted(os.listdir(DATA_DIR))):
        try:
            if not file.endswith(".pdf"):
                continue
            ticker, filing_type, accession_number, report_date = file.split(".pdf")[0].split("_")
            report_year = int(report_date[:4])
            loader = PyMuPDFLoader(f"{DATA_DIR}/{file}")
            data_pages = loader.load()
            page_number = 0
            for page_num, data_page in enumerate(data_pages):
                properties = {
                    "filing_type": filing_type,
                    "company_name": TICKER_TO_COMPANY[ticker],
                    "ticker": ticker,
                    "accession_number": accession_number,
                    "s3_doc_url": file,
                    "text": "",
                    "page_number": page_num,
                    "report_date": report_date#parser.parse(report_date)
                }
                page_content = (
                    data_page.page_content[17:]
                    if data_page.page_content.lower().startswith(
                        "table of contents"
                    )
                    else data_page.page_content
                )
                page_chunks = token_text_splitter.split_text(page_content)
                for page_chunk in page_chunks:
                    chunk_id += 1
                    properties["text"] = page_chunk
                    # embedding = get_embedding(page_chunk)
                    # properties["embedding"] = embedding
                    # dow30_collection.data.insert(properties, vector={"text": embedding})
                    chunks.append(properties)
            with open(f"/home/ubuntu/tmcc-backend/data/chunked_filings/chunking_{accession_number}.json", "w") as f:
                print(f"Finished writing file {file} to chunks.json")
                f.write(json.dumps(chunks))
                chunks = []

            with open("/home/ubuntu/tmcc-backend/chunking.log", "a") as f:
                f.write(f"Finished writing file {file} to chunks.json\n")


        except Exception as e:
            print(f"Error {e}")
            continue

        print(f"Finished loading {ticker}")




def upload_pdfs_to_s3():
    BUCKET_NAME = "sec-filings2"
    s3_FILE_PATH = "dow30/{file}"

    for file in os.listdir(DATA_DIR):
        local_file_path = f"{DATA_DIR}/{file}"

        upload_successful = upload_pdf_to_s3(local_file_path, BUCKET_NAME, s3_FILE_PATH.format(file=file))
        if upload_successful:
            print("File uploaded successfully")
        else:
            print("File upload failed")




if __name__ == '__main__':
    # download_sec_filings_as_local_pdfs()
    # upload_pdfs_to_s3()    
    # upload_data_to_weaviate()
    chunk_data()

    





