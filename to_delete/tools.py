NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or "neo4j"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = "https://api.openai.com/v1/embeddings"
HATTEN_BACKEND_ROOT = os.getenv("HATTEN_BACKEND_ROOT")

load_dotenv(f"{HATTEN_BACKEND_ROOT}/hatten_app/.env.app", override=True)

url = "http://3.140.46.146:6333"
client = QdrantClient(url=url, prefer_grpc=False)

oai_client = OpenAI(api_key="sk-proj-Tb7SWc46QhFrvgU6pPY4T3BlbkFJRyGTP45AunM8ULAu9XLA")

ENTITY_TO_NODE_MAP = {
    "cash_flow": "CASH_FLOW",
    "income_statement": "INCOME_STATEMENT",
    "balance_sheet": "BALANCE_SHEET",
    "financial_position": "FINANCIAL_POSITION" 
}


# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(2))
def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    return oai_client.embeddings.create(input=[text], model=model).data[0].embedding


class Query(BaseModel):
    __root__: str


class Question(BaseModel):
    Question: str


COLLECTION_NAME = "10K"


graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

# LLMs
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
cypher_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
qa_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in graph.structured_schema.get("relationships")
]
# ##print(f"corrector_schema: {corrector_schema}")
cypher_validation = CypherQueryCorrector(corrector_schema)
COMPANY_TO_SVG_MAP = {
    'google': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-google" viewBox="0 0 16 16"><path d="M15.545 6.558a9.4 9.4 0 0 1 .139 1.626c0 2.434-.87 4.492-2.384 5.885h.002C11.978 15.292 10.158 16 8 16A8 8 0 1 1 8 0a7.7 7.7 0 0 1 5.352 2.082l-2.284 2.284A4.35 4.35 0 0 0 8 3.166c-2.087 0-3.86 1.408-4.492 3.304a4.8 4.8 0 0 0 0 3.063h.003c.635 1.893 2.405 3.301 4.492 3.301 1.078 0 2.004-.276 2.722-.764h-.003a3.7 3.7 0 0 0 1.599-2.431H8v-3.08z"/></svg>',
    'amazon': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-google" viewBox="0 0 16 16"><path d="M15.545 6.558a9.4 9.4 0 0 1 .139 1.626c0 2.434-.87 4.492-2.384 5.885h.002C11.978 15.292 10.158 16 8 16A8 8 0 1 1 8 0a7.7 7.7 0 0 1 5.352 2.082l-2.284 2.284A4.35 4.35 0 0 0 8 3.166c-2.087 0-3.86 1.408-4.492 3.304a4.8 4.8 0 0 0 0 3.063h.003c.635 1.893 2.405 3.301 4.492 3.301 1.078 0 2.004-.276 2.722-.764h-.003a3.7 3.7 0 0 0 1.599-2.431H8v-3.08z"/></svg>',
    'microsoft': '<svg xmlns="http://www.w3.org/2000/svg" width="2500" height="534" viewBox="0 0 1033.746 220.695" id="microsoft"><path fill="#706d6e" d="M1033.746 99.838v-18.18h-22.576V53.395l-.76.234-21.205 6.488-.418.127v21.415h-33.469v-11.93c0-5.555 1.242-9.806 3.69-12.641 2.431-2.801 5.908-4.225 10.343-4.225 3.188 0 6.489.751 9.811 2.232l.833.372V36.321l-.392-.144c-3.099-1.114-7.314-1.675-12.539-1.675-6.585 0-12.568 1.433-17.786 4.273-5.221 2.844-9.328 6.904-12.205 12.066-2.867 5.156-4.322 11.111-4.322 17.701v13.116h-15.72v18.18h15.72v76.589h22.567V99.838h33.469v48.671c0 20.045 9.455 30.203 28.102 30.203 3.064 0 6.289-.359 9.582-1.062 3.352-.722 5.635-1.443 6.979-2.213l.297-.176v-18.348l-.918.607c-1.225.816-2.75 1.483-4.538 1.979-1.796.505-3.296.758-4.458.758-4.368 0-7.6-1.177-9.605-3.5-2.028-2.344-3.057-6.443-3.057-12.177V99.838h22.575zM866.635 160.26c-8.191 0-14.649-2.716-19.2-8.066-4.579-5.377-6.899-13.043-6.899-22.783 0-10.049 2.32-17.914 6.901-23.386 4.554-5.436 10.95-8.195 19.014-8.195 7.825 0 14.054 2.635 18.516 7.836 4.484 5.228 6.76 13.03 6.76 23.196 0 10.291-2.14 18.196-6.36 23.484-4.191 5.248-10.493 7.914-18.732 7.914m1.003-80.885c-15.627 0-28.039 4.579-36.889 13.61-8.844 9.032-13.328 21.531-13.328 37.153 0 14.838 4.377 26.773 13.011 35.468 8.634 8.698 20.384 13.104 34.921 13.104 15.148 0 27.313-4.643 36.159-13.799 8.845-9.146 13.326-21.527 13.326-36.784 0-15.07-4.205-27.094-12.502-35.731-8.301-8.641-19.977-13.021-34.698-13.021m-86.602 0c-10.63 0-19.423 2.719-26.14 8.08-6.758 5.393-10.186 12.466-10.186 21.025 0 4.449.74 8.401 2.196 11.753 1.465 3.363 3.732 6.324 6.744 8.809 2.989 2.465 7.603 5.047 13.717 7.674 5.14 2.115 8.973 3.904 11.408 5.314 2.38 1.382 4.069 2.771 5.023 4.124.927 1.324 1.397 3.136 1.397 5.372 0 6.367-4.768 9.465-14.579 9.465-3.639 0-7.79-.76-12.337-2.258a46.347 46.347 0 0 1-12.634-6.406l-.937-.672v21.727l.344.16c3.193 1.474 7.219 2.717 11.964 3.695 4.736.979 9.039 1.477 12.777 1.477 11.535 0 20.824-2.732 27.602-8.125 6.821-5.43 10.278-12.67 10.278-21.525 0-6.388-1.861-11.866-5.529-16.284-3.643-4.382-9.966-8.405-18.785-11.961-7.026-2.82-11.527-5.161-13.384-6.958-1.79-1.736-2.699-4.191-2.699-7.3 0-2.756 1.122-4.964 3.425-6.752 2.321-1.797 5.552-2.711 9.604-2.711 3.76 0 7.607.594 11.433 1.758 3.823 1.164 7.181 2.723 9.984 4.63l.922.63v-20.61l-.354-.152c-2.586-1.109-5.996-2.058-10.138-2.828-4.123-.765-7.863-1.151-11.116-1.151m-95.157 80.885c-8.189 0-14.649-2.716-19.199-8.066-4.58-5.377-6.896-13.041-6.896-22.783 0-10.049 2.319-17.914 6.901-23.386 4.55-5.436 10.945-8.195 19.013-8.195 7.822 0 14.051 2.635 18.514 7.836 4.485 5.228 6.76 13.03 6.76 23.196 0 10.291-2.141 18.196-6.361 23.484-4.191 5.248-10.49 7.914-18.732 7.914m1.006-80.885c-15.631 0-28.044 4.579-36.889 13.61-8.844 9.032-13.331 21.531-13.331 37.153 0 14.844 4.38 26.773 13.014 35.468 8.634 8.698 20.383 13.104 34.92 13.104 15.146 0 27.314-4.643 36.16-13.799 8.843-9.146 13.326-21.527 13.326-36.784 0-15.07-4.206-27.094-12.505-35.731-8.303-8.641-19.977-13.021-34.695-13.021M602.409 98.07V81.658h-22.292v94.767h22.292v-48.477c0-8.243 1.869-15.015 5.557-20.13 3.641-5.054 8.493-7.615 14.417-7.615 2.008 0 4.262.331 6.703.986 2.416.651 4.166 1.358 5.198 2.102l.937.679V81.496l-.361-.155c-2.076-.882-5.013-1.327-8.729-1.327-5.602 0-10.615 1.8-14.909 5.344-3.769 3.115-6.493 7.386-8.576 12.712h-.237zm-62.213-18.695c-10.227 0-19.349 2.193-27.108 6.516-7.775 4.333-13.788 10.519-17.879 18.385-4.073 7.847-6.141 17.013-6.141 27.235 0 8.954 2.005 17.171 5.968 24.413 3.965 7.254 9.577 12.929 16.681 16.865 7.094 3.931 15.293 5.924 24.371 5.924 10.594 0 19.639-2.118 26.891-6.295l.293-.168v-20.423l-.937.684c-3.285 2.393-6.956 4.303-10.906 5.679-3.94 1.375-7.532 2.07-10.682 2.07-8.747 0-15.769-2.737-20.866-8.133-5.108-5.403-7.698-12.99-7.698-22.537 0-9.607 2.701-17.389 8.024-23.131 5.307-5.725 12.342-8.629 20.908-8.629 7.327 0 14.467 2.481 21.222 7.381l.935.679V84.371l-.302-.17c-2.542-1.423-6.009-2.598-10.313-3.489-4.286-.889-8.478-1.337-12.461-1.337m-66.481 2.284h-22.292v94.766h22.292V81.659zm-10.918-40.371c-3.669 0-6.869 1.249-9.498 3.724-2.64 2.482-3.979 5.607-3.979 9.295 0 3.63 1.323 6.698 3.938 9.114 2.598 2.409 5.808 3.63 9.54 3.63 3.731 0 6.953-1.221 9.582-3.626 2.646-2.42 3.988-5.487 3.988-9.118 0-3.559-1.306-6.652-3.879-9.195-2.571-2.538-5.833-3.824-9.692-3.824m-55.62 33.379v101.758h22.75V44.189H398.44l-40.022 98.221-38.839-98.221H286.81v132.235h21.379V74.657h.734l41.013 101.768h16.134l40.373-101.758h.734z"></path><path fill="#f1511b" d="M104.868 104.868H0V0h104.868v104.868z"></path><path fill="#80cc28" d="M220.654 104.868H115.788V0h104.866v104.868z"></path><path fill="#00adef" d="M104.865 220.695H0V115.828h104.865v104.867z"></path><path fill="#fbbc09" d="M220.654 220.695H115.788V115.828h104.866v104.867z"></path></svg>',
    'apple': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-apple" viewBox="0 0 16 16"><path d="M11.182.008C11.148-.03 9.923.023 8.857 1.18c-1.066 1.156-.902 2.482-.878 2.516.024.034 1.52.087 2.475-1.258.955-1.345.762-2.391.728-2.43zm3.314 11.733c-.048-.096-2.325-1.234-2.113-3.422.212-2.189 1.675-2.789 1.698-2.854.023-.065-.597-.79-1.254-1.157a3.692 3.692 0 0 0-1.563-.434c-.108-.003-.483-.095-1.254.116-.508.139-1.653.589-1.968.607-.316.018-1.256-.522-2.267-.665-.647-.125-1.333.131-1.824.328-.49.196-1.422.754-2.074 2.237-.652 1.482-.311 3.83-.067 4.56.244.729.625 1.924 1.273 2.796.576.984 1.34 1.667 1.659 1.899.319.232 1.219.386 1.843.067.502-.308 1.408-.485 1.766-.472.357.013 1.061.154 1.782.539.571.197 1.111.115 1.652-.105.541-.221 1.324-1.059 2.238-2.758.347-.79.505-1.217.473-1.282z"/></svg>'
}


def get_entities_from_extraction_service(query):
    query = query.replace("/", " ")
    res = requests.get(f"http://127.0.0.1:8085/message/{query}/")
    ##print(f"res: {res}")
    return res.json()


def process_pipeline(ex_pipeline, complete_exe_pipeline):
    print(f"ex_pipeline: {ex_pipeline}")
    for query, func_strings in ex_pipeline.items():
        extracted_entities = get_entities_from_extraction_service(query)
        funcs = [FUNCS_MAP[f] for f in func_strings]
        # #print(f"funcs pipeline: {funcs}")
        output = query
        for func in funcs:
            # print(f"process_pipeline")
            # print(f"Output BEFORE calling {func.__name__}: {output}")
            output = func(
                output, extracted_entities, ex_pipeline, complete_exe_pipeline
            )
            # print(f"Output AFTER calling {func.__name__}: {output}")

    return complete_exe_pipeline


"""
- Show your work in UI (list the subtasks)
- Have a calculator as a separate function  (e.g. effective tax rate)

"""
"""

ticker = get_entities_from_extraction_service(question)["entities"][0]["ticker"]
    # ##print(f"ticker: {ticker}")
    # MATCH (i:INCOME_STATEMENT) WHERE i.ticker ='MSFT' RETURN (i.table)
    table = []
    for statement in ["INCOME_STATEMENT"]:
        res_from_graph = graph.query(
            f"MATCH (i:{statement}) WHERE i.ticker='{ticker}' RETURN (i.table)"
        )
        table.append(f"{statement}:\n")
        table.append("\n".join([r["(i.table)"] for r in res_from_graph]))

    # res_from_graph = graph.query(result_from_query)
    # ##print(f"res_from_graph: {res_from_graph[0]}")

    response = (
        RunnablePassthrough.assign(question=lambda x: x)
        | get_financials_prompt
        | llm
        | StrOutputParser()
    )
"""


def get_product_segments(query, extracted_data, ex_pipeline, complete_exe_pipeline):
    # ##print(f"INSIDE get_product_segment!!")
    # ##print(f"extracted_data: {extracted_data}")
    # __product_segment__
    # a, b = query.split("->")

    vector_search_response = (
        RunnablePassthrough.assign(
            Question=lambda x: x["Question"],
            extracted_data=lambda x: x["extracted_data"],
        )
        | RunnableLambda(get_context_from_vector_search_query_synth)
        | vector_search_get_list_prompt
        | cypher_llm
        | StrOutputParser()
    )

    subtask = query

    response_vector_search_from_chain = vector_search_response.invoke(
        {"Question": subtask, "extracted_data": extracted_data}
    )
    response_vector_search_from_chain = json.loads(response_vector_search_from_chain)[:]
    if complete_exe_pipeline:
        complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
            {
                "subtask": query,
                "response": response_vector_search_from_chain,
            }
        )
    # ##print(f"response_vector_search_from_chain: {response_vector_search_from_chain}")

    return {"Question": query, "subtask_responses": response_vector_search_from_chain}


# def get_competitors(query, extracted_data, ex_pipeline, complete_exe_pipeline):
#     entities = get_entities_from_extraction_service(query)
#     ticker = None
#     for ent in entities["entities"]:
#         if ent["entity"] == "ticker":
#             ticker = ent["value"]
#     res_from_graph = graph.query(
#         f"""
#     MATCH (c:COMPANY)-[r:IS_COMPETITOR]->(b:COMPANY) 
#         WHERE c.ticker='{ticker}' 
#     RETURN collect(b.ticker)
#     """
#     )
#     ##print(f"res_from_graph: {res_from_graph}")
#     response_from_query = res_from_graph[0][
#         "collect(b.ticker)"
#     ]  # json.loads(res_from_graph)
#     ##print(f"response_from_query: {response_from_query}")
#     # TODO: Remove this later
#     filtered_response_from_query = []
#     for t in response_from_query:
#         if t in ["MSFT", "GOOG", "INTC", "CSCO", "HPQ", "ORCL", "AMZN", "IBM"]:
#             filtered_response_from_query.append(t)

#     ##print(f"filtered_response_from_query: {filtered_response_from_query}")
#     if complete_exe_pipeline:
#         complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
#             {
#                 "subtask": query,
#                 "response": filtered_response_from_query,
#             }
#         )
#     return {"Question": query, "subtask_responses": filtered_response_from_query}


def filter_comps_against_available_data(competitors):
    AVAILABLE_TICKERS = ["MSFT", "AAPL", "GOOG"]
    new_comps = []
    for c in competitors:
        if c in AVAILABLE_TICKERS:
            new_comps.append(c.lower())
        elif c == "GOOGL":
            new_comps.append("goog")

    return new_comps


def get_competitors(query, extracted_data, ex_pipeline, complete_exe_pipeline):
    # print(f"complete_exe_pipeline:\n{complete_exe_pipeline}")
    entities = get_entities_from_extraction_service(query)
    company_reference = None
    for ent in entities["entities"]:
        if ent["entity"] == "ticker" or ent["entity"] == "company_name":
            company_reference = ent["value"]


    get_competitors_template = """
    who are the competitors for {CompanyReference}? your response should be a JSON list of public US company tickers and NOTHING else. Do not include Google in your response.
    """

    get_competitors_prompt = PromptTemplate(
        template=get_competitors_template,
        input_variables=["CompanyReference"],
    )
    get_competitors_response = (
        RunnablePassthrough.assign(
            CompanyReference=lambda x: x["CompanyReference"]
        )
        | get_competitors_prompt
        | cypher_llm
        | StrOutputParser()
    )

    competitors = get_competitors_response.invoke(
        {"CompanyReference": company_reference}
    )
    if '```json' in competitors:
        competitors = competitors[7:len(competitors)-3]

    competitors = ast.literal_eval(competitors)

    # TODO: REMOVE AFTER WE LOAD ALL EQUITIES
    competitors = filter_comps_against_available_data(competitors)

    ##print(f"filtered_response_from_query: {filtered_response_from_query}")
    if complete_exe_pipeline:
        complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
            {
                "subtask": query,
                "response": competitors,
            }
        )
    return {"Question": query, "subtask_responses": competitors}


def get_neo4j_date_filter_where_clause_from_extracted_entities(
    entity_dict,
):
    """NOTE: the date filter cases
    # past 3 quarters
    {
            "entity": "time",
            "from": "2023-07-01",
            "to": "2024-04-01"
    }
    """
    where_clause = None
    if entity_dict["entity"] == "time":
        if "from" in entity_dict["entity"]:
            where_clause = (
                """ ("""
                + entity_dict["entity"]["from"]
                + """ >= c.contains_data_from_date OR """
                + entity_dict["entity"]["to"]
                + """ <= i.contains_data_to_date)"""
            )
        else:
            where_clause = (
            """ ("""
            + str(entity_dict["value"])
            + """ >= i.contains_data_from_date.year AND """
            + entity_dict["value"]
            + """ <= i.contains_data_to_date.year)"""
        )

    elif entity_dict["entity"] == "date_year":
        where_clause = (
            """ ("""
            + entity_dict["value"]
            + """ >= i.contains_data_from_date.year AND """
            + entity_dict["value"]
            + """ <= i.contains_data_to_date.year)"""
        )

    return where_clause


def get_equation(question, vars_table):
    # print(f"vars_table:\n{vars_table}")
    # print(f"question:\n{question}")
    get_equation_template = """
    Given a user question and a table that contains the data necessary to answer the question, provide the equation required to answer the user's question.
    Your response should be a JSON list where the first and only element is a string that contains only the equaition required to answer the question.

    ## EXAMPLES ##
    Question: What was msft's revenue difference between the US and Non-US regions for the period 2023-07-01 to 2023-09-30?
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>country:US</th><th>us-gaap:NonUsMember</th></tr><tr><th>Period</th><th></th><th></th></tr></thead><tbody><tr><th>2023-07-01-2023-09-30</th><td>A1</td><td>B1</td></tr></tbody></table>
    Answer: ['A1 - B1']

    Question: "What's the quick ratio for amzn for 2023?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2023-03-31</th><th>2023-06-30</th><th>2023-09-30</th><th>2023-12-31</th></tr><tr><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><th>AccountsReceivableNetCurrent</th><td>A0</td><td>B0</td><td>C0</td><td>D0</td></tr><tr><th>CashAndCashEquivalentsAtCarryingValue</th><td>A1</td><td>B1</td><td>C1</td><td>D1</td></tr><tr><th>MarketableSecuritiesCurrent</th><td>A2</td><td>B2</td><td>C2</td><td>D2</td></tr><tr><th>LiabilitiesCurrent</th><td>A3</td><td>B3</td><td>C3</td><td>D3</td></tr></tbody></table>
    Answer: ['sum([A0, B0, C0, D0, A1, B1, C1, D1, A2, B2, C2, D2]) / sum([A3, B3, C3, D3])']

    Question: "What was msft's revenue difference between the US and Non-US regions for the period 2023-07-01 to 2023-09-30?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>country:US</th><th>us-gaap:NonUsMember</th></tr><tr><th>Period</th><th></th><th></th></tr></thead><tbody><tr><th>2023-07-01-2023-09-30</th><td>A1</td><td>B1</td></tr></tbody></table>
    Answer: ['A1 - B1']

    Question: "What's the quick ratio for amzn for 2023?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2023-03-31</th><th>2023-06-30</th><th>2023-09-30</th><th>2023-12-31</th></tr><tr><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><th>AccountsReceivableNetCurrent</th><td>A0</td><td>B0</td><td>C0</td><td>D0</td></tr><tr><th>CashAndCashEquivalentsAtCarryingValue</th><td>A1</td><td>B1</td><td>C1</td><td>D1</td></tr><tr><th>MarketableSecuritiesCurrent</th><td>A2</td><td>B2</td><td>C2</td><td>D2</td></tr><tr><th>LiabilitiesCurrent</th><td>A3</td><td>B3</td><td>C3</td><td>D3</td></tr></tbody></table>
    Answer: ['sum([A0, B0, C0, D0, A1, B1, C1, D1, A2, B2, C2, D2]) / sum([A3, B3, C3, D3])']

    Question: "What's the acid-test for amzn for 2023?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2023-03-31</th><th>2023-06-30</th><th>2023-09-30</th><th>2023-12-31</th></tr><tr><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><th>AccountsReceivableNetCurrent</th><td>A0</td><td>B0</td><td>C0</td><td>D0</td></tr><tr><th>CashAndCashEquivalentsAtCarryingValue</th><td>A1</td><td>B1</td><td>C1</td><td>D1</td></tr><tr><th>MarketableSecuritiesCurrent</th><td>A2</td><td>B2</td><td>C2</td><td>D2</td></tr><tr><th>LiabilitiesCurrent</th><td>A3</td><td>B3</td><td>C3</td><td>D3</td></tr></tbody></table>
    Answer: ['sum([A0, B0, C0, D0, A1, B1, C1, D1, A2, B2, C2, D2]) / sum([A3, B3, C3, D3])']
    
    Question: "What is the trend in AMZN's Retained Earnings Accumulated Deficit from '2020-12-31', '2021-06-30', '2021-09-30', '2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31', '2024-03-31'?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2020-12-31</th><th>2021-06-30</th><th>2021-09-30</th><th>2021-12-31</th><th>2022-03-31</th><th>2022-06-30</th><th>2022-09-30</th><th>2022-12-31</th><th>2023-03-31</th><th>2023-06-30</th><th>2023-09-30</th><th>2023-12-31</th><th>2024-03-31</th></tr><tr><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><th>RetainedEarningsAccumulatedDeficit</th><td>A0</td><td>B0</td><td>C0</td><td>D0</td><td>E0</td><td>F0</td><td>G0</td><td>H0</td><td>I0</td><td>J0</td><td>K0</td><td>L0</td><td>M0</td></tr></tbody></table>
    Answer: ['A0, B0, C0, D0, E0, F0, G0, H0, I0, J0, K0, L0, M0']

    Question: "What's MSFT's EPS (basic) for 2023?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2022-07-01-2022-09-30</th><th>2022-10-01-2022-12-31</th><th>2023-01-01-2023-03-31</th><th>2023-03-31-2023-06-30</th></tr><tr><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><th>EarningsPerShareBasic</th><td>A0</td><td>B0</td><td>C0</td><td>D0</td></tr></tbody></table>
    Answer: ['sum([A0, B0, C0, D0])']

    Question: "What's GOOG's EPS (basic) for '2022-01-01-2022-03-31', '2022-04-01-2022-06-30', '2022-07-01-2022-09-30'?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2022-01-01-2022-03-31</th><th>2022-04-01-2022-06-30</th><th>2022-07-01-2022-09-30</th></tr><tr><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><th>EarningsPerShareBasic</th><td>A0</td><td>B0</td><td>C0</td></tr></tbody></table>
    Answer: ['sum([A0, B0, C0])']

    Question: "What is the quick ratio for Amazon in '2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31'?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2023-03-31</th><th>2023-06-30</th><th>2023-09-30</th><th>2023-12-31</th></tr><tr><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><th>AccountsReceivableNetCurrent</th><td>A0</td><td>B0</td><td>C0</td><td>D0</td></tr><tr><th>CashAndCashEquivalentsAtCarryingValue</th><td>A1</td><td>B1</td><td>C1</td><td>D1</td></tr><tr><th>MarketableSecuritiesCurrent</th><td>A2</td><td>B2</td><td>C2</td><td>D2</td></tr><tr><th>LiabilitiesCurrent</th><td>A3</td><td>B3</td><td>C3</td><td>D3</td></tr></tbody></table>
    Answer: ['sum([A0, B0, C0, D0, A1, B1, C1, D1, A2, B2, C2, D2]) / sum([A3, B3, C3, D3])']

    Question: "what's Amazon's current ratio for Q1 2023?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2023-03-31</th></tr><tr><th></th><th></th></tr></thead><tbody><tr><th>AssetsCurrent</th><td>A0</td></tr><tr><th>LiabilitiesCurrent</th><td>A1</td></tr></tbody></table>
    Answer: ['A0 / A1']

    Question: "what's Amazon's current ratio for 2023?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2023-03-31</th><th>2023-06-30</th><th>2023-09-30</th><th>2023-12-31</th></tr><tr><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><th>AssetsCurrent</th><td>A0</td><td>B0</td><td>C0</td><td>D0</td></tr><tr><th>LiabilitiesCurrent</th><td>A1</td><td>B1</td><td>C1</td><td>D1</td></tr></tbody></table>
    Answer: ['sum([A0, B0, C0, D0]) / sum([A1, B1, C1, D1])']

    Question: "What's Amazon's current ratio for '2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31'?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2023-03-31</th><th>2023-06-30</th><th>2023-09-30</th><th>2023-12-31</th></tr><tr><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><th>AssetsCurrent</th><td>A0</td><td>B0</td><td>C0</td><td>D0</td></tr><tr><th>LiabilitiesCurrent</th><td>A1</td><td>B1</td><td>C1</td><td>D1</td></tr></tbody></table>
    Answer: ['sum([A0, B0, C0, D0]) / sum([A1, B1, C1, D1])']

    Question: "what's the cash ratio for amzn for 2023?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2023-03-31</th><th>2023-06-30</th><th>2023-09-30</th><th>2023-12-31</th></tr><tr><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><th>CashAndCashEquivalentsAtCarryingValue</th><td>A0</td><td>B0</td><td>C0</td><td>D0</td></tr><tr><th>LiabilitiesCurrent</th><td>A1</td><td>B1</td><td>C1</td><td>D1</td></tr></tbody></table>
    Answer: ['sum([A0, B0, C0, D0]) / sum([A1, B1, C1, D1])']

    Question: "What's the cash ratio for amzn for '2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31'?"
    Table: <table border="1" class="dataframe"><thead><tr style="text-align: right;"><th></th><th>2023-03-31</th><th>2023-06-30</th><th>2023-09-30</th><th>2023-12-31</th></tr><tr><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><th>CashAndCashEquivalentsAtCarryingValue</th><td>A0</td><td>B0</td><td>C0</td><td>D0</td></tr><tr><th>LiabilitiesCurrent</th><td>A1</td><td>B1</td><td>C1</td><td>D1</td></tr></tbody></table>
    Answer: ['sum([A0, B0, C0, D0]) / sum([A1, B1, C1, D1])']

    Question: {Question}
    Table: {Table}
    """

    get_equation_prompt = PromptTemplate(
        template=get_equation_template,
        input_variables=["Question", "Table"],
    )
    get_equation_response = (
        RunnablePassthrough.assign(
            Question=lambda x: x["Question"],
            Table=lambda x: x["Table"],
        )
        | get_equation_prompt
        | cypher_llm
        | StrOutputParser()
    )

    equation = get_equation_response.invoke(
        {"Question": question, "Table": vars_table}
    )
    if '```json' in equation:
        equation = equation[7:len(equation)-3]

    equation = ast.literal_eval(equation)
    print(f"equation: {equation}")

    return equation[0]


def solve_equation(equation, df_vars, df_vals):
    print(f"VARS equation: {equation}")
    print(f"df_vars: {df_vars}")
    print(f"df_vals: {df_vals}")
    # import pdb
    # pdb.set_trace()

    for ind in reversed(df_vars.index):
        for col in df_vars.columns:
            # print(f"ind: {ind}, col: {col}")
            val = df_vals[col][ind]
            var = df_vars[col][ind]
            # print(f"var: {var}, val: {val}")
            if var in equation:
                # print(f"Replacing var: {var} with str(val): {str(val)}")
                equation = equation.replace(var, str(val))
                # print(f"equation:\n{equation}")

    # print(f"NUMERICAL equation: {equation}")
    equation = equation.lower()
    if 'nan' in equation:
        # TODO: send this to equation_fixer_function()
        print(f"Found nan in equation! Replace it with ''")
        equation = equation.replace('nan', '0')
        return "Not found"
    # print(f"equation: {equation}")
    solution = ""
    try:
        solution = eval(equation)
    except Exception as e:
        print(f"Error in solve_equation: {e}")
        print(f"df_vals: \n{df_vals}")
        print(f"Not found")
        return "Not found"

    return solution


def get_explanation_with_solution(question, solution):
    get_explanation_with_solution_template = """
    Given a user question and a numerical solution to the question, use the solution to provide a resonse using language. 
    Do not make up any values. Your reponse must contain the provided numerical solution.

    ## EXAMPLES ##
    Question: What was msft's revenue difference between the US and Non-US regions for the period 2023-07-01 to 2023-09-30?
    Solution: 1107000000
    Answer: The revenue difference for msft between the US and Non-US regions for the period 2023-07-01 to 2023-09-30 was 1107000000.

    Question: What was msft's revenue difference between the US and Non-US regions for the period 2023-07-01 to 2023-09-30?
    Solution: 1107000000
    Answer: The revenue difference for msft between the US and Non-US regions for the period 2023-07-01 to 2023-09-30 was 1107000000.

    Question: {Question}
    Solution: {Solution}
    """

    get_explanation_with_solution_prompt = PromptTemplate(
        template=get_explanation_with_solution_template,
        input_variables=["Question", "Solution"],
    )
    get_explanation_with_solution_response = (
        RunnablePassthrough.assign(
            Question=lambda x: x["Question"],
            extracted_data=lambda x: x["Solution"],
        )
        | get_explanation_with_solution_prompt
        | cypher_llm
        | StrOutputParser()
    )

    explanation_with_solution = get_explanation_with_solution_response.invoke(
        {"Question": question, "Solution": solution}
    )

    print(f"explanation_with_solution: {explanation_with_solution}")
    return explanation_with_solution


def get_vars_table_from_values_table(t_vals):
    data_dict = {c: [] for c in t_vals.columns}
    data_dict[t_vals.index.name] = []
    vars_cols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, ind in enumerate(t_vals.index):
        data_dict[t_vals.index.name].append(ind)
        for j, col in enumerate(t_vals.columns):
            data_dict[col].append(f"{vars_cols[j]}{i}")

    df_vars = pd.DataFrame(data=data_dict).set_index(t_vals.index.name)
    return df_vars


def get_revenue_by_region(question, extracted_data, ex_pipeline, complete_exe_pipeline):
    entities_from_extraction_service = get_entities_from_extraction_service(question)
    where_clauses = []
    statement_types = []
    quarterly_date_year_subs = []
    ##print(f'entities_from_extraction_service["entities"]: {entities_from_extraction_service["entities"]}')
    for ent in entities_from_extraction_service["entities"]:
        if ent["entity"] == "ticker":
            ticker = ent["value"]
            where_clauses.append(f"c.ticker='{ticker.lower()}'")
        elif ent["entity"]=="company_name":
            company_name = ent["value"]
            where_clauses.append(f"c.company_name='{company_name}'")
        elif ent["entity"]=="statement_type":
            statement_type = ENTITY_TO_NODE_MAP[ent["value"]]
            statement_types.append(statement_type)
        elif ent["entity"]=="filing_type":
            continue
        elif ent["entity"]=="quarterly_date_year":
            quarterly_date_year_subs.append(ent["value"])
        else:
            continue
            # where_clauses.append(get_neo4j_date_filter_where_clause_from_extracted_entities(ent))

    table = []
    
    where_clause = ' AND '.join(where_clauses)

    fiscal_or_calendar_response = (
            RunnablePassthrough.assign(Question=lambda x: x)
            | fiscal_or_calendar_prompt
            | llm
            | StrOutputParser()
        )

    financial_reporting_calendar_response = (
            RunnablePassthrough.assign(Question=lambda x: x)
            | financial_reporting_calendar_prompt_v2
            | llm
            | StrOutputParser()
        )

    fiscal_or_calendar_response = fiscal_or_calendar_response.invoke({"Question": question})
    fiscal_or_calendar_str_val = ast.literal_eval(fiscal_or_calendar_response)[0]
    print(f"fiscal_or_calendar_str_val: {fiscal_or_calendar_str_val}")
    financial_reporting_response = financial_reporting_calendar_response.invoke({"Question": question, "Mode": fiscal_or_calendar_str_val})
    print(f"financial_reporting_response: {financial_reporting_response}")
    columns = financial_reporting_response
    if '```json' in columns:
        columns = columns[7:len(columns)-3]

    date_rows = ast.literal_eval(columns)
    
    updated_question = question
    for q, d in zip(quarterly_date_year_subs, date_rows):
        updated_question = updated_question.replace(q, f"{q} ({d})")

    res_from_graph = graph.query(
        f"""MATCH (c:company) WHERE {where_clause}
        RETURN c.revenue_by_region_table as table"""
    )
    
    df = pd.read_html(res_from_graph[-1]["table"], index_col=1, header=0)[0]
    df.drop("Unnamed: 0", axis=1, inplace=True)
    
    update_question_with_fallback = False
    date_rows_to_keep = set()
    for d in list(df.index):
        # #print(f"d:{d}")
        for row in date_rows:
            if len(d) > 10:
                if row in d[10:]:
                    date_rows_to_keep.add(d)
                    if row not in updated_question:
                        update_question_with_fallback = True          
            else:
                if row in d:
                    date_rows_to_keep.add(d)
                    if row not in updated_question:
                        update_question_with_fallback = True
    
    if update_question_with_fallback:
        updated_question = get_update_question_with_fallback(updated_question, sorted(list(date_rows_to_keep)))

    question = updated_question
    if isinstance(question, list):
        complete_exe_pipeline["UserQuestion"] += " " + question[0]
    else:
        complete_exe_pipeline["UserQuestion"] += " " + question

    cols_to_keep = select_relevant_values(question, list(df.columns))
    df.sort_index(axis=0, inplace=True)
    df = df.loc[sorted(list(date_rows_to_keep)), cols_to_keep]

    t_vals = df.copy()
    df_vars = get_vars_table_from_values_table(t_vals)
    equation = get_equation(question, df_vars.to_html())
    solution = solve_equation(equation, df_vars, df)
    explanation_with_solution = get_explanation_with_solution(question, solution)

    if FINAL_PROMPT_MAP["get_revenue_by_region"]:
        complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
            {"subtask": updated_question, "response": f"\n{explanation_with_solution}"}
        )

    return explanation_with_solution


def select_relevant_values(user_question, list_of_values):
    print(f"list_of_values:\n{list_of_values}")

    value_selector_response = (
            RunnablePassthrough.assign(Question=lambda x: x, Values=lambda x: x)
            | row_selector_prompt
            | llm
            | StrOutputParser()
        )

    value_selector_response = value_selector_response.invoke({"Question": user_question, "Values": list_of_values})
    # print(f"value_selector_response: \n{value_selector_response}")
    values = value_selector_response
    if '```json' in values:
        values = values[7:len(values)-3]

    print(f"List of financial metrics selected: {values}")

    return ast.literal_eval(values)


def get_update_question_with_fallback(updated_question, date_rows):
    update_question_fallback_response = (
            RunnablePassthrough.assign(Question=lambda x: x, Values=lambda x: x)
            | update_question_fallback_prompt
            | llm
            | StrOutputParser()
        )

    update_question_fallback_response = update_question_fallback_response.invoke({"Question": updated_question, "Values": date_rows})
    values = update_question_fallback_response
    if '```json' in values:
        values = values[7:len(values)-3]

    return ast.literal_eval(values)


def get_revenue_by_product(question, extracted_data, ex_pipeline, complete_exe_pipeline):
    entities_from_extraction_service = get_entities_from_extraction_service(question)
    where_clauses = []
    statement_types = []
    quarterly_date_year_subs = []
    for ent in entities_from_extraction_service["entities"]:
        if ent["entity"] == "ticker":
            ticker = ent["value"]
            where_clauses.append(f"c.ticker='{ticker.lower()}'")
        elif ent["entity"]=="company_name":
            company_name = ent["value"]
            where_clauses.append(f"c.company_name='{company_name}'")
        elif ent["entity"]=="statement_type":
            statement_type = ENTITY_TO_NODE_MAP[ent["value"]]
            statement_types.append(statement_type)
        elif ent["entity"]=="filing_type":
            continue
        elif ent["entity"]=="quarterly_date_year":
            quarterly_date_year_subs.append(ent["value"])
        else:
            continue
            # where_clauses.append(get_neo4j_date_filter_where_clause_from_extracted_entities(ent))

    table = []

    
    where_clause = ' AND '.join(where_clauses)

    fiscal_or_calendar_response = (
            RunnablePassthrough.assign(Question=lambda x: x)
            | fiscal_or_calendar_prompt
            | llm
            | StrOutputParser()
        )

    financial_reporting_calendar_response = (
            RunnablePassthrough.assign(Question=lambda x: x)
            | financial_reporting_calendar_prompt_v2
            | llm
            | StrOutputParser()
        )

    fiscal_or_calendar_response = fiscal_or_calendar_response.invoke({"Question": question})
    fiscal_or_calendar_str_val = ast.literal_eval(fiscal_or_calendar_response)[0]
    print(f"fiscal_or_calendar_str_val: {fiscal_or_calendar_str_val}")
    financial_reporting_response = financial_reporting_calendar_response.invoke({"Question": question, "Mode": fiscal_or_calendar_str_val})
    print(f"financial_reporting_response: {financial_reporting_response}")
    ##print(f"financial_reporting_response: {financial_reporting_response}")
    date_rows = financial_reporting_response
    if '```json' in date_rows:
        date_rows = date_rows[7:len(date_rows)-3]

    date_rows = ast.literal_eval(date_rows)
    
    updated_question = question
    for q, r   in zip(quarterly_date_year_subs, date_rows):
        updated_question = updated_question.replace(q, f"{q} ({r})")

    res_from_graph = graph.query(
        f"""MATCH (c:company) WHERE {where_clause}
        RETURN c.revenue_by_product_table as table"""
    )
    
    df = pd.read_html(res_from_graph[-1]["table"], index_col=1, header=0)[0]
    df.drop("Unnamed: 0", axis=1, inplace=True)
    
    update_question_with_fallback = False
    date_rows_to_keep = set()
    for d in list(df.index):
        for row in date_rows:
            if len(d) > 10:
                if row in d[10:]:
                    date_rows_to_keep.add(d)
                    if row not in updated_question:
                        update_question_with_fallback = True          
            else:
                if row in d:
                    date_rows_to_keep.add(d)
                    if row not in updated_question:
                        update_question_with_fallback = True
    
    if update_question_with_fallback:
        updated_question = get_update_question_with_fallback(updated_question, sorted(list(date_rows_to_keep)))

    question = updated_question
    cols_to_keep = select_relevant_values(question, list(df.columns))
    df.sort_index(axis=1, inplace=True)
    df = df.loc[sorted(list(date_rows_to_keep)), cols_to_keep]
    
    # table.append("\n".join(df.to_html()))

    # response = (
    #     RunnablePassthrough.assign(question=lambda x: x)
    #     | get_financials_prompt
    #     | llm
    #     | StrOutputParser()
    # )
    # response = response.invoke({"question": updated_question, "table": "\n".join(table)})

    # if FINAL_PROMPT_MAP["get_revenue_by_product"]:
    #     complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
    #         {"subtask": updated_question, "response": f"\n{response}"}
    #     )

    # return response
    t_vals = df.copy()
    df_vars = get_vars_table_from_values_table(t_vals)
    equation = get_equation(question, df_vars.to_html())
    solution = solve_equation(equation, df_vars, df)
    explanation_with_solution = get_explanation_with_solution(question, solution)

    if FINAL_PROMPT_MAP["get_revenue_by_region"]:
        complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
            {"subtask": updated_question, "response": f"\n{explanation_with_solution}"}
        )

    return explanation_with_solution


def get_company_financials(question, extracted_data, ex_pipeline, complete_exe_pipeline):
    entities_from_extraction_service = get_entities_from_extraction_service(question)
    where_clauses = []
    statement_types = []
    quarterly_date_year_subs = []
    for ent in entities_from_extraction_service["entities"]:
        if ent["entity"] == "ticker":
            ticker = ent["value"]
            where_clauses.append(f"c.ticker='{ticker.lower()}'")
        elif ent["entity"]=="company_name":
            company_name = ent["value"]
            where_clauses.append(f"c.company_name='{company_name.lower()}'")
        elif ent["entity"]=="statement_type":
            statement_type = ENTITY_TO_NODE_MAP[ent["value"]]
            statement_types.append(statement_type)
        elif ent["entity"]=="filing_type":
            continue
        elif ent["entity"]=="quarterly_date_year":
            quarterly_date_year_subs.append(ent["value"])
        else:
            continue

    table = []
    responses = []
    
    where_clause = ' AND '.join(where_clauses)
    fiscal_or_calendar_response = (
            RunnablePassthrough.assign(Question=lambda x: x)
            | fiscal_or_calendar_prompt
            | llm
            | StrOutputParser()
        )

    financial_reporting_calendar_response = (
            RunnablePassthrough.assign(Question=lambda x: x)
            | financial_reporting_calendar_prompt_v2
            | llm
            | StrOutputParser()
        )

    fiscal_or_calendar_response = fiscal_or_calendar_response.invoke({"Question": question})
    fiscal_or_calendar_str_val = ast.literal_eval(fiscal_or_calendar_response)[0]
    print(f"fiscal_or_calendar_str_val: {fiscal_or_calendar_str_val}")
    financial_reporting_response = financial_reporting_calendar_response.invoke({"Question": question, "Mode": fiscal_or_calendar_str_val})
    print(f"financial_reporting_response: {financial_reporting_response}")
    columns = financial_reporting_response
    if '```json' in columns:
        columns = columns[7:len(columns)-3]

    columns = ast.literal_eval(columns)
    print(f"columns: {columns}")
    
    updated_question = question
    for q, c in zip(quarterly_date_year_subs, columns):
        updated_question = updated_question.replace(q, f"{q} ({c})")

    res_from_graph = graph.query(
        f"""MATCH (c:company) WHERE {where_clause}
        RETURN c.financial_statements_table_table as table"""
    )
    df = pd.read_html(res_from_graph[-1]["table"], index_col=1, header=0)[0]
    df.drop('Unnamed: 0', axis=1,inplace=True)
    print(f"df: {df}")

    update_question_with_fallback = False
    cols_to_keep = set()
    for c in list(df.columns):
        for col in columns:
            if len(c) > 100:
                if col in c[10:]:
                    cols_to_keep.add(c)    
                    if col not in updated_question:
                        update_question_with_fallback = True     
            else:
                if col in c:
                    cols_to_keep.add(c)
                    if col not in updated_question:
                        update_question_with_fallback = True
    
    if update_question_with_fallback:
        updated_question = get_update_question_with_fallback(updated_question, sorted(list(cols_to_keep)))

    question = updated_question
    if isinstance(question, list):
        complete_exe_pipeline["UserQuestion"] += " " + question[0]
    else:
        complete_exe_pipeline["UserQuestion"] += " " + question

    rows_to_keep = select_relevant_values(question, list(df.index.values))
    print(f"rows_to_keep: {rows_to_keep}")
    # import pdb 
    # pdb.set_trace()

    df.sort_index(axis=1, inplace=True)
    df = df.loc[rows_to_keep, sorted(list(cols_to_keep))]

    # return response
    t_vals = df.copy()
    df_vars = get_vars_table_from_values_table(t_vals)
    equation = get_equation(question, df_vars.to_html())
    solution = solve_equation(equation, df_vars, df)
    explanation_with_solution = get_explanation_with_solution(question, solution)

    if FINAL_PROMPT_MAP["get_revenue_by_region"]:
        complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
            {"subtask": updated_question, "response": f"\n{explanation_with_solution}"}
        )

    return explanation_with_solution



def get_income_statement(question, extracted_data, ex_pipeline, complete_exe_pipeline):
    # print(f"Inside get_income_statement")
    entities_from_extraction_service = get_entities_from_extraction_service(question)
    where_clauses = []
    statement_types = []
    quarterly_date_year_subs = []
    for ent in entities_from_extraction_service["entities"]:
        if ent["entity"] == "ticker":
            ticker = ent["value"]
            where_clauses.append(f"c.ticker='{ticker.lower()}'")
        elif ent["entity"]=="company_name":
            company_name = ent["value"]
            where_clauses.append(f"c.company_name='{company_name.lower()}'")
        elif ent["entity"]=="statement_type":
            statement_type = ENTITY_TO_NODE_MAP[ent["value"]]
            statement_types.append(statement_type)
        elif ent["entity"]=="filing_type":
            continue
        elif ent["entity"]=="quarterly_date_year":
            quarterly_date_year_subs.append(ent["value"])
        else:
            continue
            # where_clauses.append(get_neo4j_date_filter_where_clause_from_extracted_entities(ent))

    table = []
    responses = []

    where_clause = ' AND '.join(where_clauses)
    # financial_reporting_response = (
    #         RunnablePassthrough.assign(Question=lambda x: x)
    #         | financial_reporting_calendar_prompt
    #         | llm
    #         | StrOutputParser()
    #     )

    # financial_reporting_response = financial_reporting_response.invoke({"Question": question})
    fiscal_or_calendar_response = (
            RunnablePassthrough.assign(Question=lambda x: x)
            | fiscal_or_calendar_prompt
            | llm
            | StrOutputParser()
        )

    financial_reporting_calendar_response = (
            RunnablePassthrough.assign(Question=lambda x: x)
            | financial_reporting_calendar_prompt_v2
            | llm
            | StrOutputParser()
        )

    fiscal_or_calendar_response = fiscal_or_calendar_response.invoke({"Question": question})
    fiscal_or_calendar_str_val = ast.literal_eval(fiscal_or_calendar_response)[0]
    print(f"fiscal_or_calendar_str_val: {fiscal_or_calendar_str_val}")
    financial_reporting_response = financial_reporting_calendar_response.invoke({"Question": question, "Mode": fiscal_or_calendar_str_val})
    print(f"financial_reporting_response: {financial_reporting_response}")
    columns = financial_reporting_response
    if '```json' in columns:
        columns = columns[7:len(columns)-3]

    columns = ast.literal_eval(columns)
    
    updated_question = question
    # TODO: sort quarterly_date_year_subs by order it appears in question 
    for q, c   in zip(quarterly_date_year_subs, columns):
        updated_question = updated_question.replace(q, f"{q} ({c})")

    # print(f"""\n\n\nMATCH (c:company) WHERE {where_clause}
    #     RETURN c.income_statement_table as table\n\n\n""")
    res_from_graph = graph.query(
        f"""MATCH (c:company) WHERE {where_clause}
        RETURN c.income_statement_table as table"""
    )

    df = pd.read_html(res_from_graph[-1]["table"], index_col=1, header=0)[0]
    df.index.rename('', inplace=True)

    update_question_with_fallback = False
    cols_to_keep = set()
    for c in list(df.columns):
        for col in columns:
            if len(c) > 10:
                if col in c[10:]:
                    cols_to_keep.add(c)    
                    if col not in updated_question:
                        update_question_with_fallback = True     
            else:
                if col in c:
                    cols_to_keep.add(c)
                    if col not in updated_question:
                        update_question_with_fallback = True
    
    if update_question_with_fallback:
        updated_question = get_update_question_with_fallback(updated_question, sorted(list(cols_to_keep)))

    question = updated_question
    if isinstance(question, list):
        complete_exe_pipeline["UserQuestion"] += " " + question[0]
    else:
        complete_exe_pipeline["UserQuestion"] += " " + question


    rows_to_keep = select_relevant_values(question, list(df.index.values))
    df.sort_index(axis=1, inplace=True)
    df = df.loc[rows_to_keep, sorted(list(cols_to_keep))]

    t_vals = df.copy()
    df_vars = get_vars_table_from_values_table(t_vals)
    # print(f"df_vars:\n{df_vars}")
    equation = get_equation(question, df_vars.to_html())
    solution = solve_equation(equation, df_vars, df)
    explanation_with_solution = get_explanation_with_solution(question, solution)

    complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
        {"subtask": updated_question, "response": f"\n{explanation_with_solution}"}
    )

    # print(f"End of get_income_statement")

    return explanation_with_solution

def append_user_query(question, extracted_data, ex_pipeline, complete_exe_pipeline):
    print(f"inside append_user_query!")
    complete_exe_pipeline["UserQuestion"] += " " + question

    return complete_exe_pipeline



def get_cash_flow_statement(question, extracted_data, ex_pipeline, complete_exe_pipeline):
    entities_from_extraction_service = get_entities_from_extraction_service(question)
    where_clauses = []
    statement_types = []
    quarterly_date_year_subs = []
    contain_quarterly_date_year = False
    for ent in entities_from_extraction_service["entities"]:
        if ent["entity"] == "ticker":
            ticker = ent["value"]
            where_clauses.append(f"c.ticker='{ticker.lower()}'")
        elif ent["entity"]=="company_name":
            company_name = ent["value"]
            where_clauses.append(f"c.company_name='{company_name.lower()}'")
        elif ent["entity"]=="statement_type":
            statement_type = ENTITY_TO_NODE_MAP[ent["value"]]
            statement_types.append(statement_type)
        elif ent["entity"]=="filing_type":
            continue
        elif ent["entity"]=="quarterly_date_year":
            quarterly_date_year_subs.append(ent["value"])
            contain_quarterly_date_year = True
        else:
            continue
            # where_clauses.append(get_neo4j_date_filter_where_clause_from_extracted_entities(ent))

    table = []
    responses = []
    
    where_clause = ' AND '.join(where_clauses)
    fiscal_or_calendar_response = (
            RunnablePassthrough.assign(Question=lambda x: x)
            | fiscal_or_calendar_prompt
            | llm
            | StrOutputParser()
        )

    financial_reporting_calendar_response = (
            RunnablePassthrough.assign(Question=lambda x: x)
            | financial_reporting_calendar_prompt_v2
            | llm
            | StrOutputParser()
        )

    fiscal_or_calendar_response = fiscal_or_calendar_response.invoke({"Question": question})
    fiscal_or_calendar_str_val = ast.literal_eval(fiscal_or_calendar_response)[0]
    print(f"fiscal_or_calendar_str_val: {fiscal_or_calendar_str_val}")
    financial_reporting_response = financial_reporting_calendar_response.invoke({"Question": question, "Mode": fiscal_or_calendar_str_val})
    print(f"financial_reporting_response: {financial_reporting_response}")
    columns = financial_reporting_response
    if '```json' in columns:
        columns = columns[7:len(columns)-3]

    columns = ast.literal_eval(columns)
    
    updated_question = question 
    for q, c in zip(quarterly_date_year_subs, columns):
        updated_question = updated_question.replace(q, f"{q} ({c})")

    res_from_graph = graph.query(
        f"""MATCH (c:company) WHERE {where_clause}
        RETURN c.cash_flow_table as table"""
    )
    df = pd.read_html(res_from_graph[-1]["table"], index_col=1, header=0)[0]
    df.index.rename('', inplace=True)
    # df.sort_values(axis=1, inplace=True)
    print(f"ALL df.columns:\n{df.columns}")

    update_question_with_fallback = False
    cols_to_keep = set()
    for c in list(df.columns):
        for col in columns:
            if len(c) > 10:
                if col in c[10:]:
                    cols_to_keep.add(c)    
                    if col not in updated_question:
                        update_question_with_fallback = True     
            else:
                if col in c:
                    cols_to_keep.add(c)
                    if col not in updated_question:
                        update_question_with_fallback = True
    
    if update_question_with_fallback:
        updated_question = get_update_question_with_fallback(updated_question, sorted(list(cols_to_keep)))

    question = updated_question
    print(f"updated_question: {updated_question}")
    # complete_exe_pipeline["UserQuestion"] = question
    if isinstance(question, list):
        complete_exe_pipeline["UserQuestion"] += " " + question[0]
    else:
        complete_exe_pipeline["UserQuestion"] += " " + question

    rows_to_keep = select_relevant_values(question, list(df.index.values))
    print(f"rows_to_keep:\n{rows_to_keep}")
    df.sort_index(axis=1, inplace=True)
    df = df.loc[rows_to_keep, sorted(list(cols_to_keep))]

    # return response
    t_vals = df.copy()
    df_vars = get_vars_table_from_values_table(t_vals)
    equation = get_equation(question, df_vars.to_html())
    solution = solve_equation(equation, df_vars, df)
    explanation_with_solution = get_explanation_with_solution(question, solution)

    if FINAL_PROMPT_MAP["get_revenue_by_region"]:
        complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
            {"subtask": updated_question, "response": f"\n{explanation_with_solution}"}
        )

    return explanation_with_solution
    


def interpolate_subtask(subtask, list_of_subtasks):
    expander_query, query_to_expand = subtask.split("->")
    query = expander_query[0 : expander_query.find("*<*")]
    functions = expander_query[
        expander_query.find("*<*") + 3 : expander_query.find("*>*")
    ]

    vals_for_interpolating = FUNCS_MAP[functions](subtask, "", "", {})[
        "subtask_responses"
    ]
    expanded_queries = []
    for val in vals_for_interpolating:
        # TODO: Change this to use regex on wild card
        if "__ticker__" in query_to_expand:
            expanded_queries.append(query_to_expand.replace("__ticker__", val))
        elif "__product_segment__" in query_to_expand:
            expanded_queries.append(query_to_expand.replace("__product_segment__", val))

    list_of_subtasks.extend(expanded_queries)
    
    return expanded_queries


def parse_subtask_response(response_from_task_parser):
    print(f"response_from_task_parser: {response_from_task_parser}")
    if 'json' in response_from_task_parser:
        response_from_task_parser = response_from_task_parser[7:len(response_from_task_parser)-4]
    list_of_subtasks = ast.literal_eval(response_from_task_parser)
    execution_pipelines = {"exe_pipeline": []}
    for subtask in list_of_subtasks:
        if "->" in subtask:
            interpolate_subtask(subtask, list_of_subtasks)
            continue
        subtask_exe_dict = {}
        query = subtask[0 : subtask.find("*<*")]
        functions = subtask[subtask.find("*<*") + 3 : subtask.find("*>*")]
        funcs = functions.split(", ")  # ast.literal_eval(functions)
        subtask_exe_dict[query] = funcs

        execution_pipelines["exe_pipeline"].append(subtask_exe_dict)
        # #print(f'execution_pipelines["exe_pipeline"]\n: {execution_pipelines["exe_pipeline"]}')

    return json.dumps(execution_pipelines)


def parse_a_subtask_for_query_interpolation(a_subtask):
    subtask_dict = {}
    query = a_subtask[0 : a_subtask.find("*<*")]
    functions = a_subtask[a_subtask.find("*<*") + 3 : a_subtask.find("*>*")]
    funcs = functions.split(", ")  # ast.literal_eval(functions)
    subtask_dict[query] = funcs

    return subtask_dict


def filtered_search(input, final_context):
    pass


def query_interpolation(question, extracted_entities, ex_pipeline, final_context):

    """
    Algorithm:
    1) split the query on '->'
        vector_search_query, expansion_query = question.split('->')
        vector_search_query = List the major __product__ launches from MSFT over the last 1 year
        expansion_query = How has __product__ impacted revenue growth?

    2) remove __ from vector_search_query and pass to vector search prompt.
    3) use the results to interpolate each into expansion_query
    4) Send each expanded query to vector search to get results
    5) Add each result to final_context["Context_Pieces_For_Final_Prompt"].append(
            {"subtask": expansion_query, "response": response}
        )
    """
    # ##print(f"query INSIDE query_interpolation: {question}")
    expander_query, expandee_query = question.split("->")

    expander_query_dict = parse_a_subtask_for_query_interpolation(expander_query)
    # ##print(f"interp resp: {expander_query_dict}")


final_prompt_llm = ChatOpenAI(model="gpt-4", temperature=0.0)


def prepare_final_prompt(ex_pipeline):
    ex_pipeline = json.loads(ex_pipeline)
    context = []
    context = ex_pipeline["Context_Pieces_For_Final_Prompt"]
    new_context = []
    for c in context:
        new_context.append(f"# Subtask: {c['subtask']}")
        new_context.append(f"Answer: {c['response']}")

    new_context = "\n".join(new_context)
    ex_pipeline["Context_Pieces_For_Final_Prompt"] = new_context

    # print(f'context:\n{new_context}')

    final_chain = (
        RunnablePassthrough.assign(question=lambda x: x)
        | final_prompt
        | cypher_llm
        | StrOutputParser()
    )
    response = final_chain.invoke(
        {"question": ex_pipeline["OriginalQuestion"], "context": new_context}
    )
    
    # print(f"response: {response}")
    ex_pipeline["FinalResponse"] = response

    return ex_pipeline


FILING_TYPE_TO_YAML_MAP = {"10-K": "10_k", "10-Q": "10_q", "8-K": "8_k"}
# TODO: Added the following to a config.yaml


def read_docs_from_json(json_path: str):
    """_summary_

    Args:
        json_path (str): _description_
    """
    data = None
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


embeddings = OpenAIEmbeddings()
# doc_store = Qdrant.from_texts(
#     answers, embeddings, host="localhost"
# )

openai_client = openai.Client(
    api_key="sk-proj-Tb7SWc46QhFrvgU6pPY4T3BlbkFJRyGTP45AunM8ULAu9XLA"
)


from qdrant_client import QdrantClient, models

q_client = QdrantClient(url=url)


TOP_K = 7



def get_context_from_vector_search_query(query, extracted_data):
    # ##print(f"INSIDE get_context_from_vector_search_query: {query}")
    embedding = get_embedding(query)
    filters = extracted_data["entities"] or get_entities_from_extraction_service(query)
    embedding = get_embedding(query)
    filters = extracted_data["entities"]
    musts = []
    for filter in filters:
        qdrant_filter_key = EXTRACTION_KEYS_TO_QDRANT_FIELDS_MAP[filter["entity"]]
        qdrant_filter_value = filter["value"]
        musts.append({ "key": qdrant_filter_key, "match": { "value": qdrant_filter_value } })


    data = {
        "filter": {
            "must": musts
        },
        "params":{
            "hnsw_ef":1024, 
            "exact": False
        },
        "vector": embedding,
        "with_payload": True,
        "limit": 12
    }

    response = requests.post(f"{url}/collections/sec_filings/points/search",json=data).json()
    

    texts = []
    for r in response["result"]:
        vals = r["payload"]["text"]
        ##print(f"text:\n{vals}")
        texts.append(vals)


    return {"Question": query, "Context": "\n".join(texts)}



EXTRACTION_KEYS_TO_QDRANT_FIELDS_MAP = {
    "ticker": "ticker",
    "company_name": "company_name",
    "date_year": "year_of_report",
    # TODO: add logic for duckling to period_of_report mapping

}
# def get_context_from_vector_search_query_synth(some_input):
#     ##print(f"some_input: {some_input}")
#     query = some_input["Question"]
#     extracted_data = some_input[
#         "extracted_data"
#     ] or get_entities_from_extraction_service(query)

#     embedding = get_embedding(query)
#     filters = extracted_data["entities"]
#     musts = []
#     for filter in filters:
#         qdrant_filter_key = EXTRACTION_KEYS_TO_QDRANT_FIELDS_MAP[filter["entity"]]
#         qdrant_filter_value = filter["value"]
#         musts.append({ "key": qdrant_filter_key, "match": { "value": qdrant_filter_value } })


#     data = {
#         "filter": {
#             "must": musts
#         },
#         "params":{
#             "hnsw_ef":1024, 
#             "exact": False
#         },
#         "vector": embedding,
#         "with_payload": True,
#         "limit": 12
#     }

#     response = requests.post(f"{url}/collections/sec_filings/points/search",json=data).json()
    

#     texts = []
#     for r in response["result"]:
#         vals = r["payload"]["text"]
#         ##print(f"text:\n{vals}")
#         texts.append(vals)


#     return {"Question": query, "Context": "\n".join(texts)}


rephrased_user_question_template = """
    Given a supplied Question below rephrase the question 9 different ways using variations in verbiage. 
    Make sure to keep the essence of the question. Do not modify any companies, numerical values, or dates contained in the original supplied question.
    Make sure these values are also in the rephrased questions. The result should be a JSON list that contains the original supplied user question and the 9 rephrased questions.
    Do not include any other information in your reponse besides the JSON.

    Question: {Question}
"""


rephrased_user_question_prompt = PromptTemplate(
    template=rephrased_user_question_template,
    input_variables=["Question"],
)

hypothetical_answers_template = """
    Given a supplied Question below generate a list of 10 hypothetical answers that might appear in a sec filing such as a 10K or 10Q. The answers should
    be viable answers to the supplied Question.  The answers you generate should be in complete sentences. Each answer should be unique.
    The result should be a JSON list that contains the hypothetical answers.
    Do not include any other information in your reponse besides the JSON.

    Question: {Question}
"""

hypothetical_answers_prompt = PromptTemplate(
    template=hypothetical_answers_template,
    input_variables=["Question"],
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

def get_hypothetical_questions_and_answers_from_gpt4(user_query):
    try:
        rephrased_questions_chain = (
                RunnablePassthrough.assign(Question=lambda x: x)
                | rephrased_user_question_prompt
                | llm
                | StrOutputParser()
            )

        rephrased_questions_response = rephrased_questions_chain.invoke({"Question": user_query})
        # print(f"rephrased_questions_response:\n{rephrased_questions_response}")
        if '```json' in rephrased_questions_response:
            rephrased_questions_response = rephrased_questions_response[7:len(rephrased_questions_response)-3]

        rephrased_questions_response = ast.literal_eval(rephrased_questions_response)
        rephrased_questions_response = [r["Question"] for r in rephrased_questions_response]

        # print(f"rephrased_questions_response:\n{rephrased_questions_response}")
        hypothetical_answers_chain = (
                RunnablePassthrough.assign(Question=lambda x: x)
                | hypothetical_answers_prompt
                | llm
                | StrOutputParser()
            )

        
        hypothetical_answers_response = hypothetical_answers_chain.invoke({"Question": user_query})
        if '```json' in hypothetical_answers_response:
            hypothetical_answers_response = hypothetical_answers_response[7:len(hypothetical_answers_response)-3]
        hypothetical_answers_response = ast.literal_eval(hypothetical_answers_response)
        
        return rephrased_questions_response, hypothetical_answers_response
    except:
        return None


def get_hypothetical_embeddings_from_user_query(user_query):
    # Feed text_chunk to gpt4 to get back json of questions and answers per chunk
    hype_qa = get_hypothetical_questions_and_answers_from_gpt4(user_query)
    # print(f"hype_qa: {hype_qa}")
    if hype_qa:
        rephrased_questions_list, hypothetical_answers_list = hype_qa
    
    if not rephrased_questions_list or not hypothetical_answers_list:
        return None
    
    raw_rephrased_questions_list = '\n'.join(rephrased_questions_list)
    raw_hypothetical_answers_list = '\n'.join(hypothetical_answers_list)
    
    # Take list of questions and list of answers and create an embedding of each list
    rephrased_questions_embedding = (
                    oai_client.embeddings.create(
                        input=raw_rephrased_questions_list,
                        model="text-embedding-3-small",
                    )
                    .data[0]
                    .embedding
                )
    hypothetical_answers_embedding = (
                    oai_client.embeddings.create(
                        input=raw_hypothetical_answers_list,
                        model="text-embedding-3-small",
                    )
                    .data[0]
                    .embedding
                )
    return rephrased_questions_embedding, hypothetical_answers_embedding, raw_rephrased_questions_list, raw_hypothetical_answers_list


# def get_context_from_vector_search_query_synth(some_input):
#     # #print(f"some_input: {some_input}")
#     query = some_input["Question"]
#     extracted_data = some_input[
#         "extracted_data"
#     ] or get_entities_from_extraction_service(query)

#     embedding = get_embedding(query)
#     filters = extracted_data["entities"]
#     musts = []
#     for filter in filters:
#         qdrant_filter_key = EXTRACTION_KEYS_TO_QDRANT_FIELDS_MAP[filter["entity"]]
#         qdrant_filter_value = filter["value"]
#         musts.append({ "key": qdrant_filter_key, "match": { "value": qdrant_filter_value } })


#     data = {
#         "filter": {
#             "must": musts
#         },
#         "params":{
#             "hnsw_ef":1024, 
#             "exact": False
#         },
#         "vector": embedding,
#         "with_payload": True,
#         "limit": 12
#     }

#     response = requests.post(f"{url}/collections/sec_filings/points/search",json=data).json()
    
    
#     # TODO: for HyDE:
#     # rephrased_questions_embedding, hypothetical_answers_embedding = get_hypothetical_embeddings_from_user_query(query)
#     # retrieved_docs = []
#     # for (vec_name, emb) in [("hypothetical_questions", rephrased_questions_embedding), ("hypothetical_answers", hypothetical_answers_embedding), ("text", embedding)]:
#     #     data = {
#     #         "filter": {
#     #             "must": musts
#     #         },
#     #         "params":{
#     #             "hnsw_ef":1024, 
#     #             "exact": False
#     #         },
#     #         "query_vector": {
#     #             "name": vec_name,
#     #             "vector": emb
#     #         }
            
#     #         "with_payload": True,
#     #         "limit": 5
#     #     }
#     #     response = requests.post(f"{url}/collections/sec_filings/points/search",json=data).json()
#     #     retrieved_docs.extend(response)

#     texts = []
#     ui_json = {}

#     # TODO: modify this to get it to match expected mock_data.json for upwork UI

#     ##print(f"response_vector_search_from_chain: {response_vector_search_from_chain}")
#     """
#         {
#       "user_query": "In 2023, did aapl's report on the concentration of trade receivables?",
#       "response": "Yes, in 2023, Apple Inc. reported on the concentration of trade receivables. Specifically, the company's cellular network carriers accounted for 44% of total trade receivables as of September 24, 2022, and 42% as of September 25, 2021.",
#       "sources": [
#         {
#           "title": "Annual 10-K 2023",
#           "subtitle": "Highlights the most recent annual latest sales breakdown by product line.",
#           "importance_score": 0.6090705,
#           "extracted_text": " total trade receivables, which\naccounted for 10%. The Company’s cellular network carriers accounted for 44% and 42% of total trade receivables as of\nSeptember 24, 2022 and September 25, 2021, respectively.\nVendor Non-Trade Receivables\nThe Company has non-trade receivables from certain of its manufacturing vendors resulting from the sale of components to these\nvendors who manufacture subassemblies or assemble final products for the Company. The Company purchases these components\ndirectly from suppliers. As of September 24, 2022, the Company had two vendors that individually represented 10% or more of total\nvendor non-trade receivables, which accounted for 54% and 13%. As of September 25, 2021, the Company had three vendors that\nindividually represented 10% or more of total vendor non-trade receivables, which accounted for 52%, 11% and 11%.\n(1)\n(2)\nApple Inc. | 2022 Form 10-K | 40\n",
#           "date": "2023-09-30",
#           "company": "apple",
#           "doc_download_url": "local/path/to/pdf/0000320193-21-000105.pdf"
#         },
#         {
#           "title": "Annual 10-K 2023",
#           "subtitle": "Highlights the most recent annual latest sales breakdown by product line.",
#           "importance_score": 0.5509262,
#           "extracted_text": " of\nproducts cost of sales when the related final products are sold by the Company. As of September 30, 2023, the\nCompany had two vendors that individually represented 10% or more of total vendor non-trade receivables, which\naccounted for 48% and 23%. As of September 24, 2022, the Company had two vendors that individually represented\n10% or more of total vendor non-trade receivables, which accounted for 54% and 13%.\n(1)\n(2)\nApple Inc. | 2023 Form 10-K | 38\n",
#           "date": "2023-09-30",
#           "company": "apple",
#           "doc_download_url": "local/path/to/pdf/0000320193-21-000105.pdf"
#         }
#       ]
#     }
#     """

#     print(f'response["result"]:\n{response["result"]}')

#     for retrieved_doc in retrieved_docs:

#     temp_sources = {}
#     # #print(f'response["result"]\n{response["result"]}')
#     for r in response["result"]:
#         # #print(f"r: {r}")
#         vals = r["payload"]["text"]
#         ##print(f"text:\n{vals}")
#         texts.append(vals)
#         temp_source = {}
#         temp_source["title"] = f'Annual 10-K  {r["payload"]["year_of_report"]}'
#         temp_source["subtitle"] = "Highlights the most recent annual latest sales breakdown by product line."
#         temp_source["importance_score"] = r["score"]
#         temp_source["extracted_text"] = vals
#         temp_source["date"] = r["payload"]["period_of_report"]
#         temp_source["company"] = r["payload"]["company_name"]
#         temp_source["doc_download_url"] = f'local/path/to/pdf/{r["payload"]["accession_no"]}.pdf'
#         # temp_source.append(temp_source)
#         temp_sources[vals] = temp_source

#     results = co.rerank(model="rerank-english-v3.0", query=query, documents=texts, top_n=7, return_documents=True)
    
#     results = [r.document.text for r in results.results]
#     final_sources = []
#     for r in results:
#         final_sources.append(temp_sources[r])

#     # #print(f"reranked text:\n{results}")
#     some_input["complete_exe_pipeline"]["sources"].extend(final_sources)


#     return {"Question": query, "Context": "\n".join(results)}


# TODO: THIS GUY IS FOR HyDE QA's within the vector search
# def get_context_from_vector_search_query_synth(some_input):
#     print(f"some_input:\n{some_input}")
#     query = some_input["Question"]
#     extracted_data = some_input[
#         "extracted_data"
#     ] or get_entities_from_extraction_service(query)

#     embedding = get_embedding(query)
#     filters = extracted_data["entities"]
#     musts = []
#     for filter in filters:
#         qdrant_filter_key = EXTRACTION_KEYS_TO_QDRANT_FIELDS_MAP[filter["entity"]]
#         qdrant_filter_value = filter["value"]
#         musts.append({ "key": qdrant_filter_key, "match": { "value": qdrant_filter_value } })
    
#     # TODO: for HyDE:
#     rephrased_questions_embedding, hypothetical_answers_embedding, raw_rephrased_questions_list, raw_hypothetical_answers_list = get_hypothetical_embeddings_from_user_query(query)
#     retrieved_docs_map = {
#         "hypothetical_questions": [],
#         "hypothetical_answers": [],
#         "text": []
#     }
#     final_sources_texts_map = {}
#     for (vec_name, emb) in [("hypothetical_questions", rephrased_questions_embedding), ("hypothetical_answers", hypothetical_answers_embedding), ("text", embedding)]:
#         data = {
#             "filter": {
#                 "must": musts
#             },
#             "params":{
#                 "hnsw_ef":1024, 
#                 "exact": False
#             },
#             "vector": {
#                 "name": vec_name,
#                 "vector": emb
#             },
#             "with_payload": True,
#             "limit": 10
#         }
#         response = requests.post(f"{url}/collections/sec_filings/points/search",json=data).json()
#         # response = requests.post(f"{url}/collections/sec_filings/points/search",json=data).json()
#         # print(f"response:\n{response}")
#         retrieved_docs_map[vec_name].extend(response['result'])
#         if vec_name == "hypothetical_questions":
#             for r in response["result"]:
#                 final_sources_texts_map[r["payload"]["hypothetical_questions_text"]] = r["payload"]["text"]
#         elif vec_name == "hypothetical_answers":
#             for r in response["result"]:
#                 final_sources_texts_map[r["payload"]["hypothetical_answers_text"]] = r["payload"]["text"]
#         else:
#             for r in response["result"]:
#                 final_sources_texts_map[r["payload"]["text"]] = r["payload"]["text"]

#     final_responses = []

#     for vec_type, docs in retrieved_docs_map.items():
#         rerank_query = None
#         rerank_docs = []
        
#         temp_sources = {}
#         for r in docs:
#             if vec_type == "hypothetical_questions":
#                 rerank_query = raw_rephrased_questions_list
#                 rerank_docs.append(r["payload"]["hypothetical_questions_text"])
#             elif vec_type == "hypothetical_answers":
#                 rerank_query = raw_hypothetical_answers_list
#                 rerank_docs.append(r["payload"]["hypothetical_answers_text"])
#             else:
#                 rerank_query = query
#                 rerank_docs.append(r["payload"]["text"])
            
#             temp_source = {}
#             temp_source["title"] = f'Annual 10-K {r["payload"]["year_of_report"]}'
#             temp_source["subtitle"] = "Highlights the most recent annual latest sales breakdown by product line."
#             temp_source["importance_score"] = r["score"]
#             temp_source["extracted_text"] = r["payload"]["text"]
#             temp_source["date"] = r["payload"]["period_of_report"]
#             temp_source["company"] = r["payload"]["company_name"]
#             temp_source["doc_download_url"] = f'https://sec-filings2.s3.us-east-1.amazonaws.com/{r["payload"]["accession_no"]}.pdf'
#             # temp_source["icon"] = COMPANY_TO_SVG_MAP[r["payload"]["company_name"]]
#             temp_source["page_number"] = r["payload"]["parent_id"]
#             temp_sources[r["payload"]["text"]] = temp_source


#         results = co.rerank(model="rerank-english-v3.0", query=rerank_query, documents=rerank_docs, top_n=3, return_documents=True)
#         final_responses.extend([temp_sources[final_sources_texts_map[r.document.text]] for r in results.results])

#     print(f"final_responses:\n{final_responses[0]}")
    
#     some_input["complete_exe_pipeline"]["sources"].extend(final_responses)

#     return {"Question": query, "Context": "\n".join([f["extracted_text"] for f in final_responses])}

def get_context_from_vector_search_query_synth(some_input):
    print(f"some_input:\n{some_input}")
    query = some_input["Question"]
    extracted_data = some_input[
        "extracted_data"
    ] or get_entities_from_extraction_service(query)

    embedding = get_embedding(query)
    filters = extracted_data["entities"]
    musts = []
    for filter in filters:
        qdrant_filter_key = EXTRACTION_KEYS_TO_QDRANT_FIELDS_MAP[filter["entity"]]
        qdrant_filter_value = filter["value"]
        musts.append({ "key": qdrant_filter_key, "match": { "value": qdrant_filter_value } })
    
    # TODO: for HyDE:
    # rephrased_questions_embedding, hypothetical_answers_embedding, raw_rephrased_questions_list, raw_hypothetical_answers_list = get_hypothetical_embeddings_from_user_query(query)
    retrieved_docs_map = {
        # "hypothetical_questions": [],
        # "hypothetical_answers": [],
        "text": []
    }
    final_sources_texts_map = {}
    for (vec_name, emb) in [("text", embedding)]: #[("hypothetical_questions", rephrased_questions_embedding), ("hypothetical_answers", hypothetical_answers_embedding), ("text", embedding)]:
        data = {
            "filter": {
                "must": musts
            },
            "params":{
                "hnsw_ef":1024, 
                "exact": False
            },
            "vector": {
                "name": vec_name,
                "vector": emb
            },
            "with_payload": True,
            "limit": 10
        }
        response = requests.post(f"{url}/collections/sec_filings2/points/search",json=data).json()
        # response = requests.post(f"{url}/collections/sec_filings/points/search",json=data).json()
        print(f"response:\n{response}")
        retrieved_docs_map[vec_name].extend(response['result'])
        if vec_name == "hypothetical_questions":
            for r in response["result"]:
                final_sources_texts_map[r["payload"]["hypothetical_questions_text"]] = r["payload"]["text"]
        elif vec_name == "hypothetical_answers":
            for r in response["result"]:
                final_sources_texts_map[r["payload"]["hypothetical_answers_text"]] = r["payload"]["text"]
        else:
            for r in response["result"]:
                final_sources_texts_map[r["payload"]["text"]] = r["payload"]["text"]

    final_responses = []

    for vec_type, docs in retrieved_docs_map.items():
        rerank_query = None
        rerank_docs = []
        
        temp_sources = {}
        for r in docs:
            if vec_type == "hypothetical_questions":
                rerank_query = raw_rephrased_questions_list
                rerank_docs.append(r["payload"]["hypothetical_questions_text"])
            elif vec_type == "hypothetical_answers":
                rerank_query = raw_hypothetical_answers_list
                rerank_docs.append(r["payload"]["hypothetical_answers_text"])
            else:
                rerank_query = query
                rerank_docs.append(r["payload"]["text"])
            print(f'r:\n{r}')
            
            temp_source = {}
            temp_source["title"] = f'Annual 10-K {r["payload"]["year_of_report"]}'
            temp_source["subtitle"] = "Highlights the most recent annual latest sales breakdown by product line."
            temp_source["importance_score"] = r["score"]
            temp_source["extracted_text"] = r["payload"]["text"]
            temp_source["date"] = r["payload"]["period_of_report"]
            temp_source["company"] = r["payload"]["company_name"]
            temp_source["doc_download_url"] = f'https://sec-filings2.s3.us-east-1.amazonaws.com/{r["payload"]["accession_no"]}.pdf'
            # temp_source["icon"] = COMPANY_TO_SVG_MAP[r["payload"]["company_name"]]
            temp_source["page_number"] = r["payload"]["parent_id"]
            temp_sources[r["payload"]["text"]] = temp_source


        results = co.rerank(model="rerank-english-v3.0", query=rerank_query, documents=rerank_docs, top_n=3, return_documents=True)
        final_responses.extend([temp_sources[final_sources_texts_map[r.document.text]] for r in results.results])

    print(f"final_responses:\n{final_responses[0]}")
    
    some_input["complete_exe_pipeline"]["sources"].extend(final_responses)

    return {"Question": query, "Context": "\n".join([f["extracted_text"] for f in final_responses])}


def perform_vector_search(query, extracted_data, ex_pipeline, complete_exe_pipeline):
    print(f"complete_exe_pipeline:\n{complete_exe_pipeline}")
    vector_search_system_template = """{Question}
        Use the context below to answer the question.
    """

    vector_search_human_template = """
    Context: {Context}
    """
    print(f"extracted_data: {extracted_data}")

    vector_search_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                vector_search_system_template,
            ),
            ("human", vector_search_human_template),
        ]
    )
    vector_search_response = (
        RunnablePassthrough.assign(
            Question=lambda x: x["Question"],
            extracted_data=lambda x: x["extracted_data"],
            complete_exe_pipeline=lambda x: x["complete_exe_pipeline"]

        )
        | RunnableLambda(get_context_from_vector_search_query_synth)
        | vector_search_prompt
        | cypher_llm
        | StrOutputParser()
    )

    subtask = query

    response_vector_search_from_chain = vector_search_response.invoke(
        {"Question": subtask, "extracted_data": extracted_data, "complete_exe_pipeline": complete_exe_pipeline}
    )

    complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
        {
            "subtask": subtask,
            "response": response_vector_search_from_chain,
        }
    )

    return response_vector_search_from_chain



def get_context_from_vector_search_query_synth_10Q(some_input):
    query = some_input["Question"]
    extracted_data = some_input[
        "extracted_data"
    ] or get_entities_from_extraction_service(query)

    embedding = get_embedding(query)
    filters = extracted_data["entities"]
    musts = []
    for filter in filters:
        qdrant_filter_key = EXTRACTION_KEYS_TO_QDRANT_FIELDS_MAP[filter["entity"]]
        qdrant_filter_value = filter["value"]
        musts.append({ "key": qdrant_filter_key, "match": { "value": qdrant_filter_value } })
    
    # TODO: for HyDE:
    rephrased_questions_embedding, hypothetical_answers_embedding, raw_rephrased_questions_list, raw_hypothetical_answers_list = get_hypothetical_embeddings_from_user_query(query)
    retrieved_docs_map = {
        "hypothetical_questions": [],
        "hypothetical_answers": [],
        "text": []
    }
    final_sources_texts_map = {}
    for (vec_name, emb) in [("hypothetical_questions", rephrased_questions_embedding), ("hypothetical_answers", hypothetical_answers_embedding), ("text", embedding)]:
        data = {
            "filter": {
                "must": musts
            },
            "params":{
                "hnsw_ef":1024, 
                "exact": False
            },
            "vector": {
                "name": vec_name,
                "vector": emb
            },
            "with_payload": True,
            "limit": 5
        }
        response = requests.post(f"{url}/collections/10Q3/points/search",json=data).json()
        # response = requests.post(f"{url}/collections/10Q/points/search",json=data).json()
        # print(f"response:\n{response}")
        retrieved_docs_map[vec_name].extend(response['result'])
        if vec_name == "hypothetical_questions":
            for r in response["result"]:
                final_sources_texts_map[r["payload"]["hypothetical_questions_text"]] = r["payload"]["text"]
        elif vec_name == "hypothetical_answers":
            for r in response["result"]:
                final_sources_texts_map[r["payload"]["hypothetical_answers_text"]] = r["payload"]["text"]
        else:
            for r in response["result"]:
                final_sources_texts_map[r["payload"]["text"]] = r["payload"]["text"]

    final_responses = []

    for vec_type, docs in retrieved_docs_map.items():
        rerank_query = None
        rerank_docs = []
        
        temp_sources = {}
        for r in docs:
            # print(f"r: {r}")
            if vec_type == "hypothetical_questions":
                rerank_query = raw_rephrased_questions_list
                rerank_docs.append(r["payload"]["hypothetical_questions_text"])
            elif vec_type == "hypothetical_answers":
                rerank_query = raw_hypothetical_answers_list
                rerank_docs.append(r["payload"]["hypothetical_answers_text"])
            else:
                rerank_query = query
                rerank_docs.append(r["payload"]["text"])
            
            temp_source = {}
            temp_source["title"] = f'Annual 10-K  {r["payload"]["year_of_report"]}'
            temp_source["subtitle"] = "Highlights the most recent annual latest sales breakdown by product line."
            temp_source["importance_score"] = r["score"]
            temp_source["extracted_text"] = r["payload"]["text"]
            temp_source["date"] = r["payload"]["period_of_report"]
            temp_source["company"] = r["payload"]["company_name"]
            temp_source["doc_download_url"] = f'https://sec-filings2.s3.us-east-1.amazonaws.com/{r["payload"]["accession_no"]}.pdf'
            temp_source["icon"] = COMPANY_TO_SVG_MAP[r["payload"]["company_name"]]
            temp_source["page_number"] = r["payload"]["parent_id"] -1
            temp_sources[r["payload"]["text"]] = temp_source

        results = co.rerank(model="rerank-english-v3.0", query=rerank_query, documents=rerank_docs, top_n=7, return_documents=True)
        final_responses.extend([temp_sources[final_sources_texts_map[r.document.text]] for r in results.results])

    # print(f"final_responses:\n{final_responses}")
    
    some_input["complete_exe_pipeline"]["sources"].extend(final_responses)

    return {"Question": query, "Context": "\n".join([f["extracted_text"] for f in final_responses])}


def perform_vector_search_10Q(query, extracted_data, ex_pipeline, complete_exe_pipeline):
    vector_search_system_template = """{Question}
        Use the context below to answer the question.
    """

    vector_search_human_template = """
    Context: {Context}
    """

    vector_search_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                vector_search_system_template,
            ),
            ("human", vector_search_human_template),
        ]
    )
    vector_search_response = (
        RunnablePassthrough.assign(
            Question=lambda x: x["Question"],
            extracted_data=lambda x: x["extracted_data"],
        )
        | RunnableLambda(get_context_from_vector_search_query_synth_10Q)
        | vector_search_prompt
        | cypher_llm
        | StrOutputParser()
    )

    subtask = query

    response_vector_search_from_chain = vector_search_response.invoke(
        {"Question": subtask, "extracted_data": extracted_data}
    )

    ##print(f"response_vector_search_from_chain: {response_vector_search_from_chain}")

    complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
        {
            "subtask": subtask,
            "response": response_vector_search_from_chain,
        }
    )

    return response_vector_search_from_chain


# def perform_vector_search(query, extracted_data, ex_pipeline, complete_exe_pipeline):
#     vector_search_system_template = """{Question}
#         Use the context below to answer the question.
#     """

#     vector_search_human_template = """
#     Context: {Context}
#     """

#     vector_search_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 vector_search_system_template,
#             ),
#             ("human", vector_search_human_template),
#         ]
#     )
#     vector_search_response = (
#         RunnablePassthrough.assign(
#             Question=lambda x: x["Question"],
#             extracted_data=lambda x: x["extracted_data"],
#         )
#         | RunnableLambda(get_context_from_vector_search_query_synth)
#         | vector_search_prompt
#         | cypher_llm
#         | StrOutputParser()
#     )

#     subtask = query

#     response_vector_search_from_chain = vector_search_response.invoke(
#         {"Question": subtask, "extracted_data": extracted_data}
#     )

#     complete_exe_pipeline["Context_Pieces_For_Final_Prompt"].append(
#         {
#             "subtask": subtask,
#             "response": response_vector_search_from_chain,
#         }
#     )

#     return response_vector_search_from_chain


FUNCS_MAP = {
    "filtered_search": filtered_search,
    "perform_vector_search": perform_vector_search,
    "query_interpolation": query_interpolation,
    "get_cash_flow_statement": get_cash_flow_statement,
    "get_company_financials": get_company_financials,
    "get_income_statement": get_income_statement,
    "get_competitors": get_competitors,
    "get_product_segments": get_product_segments,
    "get_revenue_by_product": get_revenue_by_product,
    "get_revenue_by_region": get_revenue_by_region,
    "append_user_query": append_user_query

}

# TODO: this should come from subtasker (gpt4) output
FINAL_PROMPT_MAP = {
    "perform_vector_search": True,
    "perform_vector_search_10Q": True,
    "filtered_search": True,
    "perform_vector_search": True,
    "query_interpolation": True,
    "get_cash_flow_statement": True,
    "get_company_financials": True,
    "get_income_statement": True,
    "get_revenue_by_product": True,
    "get_revenue_by_region": True
}