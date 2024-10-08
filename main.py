import openai
import json
from typing import List, Dict, Any, Callable, Awaitable
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Body
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import requests
from dotenv import load_dotenv
import logging
import uuid
import asyncio
import time
from scratch import get_financials, get_research_plan, get_final_analysis, get_competitors, get_list_of_companies, preprocess_user_query
from sse_starlette.sse import EventSourceResponse

MESSAGE_STREAM_DELAY = 1  # second
MESSAGE_STREAM_RETRY_TIMEOUT = 15000  # milisecond
TASK_NAME_MAP = {
    "get_sec_financials": "Get Company Financials",
    "get_final_analysis": "Generate Final Analysis",
    "get_competitors": "Identify Company Competitors",
    "get_list_of_companies": "Produce List of Companies"
}

load_dotenv()
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    query: str
    # callback_url: str

class ResearchPipeline:
    def __init__(self):
        api_key = "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"

        if not api_key:
            raise ValueError("API key not found. Please set the OAI_key environment variable.")

        self.openai_client = openai.Client(api_key=api_key)
        self.tools = self.initialize_tools()

    def initialize_tools(self) -> Dict[str, str]:
        return {
            "tasker": get_research_plan,
            "get_sec_financials": get_financials,
            "get_final_analysis": get_final_analysis,
            "get_competitors": get_competitors,
            "get_list_of_companies": get_list_of_companies
            # "vector_search": 
        }

    def generate_research_plan(self, query: str) -> List[Dict[str, Any]]:
        research_plan = get_research_plan(query)
        return research_plan

    async def execute_tasks(self, query:str, tasks: List[Dict[str, Any]], progress_callback: Callable[[str, Any], Awaitable[None]]) -> Dict[str, Any]:
        results = {"Query": query, "Context": [], "Execution": {}}
        context = {"context": []}
        
        processed_user_query, entities = preprocess_user_query(query)
        tickers_to_process = [t["value"] for t in entities if t["entity"]=="ticker"]

        context["user_query"] = processed_user_query
        tasks_contain_comps = len([t["task"] for t in tasks if t["task"]=="get_competitors"]) > 0
        if tasks_contain_comps:
            tasks.pop(0)
            await send_sse({"event": "update",
                    "type": "subtaskUpdate",
                    "status": "in-progress",
                    "id": str(uuid.uuid4()),
                    "task": TASK_NAME_MAP["get_competitors"],
                    "data": {}
                    })
            comps = get_competitors(processed_user_query)
            await send_sse({"event": "update",
                    "type": "subtaskUpdate",
                    "status": "completed",
                    "id": str(uuid.uuid4()),
                    "task": TASK_NAME_MAP["get_competitors"],
                    "data": {}
                    })
            tickers_to_process += comps[:2]

        tasks_contain_list_of_comps = len([t["task"] for t in tasks if t["task"]=="get_list_of_companies"]) > 0
        if tasks_contain_list_of_comps:
            await send_sse({"event": "update",
                    "type": "subtaskUpdate",
                    "status": "in-progress",
                    "id": str(uuid.uuid4()),
                    "task": TASK_NAME_MAP["get_list_of_companies"],
                    "data": {}
                    })
            tasks.pop(0)
            list_of_tickers = get_list_of_companies(processed_user_query)
            await send_sse({"event": "update",
                    "type": "subtaskUpdate",
                    "status": "completed",
                    "id": str(uuid.uuid4()),
                    "task": TASK_NAME_MAP["get_list_of_companies"],
                    "data": {}
                    })
            tickers_to_process += list_of_tickers

        all_queries = [processed_user_query.lower().replace(tickers_to_process[0].lower(), t.lower()) for t in tickers_to_process]
        execution_pipeline = {task.get("task"):[] for task in tasks}
        
        for t, qs in execution_pipeline.items():
            for q in all_queries:
                execution_pipeline[t].append(q)

        import pdb 
        # pdb.set_trace()
        for t, queries in execution_pipeline.items():
            print(f"t: {t}, queries: {queries}")
            task_name = t 
            if task_name not in self.tools or task_name == "get_final_analysis":
                continue

            tool = self.tools.get(task_name)
            if not tool:
                continue

            await send_sse({"event": "update",
                        "type": "subtaskUpdate",
                        "status": "in-progress",
                        "id": str(uuid.uuid4()),
                        "task": TASK_NAME_MAP[task_name],
                        "data": {}
                        })

            
            for q in queries:
            # for task in tasks:
                # task_name = task.get("task")
                

                # if tool:
                #     # Send "in_progress" status update
                #     print(f"progress_callback: {progress_callback}")

                    

                try:
                    # task_input = {
                    #     "task_description": "",#task.get("description", ""),
                    #     "context": context
                    # }
                    # results["Query"] = q
                    # print(f'results["Query"]: {results["Query"]}')
                    # task_input_json = json.dumps(task_input)
                    result_json = tool(q, results)
                except RuntimeError as e:
                    print(f"Error in execute tasks: {e}")
                    # Send completed status update
            await send_sse({"event": "update",
                "type": "subtaskUpdate",
                "status": "completed",
                "id": str(uuid.uuid4()),
                "task": TASK_NAME_MAP[task_name],
                "data": results
            })

        return results

    async def execute_research_plan(self, query: str, tasks: List[Dict[str, Any]], callback_url: Callable) -> str:
        results = await self.execute_tasks(query, tasks, callback_url)
        
        await send_sse({"event": "update",
                        "type": "subtaskUpdate",
                        "status": "in-progress",
                        "id": str(uuid.uuid4()),
                        "task": "Generate Final Analysis",
                        "data": {}
                        })
        results = get_final_analysis(query, results)
        await send_sse({"event": "update",
                        "type": "subtaskUpdate",
                        "status": "completed",
                        "id": str(uuid.uuid4()),
                        "task": "Generate Final Analysis",
                        "data": {}
                        })

        # Send final result via webhook
        import pdb
        # pdb.set_trace()
        try:
            print(f"Inside execute_research_plan: {callback_url}")
            await send_sse({"event": "update",
                    "type": "finalAnalysis",
                    "status": "completed",
                    "id": str(uuid.uuid4()),
                    "finalAnalysis": results["finalAnalysis"],
                    "workbookData": results["finalAnalysis"]["workbookData"]
                })
            if "charts" not in results["finalAnalysis"]:
                results["finalAnalysis"]["charts"] = {}
            if "tables" not in results["finalAnalysis"]:
                results["finalAnalysis"]["tables"] = {}
            if "insights" not in results["finalAnalysis"]:
                results["finalAnalysis"]["insights"] = []

        except requests.RequestException as e:
            logging.error(f"Failed to send final result: {str(e)}")

        print(f"{results['workbookData']}")

        return results


async def run_research(query: str, callback_url: Callable):
    pipeline = ResearchPipeline()
    try:
        plan = pipeline.generate_research_plan(query)
        front_end_plan_map = [{"task": TASK_NAME_MAP[t["task"]], "description": t["description"]} for t in plan]
        await send_sse(
            {"event": "update",
            "type": "researchPlan",
            "data": {
                "researchPlan": [{"task": TASK_NAME_MAP[t["task"]], "description": t["description"]} for t in plan]
            }}
        )
        await pipeline.execute_research_plan(query, plan, callback_url)
    except Exception as e:
        logging.error(f"An error occurred during research execution: {str(e)}", exc_info=True)
        try:
            await callback_url({
                "status": "error",
                "error_message": str(e)
            })
        except requests.RequestException:
            logging.error("Failed to send error notification to callback URL")


@app.post("/api/research/start")
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    print(f"request: {request}")
    task_id = str(uuid.uuid4())
    background_tasks.add_task(run_research, request.query, send_sse)
    return {"task_id": task_id, "message": "Research started"}


latest_sse_data = None

@app.get("/stream")
async def message_stream(request: Request):
    async def event_generator():
        global latest_sse_data
        while True:
            if await request.is_disconnected():
                break

            if latest_sse_data:
                yield {
                    "event": "update",
                    "id": str(uuid.uuid4()),
                    "data": json.dumps(latest_sse_data)
                }
                latest_sse_data = None  # Clear the data after sending
            
            await asyncio.sleep(1)  # Wait a second before checking for new data

    return EventSourceResponse(event_generator())


@app.post("/send-sse")
async def send_sse(data: dict = Body(...)):
    print(f"INSIDE send_sse!!")
    global latest_sse_data
    latest_sse_data = data
    await asyncio.sleep(1)  # Wait a second before checking for new data
    return {"status": "Message queued for SSE"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)