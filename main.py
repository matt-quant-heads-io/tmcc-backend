import openai
import json
from typing import List, Dict, Any
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import requests
from dotenv import load_dotenv
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    callback_url: str

class ResearchPipeline:
    def __init__(self):
        api_key = "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"

        if not api_key:
            raise ValueError("API key not found. Please set the OAI_key environment variable.")

        self.openai_client = openai.Client(api_key=api_key)
        self.assistants = self.initialize_assistants()

    def initialize_assistants(self) -> Dict[str, str]:
        return {
            "tasker": "asst_H6mNWRlBiVIv4eX9BjRFY3QB",
            "company_or_competitor_finder": "asst_nRSy7qQGJ4h39IxzvXsqDeYr",
            "search_filing_unstructured_data": "asst_ESuHdnz5YbIwCI7W9C86lPB9",
            "get_sec_financials": "asst_1kRoE8uVe8P2PIfyXmr3dIrT",
            "stock_price": "asst_GHWBbteHIuzTlUfePKzE8nDS",
            "calculate": "asst_1eKddz6uZjsXtDmtxnzOjwI6",
            "analysis_generator": "asst_1GolhtrZofI60hQ51csDwb9H"
        }

    def generate_research_plan(self, query: str) -> List[Dict[str, Any]]:
        tasker_thread = self.openai_client.beta.threads.create()
        self.openai_client.beta.threads.messages.create(
            thread_id=tasker_thread.id,
            role="user",
            content=query
        )
        run = self.openai_client.beta.threads.runs.create(
            thread_id=tasker_thread.id,
            assistant_id=self.assistants["tasker"]
        )

        while run.status != "completed":
            run = self.openai_client.beta.threads.runs.retrieve(
                thread_id=tasker_thread.id,
                run_id=run.id
            )

        messages = self.openai_client.beta.threads.messages.list(
            thread_id=tasker_thread.id
        )
        raw_response = messages.data[0].content[0].text.value

        try:
            plan_json = json.loads(raw_response)
            return plan_json["queries"][0]["tasks"]
        except json.JSONDecodeError:
            raise ValueError("Failed to parse the research plan")
        except KeyError as e:
            raise ValueError(f"Unexpected response format: {str(e)}")

    def execute_tasks(self, tasks: List[Dict[str, Any]], callback_url: str) -> Dict[str, Any]:
        results = {}
        context = {}

        for task in tasks:
            task_name = task.get("task")
            if not task_name:
                continue

            assistant_id = self.assistants.get(task_name)

            if assistant_id:
                task_input = {
                    "task_description": task.get("description", ""),
                    "context": context
                }
                task_input_json = json.dumps(task_input)

                result = self.execute_single_task(assistant_id, task_input_json)

                try:
                    result_json = json.loads(result)
                    results[task_name] = result_json
                    context[task_name] = result_json
                except json.JSONDecodeError:
                    results[task_name] = result
                    context[task_name] = result

                # Send progress update via webhook
                self.send_progress_update(callback_url, task_name, results[task_name])

        return results

    def execute_single_task(self, assistant_id: str, task_input: str) -> str:
        thread = self.openai_client.beta.threads.create()
        self.openai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Here's the task and context: {task_input}"
        )
        run = self.openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        while run.status != "completed":
            run = self.openai_client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

        messages = self.openai_client.beta.threads.messages.list(
            thread_id=thread.id
        )

        return messages.data[0].content[0].text.value

    def generate_final_analysis(self, results: Dict[str, Any]) -> str:
        analysis_assistant_id = self.assistants.get("analysis_generator")
        if not analysis_assistant_id:
            raise ValueError("Analysis generator assistant not found.")

        analysis_thread = self.openai_client.beta.threads.create()
        self.openai_client.beta.threads.messages.create(
            thread_id=analysis_thread.id,
            role="user",
            content=f"Generate a final analysis based on these results: {json.dumps(results, indent=2)}"
        )
        run = self.openai_client.beta.threads.runs.create(
            thread_id=analysis_thread.id,
            assistant_id=analysis_assistant_id
        )

        while run.status != "completed":
            run = self.openai_client.beta.threads.runs.retrieve(
                thread_id=analysis_thread.id,
                run_id=run.id
            )

        messages = self.openai_client.beta.threads.messages.list(
            thread_id=analysis_thread.id
        )

        return messages.data[0].content[0].text.value

    def send_progress_update(self, callback_url: str, task_name: str, task_result: Any):
        try:
            requests.post(callback_url, json={
                "status": "in_progress",
                "task_completed": task_name,
                "task_result": task_result
            })
        except requests.RequestException as e:
            logging.error(f"Failed to send progress update: {str(e)}")

    def execute_research_plan(self, tasks: List[Dict[str, Any]], callback_url: str) -> str:
        results = self.execute_tasks(tasks, callback_url)
        final_analysis = self.generate_final_analysis(results)

        # Send final result via webhook
        try:
            requests.post(callback_url, json={
                "status": "completed",
                "final_analysis": final_analysis
            })
        except requests.RequestException as e:
            logging.error(f"Failed to send final result: {str(e)}")

        return final_analysis

def run_research(query: str, callback_url: str):
    pipeline = ResearchPipeline()
    try:
        plan = pipeline.generate_research_plan(query)
        pipeline.execute_research_plan(plan, callback_url)
    except Exception as e:
        logging.error(f"An error occurred during research execution: {str(e)}", exc_info=True)
        try:
            requests.post(callback_url, json={
                "status": "error",
                "error_message": str(e)
            })
        except requests.RequestException:
            logging.error("Failed to send error notification to callback URL")

@app.post("/api/research/start")
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    print(f"ResearchRequest: {ResearchRequest}")
    print(f"BackgroundTasks: {BackgroundTasks}")
    background_tasks.add_task(run_research, request.query, request.callback_url)
    return {"task_id": task_id, "message": "Research started"}


@app.get("/test")
async def test(request: ResearchRequest=None):
    return {"result": "This works!"}

if __name__ == '__main__':
    import uvicorn
    logging.info("Starting the FastAPI app...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=5001)
    except Exception as e:
        logging.error(f"An error occurred while running the FastAPI app: {e}")
    logging.info("FastAPI app has finished running.")