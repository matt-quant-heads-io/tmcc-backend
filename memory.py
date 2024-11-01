from openai import OpenAI
from mem0 import Memory
import os
from dotenv import load_dotenv


load_dotenv()


class NashMemory:
    def __init__(self):
        """
        Initialize CustomerSupportAIAgent, configure memory and OpenAI client.
        """
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.2,
                    "max_tokens": 1500,
                }
            },
            "history_db_path": "/home/ubuntu/tmcc-backend/history.db",
            "version": "v1.1"
            }
        self.memory = Memory.from_config(config)
        self.messages = []
        self.client = OpenAI()
        self.app_id = "tmcc-backend"

    def add(self, data_to_add, metadata_dict, user_id=None):
        """
        Add data to memory.

        :param data_to_add: The data to add to memory.
        :param user_id: Optional user ID to associate with memory.
        """
        self.memory.add(data_to_add, user_id=user_id, metadata=metadata_dict)

    def get_memories(self, user_id=None):
        """
        Retrieve all memories associated with a given customer ID.

        :param user_id: Optional user ID to filter memories.
        :return: List of memories.
        """
        return self.memory.get_all(user_id=user_id)

# Instantiate CustomerSupportAIAgent
nash_memory = NashMemory()

# Define a customer ID
customer_id = "matt"
metadata_dict = {"checkpoint_type": "current_workbook"}

# Handle customer query
nash_memory.add('{"revenues":{"Q1 2022":116444000000.0,"Q2 2022":121234000000.0,"Q3 2022":127101000000.0,"Q4 2022":149204000000.0}}', metadata_dict, user_id=customer_id)

memories = nash_memory.get_memories(user_id=customer_id)
print(f"memories: {memories}")
for m in memories:
    print(dir(m))