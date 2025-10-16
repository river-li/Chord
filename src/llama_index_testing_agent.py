import json
from typing import Sequence, List

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool
from llama_index.core.agent import FunctionCallingAgent


class TestingAgent:
    def __init__(
            self,
            llm: FunctionCallingLLM,
            tools: Sequence[BaseTool] = [],
            chat_history: List[ChatMessage] = [],
    ) -> None:
        self.agent = FunctionCallingAgent.from_tools(
            tools,
            llm=llm,
            allow_parallel_tool_calls=False
        )
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self.system_prompt = chat_history
        self._chat_history = chat_history

    def reset(self) -> None:
        self.agent.reset()

    def chat(self, message: str) -> str:
        response = self.agent.chat(message, self._chat_history)
        # self._chat_history = self.system_prompt + self.agent.chat_history
        self._chat_history = self.agent.chat_history
        return response
