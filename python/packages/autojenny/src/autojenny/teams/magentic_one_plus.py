import warnings
from typing import Awaitable, Callable, List, Optional, Union
import os
import shutil

from autogen_agentchat.agents import CodeExecutorAgent, UserProxyAgent
from autogen_agentchat.base import ChatAgent
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_group_chat_plus import MagenticOneGroupChatPlus
from autogen_core import CancellationToken, AgentId, MessageContext, event, TypeSubscription
from autogen_core.models import ChatCompletionClient
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    HandoffMessage,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams._group_chat._events import (
    GroupChatAgentResponse,
    GroupChatMessage,
    GroupChatRequestPublish,
    GroupChatReset,
    GroupChatStart,
    GroupChatTermination,
)
from autogen_core.code_executor import CodeBlock

from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai._openai_client import BaseOpenAIChatCompletionClient


SyncInputFunc = Callable[[str], str]
AsyncInputFunc = Callable[[str, Optional[CancellationToken]], Awaitable[str]]
InputFuncType = Union[SyncInputFunc, AsyncInputFunc]

working_directory = "./magentic_workspace"


class OrchestratorSubscriber:
    def __init__(self, orchestrator_id: AgentId):
        self.orchestrator_id = orchestrator_id
        self.task = ""
        self.facts = ""
        self.plan = ""
        self.n_rounds = 0
        self.n_stalls = 0

    @event
    async def handle_group_chat_start(self, message: GroupChatStart, ctx: MessageContext) -> None:
        print(f"=== Group Chat Started ===")
        if message.messages:
            # Combine all message contents for task
            self.task = " ".join(
                [str(msg.content) for msg in message.messages if msg.content is not None])
            print(f"Initial Task: {self.task}")

    @event
    async def handle_group_chat_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        print(f"\n=== Group Chat Message (from {message.message.source}) ===")
        if isinstance(message.message, ChatMessage):
            self._print_chat_message(message.message)

    @event
    async def handle_group_chat_agent_response(self, message: GroupChatAgentResponse, ctx: MessageContext) -> None:
        print(
            f"\n=== Agent Response (from {message.agent_response.chat_message.source}) ===")

        # Handle inner messages
        if message.agent_response.inner_messages:
            print("--- Inner Messages ---")
            for inner_message in message.agent_response.inner_messages:
                if isinstance(inner_message, AgentEvent):
                    self._print_agent_event(inner_message)
                elif isinstance(inner_message, ChatMessage):
                    self._print_chat_message(inner_message)

        # Handle the main chat message
        print("--- Chat Message ---")
        self._print_chat_message(message.agent_response.chat_message)
        await self.reconstruct_state(message.agent_response.chat_message)

    @event
    async def handle_group_chat_termination(self, message: GroupChatTermination, ctx: MessageContext) -> None:
        print(f"\n=== Group Chat Terminated ===")
        print(f"Reason: {message.message.content}")
        print(f"Task: {self.task}")
        print(f"Facts: {self.facts}")
        print(f"Plan: {self.plan}")
        print(f"Rounds: {self.n_rounds}")
        print(f"Stalls: {self.n_stalls}")

    @event
    async def handle_group_chat_reset(self, message: GroupChatReset, ctx: MessageContext) -> None:
        print(f"\n=== Group Chat Reset ===")

    @event
    async def handle_group_chat_request_publish(self, message: GroupChatRequestPublish, ctx: MessageContext) -> None:
        print(f"\n=== Group Chat Request Publish ===")

    def _print_chat_message(self, message: ChatMessage) -> None:
        if isinstance(message, TextMessage):
            print(
                f"  Type: TextMessage\n  Content: {message.content}\n  Source: {message.source}")
        elif isinstance(message, MultiModalMessage):
            print(
                f"  Type: MultiModalMessage\n  Content: {message.content}\n  Source: {message.source}")
        elif isinstance(message, ToolCallSummaryMessage):
            print(
                f"  Type: ToolCallSummaryMessage\n  Content: {message.content}\n  Source: {message.source}")
        elif isinstance(message, HandoffMessage):
            print(
                f"  Type: HandoffMessage\n  Content: {message.content}\n  Source: {message.source}")
        elif isinstance(message, StopMessage):
            print(
                f"  Type: StopMessage\n  Content: {message.content}\n  Source: {message.source}")
        else:
            print(f"  Type: Unknown ChatMessage\n  Message: {message}")

    def _print_agent_event(self, event: AgentEvent) -> None:
        if isinstance(event, ToolCallRequestEvent):
            print(
                f"  Type: ToolCallRequestEvent\n  Tool Name: {event.content[0].name}\n  Arguments: {event.content[0].arguments}")
        elif isinstance(event, ToolCallExecutionEvent):
            print(
                f"  Type: ToolCallExecutionEvent\n  Result: {event.content[0].content}")
        # Recurse, as inner messages can be chat messages
        elif isinstance(event, ChatMessage):
            self._print_chat_message(event)
        else:
            print(f"  Type: Unknown AgentEvent\n  Event: {event}")

    async def reconstruct_state(self, message: ChatMessage) -> None:
        if message.source == self.orchestrator_id.type:
            self.n_rounds += 1
            if "facts" in str(message.content).lower():
                self.facts = str(message.content)
            elif "plan" in str(message.content).lower():
                self.plan = str(message.content)
            elif "stalls" in str(message.content).lower():  # This should never occur
                self.n_stalls += 1
            print(f"Task: {self.task}")
            print(f"Facts: {self.facts}")
            print(f"Plan: {self.plan}")
            print(f"Rounds: {self.n_rounds}")
            print(f"Stalls: {self.n_stalls}")


class MagenticOnePlus(MagenticOneGroupChatPlus):
    def __init__(
        self,
        orchestrator_client: ChatCompletionClient,
        file_surfer_client: ChatCompletionClient,
        web_surfer_client: ChatCompletionClient,
        coder_client: ChatCompletionClient,
        hil_mode: bool = False,
        input_func: InputFuncType | None = None,
        work_dir: str = "./magentic_workspace",
        browser_data_dir: str | None = None,
        web_surfer_config: dict = {},
        docker_image: str = "python:3-slim"  # Allow custom Docker image
    ):
        # Store the clients as instance variables so we can access their configs
        self.orchestrator_client = orchestrator_client
        self.file_surfer_client = file_surfer_client
        self.web_surfer_client = web_surfer_client
        self.coder_client = coder_client

        if browser_data_dir is None:
            browser_data_dir = os.path.join(work_dir, "browser_data")

        if os.path.exists(browser_data_dir):
            shutil.rmtree(browser_data_dir)
        os.makedirs(browser_data_dir, exist_ok=True)

        self._validate_client_capabilities([
            orchestrator_client,
            file_surfer_client,
            web_surfer_client,
            coder_client
        ])

        # Create agents with their respective client's extra_create_args
        downloads_folder = os.path.join(browser_data_dir, "downloads")
        os.makedirs(downloads_folder, exist_ok=True)

        file_surfer = FileSurfer(
            "FileSurfer",
            model_client=file_surfer_client,
            description=f"An agent that can handle local files. Files downloaded by the web surfer will be in {downloads_folder}."
        )

        web_surfer = MultimodalWebSurfer(
            "WebSurfer",
            model_client=web_surfer_client,
            downloads_folder=downloads_folder,
            debug_dir=os.path.join(browser_data_dir, "debug"),
            browser_data_dir=browser_data_dir,
            **web_surfer_config
        )

        coder = MagenticOneCoderAgent(
            "Coder",
            model_client=coder_client
        )

        self.docker_executor = DockerCommandLineCodeExecutor(
            image=docker_image,
            work_dir=work_dir,
            timeout=300,  # 5 minute timeout
            auto_remove=True
        )

        # Install common dependencies on first use
        self._common_deps = [
            "matplotlib",
            "pandas",
            "numpy",
            "requests",
            "beautifulsoup4",
            "pillow"
        ]
        self._deps_installed = False

        self._executor_agent = CodeExecutorAgent(
            "Executor",
            code_executor=self.docker_executor
        )

        agents: List[ChatAgent] = [file_surfer,
                                   web_surfer, coder, self._executor_agent]

        if hil_mode:
            user_proxy = UserProxyAgent("User", input_func=input_func)
            agents.append(user_proxy)

        # Pass the orchestrator client to the parent class
        super().__init__(
            agents,
            model_client=orchestrator_client
        )

    async def __aenter__(self):
        await self.docker_executor.__aenter__()

        # Install common dependencies on first use
        if not self._deps_installed:
            try:
                packages = " ".join(self._common_deps)
                result = await self.docker_executor.execute_code_blocks(
                    [CodeBlock(code=f"pip install {packages}", language="sh")],
                    CancellationToken()
                )
                if result.exit_code == 0:
                    self._deps_installed = True
                else:
                    print(
                        f"Warning: Failed to install common dependencies: {result.output}")
            except Exception as e:
                print(f"Warning: Error installing dependencies: {str(e)}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup of resources"""
        if hasattr(self, 'docker_executor'):
            await self.docker_executor.__aexit__(exc_type, exc_val, exc_tb)

        # Clean up any agents that need cleanup
        for agent in self._participants:  # Access through parent class's protected member
            if hasattr(agent, 'close') and callable(agent.close):
                await agent.close()

    def _validate_client_capabilities(self, clients: List[ChatCompletionClient]) -> None:
        for client in clients:
            capabilities = getattr(client, "model_info", {})
            required_capabilities = [
                "vision", "function_calling", "json_output"]

            if capabilities and not all(capabilities.get(cap) for cap in required_capabilities):
                warnings.warn(
                    f"Client {client} capabilities for MagenticOne should include vision, function calling, and json output.",
                    stacklevel=2,
                )

            if not isinstance(client, BaseOpenAIChatCompletionClient):
                warnings.warn(
                    f"MagenticOne performs best with OpenAI GPT-4o model either through OpenAI or Azure OpenAI. Client {client} may not perform optimally.",
                    stacklevel=2,
                )
