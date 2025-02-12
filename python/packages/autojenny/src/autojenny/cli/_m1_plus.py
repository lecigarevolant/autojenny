from autogen_ext.ui import AutogenFastAPIViewer
import argparse
import asyncio
import os
import sys
import warnings
import logging
from typing import Any, Dict, Optional

import yaml
from autogen_agentchat.ui import Console, UserInputManager
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.teams.magentic_one_plus import MagenticOnePlus
from autogen_ext.ui import RichConsole, AutogenFastAPIViewer, app, task_handlers

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='magentic_one_plus.log',
    filemode='w'
)
logger = logging.getLogger("autogen_agentchat")
logger.setLevel(logging.DEBUG)

# Suppress warnings about the requests.Session() not being closed
warnings.filterwarnings(
    action="ignore", message="unclosed", category=ResourceWarning)

DEFAULT_CONFIG_FILE = "config.yaml"
DEFAULT_CONFIG_CONTENTS = """# config.yaml
clients:
  orchestrator:
    provider: autogen_ext.models.openai.OpenAIChatCompletionClient
    config:
      model: gpt-4o
  
  file_surfer:
    provider: autogen_ext.models.openai.OpenAIChatCompletionClient
    config:
      model: gpt-4o
  
  web_surfer:
    provider: autogen_ext.models.openai.OpenAIChatCompletionClient
    config:
      model: gpt-4o
  
  coder:
    provider: autogen_ext.models.openai.OpenAIChatCompletionClient
    config:
      model: gpt-4o

workspace:
  working_directory: "./magentic_workspace"
  browser_data_dir: "./magentic_workspace/browser_data"

web_surfer:
  headless: true
  to_save_screenshots: true
  use_ocr: true
  browser_channel: "chromium"
  to_resize_viewport: true
  animate_actions: true
"""


async def cancellable_input(prompt: str, cancellation_token: Optional[CancellationToken]) -> str:
    task: asyncio.Task[str] = asyncio.create_task(
        asyncio.to_thread(input, prompt))
    if cancellation_token is not None:
        cancellation_token.link_future(task)
    return await task


def main() -> None:
    """
    Command-line interface for running a complex task using MagenticOnePlus.

    This script accepts a single task string and optional flags:
    --no-hil: Disable human-in-the-loop mode
    --rich: Enable rich console output
    --web: Enable web UI using FastAPI
    --config: Specify an alternate model configuration

    Example usage:
    python magentic_one_cli.py "example task"
    python magentic_one_cli.py --no-hil "example task"
    python magentic_one_cli.py --rich "example task"
    python magentic_one_cli.py --web "example task"
    python magentic_one_cli.py --config config.yaml "example task"

    Use --sample-config to print a sample configuration file.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run a complex task using MagenticOne.\n\n"
            "For more information, refer to the following paper: https://arxiv.org/abs/2411.04468"
        )
    )
    parser.add_argument("task", type=str, nargs="?",
                        help="The task to be executed by MagenticOne.")
    parser.add_argument("--no-hil", action="store_true",
                        help="Disable human-in-the-loop mode.")
    parser.add_argument(
        "--rich",
        action="store_true",
        help="Enable rich console output",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Enable web UI using FastAPI",
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs=1,
        help="The model configuration file to use. Leave empty to print a sample configuration.",
    )
    parser.add_argument("--sample-config", action="store_true",
                        help="Print a sample configuration to console.")

    args = parser.parse_args()

    if args.sample_config:
        sys.stdout.write(DEFAULT_CONFIG_CONTENTS + "\n")
        return

    # We're not printing a sample, so we need a task
    if args.task is None and not args.web:
        parser.print_usage()
        return

    # Load the configuration
    config: Dict[str, Any] = {}

    if args.config is None:
        if os.path.isfile(DEFAULT_CONFIG_FILE):
            with open(DEFAULT_CONFIG_FILE, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = yaml.safe_load(DEFAULT_CONFIG_CONTENTS)
    else:
        with open(args.config if isinstance(args.config, str) else args.config[0], "r") as f:
            config = yaml.safe_load(f)

    if args.web:
        # Import here to avoid dependency if not using web UI
        import uvicorn

        # Run the task in the background when a WebSocket connection is established
        async def run_task_for_client(client_id: str) -> None:
            input_manager = UserInputManager(callback=cancellable_input)

            # Load all clients
            clients_config = config.get("clients", {})

            # Modified client loading to preserve extra_create_args from config
            orchestrator_client = ChatCompletionClient.load_component(
                clients_config["orchestrator"])
            file_surfer_client = ChatCompletionClient.load_component(
                clients_config["file_surfer"])
            web_surfer_client = ChatCompletionClient.load_component(
                clients_config["web_surfer"])
            coder_client = ChatCompletionClient.load_component(
                clients_config["coder"])

            # Get workspace config with defaults
            workspace_config = config.get("workspace", {})
            work_dir = workspace_config.get(
                "working_directory", "./magentic_workspace")
            browser_data_dir = workspace_config.get(
                "browser_data_dir", os.path.join(work_dir, "browser_data"))

            # The extra_create_args will be automatically loaded as part of the client configs
            # and available in the client instances through their config attribute

            m1 = MagenticOnePlus(
                orchestrator_client=orchestrator_client,
                file_surfer_client=file_surfer_client,
                web_surfer_client=web_surfer_client,
                coder_client=coder_client,
                hil_mode=False,  # No HIL in web mode
                input_func=input_manager.get_wrapped_callback(),
                work_dir=work_dir,
                browser_data_dir=browser_data_dir,
                web_surfer_config=config.get("web_surfer", {})
            )

            try:
                await m1.__aenter__()
                await AutogenFastAPIViewer(
                    m1.run_stream(task=args.task),
                    client_id=client_id,
                    output_stats=True
                )
            finally:
                await m1.docker_executor.__aexit__(None, None, None)

        # Register the task handler for any client that connects
        task_handlers["default"] = run_task_for_client

        # Start the FastAPI server
        uvicorn.run(app, host="127.0.0.1", port=8000)
        return

    # Run the task
    async def run_task(task: str, hil_mode: bool, use_rich_console: bool) -> None:
        input_manager = UserInputManager(callback=cancellable_input)

        # Load all clients
        clients_config = config.get("clients", {})

        # Modified client loading to preserve extra_create_args from config
        orchestrator_client = ChatCompletionClient.load_component(
            clients_config["orchestrator"])
        file_surfer_client = ChatCompletionClient.load_component(
            clients_config["file_surfer"])
        web_surfer_client = ChatCompletionClient.load_component(
            clients_config["web_surfer"])
        coder_client = ChatCompletionClient.load_component(
            clients_config["coder"])

        # Get workspace config with defaults
        workspace_config = config.get("workspace", {})
        work_dir = workspace_config.get(
            "working_directory", "./magentic_workspace")
        browser_data_dir = workspace_config.get(
            "browser_data_dir", os.path.join(work_dir, "browser_data"))

        # The extra_create_args will be automatically loaded as part of the client configs
        # and available in the client instances through their config attribute

        m1 = MagenticOnePlus(
            orchestrator_client=orchestrator_client,
            file_surfer_client=file_surfer_client,
            web_surfer_client=web_surfer_client,
            coder_client=coder_client,
            hil_mode=hil_mode,
            input_func=input_manager.get_wrapped_callback(),
            work_dir=work_dir,
            browser_data_dir=browser_data_dir,
            web_surfer_config=config.get("web_surfer", {})
        )

        try:
            await m1.__aenter__()
            if use_rich_console:
                await RichConsole(m1.run_stream(task=task), output_stats=False, user_input_manager=input_manager)
            else:
                await Console(m1.run_stream(task=task), output_stats=False, user_input_manager=input_manager)
        finally:
            await m1.docker_executor.__aexit__(None, None, None)

    task = args.task if isinstance(args.task, str) else args.task[0]
    asyncio.run(run_task(task, not args.no_hil, args.rich))


if __name__ == "__main__":
    main()
