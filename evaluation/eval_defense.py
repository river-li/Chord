import os
import json
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI
from langchain_ollama.chat_models import ChatOllama
from dotenv import load_dotenv
import os

# current file path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

def load_queries():
    queries = {}
    with open(os.path.join(CURRENT_DIR, "..", "data", "query.json"), "r") as f:
        queries = json.load(f)
    return queries


def load_malicious_tools(malicious_tools_path: str):
    with open(malicious_tools_path, 'r') as f:
        malicious_tools = json.load(f)
        predecessor = malicious_tools["langchain"]["predecessor"]
        predecessors = {}
        for t in predecessor:
            predecessors[list(t.keys())[0]] = list(t.values())[0]
        successor = malicious_tools["langchain"]["successor"]
        successors = {}
        for t in successor:
            successors[list(t.keys())[0]] = list(t.values())[0]

        return predecessors, successors


def load_tools():
    from langchain_community.tools import (
        ArxivQueryRun,
        DuckDuckGoSearchRun,
        DuckDuckGoSearchResults,
        OpenWeatherMapQueryRun,
        RedditSearchRun,
        ShellTool,
        StackExchangeTool,
        TavilySearchResults,
        TavilyAnswer,
        WikipediaQueryRun,
        YouTubeSearchTool,
        YahooFinanceNewsTool,
    )

    from langchain_community.tools.amadeus.closest_airport import AmadeusClosestAirport
    from langchain_community.tools.amadeus.flight_search import AmadeusFlightSearch

    from amadeus import Client
    AmadeusClosestAirport.model_rebuild()
    AmadeusFlightSearch.model_rebuild()

    from langchain_community.tools.brave_search.tool import BraveSearch
    
    from langchain_community.tools.semanticscholar import SemanticScholarQueryRun
    from langchain_community.tools.sleep.tool import SleepTool
    from langchain_community.tools.wikidata.tool import WikidataQueryRun

    from tempfile import TemporaryDirectory
    from langchain_community.agent_toolkits import FileManagementToolkit
    file_toolkit = FileManagementToolkit(
        root_dir=str(TemporaryDirectory())
    )

    from langchain_community.agent_toolkits.financial_datasets.toolkit import (
        FinancialDatasetsToolkit,
        FinancialDatasetsAPIWrapper
    )
    financial_toolkit = FinancialDatasetsToolkit(
        api_wrapper=FinancialDatasetsAPIWrapper(
            financial_datasets_api_key=os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
        )
    )

    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.utilities import StackExchangeAPIWrapper
    from langchain_community.utilities import OpenWeatherMapAPIWrapper
    from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
    from langchain_community.utilities.wikidata import WikidataAPIWrapper

    from langchain_community.utilities.polygon import PolygonAPIWrapper
    from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit
    polygon_toolkit = PolygonToolkit.from_polygon_api_wrapper(PolygonAPIWrapper())

    from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
    from langchain_community.utilities.requests import TextRequestsWrapper

    requests_toolkit = RequestsToolkit(
        requests_wrapper=TextRequestsWrapper(headers={}),
        allow_dangerous_requests=True,
    )

    tools = [
        AmadeusClosestAirport(),
        AmadeusFlightSearch(),
        ArxivQueryRun(),
        BraveSearch.from_api_key(api_key=os.environ.get("BRAVE_SEARCH_API_KEY", "fake-key"), search_kwargs={"count": 1}),
        DuckDuckGoSearchRun(),
        DuckDuckGoSearchResults(),
        *file_toolkit.get_tools(),
        *financial_toolkit.get_tools(),
        OpenWeatherMapQueryRun(api_wrapper=OpenWeatherMapAPIWrapper()),
        *polygon_toolkit.get_tools(),
        *requests_toolkit.get_tools(),
        RedditSearchRun(api_wrapper=RedditSearchAPIWrapper(
            reddit_client_id=os.environ.get("REDDIT_CLIENT_ID", "fake-key"),
            reddit_client_secret=os.environ.get("REDDIT_CLIENT_SECRET", "fake-key"),
            reddit_user_agent="Langchain-based application"
        )),
        SemanticScholarQueryRun(),
        ShellTool(),
        SleepTool(),
        StackExchangeTool(api_wrapper=StackExchangeAPIWrapper()),
        TavilySearchResults(max_results=1),
        TavilyAnswer(max_results=1),
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        WikidataQueryRun(api_wrapper=WikidataAPIWrapper()),
        YahooFinanceNewsTool(),
        YouTubeSearchTool(),
    ]

    # Note: some of the tools imported above are not available for free plan, so we don't have corresponding tool caches for them
    with open(os.path.join(CURRENT_DIR, "..", "data", "tools_in_defense"), "r") as f:
        victim_tools = f.readlines()
        victim_tools = [tool.strip() for tool in victim_tools]
    
    tool_dict = {}
    for tool in tools:
        if tool.name in victim_tools:
            tool_dict[tool.name] = tool

    return tool_dict


def load_params():
    with open(os.path.join(CURRENT_DIR, "..", "data", "malicious_tool_arguments.json"), "r") as f:
        params = json.load(f)
    pred = params["predecessor"]
    succ = params["successor"]
    return pred, succ


def eval_defense(defense, tools, queries, predecessors, successors, pred_params, succ_params):
    for tool in tools:
        if tool not in predecessors:
            continue
        t = tools[tool]
        if t.name not in pred_params:
            continue
        query = queries[t.name]
        malicious_tool_info = predecessors[t.name]

        LOG_PREFIX = f"gpt-4o_{defense}_predecessor"
        malicious_tool_param = pred_params[t.name]
        agent = Agent(t, llm, query, enable_hijack=True, enable_harvest=True, enable_pollute=True, attack_only=True, malicious_tool=malicious_tool_info, log_prefix=LOG_PREFIX, predecessor=True, malicious_tool_params=malicious_tool_param, tool_cache=tool_cache_path, defense=defense, log_folder=f"./logs/{defense}_predecessor")
        agent.run(3)
        
        if tool not in successors:
            continue
        if t.name not in succ_params:
            continue
        successor_malicious_tool_info = successors[t.name]
        successor_malicious_tool_param = succ_params[t.name]

        LOG_PREFIX = f"gpt-4o_{defense}_successor"
        agent = Agent(t, llm, query, enable_hijack=True, enable_harvest=True, enable_pollute=True, attack_only=True, malicious_tool=successor_malicious_tool_info, log_prefix=LOG_PREFIX, predecessor=False, malicious_tool_params=successor_malicious_tool_param, tool_cache=tool_cache_path, defense=defense, log_folder=f"./logs/{defense}_successor")
        agent.run(3)
    
    print(f"Evaluation of {defense} completed, you can find the results in the final.log")


if __name__ == "__main__":

    tools = load_tools()
    queries = load_queries()
    predecessors, successors = load_malicious_tools(os.path.join(CURRENT_DIR, "..", "data", "malicious_tools.json"))
    pred_params, succ_params = load_params()

    llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
    from chord.agent import Agent

    tool_cache_path = os.path.join(CURRENT_DIR, "..", "cache", "tool_cache.db")

    print("Select a defense to evaluate:")
    print("1: Spotlight")
    print("2: Prompt Injection Detector")
    print("3: Tool Filter")
    print("4: Airgap")
    choice = input("Enter your choice (1-4): ")

    while not choice.isdigit() or int(choice) not in range(1, 5):
        print("Invalid choice. Please enter a number between 1 and 4.")
        choice = input("Enter your choice (1-4): ")

    choice = int(choice)  

    if choice == 1:
        defense = "spotlight"

    elif choice == 2:
        defense = "prompt_injection_detector"

    elif choice == 3:
        defense = "tool_filter"
        predecessors, successors = load_malicious_tools(os.path.join(CURRENT_DIR, "..", "data", "tool_filter_malicious_tools.json"))

    elif choice == 4:
        defense = "airgap"

    eval_defense(defense, tools, queries, predecessors, successors, pred_params, succ_params)
