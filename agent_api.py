from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import httpx
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables at startup
load_dotenv(dotenv_path="../../Personal_AI_Trip_Planner/.env")

app = FastAPI(title="Multi-Agent Calculator API")

# Configuration storage with BOTH agents
agent_config = {
    "mcp_server_1_url": "http://localhost:8001",
    "mcp_server_2_url": "http://localhost:8002",

    # Agent 1: Kimi K2 for Addition and Subtraction
    "kimi_k2_token": os.getenv("Kimi_K2_HF_Token"),
    "kimi_k2_base": os.getenv("Kimi_K2_HF_Base"),
    "kimi_k2_model": os.getenv("Kimi_K2_HF_Model", "moonshotai/Kimi-K2-Instruct:fireworks-ai"),
    "kimi_k2_agent": None,  # FIX: Added this

    # Agent 2: OpenAI GPT for Multiplication/Division
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "openai_model": os.getenv("OPENAI_MODEL", "gpt-4"),
    "openai_agent": None,

    "timeout": 30
}


# ==============================================================================
# AGENT 1: KIMI K2 CLIENT
# ==============================================================================
class KimiK2Agent:  # FIX: Renamed from KimiK2Client
    """Agent 1: Kimi K2 for Basic Operations"""

    def __init__(self, base_url: str, token: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=token)
        self.model = model
        self.name = "Kimi K2 Agent (Basic Ops)"  # FIX: Added name attribute

    def get_response(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content


# ==============================================================================
# AGENT 2: OPENAI GPT CLIENT
# ==============================================================================
class OpenAIAgent:
    """Agent 2: OpenAI GPT for Advanced Operations"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.name = "OpenAI GPT Agent (Advanced Ops)"

    def get_response(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content


# ==============================================================================
# AGENT INITIALIZATION
# ==============================================================================
def load_kimi_k2_agent():  # FIX: Renamed function
    """Initialize Kimi K2 Agent"""
    print("Loading Kimi K2 Agent for Addition/Subtraction...")

    token = agent_config["kimi_k2_token"]
    base_url = agent_config["kimi_k2_base"]
    model = agent_config["kimi_k2_model"]

    if not token or not base_url:
        raise RuntimeError("Kimi K2 credentials not set in environment")

    return KimiK2Agent(  # FIX: Use renamed class
        base_url=base_url,
        token=token,
        model=model
    )


def load_openai_agent():
    """Initialize OpenAI Agent"""
    print("Loading OpenAI GPT Agent for Multiplication/Division...")
    api_key = agent_config["openai_api_key"]
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAIAgent(
        api_key,
        agent_config["openai_base_url"],
        agent_config["openai_model"]
    )


def get_agent_for_operation(operation: str):
    """Route to appropriate AI AGENT based on operation"""
    if operation in ["addition", "subtraction"]:
        # FIX: Use kimi_k2_agent instead of groq_agent
        if agent_config["kimi_k2_agent"] is None:
            agent_config["kimi_k2_agent"] = load_kimi_k2_agent()
        return agent_config["kimi_k2_agent"]
    else:
        # Use OpenAI Agent
        if agent_config["openai_agent"] is None:
            agent_config["openai_agent"] = load_openai_agent()
        return agent_config["openai_agent"]


# Cache storage
calculation_cache = {}


# Models
class QueryRequest(BaseModel):
    query: Literal["addition", "subtraction", "multiplication", "division"]
    number1: float
    number2: float


class AgentResponse(BaseModel):
    result: float
    operation: str
    llm_explanation: str
    agent_used: str
    raw_tool_output: dict
    timestamp: str


class ConfigUpdate(BaseModel):
    mcp_server_1_url: str = None
    mcp_server_2_url: str = None
    kimi_k2_token: str = None  # FIX: Changed from groq
    kimi_k2_base: str = None  # FIX: Added
    kimi_k2_model: str = None  # FIX: Changed from groq
    openai_api_key: str = None
    openai_model: str = None
    timeout: int = None


class PartialConfigUpdate(BaseModel):
    setting_key: str
    setting_value: str


# ==============================================================================
# MULTI-AGENT ORCHESTRATION LOGIC
# ==============================================================================

async def call_mcp_tool(operation: str, number1: float, number2: float):
    """Route to appropriate MCP server"""
    if operation in ["addition", "subtraction"]:
        base_url = agent_config["mcp_server_1_url"]
        endpoint = "/tools/add" if operation == "addition" else "/tools/subtract"
    else:
        base_url = agent_config["mcp_server_2_url"]
        endpoint = "/tools/multiply" if operation == "multiplication" else "/tools/divide"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}{endpoint}",
            json={"number1": number1, "number2": number2},
            timeout=agent_config["timeout"]
        )
        response.raise_for_status()
        return response.json()


def generate_agent_explanation(tool_result: dict, operation: str) -> tuple[str, str]:
    """
    Multi-Agent System: Route to appropriate AI agent for explanation
    Returns: (explanation, agent_name)
    """
    result = tool_result["result"]

    try:
        # Get the appropriate agent for this operation
        agent = get_agent_for_operation(operation)

        # Create prompt
        prompt = f"""You are a friendly math tutor. Explain this calculation result clearly and encouragingly (2-3 sentences):
        

Operation: {operation}
Result: {result}

Make it easy to understand and add a helpful tip if relevant."""

        # Get response from the assigned agent
        explanation = agent.get_response(prompt)
        print(f"\n✓ {agent.name} responded")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Response: {explanation[:100]}...")

        return explanation, agent.name

    except Exception as e:
        print(f"Agent error: {e}, using fallback")
        fallback = {
            "addition": f"The sum is {result}.",
            "subtraction": f"The difference is {result}.",
            "multiplication": f"The product is {result}.",
            "division": f"The quotient is {result}."
        }
        return fallback.get(operation, f"Result: {result}"), "Fallback (no agent)"


# ==============================================================================
# HTTP ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    return {
        "message": "Multi-Agent Calculator API",
        "architecture": "2 AI Agents + 2 MCP Servers",
        "agents": {
            "agent_1": "Kimi K2 (Addition/Subtraction)",  # FIX: Updated
            "agent_2": "OpenAI GPT (Multiplication/Division)"
        },
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "settings": "/settings (PUT/PATCH)",
            "cache": "/cache (DELETE)"
        }
    }


@app.get("/health")
async def health_check():
    """Check all agents and servers"""
    mcp_statuses = {}

    async with httpx.AsyncClient() as client:
        try:
            resp1 = await client.get(f"{agent_config['mcp_server_1_url']}/health", timeout=5)
            mcp_statuses["server_1"] = resp1.json()
        except Exception as e:
            mcp_statuses["server_1"] = {"status": "unreachable", "error": str(e)}

        try:
            resp2 = await client.get(f"{agent_config['mcp_server_2_url']}/health", timeout=5)
            mcp_statuses["server_2"] = resp2.json()
        except Exception as e:
            mcp_statuses["server_2"] = {"status": "unreachable", "error": str(e)}

    # FIX: Updated agent status
    agent_status = {
        "kimi_k2_agent": "loaded" if agent_config["kimi_k2_agent"] else "not initialized",
        "openai_agent": "loaded" if agent_config["openai_agent"] else "not initialized"
    }

    # FIX: Hide correct API keys
    safe_config = {k: v for k, v in agent_config.items()
                   if k not in ["kimi_k2_token", "openai_api_key", "kimi_k2_agent", "openai_agent"]}
    safe_config["kimi_k2_token"] = "***" if agent_config["kimi_k2_token"] else None
    safe_config["openai_api_key"] = "***" if agent_config["openai_api_key"] else None

    return {
        "status": "healthy",
        "service": "Multi-Agent Calculator API",
        "agents": agent_status,
        "mcp_servers": mcp_statuses,
        "config": safe_config
    }


@app.post("/query", response_model=AgentResponse)
async def process_query(request: QueryRequest):
    """
    Multi-Agent Processing:
    1. Route to appropriate MCP server for calculation
    2. Route to appropriate AI AGENT for explanation
    3. Return response with agent information
    """
    try:
        # Step 1: Call MCP tool
        tool_result = await call_mcp_tool(
            request.query,
            request.number1,
            request.number2
        )

        # Step 2: Get explanation from appropriate AGENT
        llm_explanation, agent_name = generate_agent_explanation(
            tool_result,
            request.query
        )

        # Step 3: Return response with agent info
        response = AgentResponse(
            result=tool_result["result"],
            operation=tool_result["operation"],
            llm_explanation=llm_explanation,
            agent_used=agent_name,
            raw_tool_output=tool_result,
            timestamp=datetime.now().isoformat()
        )

        # Cache result
        cache_key = f"{request.query}_{request.number1}_{request.number2}"
        calculation_cache[cache_key] = response

        return response

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-agent error: {str(e)}")


@app.put("/settings")
async def update_all_settings(config: ConfigUpdate):
    """Update agent configurations"""
    if config.mcp_server_1_url:
        agent_config["mcp_server_1_url"] = config.mcp_server_1_url
    if config.mcp_server_2_url:
        agent_config["mcp_server_2_url"] = config.mcp_server_2_url

    # FIX: Reset correct agents
    if config.kimi_k2_token:
        agent_config["kimi_k2_token"] = config.kimi_k2_token
        agent_config["kimi_k2_agent"] = None
    if config.kimi_k2_base:
        agent_config["kimi_k2_base"] = config.kimi_k2_base
        agent_config["kimi_k2_agent"] = None
    if config.kimi_k2_model:
        agent_config["kimi_k2_model"] = config.kimi_k2_model
        agent_config["kimi_k2_agent"] = None
    if config.openai_api_key:
        agent_config["openai_api_key"] = config.openai_api_key
        agent_config["openai_agent"] = None
    if config.openai_model:
        agent_config["openai_model"] = config.openai_model
        agent_config["openai_agent"] = None
    if config.timeout:
        agent_config["timeout"] = config.timeout

    return {"message": "Multi-agent configuration updated", "config": agent_config}


@app.patch("/settings")
async def update_single_setting(update: PartialConfigUpdate):
    """Update single setting"""
    if update.setting_key not in agent_config:
        raise HTTPException(status_code=404, detail=f"Setting '{update.setting_key}' not found")

    agent_config[update.setting_key] = update.setting_value

    # FIX: Reset correct agents
    if update.setting_key in ["kimi_k2_token", "kimi_k2_base", "kimi_k2_model"]:
        agent_config["kimi_k2_agent"] = None
    if update.setting_key in ["openai_api_key", "openai_model", "openai_base_url"]:
        agent_config["openai_agent"] = None

    return {
        "message": f"Setting '{update.setting_key}' updated",
        "updated_setting": {update.setting_key: update.setting_value}
    }


@app.delete("/cache")
async def clear_cache():
    """Clear calculation cache"""
    cache_size = len(calculation_cache)
    calculation_cache.clear()
    return {
        "message": "Cache cleared",
        "items_removed": cache_size
    }








#
#
# def get_llm():
#     """Get or initialize the LLM model (lazy loading)"""
#     if agent_config["llm_model"] is None:
#         agent_config["llm_model"] = load_Kimi_K2_llm()
#     return agent_config["llm_model"]
#
#
# # Cache storage
# calculation_cache = {}
#
#
# # Models
# class QueryRequest(BaseModel):
#     query: Literal["addition", "subtraction", "multiplication", "division"]
#     number1: float
#     number2: float
#
#
# class AgentResponse(BaseModel):
#     result: float
#     operation: str
#     llm_explanation: str
#     raw_tool_output: dict
#     timestamp: str
#
#
# class ConfigUpdate(BaseModel):
#     mcp_server_1_url: str = None
#     mcp_server_2_url: str = None
#     kimi_k2_token: str = None
#     kimi_k2_base: str = None
#     kimi_k2_model: str = None
#     timeout: int = None
#
#
# class PartialConfigUpdate(BaseModel):
#     setting_key: str
#     setting_value: str
#
#
# # ==============================================================================
# # AI AGENT LOGIC
# # ==============================================================================
#
# async def call_mcp_tool(operation: str, number1: float, number2: float):
#     """Agent decides which MCP server to use based on operation"""
#
#     if operation in ["addition", "subtraction"]:
#         base_url = agent_config["mcp_server_1_url"]
#         endpoint = "/tools/add" if operation == "addition" else "/tools/subtract"
#     else:
#         base_url = agent_config["mcp_server_2_url"]
#         endpoint = "/tools/multiply" if operation == "multiplication" else "/tools/divide"
#
#     async with httpx.AsyncClient() as client:
#         response = await client.post(
#             f"{base_url}{endpoint}",
#             json={"number1": number1, "number2": number2},
#             timeout=agent_config["timeout"]
#         )
#         response.raise_for_status()
#         return response.json()
#
#
# def generate_llm_explanation(tool_result: dict) -> str:
#     """Use Kimi K2 LLM to generate explanation"""
#
#     result = tool_result["result"]
#     operation = tool_result["operation"]
#     number1 = tool_result.get("explanation", "").split()[1] if "explanation" in tool_result else ""
#     number2 = tool_result.get("explanation", "").split()[-1] if "explanation" in tool_result else ""
#
#     try:
#         llm = get_llm()
#
#         # Create prompt for LLM
#         prompt = f"Explain this calculation result in a friendly, conversational way (2-3 sentences): The {operation} operation resulted in {result}. Make it easy to understand."
#
#         # Call Kimi K2 LLM
#         response = llm.get_response(prompt)
#         print(f"> Prompt: {prompt}\n< Response: {response}")
#         return response
#
#     except Exception as e:
#         # Fallback to template if LLM fails
#         print(f"LLM error: {e}, using fallback")
#         explanations = {
#             "addition": f"The sum is {result}.",
#             "subtraction": f"The difference is {result}.",
#             "multiplication": f"The product is {result}.",
#             "division": f"The quotient is {result}."
#         }
#         return explanations.get(operation, f"The result is {result}")
#
#
# # ==============================================================================
# # HTTP ENDPOINTS
# # ==============================================================================
#
# @app.get("/")
# async def root():
#     """Root endpoint"""
#     return {
#         "message": "Calculator Agent API with Kimi K2",
#         "endpoints": {
#             "health": "/health",
#             "query": "/query (POST)",
#             "settings": "/settings (PUT/PATCH)",
#             "cache": "/cache (DELETE)"
#         }
#     }
#
#
# @app.get("/health")
# async def health_check():
#     """Check if the agent API is running and MCP servers are reachable"""
#     mcp_statuses = {}
#
#     async with httpx.AsyncClient() as client:
#         try:
#             resp1 = await client.get(f"{agent_config['mcp_server_1_url']}/health", timeout=5)
#             mcp_statuses["server_1"] = resp1.json()
#         except Exception as e:
#             mcp_statuses["server_1"] = {"status": "unreachable", "error": str(e)}
#
#         try:
#             resp2 = await client.get(f"{agent_config['mcp_server_2_url']}/health", timeout=5)
#             mcp_statuses["server_2"] = resp2.json()
#         except Exception as e:
#             mcp_statuses["server_2"] = {"status": "unreachable", "error": str(e)}
#
#     # Don't expose API token in health check
#     safe_config = {k: v for k, v in agent_config.items() if k != "kimi_k2_token"}
#     safe_config["kimi_k2_token"] = "***" if agent_config["kimi_k2_token"] else None
#
#     return {
#         "status": "healthy",
#         "service": "Calculator Agent API",
#         "llm": "Kimi K2",
#         "mcp_servers": mcp_statuses,
#         "config": safe_config
#     }
#
#
# @app.post("/query", response_model=AgentResponse)
# async def process_query(request: QueryRequest):
#     """Main endpoint - Agent processes calculation request"""
#     try:
#         # Step 1: Call MCP tool
#         tool_result = await call_mcp_tool(
#             request.query,
#             request.number1,
#             request.number2
#         )
#
#         # Step 2: Kimi K2 LLM generates explanation
#         llm_explanation = generate_llm_explanation(tool_result)
#
#         # Step 3: Return response
#         response = AgentResponse(
#             result=tool_result["result"],
#             operation=tool_result["operation"],
#             llm_explanation=llm_explanation,
#             raw_tool_output=tool_result,
#             timestamp=datetime.now().isoformat()
#         )
#
#         # Cache result
#         cache_key = f"{request.query}_{request.number1}_{request.number2}"
#         calculation_cache[cache_key] = response
#
#         return response
#
#     except httpx.HTTPStatusError as e:
#         raise HTTPException(status_code=e.response.status_code, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
#
#
# @app.put("/settings")
# async def update_all_settings(config: ConfigUpdate):
#     """Replace configuration settings"""
#     if config.mcp_server_1_url:
#         agent_config["mcp_server_1_url"] = config.mcp_server_1_url
#     if config.mcp_server_2_url:
#         agent_config["mcp_server_2_url"] = config.mcp_server_2_url
#     if config.kimi_k2_token:
#         agent_config["kimi_k2_token"] = config.kimi_k2_token
#         agent_config["llm_model"] = None  # Reset LLM
#     if config.kimi_k2_base:
#         agent_config["kimi_k2_base"] = config.kimi_k2_base
#         agent_config["llm_model"] = None  # Reset LLM
#     if config.kimi_k2_model:
#         agent_config["kimi_k2_model"] = config.kimi_k2_model
#         agent_config["llm_model"] = None  # Reset LLM
#     if config.timeout:
#         agent_config["timeout"] = config.timeout
#
#     return {"message": "Configuration updated", "config": agent_config}
#
#
# @app.patch("/settings")
# async def update_single_setting(update: PartialConfigUpdate):
#     """Update a single configuration setting"""
#     if update.setting_key not in agent_config:
#         raise HTTPException(status_code=404, detail=f"Setting '{update.setting_key}' not found")
#
#     agent_config[update.setting_key] = update.setting_value
#
#     # Reset LLM if credentials changed
#     if update.setting_key in ["kimi_k2_token", "kimi_k2_base", "kimi_k2_model"]:
#         agent_config["llm_model"] = None
#
#     return {
#         "message": f"Setting '{update.setting_key}' updated",
#         "updated_setting": {update.setting_key: update.setting_value}
#     }
#
#
# @app.delete("/cache")
# async def clear_cache():
#     """Delete all cached calculations"""
#     cache_size = len(calculation_cache)
#     calculation_cache.clear()
#     return {
#         "message": "Cache cleared",
#         "items_removed": cache_size
#     }




































# from typing import Literal
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# import httpx
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os
#
# # Load environment variables at startup
# load_dotenv()
#
# app = FastAPI(title="Calculator Agent API")
#
# # Configuration storage with Groq credentials
# agent_config = {
#     "mcp_server_1_url": "http://localhost:8001",
#     "mcp_server_2_url": "http://localhost:8002",
#     "groq_api_url": os.getenv("GROQ_MODEL", "https://api.groq.com/openai/v1"),  # Default Groq URL
#     "groq_api_key": os.getenv("GROQ_API_KEY"),
#     "groq_model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),  # Default model
#     "llm_model": None,  # Will be initialized lazily
#     "timeout": 30
# }
#
#
# def load_llm():
#     """
#     Load and return the LLM model using config from agent_config.
#     """
#     print("LLM loading...")
#
#     groq_api_key = agent_config["groq_api_key"]
#     if not groq_api_key:
#         raise RuntimeError("GROQ_API_KEY not set in environment or agent_config")
#
#     model_name = agent_config["groq_model"]
#
#     print(f"Loading LLM from Groq with model: {model_name}")
#
#     # Initialize ChatGroq with configuration
#     llm = ChatGroq(
#         model=model_name,
#         api_key=groq_api_key,
#         # base_url=agent_config["groq_api_url"]  # Uncomment if ChatGroq supports custom URL
#     )
#
#     return llm
#
#
# def get_llm():
#     """Get or initialize the LLM model (lazy loading)"""
#     if agent_config["llm_model"] is None:
#         agent_config["llm_model"] = load_llm()
#     return agent_config["llm_model"]
#
#
# # Cache storage
# calculation_cache = {}
#
#
# # Models
# class QueryRequest(BaseModel):
#     query: Literal["addition", "subtraction", "multiplication", "division"]
#     number1: float
#     number2: float
#
#
# class AgentResponse(BaseModel):
#     result: float
#     operation: str
#     llm_explanation: str
#     raw_tool_output: dict
#     timestamp: str
#
#
# class ConfigUpdate(BaseModel):
#     mcp_server_1_url: str = None
#     mcp_server_2_url: str = None
#     groq_api_url: str = None
#     groq_api_key: str = None
#     groq_model: str = None
#     timeout: int = None
#
#
# class PartialConfigUpdate(BaseModel):
#     setting_key: str
#     setting_value: str
#
#
# # ==============================================================================
# # AI AGENT LOGIC
# # ==============================================================================
#
# async def call_mcp_tool(operation: str, number1: float, number2: float):
#     """Agent decides which MCP server to use based on operation"""
#
#     if operation in ["addition", "subtraction"]:
#         base_url = agent_config["mcp_server_1_url"]
#         endpoint = "/tools/add" if operation == "addition" else "/tools/subtract"
#     else:
#         base_url = agent_config["mcp_server_2_url"]
#         endpoint = "/tools/multiply" if operation == "multiplication" else "/tools/divide"
#
#     async with httpx.AsyncClient() as client:
#         response = await client.post(
#             f"{base_url}{endpoint}",
#             json={"number1": number1, "number2": number2},
#             timeout=agent_config["timeout"]
#         )
#         response.raise_for_status()
#         return response.json()
#
#
# def generate_llm_explanation(tool_result: dict) -> str:
#     """Use actual LLM to generate explanation"""
#
#     result = tool_result["result"]
#     operation = tool_result["operation"]
#
#     # Get the LLM instance
#     llm = get_llm()
#
#     # Create prompt for LLM
#     prompt = f"Explain this calculation result in a friendly way: {operation} resulted in {result}"
#
#     # Call LLM (adjust based on langchain_groq API)
#     try:
#         response = llm.invoke(prompt)
#         return response.content if hasattr(response, 'content') else str(response)
#     except Exception as e:
#         # Fallback to template if LLM fails
#         print(f"LLM error: {e}, using fallback")
#         explanations = {
#             "addition": f"The sum is {result}.",
#             "subtraction": f"The difference is {result}.",
#             "multiplication": f"The product is {result}.",
#             "division": f"The quotient is {result}."
#         }
#         return explanations.get(operation, f"The result is {result}")
#
#
# # ==============================================================================
# # HTTP ENDPOINTS
# # ==============================================================================
#
# @app.get("/health")
# async def health_check():
#     """Check if the agent API is running and MCP servers are reachable"""
#     mcp_statuses = {}
#
#     async with httpx.AsyncClient() as client:
#         try:
#             resp1 = await client.get(f"{agent_config['mcp_server_1_url']}/health", timeout=5)
#             mcp_statuses["server_1"] = resp1.json()
#         except:
#             mcp_statuses["server_1"] = {"status": "unreachable"}
#
#         try:
#             resp2 = await client.get(f"{agent_config['mcp_server_2_url']}/health", timeout=5)
#             mcp_statuses["server_2"] = resp2.json()
#         except:
#             mcp_statuses["server_2"] = {"status": "unreachable"}
#
#     # Don't expose API key in health check
#     safe_config = {k: v for k, v in agent_config.items() if k != "groq_api_key"}
#     safe_config["groq_api_key"] = "***" if agent_config["groq_api_key"] else None
#
#     return {
#         "status": "healthy",
#         "service": "Calculator Agent API",
#         "mcp_servers": mcp_statuses,
#         "config": safe_config
#     }
#
#
# @app.post("/query", response_model=AgentResponse)
# async def process_query(request: QueryRequest):
#     """Main endpoint - Agent processes calculation request"""
#     try:
#         # Step 1: Call MCP tool
#         tool_result = await call_mcp_tool(
#             request.query,
#             request.number1,
#             request.number2
#         )
#
#         # Step 2: LLM generates explanation
#         llm_explanation = generate_llm_explanation(tool_result)
#
#         # Step 3: Return response
#         response = AgentResponse(
#             result=tool_result["result"],
#             operation=tool_result["operation"],
#             llm_explanation=llm_explanation,
#             raw_tool_output=tool_result,
#             timestamp=datetime.now().isoformat()
#         )
#
#         # Cache result
#         cache_key = f"{request.query}_{request.number1}_{request.number2}"
#         calculation_cache[cache_key] = response
#
#         return response
#
#     except httpx.HTTPStatusError as e:
#         raise HTTPException(status_code=e.response.status_code, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
#
#
# @app.put("/settings")
# async def update_all_settings(config: ConfigUpdate):
#     """Replace configuration settings"""
#     if config.mcp_server_1_url:
#         agent_config["mcp_server_1_url"] = config.mcp_server_1_url
#     if config.mcp_server_2_url:
#         agent_config["mcp_server_2_url"] = config.mcp_server_2_url
#     if config.groq_api_url:
#         agent_config["groq_api_url"] = config.groq_api_url
#     if config.groq_api_key:
#         agent_config["groq_api_key"] = config.groq_api_key
#         agent_config["llm_model"] = None  # Reset LLM to reload with new key
#     if config.groq_model:
#         agent_config["groq_model"] = config.groq_model
#         agent_config["llm_model"] = None  # Reset LLM to reload with new model
#     if config.timeout:
#         agent_config["timeout"] = config.timeout
#
#     return {"message": "Configuration updated", "config": agent_config}
#
#
# @app.patch("/settings")
# async def update_single_setting(update: PartialConfigUpdate):
#     """Update a single configuration setting"""
#     if update.setting_key not in agent_config:
#         raise HTTPException(status_code=404, detail=f"Setting '{update.setting_key}' not found")
#
#     agent_config[update.setting_key] = update.setting_value
#
#     # Reset LLM if credentials changed
#     if update.setting_key in ["groq_api_key", "groq_model", "groq_api_url"]:
#         agent_config["llm_model"] = None
#
#     return {
#         "message": f"Setting '{update.setting_key}' updated",
#         "updated_setting": {update.setting_key: update.setting_value}
#     }
#
#
# @app.delete("/cache")
# async def clear_cache():
#     """Delete all cached calculations"""
#     cache_size = len(calculation_cache)
#     calculation_cache.clear()
#     return {
#         "message": "Cache cleared",
#         "items_removed": cache_size
#     }
#
#
# # # ==============================================================================
# # # PART 3: MAIN AGENT API - Orchestrates MCP Servers and LLM
# # # File: agent_api.py


# # # ==============================================================================
# # from typing import Literal
# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel
# # from datetime import datetime
# # import httpx
# # from langchain_groq import ChatGroq
# # from dotenv import load_dotenv
# # import os
# #
# #
# # def load_llm():
# #     """
# #     Load and return the LLM model.
# #     """
# #     # Load environment variables
# #     load_dotenv()
# #
# #     model_provider: Literal["groq", "openai", "kimi_k2"] = "groq"
# #     print("LLM loading...")
# #     print(f"Loading model from provider: {model_provider}")
# #
# #     if model_provider == "groq":
# #         print("Loading LLM from Groq..............")
# #         groq_api_key = os.getenv("GROQ_API_KEY")
# #         if not groq_api_key:
# #             raise RuntimeError("GROQ_API_KEY not set in environment")
# #
# #         # Use a hardcoded model name or get from environment
# #         model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # default model
# #         llm = ChatGroq(model=model_name, api_key=groq_api_key)
# #         return llm
# #
# #     raise ValueError(f"Unsupported model provider: {model_provider}")
# #
# # app = FastAPI(title="Calculator Agent API")
# #
# #
# #
# # # Configuration storage (in-memory for demo)
# # agent_config = {
# #     "mcp_server_1_url": "http://localhost:8001",
# #     "mcp_server_2_url": "http://localhost:8002",
# #     "llm_model": None,
# #     "timeout": 30
# # }
# #
# # # Cache storage
# # calculation_cache = {}
# #
# #
# # # Models
# # class QueryRequest(BaseModel):
# #     query: Literal["addition", "subtraction", "multiplication", "division"]
# #     number1: float
# #     number2: float
# #
# #
# # class AgentResponse(BaseModel):
# #     result: float
# #     operation: str
# #     llm_explanation: str
# #     raw_tool_output: dict
# #     timestamp: str
# #
# #
# # class ConfigUpdate(BaseModel):
# #     mcp_server_1_url: str = None
# #     mcp_server_2_url: str = None
# #     llm_model: str = None
# #     timeout: int = None
# #
# #
# # class PartialConfigUpdate(BaseModel):
# #     setting_key: str
# #     setting_value: str
# #
# #
# # # ==============================================================================
# # # AI AGENT LOGIC - Routes requests to appropriate MCP server
# # # ==============================================================================
# #
# # async def call_mcp_tool(operation: str, number1: float, number2: float):
# #     """Agent decides which MCP server to use based on operation"""
# #
# #     # Agent routing logic
# #     if operation in ["addition", "subtraction"]:
# #         base_url = agent_config["mcp_server_1_url"]
# #         endpoint = "/tools/add" if operation == "addition" else "/tools/subtract"
# #     else:  # multiplication, division
# #         base_url = agent_config["mcp_server_2_url"]
# #         endpoint = "/tools/multiply" if operation == "multiplication" else "/tools/divide"
# #
# #     # Call the appropriate MCP server tool
# #     async with httpx.AsyncClient() as client:
# #         response = await client.post(
# #             f"{base_url}{endpoint}",
# #             json={"number1": number1, "number2": number2},
# #             timeout=agent_config["timeout"]
# #         )
# #         response.raise_for_status()
# #         return response.json()
# #
# #
# # def generate_llm_explanation(tool_result: dict) -> str:
# #     """Simulated LLM processing - explains the result in natural language"""
# #
# #     # In real implementation, this would call an actual LLM API
# #     # For tutorial purposes, we'll create a template-based explanation
# #
# #     result = tool_result["result"]
# #     operation = tool_result["operation"]
# #
# #     explanations = {
# #         "addition": f"I've calculated the sum for you. When we add these two numbers together, we get {result}. This is the total you get when combining both values.",
# #         "subtraction": f"I've performed the subtraction. The difference between these numbers is {result}. This represents how much remains after removing the second number from the first.",
# #         "multiplication": f"I've multiplied the numbers. The product is {result}. This represents the total when you have the first number repeated the second number of times.",
# #         "division": f"I've divided the numbers. The quotient is {result}. This shows how many times the second number fits into the first number."
# #     }
# #
# #     return explanations.get(operation, f"The result of {operation} is {result}")
# #
# #
# # # ==============================================================================
# # # HTTP METHOD ENDPOINTS
# # # ==============================================================================
# #
# # # GET - Health check
# # @app.get("/health")
# # async def health_check():
# #     """Check if the agent API is running and MCP servers are reachable"""
# #     mcp_statuses = {}
# #
# #     async with httpx.AsyncClient() as client:
# #         try:
# #             resp1 = await client.get(f"{agent_config['mcp_server_1_url']}/health", timeout=5)
# #             mcp_statuses["server_1"] = resp1.json()
# #         except:
# #             mcp_statuses["server_1"] = {"status": "unreachable"}
# #
# #         try:
# #             resp2 = await client.get(f"{agent_config['mcp_server_2_url']}/health", timeout=5)
# #             mcp_statuses["server_2"] = resp2.json()
# #         except:
# #             mcp_statuses["server_2"] = {"status": "unreachable"}
# #
# #     return {
# #         "status": "healthy",
# #         "service": "Calculator Agent API",
# #         "mcp_servers": mcp_statuses,
# #         "config": agent_config
# #     }
# #
# #
# # # POST - Submit calculation query
# # @app.post("/query", response_model=AgentResponse)
# # async def process_query(request: QueryRequest):
# #     """
# #     Main endpoint - Agent processes the request:
# #     1. Routes to appropriate MCP server tool
# #     2. Gets result from tool
# #     3. Passes result to LLM for explanation
# #     4. Returns enriched response
# #     """
# #     try:
# #         # Step 1: Agent calls the appropriate MCP tool
# #         tool_result = await call_mcp_tool(
# #             request.query,
# #             request.number1,
# #             request.number2
# #         )
# #
# #         # Step 2: LLM processes the tool result and generates explanation
# #         llm_explanation = generate_llm_explanation(tool_result)
# #
# #         # Step 3: Return complete agent response
# #         response = AgentResponse(
# #             result=tool_result["result"],
# #             operation=tool_result["operation"],
# #             llm_explanation=llm_explanation,
# #             raw_tool_output=tool_result,
# #             timestamp=datetime.now().isoformat()
# #         )
# #
# #         # Cache the result
# #         cache_key = f"{request.query}_{request.number1}_{request.number2}"
# #         calculation_cache[cache_key] = response
# #
# #         return response
# #
# #     except httpx.HTTPStatusError as e:
# #         raise HTTPException(status_code=e.response.status_code, detail=str(e))
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
# #
# #
# # # PUT - Replace entire configuration
# # @app.put("/settings")
# # async def update_all_settings(config: ConfigUpdate):
# #     """Replace all configuration settings at once"""
# #     if config.mcp_server_1_url:
# #         agent_config["mcp_server_1_url"] = config.mcp_server_1_url
# #     if config.mcp_server_2_url:
# #         agent_config["mcp_server_2_url"] = config.mcp_server_2_url
# #     if config.llm_model:
# #         agent_config["llm_model"] = config.llm_model
# #     if config.timeout:
# #         agent_config["timeout"] = config.timeout
# #
# #     return {"message": "Configuration updated", "config": agent_config}
# #
# #
# # # PATCH - Update single configuration setting
# # @app.patch("/settings")
# # async def update_single_setting(update: PartialConfigUpdate):
# #     """Update a single configuration setting"""
# #     if update.setting_key not in agent_config:
# #         raise HTTPException(status_code=404, detail=f"Setting '{update.setting_key}' not found")
# #
# #     agent_config[update.setting_key] = update.setting_value
# #
# #     return {
# #         "message": f"Setting '{update.setting_key}' updated",
# #         "updated_setting": {update.setting_key: update.setting_value}
# #     }
# #
# #
# # # DELETE - Clear cache
# # @app.delete("/cache")
# # async def clear_cache():
# #     """Delete all cached calculations"""
# #     cache_size = len(calculation_cache)
# #     calculation_cache.clear()
# #     return {
# #         "message": "Cache cleared",
# #         "items_removed": cache_size
# #     }