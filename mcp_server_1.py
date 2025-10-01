# ==============================================================================
# PART 1: MCP SERVER 1 - Addition and Subtraction Tools
# File: mcp_server_1.py
# ==============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

app_mcp1 = FastAPI(title="MCP Server 1 - Basic Operations")

# Request models
class OperationRequest(BaseModel):
    number1: float
    number2: float

class OperationResponse(BaseModel):
    result: float
    operation: str
    explanation: str

# Tool 1: Addition
@app_mcp1.post("/tools/add", response_model=OperationResponse)
async def add_numbers(request: OperationRequest):
    """Addition tool - adds two numbers"""
    result = request.number1 + request.number2
    return {
        "result": result,
        "operation": "addition",
        "explanation": f"Added {request.number1} and {request.number2}"
    }

# Tool 2: Subtraction
@app_mcp1.post("/tools/subtract", response_model=OperationResponse)
async def subtract_numbers(request: OperationRequest):
    """Subtraction tool - subtracts second number from first"""
    result = request.number1 - request.number2
    return {
        "result": result,
        "operation": "subtraction",
        "explanation": f"Subtracted {request.number2} from {request.number1}"
    }

# Health check
@app_mcp1.get("/health")
async def health_check():
    """GET - Check if server is running"""
    return {"status": "healthy", "server": "MCP Server 1", "tools": ["add", "subtract"]}
