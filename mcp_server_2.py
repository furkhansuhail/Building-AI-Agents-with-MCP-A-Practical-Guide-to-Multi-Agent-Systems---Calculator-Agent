# ==============================================================================
# PART 2: MCP SERVER 2 - Multiplication and Division Tools
# File: mcp_server_2.py
# ==============================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
app_mcp2 = FastAPI(title="MCP Server 2 - Advanced Operations")

# Request models
class OperationRequest(BaseModel):
    number1: float
    number2: float

class OperationResponse(BaseModel):
    result: float
    operation: str
    explanation: str



# Tool 3: Multiplication
@app_mcp2.post("/tools/multiply", response_model=OperationResponse)
async def multiply_numbers(request: OperationRequest):
    """Multiplication tool - multiplies two numbers"""
    result = request.number1 * request.number2
    return {
        "result": result,
        "operation": "multiplication",
        "explanation": f"Multiplied {request.number1} by {request.number2}"
    }

# Tool 4: Division
@app_mcp2.post("/tools/divide", response_model=OperationResponse)
async def divide_numbers(request: OperationRequest):
    """Division tool - divides first number by second"""
    if request.number2 == 0:
        raise HTTPException(status_code=400, detail="Cannot divide by zero")
    result = request.number1 / request.number2
    return {
        "result": result,
        "operation": "division",
        "explanation": f"Divided {request.number1} by {request.number2}"
    }

# Health check
@app_mcp2.get("/health")
async def health_check():
    """GET - Check if server is running"""
    return {"status": "healthy", "server": "MCP Server 2", "tools": ["multiply", "divide"]}
