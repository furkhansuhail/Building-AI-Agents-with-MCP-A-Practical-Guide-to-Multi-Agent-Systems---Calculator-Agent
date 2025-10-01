# Building-AI-Agents-with-MCP-A-Practical-Guide-to-Multi-Agent-Systems---Calculator-Agent
A practical multi-agent calculator system


# 🤖 Multi-Agent Calculator with MCP

A production-ready demonstration of AI Agent architecture using the Model Context Protocol (MCP). This project showcases how to build modular, scalable agent systems where specialized AI models work together with dedicated tool servers.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project demonstrates a multi-agent system where:

- **2 AI Agents** (Kimi K2 and OpenAI GPT) provide natural language explanations
- **2 MCP Servers** handle mathematical computations
- **1 Orchestration Layer** routes requests intelligently
- **1 Streamlit UI** provides user interaction

The system showcases key concepts in modern AI architecture:
- Agent specialization and routing
- Model Context Protocol (MCP) for tool integration
- Separation of concerns between computation and language generation
- RESTful API design patterns

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│                    (User Interface)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP POST
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    Agent API (Port 8000)                     │
│                    [Orchestration Layer]                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Request Router                                     │     │
│  │  • Analyzes operation type                         │     │
│  │  • Selects appropriate agent & MCP server          │     │
│  └────────────────────────────────────────────────────┘     │
└───────────────┬──────────────────────┬──────────────────────┘
                │                      │
        ┌───────┴────────┐    ┌────────┴────────┐
        │ Kimi K2 Agent  │    │ OpenAI Agent    │
        │ (Add/Subtract) │    │ (Multiply/Div)  │
        └───────┬────────┘    └────────┬────────┘
                │                      │
        ┌───────┴────────┐    ┌────────┴────────┐
        │ MCP Server 1   │    │ MCP Server 2    │
        │ (Port 8001)    │    │ (Port 8002)     │
        │ • Addition     │    │ • Multiplication│
        │ • Subtraction  │    │ • Division      │
        └────────────────┘    └─────────────────┘
```

## ✨ Features

### Core Functionality
- ✅ Multi-agent system with intelligent routing
- ✅ Modular MCP servers for specific operations
- ✅ Natural language explanations of results
- ✅ Response caching for performance
- ✅ Comprehensive error handling
- ✅ Health monitoring endpoints

### API Features
- ✅ RESTful API design (GET, POST, PUT, PATCH, DELETE)
- ✅ Configuration management
- ✅ Cache management
- ✅ Async operations for better performance

### UI Features
- ✅ Clean, intuitive Streamlit interface
- ✅ Real-time calculations
- ✅ Technical details view
- ✅ System health checks
- ✅ Cache clearing

## 📦 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- API keys for:
  - Kimi K2 (or alternative LLM)
  - OpenAI GPT

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multi-agent-calculator.git
cd multi-agent-calculator
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
httpx==0.25.0
python-dotenv==1.0.0
openai==1.3.0
streamlit==1.28.0
requests==2.31.0
```

## ⚙️ Configuration

### 1. Environment Variables

Copy the template and fill in your API keys:

```bash
cp .env.template .env
```

Edit `.env` with your actual credentials:

```bash
# Agent 1: Kimi K2
Kimi_K2_HF_Token=your_actual_token_here
Kimi_K2_HF_Base=https://your-api-endpoint.com/v1
Kimi_K2_HF_Model=moonshotai/Kimi-K2-Instruct:fireworks-ai

# Agent 2: OpenAI
OPENAI_API_KEY=sk-your_actual_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
```

### 2. Port Configuration (Optional)

Default ports:
- MCP Server 1: `8001`
- MCP Server 2: `8002`
- Agent API: `8000`
- Streamlit: `8501`

Modify in code if needed.

## 🎮 Running the Application

You'll need **4 terminal windows** (or use `tmux`/`screen`):

### Terminal 1: MCP Server 1 (Basic Operations)

```bash
cd path/to/project
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn mcp_server_1:app_mcp1 --host 0.0.0.0 --port 8001 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Application startup complete.
```

### Terminal 2: MCP Server 2 (Advanced Operations)

```bash
cd path/to/project
source venv/bin/activate
uvicorn mcp_server_2:app_mcp2 --host 0.0.0.0 --port 8002 --reload
```

### Terminal 3: Agent API (Orchestrator)

```bash
cd path/to/project
source venv/bin/activate
uvicorn agent_api:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:
```
Loading Kimi K2 Agent for Addition/Subtraction...
Loading OpenAI GPT Agent for Multiplication/Division...
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 4: Streamlit UI

```bash
cd path/to/project
source venv/bin/activate
streamlit run streamlit_app.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.1.x:8501
```

### ✅ Verify All Services

Open your browser:

1. **Streamlit UI**: http://localhost:8501
2. **Agent API**: http://localhost:8000/health
3. **MCP Server 1**: http://localhost:8001/health
4. **MCP Server 2**: http://localhost:8002/health

All health checks should return `"status": "healthy"`

## 📚 API Documentation

### Agent API Endpoints

#### GET /health
Check system status and connectivity to MCP servers.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "Multi-Agent Calculator API",
  "agents": {
    "kimi_k2_agent": "loaded",
    "openai_agent": "not initialized"
  },
  "mcp_servers": {
    "server_1": {"status": "healthy"},
    "server_2": {"status": "healthy"}
  }
}
```

#### POST /query
Submit a calculation request.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "addition",
    "number1": 10,
    "number2": 5
  }'
```

Response:
```json
{
  "result": 15.0,
  "operation": "addition",
  "llm_explanation": "Great calculation! When you add 10 and 5 together...",
  "agent_used": "Kimi K2 Agent (Basic Ops)",
  "raw_tool_output": {...},
  "timestamp": "2025-01-15T10:30:45.123456"
}
```

#### PUT /settings
Update all configuration settings.

```bash
curl -X PUT http://localhost:8000/settings \
  -H "Content-Type: application/json" \
  -d '{
    "mcp_server_1_url": "http://new-server:8001",
    "timeout": 60
  }'
```

#### PATCH /settings
Update a single configuration setting.

```bash
curl -X PATCH http://localhost:8000/settings \
  -H "Content-Type: application/json" \
  -d '{
    "setting_key": "timeout",
    "setting_value": "45"
  }'
```

#### DELETE /cache
Clear all cached calculations.

```bash
curl -X DELETE http://localhost:8000/cache
```

### MCP Server Endpoints

#### MCP Server 1 (Basic Operations)

**POST /tools/add**
```bash
curl -X POST http://localhost:8001/tools/add \
  -H "Content-Type: application/json" \
  -d '{"number1": 10, "number2": 5}'
```

**POST /tools/subtract**
```bash
curl -X POST http://localhost:8001/tools/subtract \
  -H "Content-Type: application/json" \
  -d '{"number1": 10, "number2": 5}'
```

#### MCP Server 2 (Advanced Operations)

**POST /tools/multiply**
**POST /tools/divide**

## 📁 Project Structure

```
multi-agent-calculator/
│
├── mcp_server_1.py          # MCP Server for addition/subtraction
├── mcp_server_2.py          # MCP Server for multiplication/division
├── agent_api.py             # Main orchestration layer with AI agents
├── streamlit_app.py         # Frontend user interface
│
├── .env                     # Environment variables (not in git)
├── .env.template            # Template for environment setup
├── .gitignore              # Git ignore rules
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
└── docs/                   # Additional documentation
    ├── architecture.md     # Detailed architecture docs
    └── api-reference.md    # Complete API reference
```

## 🔍 How It Works

### Request Flow Example: Adding 10 + 5

1. **User Input** (Streamlit UI)
   - User selects "addition"
   - Enters numbers: 10 and 5
   - Clicks "Calculate"

2. **API Request** (HTTP)
   ```
   POST http://localhost:8000/query
   Body: {"query": "addition", "number1": 10, "number2": 5}
   ```

3. **Orchestration** (Agent API)
   - Analyzes operation type: "addition"
   - Routes to MCP Server 1
   - Calls `/tools/add` endpoint

4. **Computation** (MCP Server 1)
   - Receives: `{"number1": 10, "number2": 5}`
   - Calculates: `10 + 5 = 15`
   - Returns: `{"result": 15.0, "operation": "addition", ...}`

5. **AI Explanation** (Kimi K2 Agent)
   - Receives result: 15.0
   - Generates prompt for LLM
   - LLM creates friendly explanation
   - Returns: "Great calculation! When you add 10 and 5..."

6. **Response Assembly**
   - Combines computation result + AI explanation
   - Adds metadata (agent used, timestamp)
   - Caches response

7. **Display** (Streamlit UI)
   - Shows result prominently: **15.00**
   - Displays AI explanation
   - Shows technical details

### Agent Routing Logic

```python
Addition/Subtraction → Kimi K2 Agent → MCP Server 1
Multiplication/Division → OpenAI Agent → MCP Server 2
```

This demonstrates how different specialized agents can handle different types of operations.

## 🛠️ Development

### Adding a New Operation

1. **Create the tool** in an MCP server:

```python
@app_mcp1.post("/tools/power")
async def power_numbers(request: OperationRequest):
    result = request.number1 ** request.number2
    return {
        "result": result,
        "operation": "exponentiation",
        "explanation": f"Raised {request.number1} to power {request.number2}"
    }
```

2. **Update routing** in `agent_api.py`:

```python
async def call_mcp_tool(operation: str, number1: float, number2: float):
    if operation in ["addition", "subtraction"]:
        # ... existing code ...
    elif operation == "power":
        base_url = agent_config["mcp_server_1_url"]
        endpoint = "/tools/power"
    # ... rest of code
```

3. **Update the request model** to accept the new operation:

```python
class QueryRequest(BaseModel):
    query: Literal["addition", "subtraction", "multiplication", "division", "power"]
    number1: float
    number2: float
```

4. **Update the Streamlit UI** to include the new option.

### Adding a New Agent

To add a third agent (e.g., for specialized operations):

1. **Create agent class** in `agent_api.py`:

```python
class ClaudeAgent:
    """Agent 3: Claude for special operations"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.name = "Claude Agent (Special Ops)"
    
    def get_response(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
```

2. **Update routing logic**:

```python
def get_agent_for_operation(operation: str):
    if operation in ["addition", "subtraction"]:
        return get_kimi_k2_agent()
    elif operation in ["multiplication", "division"]:
        return get_openai_agent()
    elif operation in ["power", "root"]:
        return get_claude_agent()
```

### Testing

Run tests (when available):

```bash
pytest tests/
```

Manual testing checklist:
- [ ] All 4 services start without errors
- [ ] Health endpoints return 200 OK
- [ ] Each operation type works (add, subtract, multiply, divide)
- [ ] AI explanations are generated
- [ ] Cache is working
- [ ] Error handling works (try dividing by zero)

## 🐛 Troubleshooting

### Issue: MCP Server won't start

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using the port
lsof -i :8001  # On macOS/Linux
netstat -ano | findstr :8001  # On Windows

# Kill the process
kill -9 <PID>  # On macOS/Linux
taskkill /PID <PID> /F  # On Windows
```

### Issue: Agent API can't reach MCP servers

**Error:** `Connection refused` or `unreachable`

**Checklist:**
1. Verify MCP servers are running: `curl http://localhost:8001/health`
2. Check firewall settings
3. Verify ports in `.env` match running services
4. Check logs for startup errors

### Issue: LLM not responding

**Error:** `401 Unauthorized` or `Agent error`

**Checklist:**
1. Verify API keys in `.env` are correct
2. Check API key has sufficient credits/quota
3. Test API key directly with curl:
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_KEY"
```
4. Check for rate limiting

### Issue: Streamlit can't connect to Agent API

**Solution:**
1. Verify Agent API is running on port 8000
2. Check `agent_url` in Streamlit sidebar
3. Look at browser console for errors (F12)
4. Verify no CORS issues in Agent API logs

### Issue: Import errors

**Error:** `ModuleNotFoundError`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Or install missing package specifically
pip install package-name
```

### Issue: Division by zero not handled

This is expected behavior! The MCP server returns an HTTP 400 error, which the Agent API catches and returns to the user. This demonstrates proper error handling.

## 📊 Performance Considerations

### Caching
The system caches responses to avoid redundant calculations:
- Cache key format: `{operation}_{number1}_{number2}`
- Clear cache via UI or API: `DELETE /cache`
- Consider Redis for production

### Async Operations
All I/O operations (HTTP requests, LLM calls) are async for better performance:
```python
async def call_mcp_tool(...)  # Non-blocking
await client.post(...)        # Concurrent requests possible
```

### Scaling Options

**Horizontal Scaling:**
- Run multiple instances of each MCP server
- Use load balancer (nginx, HAProxy)
- Add Redis for shared cache

**Optimization:**
- Implement request queuing (Celery, RabbitMQ)
- Add connection pooling
- Use async LLM libraries where available

## 🔒 Security Best Practices

### For Development

1. **Never commit `.env` file**
   ```bash
   # Add to .gitignore
   .env
   *.env
   ```

2. **Use environment-specific configs**
   ```
   .env.development
   .env.staging
   .env.production
   ```

3. **Validate all inputs**
   - Pydantic models handle basic validation
   - Add custom validators for business logic

### For Production

1. **Use secrets management**
   - AWS Secrets Manager
   - HashiCorp Vault
   - Azure Key Vault

2. **Add authentication**
   ```python
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   @app.post("/query")
   async def process_query(
       request: QueryRequest,
       credentials: HTTPAuthorizationCredentials = Depends(security)
   ):
       # Verify token
       ...
   ```

3. **Rate limiting**
   ```python
   from slowapi import Limiter
   
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/query")
   @limiter.limit("10/minute")
   async def process_query(...):
       ...
   ```

4. **HTTPS only**
   - Use reverse proxy (nginx, Caddy)
   - Enable HTTPS/TLS
   - Set secure headers

## 📈 Monitoring and Logging

### Adding Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.post("/query")
async def process_query(request: QueryRequest):
    logger.info(f"Processing {request.query} for {request.number1}, {request.number2}")
    # ... rest of code
```

### Metrics to Track

- Request count by operation type
- Response times (p50, p95, p99)
- Agent selection distribution
- Cache hit rate
- Error rate by endpoint
- LLM token usage

### Recommended Tools

- **Logging:** Loguru, structlog
- **Monitoring:** Prometheus + Grafana
- **APM:** New Relic, DataDog
- **Error Tracking:** Sentry

## 🚢 Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose all ports
EXPOSE 8000 8001 8002

CMD ["uvicorn", "agent_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mcp-server-1:
    build: .
    command: uvicorn mcp_server_1:app_mcp1 --host 0.0.0.0 --port 8001
    ports:
      - "8001:8001"
    
  mcp-server-2:
    build: .
    command: uvicorn mcp_server_2:app_mcp2 --host 0.0.0.0 --port 8002
    ports:
      - "8002:8002"
    
  agent-api:
    build: .
    command: uvicorn agent_api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - mcp-server-1
      - mcp-server-2
    
  streamlit:
    build: .
    command: streamlit run streamlit_app.py --server.port 8501
    ports:
      - "8501:8501"
    depends_on:
      - agent-api
```

Run with:
```bash
docker-compose up -d
```

### Cloud Deployment

**AWS:**
- Use ECS/Fargate for containers
- API Gateway + Lambda (serverless)
- Store secrets in AWS Secrets Manager

**Google Cloud:**
- Cloud Run for containers
- Cloud Functions for serverless
- Secret Manager for credentials

**Heroku (Quick deployment):**
```bash
# Install Heroku CLI
# Create Procfile
web: uvicorn agent_api:app --host 0.0.0.0 --port $PORT

# Deploy
heroku create your-app-name
git push heroku main
heroku config:set OPENAI_API_KEY=your_key
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Write tests** (when test suite exists)
5. **Follow code style**
   - Use type hints
   - Follow PEP 8
   - Add docstrings
6. **Commit with clear messages**
   ```bash
   git commit -m "Add: Support for square root operation"
   ```
7. **Push and create Pull Request**

### Code Style

```python
# Good
async def calculate_result(
    operation: str,
    number1: float,
    number2: float
) -> dict:
    """
    Calculate result using appropriate MCP server.
    
    Args:
        operation: Type of mathematical operation
        number1: First operand
        number2: Second operand
        
    Returns:
        Dictionary containing result and metadata
    """
    ...
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- OpenAI and Kimi K2 for AI capabilities
- Streamlit for rapid UI development
- The open-source community

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/multi-agent-calculator/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/multi-agent-calculator/discussions)
- **Email:** your.email@example.com

## 🗺️ Roadmap

- [ ] Add more mathematical operations (sqrt, power, logarithm)
- [ ] Implement agent memory/conversation history
- [ ] Add authentication and authorization
- [ ] Create comprehensive test suite
- [ ] Add monitoring and observability
- [ ] Support for batch calculations
- [ ] WebSocket support for real-time updates
- [ ] Multi-language support
- [ ] Export calculation history
- [ ] GraphQL API option

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [AI Agent Design Patterns](https://www.anthropic.com/research)

---

**Built with ❤️ by Furkhan Suhail**

*Star this repo if you find it helpful!*