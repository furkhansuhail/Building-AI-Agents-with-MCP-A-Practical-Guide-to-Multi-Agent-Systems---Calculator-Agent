# ==============================================================================
# MULTI-AGENT CALCULATOR - ENVIRONMENT VARIABLES
# ==============================================================================
# Copy this file to .env and fill in your actual API keys
# Never commit the .env file to version control!

# ------------------------------------------------------------------------------
# AGENT 1: KIMI K2 (Handles Addition & Subtraction)
# ------------------------------------------------------------------------------
# Get your Kimi K2 API credentials from your provider
Kimi_K2_HF_Token=your_kimi_k2_api_token_here
Kimi_K2_HF_Base=https://your-kimi-k2-api-endpoint.com/v1
Kimi_K2_HF_Model=moonshotai/Kimi-K2-Instruct:fireworks-ai

# ------------------------------------------------------------------------------
# AGENT 2: OPENAI GPT (Handles Multiplication & Division)
# ------------------------------------------------------------------------------
# Get your OpenAI API key from https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# ------------------------------------------------------------------------------
# MCP SERVER CONFIGURATION (Optional - defaults to localhost)
# ------------------------------------------------------------------------------
MCP_SERVER_1_URL=http://localhost:8001
MCP_SERVER_2_URL=http://localhost:8002

# ------------------------------------------------------------------------------
# AGENT API CONFIGURATION (Optional)
# ------------------------------------------------------------------------------
AGENT_API_URL=http://localhost:8000
TIMEOUT=30

# ------------------------------------------------------------------------------
# NOTES
# ------------------------------------------------------------------------------
# 1. Make sure to replace all "your_*_here" placeholders with actual values
# 2. Keep this file secure - never share or commit it
# 3. For production, use environment-specific .env files (.env.production, .env.staging)
# 4. If using Docker, you can pass these as environment variables or use docker-compose
