# ==============================================================================
# PART 4: STREAMLIT FRONTEND
# File: streamlit_app.py
# ==============================================================================


import streamlit as st
import requests
import json

# Streamlit App Configuration
st.set_page_config(page_title="AI Calculator Agent", page_icon="🤖", layout="wide")

st.title("🤖 AI-Powered Calculator Agent")
st.markdown("### Learn AI Agents with MCP Servers")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    agent_url = st.text_input("Agent API URL", "http://localhost:8000")

    if st.button("Check Health"):
        try:
            response = requests.get(f"{agent_url}/health")
            st.json(response.json())
        except Exception as e:
            st.error(f"Error: {e}")

# Main calculator interface
col1, col2, col3 = st.columns(3)

with col1:
    operation = st.selectbox(
        "Operation",
        ["addition", "subtraction", "multiplication", "division"],
        help="Select the mathematical operation"
    )

with col2:
    number1 = st.number_input("Number 1", value=10.0, format="%.2f")

with col3:
    number2 = st.number_input("Number 2", value=5.0, format="%.2f")

# Calculate button
if st.button("🧮 Calculate", type="primary"):
    with st.spinner("Agent is processing your request..."):
        try:
            # Prepare request
            payload = {
                "query": operation,
                "number1": number1,
                "number2": number2
            }

            print(payload)

            # Call agent API
            response = requests.post(f"{agent_url}/query", json=payload)
            response.raise_for_status()
            result = response.json()

            # Display results
            st.success("✅ Calculation Complete!")

            # Show result prominently
            st.metric(label="Result", value=f"{result['result']:.2f}")

            # Show LLM explanation
            st.info(f"**AI Explanation:** {result['llm_explanation']}")

            # Show technical details in expander
            with st.expander("🔍 Technical Details"):
                st.write("**Operation:**", result['operation'])
                st.write("**Timestamp:**", result['timestamp'])
                st.json(result['raw_tool_output'])

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error calling agent API: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# Cache management
st.divider()
col1, col2 = st.columns(2)

with col1:
    if st.button("🗑️ Clear Cache"):
        try:
            response = requests.delete(f"{agent_url}/cache")
            st.success(response.json()["message"])
        except Exception as e:
            st.error(f"Error: {e}")

# Show how the system works
with st.expander("📚 How This AI Agent System Works"):
    st.markdown('''
    **Architecture Flow:**

    1. **Streamlit Frontend** → User enters operation and numbers
    2. **Agent API (FastAPI)** → Receives request via POST /query
    3. **Agent Decision** → Routes to appropriate MCP server
    4. **MCP Server 1 or 2** → Executes the tool (add/subtract/multiply/divide)
    5. **Tool Result** → Returns to Agent API
    6. **LLM Processing** → Generates natural language explanation
    7. **Response** → Sent back to Streamlit for display

    **MCP Servers:**
    - Server 1 (port 8001): Addition & Subtraction tools
    - Server 2 (port 8002): Multiplication & Division tools

    **HTTP Methods Used:**
    - GET /health - Check system status
    - POST /query - Submit calculations
    - PUT /settings - Update all config
    - PATCH /settings - Update one setting
    - DELETE /cache - Clear cache
    ''')
