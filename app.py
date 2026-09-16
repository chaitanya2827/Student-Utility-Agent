import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


# =========================
# 1. STUDENT UTILITY TOOLS
# =========================

@tool
def calculate_percentage(marks: float, total: float) -> str:
    """Calculate percentage from obtained marks and total marks."""
    if total <= 0:
        return "Total marks must be greater than 0."

    percentage = (marks / total) * 100
    return f"Percentage = {percentage:.2f}%"


@tool
def calculate_cgpa(grades: str) -> str:
    """Calculate average CGPA from comma-separated grade points."""
    try:
        values = [float(x.strip()) for x in grades.split(",")]

        if not values:
            return "Please provide grade points."

        cgpa = sum(values) / len(values)
        return f"CGPA = {cgpa:.2f}"

    except ValueError:
        return "Please provide grade points like: 8.5, 9, 7.5, 8"


@tool
def calculate_attendance(attended: float, total: float) -> str:
    """Calculate attendance percentage."""
    if total <= 0:
        return "Total classes must be greater than 0."

    percentage = (attended / total) * 100
    return f"Attendance = {percentage:.2f}%"


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert common units such as km/miles, kg/pounds, meters/feet."""

    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    conversions = {
        ("km", "miles"): value * 0.621371,
        ("miles", "km"): value * 1.60934,
        ("kg", "pounds"): value * 2.20462,
        ("pounds", "kg"): value * 0.453592,
        ("meters", "feet"): value * 3.28084,
        ("feet", "meters"): value * 0.3048,
        ("cm", "inches"): value * 0.393701,
        ("inches", "cm"): value * 2.54,
    }

    key = (from_unit, to_unit)

    if key not in conversions:
        return "Conversion not supported."

    result = conversions[key]

    return f"{value} {from_unit} = {result:.2f} {to_unit}"


tools = [
    calculate_percentage,
    calculate_cgpa,
    calculate_attendance,
    unit_converter
]


# =========================
# 2. GEMINI MODEL
# =========================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# =========================
# 3. CREATE AGENT
# =========================

student_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a Student Utility Agent.

You help students with:
1. Percentage calculations
2. CGPA calculations
3. Attendance calculations
4. Unit conversions

Always use the appropriate tool when a calculation is required.

Give simple, clear answers suitable for students.

For percentage:
Use obtained marks and total marks.

For attendance:
Use attended classes and total classes.

For CGPA:
Calculate the average of the provided grade points unless
the user specifies a different grading method.

For unit conversion:
Identify the units and use the unit conversion tool.

If the question is unrelated to these student utilities,
politely say that you are specialized in student utility tasks.
"""
)


# =========================
# 4. LANGSERVE INPUT
# =========================

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_agent(x):
    user_input = x["input"] if isinstance(x, dict) else x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


def extract_text_response(agent_output):
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages:
        last = messages[-1]

        content = getattr(last, "content", None)

        if content is not None:
            return str(content)

        return str(last)

    return str(agent_output)


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | student_agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# =========================
# 5. FASTAPI APP
# =========================

app = FastAPI(
    title="Student Utility Agent",
    description="AI-powered student utility assistant"
)


# =========================
# 6. WEB UI
# =========================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>
<html>
<head>

    <title>Student Utility Agent</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: Arial, sans-serif;
            background: #f4f7fb;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: 95%;
            max-width: 750px;
            background: white;
            border-radius: 18px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
            overflow: hidden;
        }

        .header {
            background: #2563eb;
            color: white;
            padding: 25px;
            text-align: center;
        }

        .header h1 {
            font-size: 28px;
            margin-bottom: 8px;
        }

        .header p {
            font-size: 14px;
            opacity: 0.9;
        }

        .features {
            display: flex;
            gap: 10px;
            padding: 15px;
            flex-wrap: wrap;
            justify-content: center;
        }

        .feature {
            background: #eef4ff;
            padding: 8px 13px;
            border-radius: 20px;
            font-size: 13px;
            color: #1e40af;
        }

        .chat {
            height: 350px;
            overflow-y: auto;
            padding: 20px;
            border-top: 1px solid #eee;
            border-bottom: 1px solid #eee;
        }

        .message {
            margin-bottom: 15px;
            padding: 12px 15px;
            border-radius: 12px;
            max-width: 80%;
            line-height: 1.5;
        }

        .bot {
            background: #f1f5f9;
            margin-right: auto;
        }

        .user {
            background: #2563eb;
            color: white;
            margin-left: auto;
        }

        .input-area {
            display: flex;
            padding: 15px;
            gap: 10px;
        }

        #message {
            flex: 1;
            padding: 13px;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            outline: none;
            font-size: 15px;
        }

        #send {
            padding: 13px 22px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
        }

        #send:hover {
            background: #1d4ed8;
        }

        #send:disabled {
            background: #9ca3af;
            cursor: not-allowed;
        }

        .examples {
            padding: 0 15px 15px;
            font-size: 13px;
            color: #64748b;
        }

    </style>

</head>

<body>

<div class="container">

    <div class="header">

        <h1>🎓 Student Utility Agent</h1>

        <p>
            Your AI assistant for everyday student calculations
        </p>

    </div>


    <div class="features">

        <span class="feature">📊 Percentage</span>
        <span class="feature">🎯 CGPA</span>
        <span class="feature">📅 Attendance</span>
        <span class="feature">📏 Unit Conversion</span>

    </div>


    <div class="chat" id="chat">

        <div class="message bot">

            👋 Hi! I'm your Student Utility Agent.
            <br><br>
            Ask me something like:
            <br>
            <b>"I scored 435 out of 500. What is my percentage?"</b>

        </div>

    </div>


    <div class="input-area">

        <input
            type="text"
            id="message"
            placeholder="Ask your question..."
            onkeypress="handleKey(event)"
        >

        <button id="send" onclick="sendMessage()">
            Send
        </button>

    </div>


    <div class="examples">

        Try: "My attendance is 42 out of 50. What is my attendance percentage?"

    </div>

</div>


<script>

async function sendMessage() {

    const input = document.getElementById("message");
    const button = document.getElementById("send");
    const chat = document.getElementById("chat");

    const message = input.value.trim();

    if (!message) {
        return;
    }


    // Show user message

    const userMessage = document.createElement("div");

    userMessage.className = "message user";

    userMessage.textContent = message;

    chat.appendChild(userMessage);


    input.value = "";

    button.disabled = true;

    button.textContent = "Thinking...";


    // Loading message

    const loading = document.createElement("div");

    loading.className = "message bot";

    loading.textContent = "🤔 Thinking...";

    chat.appendChild(loading);

    chat.scrollTop = chat.scrollHeight;


    try {

        const response = await fetch("/agent/invoke", {

            method: "POST",

            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },

            body: JSON.stringify({
                input: message
            })

        });


        const data = await response.json();

        loading.remove();


        const botMessage = document.createElement("div");

        botMessage.className = "message bot";


        if (response.ok) {

            botMessage.textContent =
                data.output || "No response received.";

        } else {

            botMessage.textContent =
                "❌ Error: " +
                (data.detail || "Something went wrong.");

        }


        chat.appendChild(botMessage);

    }

    catch (error) {

        loading.remove();

        const errorMessage = document.createElement("div");

        errorMessage.className = "message bot";

        errorMessage.textContent =
            "❌ Could not connect to the agent.";

        chat.appendChild(errorMessage);

    }


    button.disabled = false;

    button.textContent = "Send";

    chat.scrollTop = chat.scrollHeight;

}


function handleKey(event) {

    if (event.key === "Enter") {

        sendMessage();

    }

}

</script>

</body>
</html>
"""


# =========================
# 7. LANGSERVE API ROUTES
# =========================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# =========================
# 8. START SERVER
# =========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
