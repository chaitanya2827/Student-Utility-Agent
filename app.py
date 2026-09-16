import os
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field


# ============================================================
# 1. STUDENT UTILITY TOOLS
# ============================================================

@tool
def calculate_percentage(marks: float, total: float) -> str:
    """Calculate percentage from obtained marks and total marks."""

    if total <= 0:
        return "Total marks must be greater than 0."

    percentage = (marks / total) * 100

    return f"Percentage = {percentage:.2f}%"


@tool
def calculate_cgpa(grades: str) -> str:
    """Calculate CGPA from comma-separated grade points."""

    try:

        values = [
            float(x.strip())
            for x in grades.split(",")
            if x.strip()
        ]

        if not values:
            return "Please provide grade points."

        cgpa = sum(values) / len(values)

        return f"CGPA = {cgpa:.2f}"

    except ValueError:

        return (
            "Please provide grade points like: "
            "8.5, 9, 7.5, 8"
        )


@tool
def calculate_attendance(
    attended: float,
    total: float
) -> str:
    """Calculate attendance percentage."""

    if total <= 0:
        return "Total classes must be greater than 0."

    percentage = (attended / total) * 100

    return f"Attendance = {percentage:.2f}%"


@tool
def unit_converter(
    value: float,
    from_unit: str,
    to_unit: str
) -> str:
    """Convert common units."""

    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    conversions = {

        ("km", "miles"):
            value * 0.621371,

        ("miles", "km"):
            value * 1.60934,

        ("kg", "pounds"):
            value * 2.20462,

        ("pounds", "kg"):
            value * 0.453592,

        ("meters", "feet"):
            value * 3.28084,

        ("feet", "meters"):
            value * 0.3048,

        ("m", "ft"):
            value * 3.28084,

        ("ft", "m"):
            value * 0.3048,

        ("cm", "inches"):
            value * 0.393701,

        ("inches", "cm"):
            value * 2.54,

        ("kg", "g"):
            value * 1000,

        ("g", "kg"):
            value / 1000,

        ("liters", "ml"):
            value * 1000,

        ("ml", "liters"):
            value / 1000
    }

    key = (from_unit, to_unit)

    if key not in conversions:

        return (
            f"Conversion from {from_unit} "
            f"to {to_unit} is not supported."
        )

    result = conversions[key]

    return (
        f"{value} {from_unit} = "
        f"{result:.2f} {to_unit}"
    )


# ============================================================
# 2. TOOLS
# ============================================================

tools = [
    calculate_percentage,
    calculate_cgpa,
    calculate_attendance,
    unit_converter
]


# ============================================================
# 3. GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# ============================================================
# 4. GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(

    model="gemini-3.1-flash-lite-preview",

    google_api_key=GEMINI_API_KEY,

    temperature=0
)


# ============================================================
# 5. STUDENT UTILITY AGENT
# ============================================================

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

Always use the appropriate tool whenever
a calculation is required.

For percentage:

Use obtained marks and total marks.

For attendance:

Use attended classes and total classes.

For CGPA:

Calculate the average of the provided
grade points unless the user specifies
another grading method.

For unit conversion:

Identify the units and use the
unit conversion tool.

Give simple and clear answers suitable
for students.

If the question is unrelated to these
student utility tasks, politely explain
that you are specialized in student
utility calculations.

Do not invent calculation results.

"""
)


# ============================================================
# 6. LANGSERVE INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Your message to the agent"
    )


# ============================================================
# 7. FORMAT INPUT
# ============================================================

def format_for_agent(x):

    user_input = (
        x["input"]
        if isinstance(x, dict)
        else x.input
    )

    return {

        "messages": [
            ("user", user_input)
        ]

    }


# ============================================================
# 8. EXTRACT CLEAN TEXT RESPONSE
# ============================================================

def extract_text_response(agent_output):

    if not isinstance(agent_output, dict):

        return str(agent_output)


    messages = agent_output.get("messages")


    if not messages:

        return str(agent_output)


    last = messages[-1]


    content = getattr(
        last,
        "content",
        ""
    )


    # --------------------------------------------------------
    # Gemini can return content as a list
    # --------------------------------------------------------

    if isinstance(content, list):

        text_parts = []


        for block in content:

            # Dictionary content block

            if isinstance(block, dict):

                if block.get("type") == "text":

                    text = block.get(
                        "text",
                        ""
                    )

                    if text:

                        text_parts.append(
                            str(text)
                        )


            # String content block

            elif isinstance(block, str):

                text_parts.append(
                    block
                )


        if text_parts:

            return "".join(
                text_parts
            ).strip()


    # --------------------------------------------------------
    # Normal string response
    # --------------------------------------------------------

    if isinstance(content, str):

        return content.strip()


    return str(content)


# ============================================================
# 9. LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (

    RunnableLambda(
        format_for_agent
    )

    | student_agent

    | RunnableLambda(
        extract_text_response
    )

).with_types(

    input_type=AgentInput,

    output_type=str

)


# ============================================================
# 10. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Student Utility Agent",

    description=(
        "AI-powered Student Utility Agent "
        "for percentage, CGPA, attendance "
        "and unit conversion."
    )

)


# ============================================================
# 11. CHAT REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    message: str


# ============================================================
# 12. CHAT API
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    try:

        result = student_agent.invoke({

            "messages": [

                (
                    "user",
                    request.message
                )

            ]

        })


        answer = extract_text_response(
            result
        )


        return {

            "response": answer

        }


    except Exception as e:

        return {

            "response":
                f"Error: {str(e)}"

        }


# ============================================================
# 13. WEB UI
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Student Utility Agent
</title>


<style>

/* ==========================================================
   RESET
   ========================================================== */

* {

    box-sizing: border-box;

    margin: 0;

    padding: 0;

}


/* ==========================================================
   BODY
   ========================================================== */

body {

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    min-height: 100vh;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #f8fafc
        );

    display: flex;

    justify-content: center;

    align-items: center;

    padding: 20px;

}


/* ==========================================================
   MAIN CONTAINER
   ========================================================== */

.container {

    width: 100%;

    max-width: 850px;

    background: white;

    border-radius: 22px;

    overflow: hidden;

    box-shadow:
        0 12px 45px
        rgba(0, 0, 0, 0.12);

}


/* ==========================================================
   HEADER
   ========================================================== */

.header {

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );

    color: white;

    padding: 32px 25px;

    text-align: center;

}


.header h1 {

    font-size: 30px;

    margin-bottom: 8px;

}


.header p {

    font-size: 15px;

    opacity: 0.92;

}


/* ==========================================================
   FEATURES
   ========================================================== */

.features {

    display: flex;

    justify-content: center;

    align-items: center;

    flex-wrap: wrap;

    gap: 10px;

    padding: 18px;

    border-bottom:
        1px solid #e5e7eb;

}


.feature {

    padding:
        8px 14px;

    background: #eef2ff;

    color: #3730a3;

    border-radius: 20px;

    font-size: 13px;

    font-weight: 600;

}


/* ==========================================================
   CHAT
   ========================================================== */

.chat {

    height: 420px;

    overflow-y: auto;

    padding: 22px;

    background: white;

}


.message {

    max-width: 80%;

    padding:
        14px 17px;

    margin-bottom: 16px;

    border-radius: 15px;

    line-height: 1.55;

    font-size: 15px;

    white-space: pre-wrap;

    word-wrap: break-word;

}


.bot {

    background: #f1f5f9;

    color: #111827;

    margin-right: auto;

}


.user {

    background: #2563eb;

    color: white;

    margin-left: auto;

}


/* ==========================================================
   INPUT
   ========================================================== */

.input-area {

    display: flex;

    gap: 10px;

    padding: 16px;

    border-top:
        1px solid #e5e7eb;

}


#message {

    flex: 1;

    padding:
        14px 16px;

    border:
        1px solid #d1d5db;

    border-radius: 12px;

    outline: none;

    font-size: 15px;

}


#message:focus {

    border-color: #2563eb;

    box-shadow:
        0 0 0 3px
        rgba(37, 99, 235, 0.1);

}


#send {

    padding:
        14px 25px;

    background: #2563eb;

    color: white;

    border: none;

    border-radius: 12px;

    cursor: pointer;

    font-size: 15px;

    font-weight: bold;

}


#send:hover {

    background: #1d4ed8;

}


#send:disabled {

    background: #94a3b8;

    cursor: not-allowed;

}


/* ==========================================================
   EXAMPLES
   ========================================================== */

.examples {

    padding:
        0 18px 15px;

    color: #64748b;

    font-size: 13px;

}


/* ==========================================================
   FOOTER
   ========================================================== */

.status {

    text-align: center;

    padding:
        0 15px 18px;

    color: #94a3b8;

    font-size: 12px;

}


/* ==========================================================
   MOBILE
   ========================================================== */

@media (max-width: 600px) {

    body {

        padding: 10px;

    }


    .header {

        padding: 25px 15px;

    }


    .header h1 {

        font-size: 24px;

    }


    .chat {

        height: 380px;

        padding: 15px;

    }


    .message {

        max-width: 90%;

    }


    .input-area {

        padding: 12px;

    }


    #send {

        padding:
            14px 18px;

    }

}

</style>

</head>


<body>


<div class="container">


<!-- ========================================================
     HEADER
     ======================================================== -->

<div class="header">

    <h1>
        🎓 Student Utility Agent
    </h1>

    <p>
        Your AI assistant for everyday
        student calculations
    </p>

</div>


<!-- ========================================================
     FEATURES
     ======================================================== -->

<div class="features">

    <span class="feature">
        📊 Percentage
    </span>

    <span class="feature">
        🎯 CGPA
    </span>

    <span class="feature">
        📅 Attendance
    </span>

    <span class="feature">
        📏 Unit Conversion
    </span>

</div>


<!-- ========================================================
     CHAT
     ======================================================== -->

<div
    class="chat"
    id="chat"
>

    <div
        class="message bot"
    >

👋 Hi! I'm your Student Utility Agent.

I can help you with:

• Percentage
• CGPA
• Attendance
• Unit conversion

Try asking:

"I scored 435 out of 500.
What is my percentage?"

    </div>

</div>


<!-- ========================================================
     INPUT
     ======================================================== -->

<div class="input-area">

    <input

        type="text"

        id="message"

        placeholder="Ask your question..."

        autocomplete="off"

    >


    <button
        id="send"
        onclick="sendMessage()"
    >

        Send

    </button>

</div>


<!-- ========================================================
     EXAMPLE
     ======================================================== -->

<div class="examples">

    Try:

    "My attendance is 42 out of 50.
    What is my attendance percentage?"

</div>


<!-- ========================================================
     FOOTER
     ======================================================== -->

<div class="status">

    Powered by Gemini + LangChain

</div>


</div>


<script>


// ============================================================
// SEND MESSAGE
// ============================================================

async function sendMessage() {


    const input =
        document.getElementById(
            "message"
        );


    const button =
        document.getElementById(
            "send"
        );


    const chat =
        document.getElementById(
            "chat"
        );


    const message =
        input.value.trim();


    if (!message) {

        return;

    }


    // ========================================================
    // USER MESSAGE
    // ========================================================

    const userMessage =
        document.createElement(
            "div"
        );


    userMessage.className =
        "message user";


    userMessage.textContent =
        message;


    chat.appendChild(
        userMessage
    );


    input.value = "";


    // ========================================================
    // DISABLE SEND BUTTON
    // ========================================================

    button.disabled = true;

    button.textContent =
        "Thinking...";


    // ========================================================
    // LOADING
    // ========================================================

    const loading =
        document.createElement(
            "div"
        );


    loading.className =
        "message bot";


    loading.textContent =
        "🤔 Thinking...";


    chat.appendChild(
        loading
    );


    chat.scrollTop =
        chat.scrollHeight;


    // ========================================================
    // CALL BACKEND
    // ========================================================

    try {


        const response =
            await fetch(
                "/chat",
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body: JSON.stringify({

                        message:
                            message

                    })

                }
            );


        const data =
            await response.json();


        loading.remove();


        // ====================================================
        // BOT RESPONSE
        // ====================================================

        const botMessage =
            document.createElement(
                "div"
            );


        botMessage.className =
            "message bot";


        if (response.ok) {

            botMessage.textContent =
                data.response ||
                "No response received.";

        }

        else {

            botMessage.textContent =
                "❌ Error: " +
                (
                    data.detail ||
                    data.response ||
                    "Something went wrong."
                );

        }


        chat.appendChild(
            botMessage
        );


    }


    catch (error) {


        loading.remove();


        const errorMessage =
            document.createElement(
                "div"
            );


        errorMessage.className =
            "message bot";


        errorMessage.textContent =
            "❌ Could not connect to the agent.";


        chat.appendChild(
            errorMessage
        );


        console.error(
            "Connection error:",
            error
        );

    }


    // ========================================================
    // ENABLE SEND BUTTON
    // ========================================================

    button.disabled = false;

    button.textContent =
        "Send";


    chat.scrollTop =
        chat.scrollHeight;

}


// ============================================================
// ENTER KEY
// ============================================================

document
    .getElementById("message")
    .addEventListener(
        "keypress",
        function(event) {

            if (
                event.key === "Enter"
            ) {

                sendMessage();

            }

        }
    );


</script>


</body>

</html>

"""


# ============================================================
# 14. LANGSERVE ROUTES
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# ============================================================
# 15. START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
