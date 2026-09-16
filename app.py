import os
import re
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from langserve import add_routes
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field


# ============================================================
# API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# ============================================================
# TOOLS
# ============================================================

@tool
def calculate_percentage(marks: float, total: float) -> str:
    """Calculate percentage from obtained marks and total marks."""

    if total <= 0:
        return "Total marks must be greater than zero."

    result = (marks / total) * 100

    return f"Your percentage is {result:.2f}%."


@tool
def calculate_cgpa(grades: str) -> str:
    """Calculate average CGPA from grade points."""

    try:
        values = [
            float(x.strip())
            for x in grades.split(",")
            if x.strip()
        ]

        if not values:
            return "Please provide grade points."

        result = sum(values) / len(values)

        return f"Your CGPA is {result:.2f}."

    except Exception:
        return "Enter grade points like 8.5, 9, 7.5, 8."


@tool
def calculate_attendance(attended: float, total: float) -> str:
    """Calculate attendance percentage."""

    if total <= 0:
        return "Total classes must be greater than zero."

    result = (attended / total) * 100

    return f"Your attendance is {result:.2f}%."


@tool
def unit_converter(
    value: float,
    from_unit: str,
    to_unit: str
) -> str:
    """Convert common units."""

    f = from_unit.lower().strip()
    t = to_unit.lower().strip()

    aliases = {

        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",

        "centimeter": "cm",
        "centimeters": "cm",
        "centimetre": "cm",
        "centimetres": "cm",

        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",

        "kilometer": "km",
        "kilometers": "km",
        "kilometre": "km",
        "kilometres": "km",

        "inch": "in",
        "inches": "in",

        "foot": "ft",
        "feet": "ft",

        "yard": "yd",
        "yards": "yd",

        "mile": "mi",
        "miles": "mi",

        "milligram": "mg",
        "milligrams": "mg",

        "gram": "g",
        "grams": "g",

        "kilogram": "kg",
        "kilograms": "kg",

        "ounce": "oz",
        "ounces": "oz",

        "pound": "lb",
        "pounds": "lb",

        "milliliter": "ml",
        "milliliters": "ml",
        "millilitre": "ml",
        "millilitres": "ml",

        "liter": "l",
        "liters": "l",
        "litre": "l",
        "litres": "l",

        "gallon": "gal",
        "gallons": "gal",

        "celsius": "c",
        "centigrade": "c",
        "degree celsius": "c",
        "degrees celsius": "c",

        "fahrenheit": "f",
        "degree fahrenheit": "f",
        "degrees fahrenheit": "f",

        "kelvin": "k",

        "second": "s",
        "seconds": "s",
        "sec": "s",

        "minute": "min",
        "minutes": "min",
        "mins": "min",

        "hour": "h",
        "hours": "h",
        "hr": "h",
        "hrs": "h",

        "day": "day",
        "days": "day",

        "byte": "byte",
        "bytes": "byte",

        "kilobyte": "kb",
        "kilobytes": "kb",

        "megabyte": "mb",
        "megabytes": "mb",

        "gigabyte": "gb",
        "gigabytes": "gb",

        "terabyte": "tb",
        "terabytes": "tb",
    }

    f = aliases.get(f, f)
    t = aliases.get(t, t)

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    if f == "c" and t == "f":
        result = (value * 9 / 5) + 32
        return f"{value:g}°C = {result:.2f}°F"

    if f == "f" and t == "c":
        result = (value - 32) * 5 / 9
        return f"{value:g}°F = {result:.2f}°C"

    if f == "c" and t == "k":
        result = value + 273.15
        return f"{value:g}°C = {result:.2f} K"

    if f == "k" and t == "c":
        result = value - 273.15
        return f"{value:g} K = {result:.2f}°C"

    if f == "f" and t == "k":
        result = ((value - 32) * 5 / 9) + 273.15
        return f"{value:g}°F = {result:.2f} K"

    if f == "k" and t == "f":
        result = ((value - 273.15) * 9 / 5) + 32
        return f"{value:g} K = {result:.2f}°F"

    # --------------------------------------------------------
    # LENGTH
    # --------------------------------------------------------

    length = {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1,
        "km": 1000,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.344
    }

    if f in length and t in length:
        meters = value * length[f]
        result = meters / length[t]

        return f"{value:g} {f} = {result:.4f} {t}"

    # --------------------------------------------------------
    # WEIGHT
    # --------------------------------------------------------

    weight = {
        "mg": 0.000001,
        "g": 0.001,
        "kg": 1,
        "oz": 0.0283495,
        "lb": 0.45359237
    }

    if f in weight and t in weight:
        kg = value * weight[f]
        result = kg / weight[t]

        return f"{value:g} {f} = {result:.4f} {t}"

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume = {
        "ml": 0.001,
        "l": 1,
        "gal": 3.785411784
    }

    if f in volume and t in volume:
        liters = value * volume[f]
        result = liters / volume[t]

        return f"{value:g} {f} = {result:.4f} {t}"

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    time_units = {
        "s": 1,
        "min": 60,
        "h": 3600,
        "day": 86400
    }

    if f in time_units and t in time_units:
        seconds = value * time_units[f]
        result = seconds / time_units[t]

        return f"{value:g} {f} = {result:.4f} {t}"

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    data = {
        "byte": 1,
        "kb": 1024,
        "mb": 1024 ** 2,
        "gb": 1024 ** 3,
        "tb": 1024 ** 4
    }

    if f in data and t in data:
        bytes_value = value * data[f]
        result = bytes_value / data[t]

        return f"{value:g} {f} = {result:.4f} {t}"

    return (
        f"Sorry, conversion from {from_unit} "
        f"to {to_unit} is not supported."
    )


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# AGENT
# ============================================================

student_agent = create_agent(
    model=llm,
    tools=[
        calculate_percentage,
        calculate_cgpa,
        calculate_attendance,
        unit_converter
    ],
    system_prompt="""
You are a Student Utility Agent.

You help students with:

- Percentage
- CGPA
- Attendance
- Unit conversion

Always use the appropriate tool when calculation is required.

Give short, clear and accurate answers.

Do not invent calculation results.
"""
)


# ============================================================
# AGENT RESPONSE
# ============================================================

def extract_text_response(result):

    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])

    if not messages:
        return str(result)

    content = getattr(
        messages[-1],
        "content",
        ""
    )

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text = item.get("text", "")

                    if text:
                        parts.append(text)

            elif isinstance(item, str):

                parts.append(item)

        return "".join(parts).strip()

    return str(content)


# ============================================================
# LANGSERVE
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Message for the student utility agent"
    )


def format_for_agent(x):

    message = (
        x["input"]
        if isinstance(x, dict)
        else x.input
    )

    return {
        "messages": [
            ("user", message)
        ]
    }


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | student_agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Student Utility Agent",
    description="Student Utility Agent powered by Gemini + LangChain"
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# CHAT MODEL
# ============================================================

class ChatRequest(BaseModel):

    message: str


# ============================================================
# DETECT SIMPLE CALCULATIONS
# ============================================================

def process_simple_question(message):

    text = message.lower().strip()


    # --------------------------------------------------------
    # PERCENTAGE
    # --------------------------------------------------------

    match = re.search(
        r"(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)"
        r".*(?:percentage|percent|%)",
        text
    )

    if match:

        marks = float(match.group(1))
        total = float(match.group(2))

        return calculate_percentage.invoke({
            "marks": marks,
            "total": total
        })


    # --------------------------------------------------------
    # ATTENDANCE
    # --------------------------------------------------------

    match = re.search(
        r"(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)"
        r".*attendance",
        text
    )

    if match:

        attended = float(match.group(1))
        total = float(match.group(2))

        return calculate_attendance.invoke({
            "attended": attended,
            "total": total
        })


    # --------------------------------------------------------
    # CONVERSION
    # --------------------------------------------------------

    conversion = detect_conversion(text)

    if conversion:

        value, from_unit, to_unit = conversion

        return unit_converter.invoke({
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit
        })


    return None


# ============================================================
# CONVERSION DETECTOR
# ============================================================

def detect_conversion(text):

    text = text.lower().strip()

    text = text.replace("?", "")

    # 30 celsius to fahrenheit
    # 2 feet to centimeters
    # 5 km to miles

    pattern1 = re.search(
        r"(-?\d+(?:\.\d+)?)\s*"
        r"([a-zA-Z°²/ ]+?)"
        r"\s+(?:to|into|in)\s+"
        r"([a-zA-Z°²/ ]+?)$",
        text
    )

    if pattern1:

        return (
            float(pattern1.group(1)),
            pattern1.group(2).strip(),
            pattern1.group(3).strip()
        )


    # 2 feet is how many cm

    pattern2 = re.search(
        r"(-?\d+(?:\.\d+)?)\s*"
        r"([a-zA-Z°²/ ]+?)"
        r"\s+(?:is|are)\s+how\s+many\s+"
        r"([a-zA-Z°²/ ]+?)$",
        text
    )

    if pattern2:

        return (
            float(pattern2.group(1)),
            pattern2.group(2).strip(),
            pattern2.group(3).strip()
        )


    # how many cm is 2 feet

    pattern3 = re.search(
        r"how\s+many\s+"
        r"([a-zA-Z°²/ ]+?)\s+"
        r"(?:is|are)\s+"
        r"(-?\d+(?:\.\d+)?)\s*"
        r"([a-zA-Z°²/ ]+?)$",
        text
    )

    if pattern3:

        return (
            float(pattern3.group(2)),
            pattern3.group(3).strip(),
            pattern3.group(1).strip()
        )


    return None


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:

        return JSONResponse(
            status_code=400,
            content={
                "response": "Please enter a question."
            }
        )


    # First handle calculations/conversions directly

    simple_answer = process_simple_question(
        message
    )

    if simple_answer:

        return {
            "response": simple_answer
        }


    # Otherwise use Gemini agent

    try:

        result = student_agent.invoke({

            "messages": [
                (
                    "user",
                    message
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

        return JSONResponse(
            status_code=500,
            content={
                "response":
                    "Agent error: " + str(e)
            }
        )


# ============================================================
# WEB PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
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

<title>Student Utility Agent</title>


<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family: Arial, sans-serif;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #f8fafc
        );

    min-height: 100vh;

    display: flex;

    justify-content: center;

    align-items: center;

    padding: 20px;
}

.container {

    width: 100%;

    max-width: 850px;

    background: white;

    border-radius: 22px;

    overflow: hidden;

    box-shadow:
        0 15px 45px
        rgba(0,0,0,0.12);
}

.header {

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );

    color: white;

    text-align: center;

    padding: 28px;
}

.header h1 {

    margin: 0 0 8px;

}

.header p {

    margin: 0;

}

.features {

    display: flex;

    justify-content: center;

    flex-wrap: wrap;

    gap: 10px;

    padding: 15px;

}

.feature {

    background: #eef2ff;

    color: #3730a3;

    padding: 8px 14px;

    border-radius: 20px;

    font-size: 13px;

    font-weight: bold;

}

.chat {

    height: 430px;

    overflow-y: auto;

    padding: 20px;

    border-top:
        1px solid #eee;

    border-bottom:
        1px solid #eee;
}

.message {

    max-width: 80%;

    padding: 14px 17px;

    margin-bottom: 15px;

    border-radius: 15px;

    line-height: 1.5;

    white-space: pre-wrap;

    word-break: break-word;
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

.chat-form {

    display: flex;

    gap: 10px;

    padding: 16px;
}

#message {

    flex: 1;

    padding: 14px;

    border:
        1px solid #cbd5e1;

    border-radius: 12px;

    font-size: 15px;

    outline: none;
}

#message:focus {

    border-color: #2563eb;
}

#send {

    width: 100px;

    border: none;

    border-radius: 12px;

    background: #2563eb;

    color: white;

    font-size: 15px;

    font-weight: bold;

    cursor: pointer;
}

#send:hover {

    background: #1d4ed8;
}

#send:disabled {

    background: #94a3b8;

    cursor: wait;
}

.status {

    text-align: center;

    color: #64748b;

    font-size: 12px;

    padding-bottom: 18px;
}

@media(max-width:600px) {

    body {
        padding: 10px;
    }

    .message {
        max-width: 90%;
    }

    #send {
        width: 80px;
    }

}

</style>

</head>


<body>


<div class="container">


<div class="header">

    <h1>🎓 Student Utility Agent</h1>

    <p>
        Your AI assistant for student calculations
    </p>

</div>


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


<div
    class="chat"
    id="chat"
>

    <div class="message bot">

👋 Hi! I'm your Student Utility Agent.

I can help with:

📊 Percentage
🎯 CGPA
📅 Attendance
📏 Unit Conversion

Try:

435 out of 500 percentage

42 out of 50 attendance

2 feet to centimeters

30 Celsius to Fahrenheit

5 km to miles

2 hours to minutes

1 GB to MB

    </div>

</div>


<!-- IMPORTANT: NORMAL FORM -->

<form
    id="chatForm"
    class="chat-form"
>

    <input
        id="message"
        name="message"
        type="text"
        placeholder="Ask your question..."
        autocomplete="off"
    >

    <button
        id="send"
        type="submit"
    >
        Send
    </button>

</form>


<div class="status">

    Powered by Gemini + LangChain

</div>


</div>


<script>

(function() {

    const form =
        document.getElementById("chatForm");

    const input =
        document.getElementById("message");

    const button =
        document.getElementById("send");

    const chat =
        document.getElementById("chat");


    function addMessage(
        text,
        type
    ) {

        const div =
            document.createElement("div");

        div.className =
            "message " + type;

        div.textContent =
            text;

        chat.appendChild(div);

        chat.scrollTop =
            chat.scrollHeight;

        return div;
    }


    form.addEventListener(
        "submit",
        async function(event) {

            event.preventDefault();


            const message =
                input.value.trim();


            if (!message) {

                input.focus();

                return;

            }


            // Show user message

            addMessage(
                message,
                "user"
            );


            // Clear input

            input.value = "";


            // Disable button

            button.disabled = true;

            button.textContent =
                "Wait...";


            // Loading

            const loading =
                addMessage(
                    "🤔 Thinking...",
                    "bot"
                );


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

                            body:
                                JSON.stringify({
                                    message:
                                        message
                                })

                        }
                    );


                const data =
                    await response.json();


                loading.remove();


                if (
                    data &&
                    data.response
                ) {

                    addMessage(
                        data.response,
                        "bot"
                    );

                }

                else {

                    addMessage(
                        "No response received.",
                        "bot"
                    );

                }


            }

            catch(error) {

                loading.remove();


                addMessage(
                    "❌ Connection error. Please refresh the page and try again.",
                    "bot"
                );


                console.error(
                    error
                );

            }


            button.disabled = false;

            button.textContent =
                "Send";

            input.focus();

        }
    );


})();

</script>


</body>

</html>
"""


# ============================================================
# LANGSERVE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# ============================================================
# SERVER
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
