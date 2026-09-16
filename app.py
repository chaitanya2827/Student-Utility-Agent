import os
import re
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
# 1. PERCENTAGE TOOL
# ============================================================

@tool
def calculate_percentage(marks: float, total: float) -> str:
    """Calculate percentage from obtained marks and total marks."""

    if total <= 0:
        return "Total marks must be greater than 0."

    percentage = (marks / total) * 100

    return f"Your percentage is {percentage:.2f}%."


# ============================================================
# 2. CGPA TOOL
# ============================================================

@tool
def calculate_cgpa(grades: str) -> str:
    """Calculate average CGPA from comma-separated grade points."""

    try:
        values = [
            float(x.strip())
            for x in grades.split(",")
            if x.strip()
        ]

        if not values:
            return "Please provide your grade points."

        cgpa = sum(values) / len(values)

        return f"Your CGPA is {cgpa:.2f}."

    except ValueError:
        return "Please provide grade points like 8.5, 9, 7.5, 8."


# ============================================================
# 3. ATTENDANCE TOOL
# ============================================================

@tool
def calculate_attendance(
    attended: float,
    total: float
) -> str:
    """Calculate attendance percentage."""

    if total <= 0:
        return "Total classes must be greater than 0."

    percentage = (attended / total) * 100

    return f"Your attendance is {percentage:.2f}%."


# ============================================================
# 4. UNIT CONVERTER TOOL
# ============================================================

@tool
def unit_converter(
    value: float,
    from_unit: str,
    to_unit: str
) -> str:
    """Convert length, weight, volume, temperature, time,
    speed, area and digital storage units."""

    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    aliases = {

        # Length
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",

        "centimeter": "cm",
        "centimeters": "cm",
        "centimetre": "cm",
        "centimetres": "cm",

        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",

        "kilometer": "km",
        "kilometers": "km",
        "kilometre": "km",
        "kilometres": "km",

        "foot": "ft",
        "feet": "ft",

        "inch": "in",
        "inches": "in",

        "yard": "yd",
        "yards": "yd",

        "mile": "mi",
        "miles": "mi",

        # Weight
        "kilogram": "kg",
        "kilograms": "kg",

        "gram": "g",
        "grams": "g",

        "milligram": "mg",
        "milligrams": "mg",

        "pound": "lb",
        "pounds": "lb",

        "ounce": "oz",
        "ounces": "oz",

        # Volume
        "liter": "l",
        "liters": "l",
        "litre": "l",
        "litres": "l",

        "milliliter": "ml",
        "milliliters": "ml",
        "millilitre": "ml",
        "millilitres": "ml",

        "gallon": "gal",
        "gallons": "gal",

        # Temperature
        "celsius": "c",
        "centigrade": "c",

        "fahrenheit": "f",

        "kelvin": "k",

        # Time
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

        # Speed
        "kmph": "km/h",
        "kph": "km/h",
        "km/hr": "km/h",

        "mph": "mph",

        "m/s": "m/s",

        # Area
        "square meter": "m2",
        "square meters": "m2",
        "sq meter": "m2",
        "sq meters": "m2",

        "square kilometer": "km2",
        "square kilometers": "km2",

        "square foot": "ft2",
        "square feet": "ft2",

        "acre": "acre",
        "acres": "acre",

        "hectare": "hectare",
        "hectares": "hectare",

        # Data
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

    from_unit = aliases.get(from_unit, from_unit)
    to_unit = aliases.get(to_unit, to_unit)

    # ========================================================
    # SAME UNIT
    # ========================================================

    if from_unit == to_unit:
        return f"{value:g} {from_unit} = {value:g} {to_unit}"

    # ========================================================
    # TEMPERATURE
    # ========================================================

    if from_unit == "c" and to_unit == "f":
        result = (value * 9 / 5) + 32
        return f"{value:g}°C = {result:.2f}°F"

    if from_unit == "f" and to_unit == "c":
        result = (value - 32) * 5 / 9
        return f"{value:g}°F = {result:.2f}°C"

    if from_unit == "c" and to_unit == "k":
        result = value + 273.15
        return f"{value:g}°C = {result:.2f} K"

    if from_unit == "k" and to_unit == "c":
        result = value - 273.15
        return f"{value:g} K = {result:.2f}°C"

    if from_unit == "f" and to_unit == "k":
        result = ((value - 32) * 5 / 9) + 273.15
        return f"{value:g}°F = {result:.2f} K"

    if from_unit == "k" and to_unit == "f":
        result = ((value - 273.15) * 9 / 5) + 32
        return f"{value:g} K = {result:.2f}°F"

    # ========================================================
    # LENGTH
    # ========================================================

    length_to_meter = {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1,
        "km": 1000,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.344
    }

    if from_unit in length_to_meter and to_unit in length_to_meter:

        meters = value * length_to_meter[from_unit]

        result = meters / length_to_meter[to_unit]

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )

    # ========================================================
    # WEIGHT
    # ========================================================

    weight_to_kg = {
        "mg": 0.000001,
        "g": 0.001,
        "kg": 1,
        "oz": 0.0283495,
        "lb": 0.45359237
    }

    if from_unit in weight_to_kg and to_unit in weight_to_kg:

        kilograms = value * weight_to_kg[from_unit]

        result = kilograms / weight_to_kg[to_unit]

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )

    # ========================================================
    # VOLUME
    # ========================================================

    volume_to_liter = {
        "ml": 0.001,
        "l": 1,
        "gal": 3.785411784
    }

    if from_unit in volume_to_liter and to_unit in volume_to_liter:

        liters = value * volume_to_liter[from_unit]

        result = liters / volume_to_liter[to_unit]

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )

    # ========================================================
    # TIME
    # ========================================================

    time_to_seconds = {
        "s": 1,
        "min": 60,
        "h": 3600,
        "day": 86400
    }

    if from_unit in time_to_seconds and to_unit in time_to_seconds:

        seconds = value * time_to_seconds[from_unit]

        result = seconds / time_to_seconds[to_unit]

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )

    # ========================================================
    # SPEED
    # ========================================================

    speed_to_mps = {
        "m/s": 1,
        "km/h": 1000 / 3600,
        "mph": 1609.344 / 3600
    }

    if from_unit in speed_to_mps and to_unit in speed_to_mps:

        meters_per_second = (
            value * speed_to_mps[from_unit]
        )

        result = (
            meters_per_second /
            speed_to_mps[to_unit]
        )

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )

    # ========================================================
    # AREA
    # ========================================================

    area_to_m2 = {
        "m2": 1,
        "km2": 1000000,
        "ft2": 0.09290304,
        "acre": 4046.8564224,
        "hectare": 10000
    }

    if from_unit in area_to_m2 and to_unit in area_to_m2:

        square_meters = (
            value * area_to_m2[from_unit]
        )

        result = (
            square_meters /
            area_to_m2[to_unit]
        )

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )

    # ========================================================
    # DIGITAL STORAGE
    # ========================================================

    data_to_bytes = {
        "byte": 1,
        "kb": 1024,
        "mb": 1024 ** 2,
        "gb": 1024 ** 3,
        "tb": 1024 ** 4
    }

    if from_unit in data_to_bytes and to_unit in data_to_bytes:

        bytes_value = (
            value * data_to_bytes[from_unit]
        )

        result = (
            bytes_value /
            data_to_bytes[to_unit]
        )

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )

    return (
        f"Sorry, conversion from "
        f"{from_unit} to {to_unit} "
        f"is not supported."
    )


# ============================================================
# 5. TOOLS
# ============================================================

tools = [
    calculate_percentage,
    calculate_cgpa,
    calculate_attendance,
    unit_converter
]


# ============================================================
# 6. GEMINI
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 7. STUDENT AGENT
# ============================================================

student_agent = create_agent(

    model=llm,

    tools=tools,

    system_prompt="""

You are a Student Utility Agent.

You help students with:

- Percentage
- CGPA
- Attendance
- Unit conversions

For calculations, always use the appropriate tool.

For unit conversions, the backend may directly
handle the conversion before reaching you.

Give simple, clear answers.

For unrelated questions, explain that you
specialize in student utility tasks.

"""
)


# ============================================================
# 8. RESPONSE EXTRACTOR
# ============================================================

def extract_text_response(agent_output):

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if not messages:
        return str(agent_output)

    last = messages[-1]

    content = getattr(last, "content", "")

    # Gemini sometimes returns:
    #
    # [
    #   {
    #       "type": "text",
    #       "text": "Your answer"
    #   }
    # ]

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":

                    text = block.get("text", "")

                    if text:
                        text_parts.append(
                            str(text)
                        )

            elif isinstance(block, str):

                text_parts.append(block)

        if text_parts:
            return "".join(text_parts).strip()

    if isinstance(content, str):
        return content.strip()

    return str(content)


# ============================================================
# 9. LANGSERVE CHAIN
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Your message to the agent"
    )


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


formatted_agent_chain = (

    RunnableLambda(format_for_agent)

    | student_agent

    | RunnableLambda(
        extract_text_response
    )

).with_types(

    input_type=AgentInput,

    output_type=str

)


# ============================================================
# 10. FASTAPI
# ============================================================

app = FastAPI(
    title="Student Utility Agent",
    description="AI-powered Student Utility Agent"
)


# ============================================================
# 11. CHAT MODEL
# ============================================================

class ChatRequest(BaseModel):

    message: str


# ============================================================
# 12. DIRECT CONVERSION DETECTOR
# ============================================================

def detect_conversion(text):

    text = text.lower().strip()

    patterns = [

        r"(-?\d+(?:\.\d+)?)\s*(?:degrees?\s*)?([a-zA-Z°/ ]+?)\s+(?:to|into|in)\s+([a-zA-Z°/ ]+)",

        r"(-?\d+(?:\.\d+)?)\s+([a-zA-Z°/ ]+?)\s+(?:is|are)\s+how\s+many\s+([a-zA-Z°/ ]+)",

        r"how\s+many\s+([a-zA-Z°/ ]+)\s+(?:is|are)\s+(-?\d+(?:\.\d+)?)\s+([a-zA-Z°/ ]+)",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            groups = match.groups()

            if len(groups) == 3:

                # Normal:
                # 30 celsius to fahrenheit

                try:

                    value = float(groups[0])

                    from_unit = groups[1].strip()

                    to_unit = groups[2].strip()

                    return (
                        value,
                        from_unit,
                        to_unit
                    )

                except ValueError:
                    pass

    # ========================================================
    # Special pattern:
    #
    # "how many cm is 2 feet"
    # ========================================================

    pattern = (
        r"how\s+many\s+"
        r"([a-zA-Z°/ ]+)\s+"
        r"(?:is|are)\s+"
        r"(-?\d+(?:\.\d+)?)\s+"
        r"([a-zA-Z°/ ]+)"
    )

    match = re.search(
        pattern,
        text
    )

    if match:

        to_unit = match.group(1).strip()

        value = float(
            match.group(2)
        )

        from_unit = match.group(3).strip()

        return (
            value,
            from_unit,
            to_unit
        )

    return None


# ============================================================
# 13. CHAT ENDPOINT
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    user_message = request.message.strip()

    if not user_message:

        return {
            "response": "Please enter a question."
        }

    # ========================================================
    # DIRECT CONVERSION
    # ========================================================

    conversion = detect_conversion(
        user_message
    )

    if conversion:

        value, from_unit, to_unit = conversion

        try:

            result = unit_converter.invoke({

                "value": value,

                "from_unit": from_unit,

                "to_unit": to_unit

            })

            return {
                "response": result
            }

        except Exception as e:

            return {
                "response":
                    f"Conversion error: {str(e)}"
            }

    # ========================================================
    # AI AGENT
    # ========================================================

    try:

        result = student_agent.invoke({

            "messages": [

                (
                    "user",
                    user_message
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
# 14. WEB UI
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

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

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

.examples {

    padding:
        0 18px 15px;

    color: #64748b;

    font-size: 13px;

}

.status {

    text-align: center;

    padding:
        0 15px 18px;

    color: #94a3b8;

    font-size: 12px;

}

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


<div class="header">

    <h1>
        🎓 Student Utility Agent
    </h1>

    <p>
        Your AI assistant for everyday
        student calculations
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

I can help you with:

📊 Percentage
🎯 CGPA
📅 Attendance
📏 Unit Conversion

Examples:

"435 out of 500 percentage"

"42 out of 50 attendance"

"2 feet to centimeters"

"30 Celsius to Fahrenheit"

"5 km to miles"

    </div>

</div>


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


<div class="examples">

    Try:
    "30 Celsius to Fahrenheit"

</div>


<div class="status">

    Powered by Gemini + LangChain

</div>


</div>


<script>


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
    // LOADING
    // ========================================================

    button.disabled = true;

    button.textContent =
        "Thinking...";

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
                "❌ " +
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
            error
        );

    }


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
# 15. LANGSERVE ROUTES
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/agent"

)


# ============================================================
# 16. START SERVER
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
