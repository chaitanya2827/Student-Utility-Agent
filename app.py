import os
import re
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

# IMPORTANT: THIS IS THE CORRECT IMPORT
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.agents import create_agent

from pydantic import BaseModel, Field


# ============================================================
# GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# ============================================================
# PERCENTAGE TOOL
# ============================================================

@tool
def calculate_percentage(marks: float, total: float) -> str:
    """Calculate percentage."""

    if total <= 0:
        return "Total marks must be greater than zero."

    percentage = (marks / total) * 100

    return f"Your percentage is {percentage:.2f}%."


# ============================================================
# CGPA TOOL
# ============================================================

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

        cgpa = sum(values) / len(values)

        return f"Your CGPA is {cgpa:.2f}."

    except Exception:

        return (
            "Please enter grade points like "
            "8.5, 9, 7.5, 8."
        )


# ============================================================
# ATTENDANCE TOOL
# ============================================================

@tool
def calculate_attendance(
    attended: float,
    total: float
) -> str:
    """Calculate attendance percentage."""

    if total <= 0:
        return "Total classes must be greater than zero."

    percentage = (attended / total) * 100

    return f"Your attendance is {percentage:.2f}%."


# ============================================================
# UNIT CONVERTER TOOL
# ============================================================

@tool
def unit_converter(
    value: float,
    from_unit: str,
    to_unit: str
) -> str:
    """Convert common units."""

    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    aliases = {

        # LENGTH
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

        # WEIGHT
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

        # VOLUME
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

        # TEMPERATURE
        "celsius": "c",
        "centigrade": "c",
        "degree celsius": "c",
        "degrees celsius": "c",
        "°c": "c",

        "fahrenheit": "f",
        "degree fahrenheit": "f",
        "degrees fahrenheit": "f",
        "°f": "f",

        "kelvin": "k",
        "kelvins": "k",

        # TIME
        "second": "s",
        "seconds": "s",
        "sec": "s",
        "secs": "s",

        "minute": "min",
        "minutes": "min",
        "mins": "min",

        "hour": "h",
        "hours": "h",
        "hr": "h",
        "hrs": "h",

        "day": "day",
        "days": "day",

        # SPEED
        "meter per second": "m/s",
        "meters per second": "m/s",

        "kilometer per hour": "km/h",
        "kilometers per hour": "km/h",
        "km per hour": "km/h",
        "kmph": "km/h",
        "kph": "km/h",

        "mile per hour": "mph",
        "miles per hour": "mph",

        # AREA
        "square meter": "m2",
        "square meters": "m2",
        "square metre": "m2",
        "square metres": "m2",

        "square kilometer": "km2",
        "square kilometers": "km2",

        "square foot": "ft2",
        "square feet": "ft2",

        "acre": "acre",
        "acres": "acre",

        "hectare": "hectare",
        "hectares": "hectare",

        # DATA
        "byte": "byte",
        "bytes": "byte",

        "kilobyte": "kb",
        "kilobytes": "kb",

        "megabyte": "mb",
        "megabytes": "mb",

        "gigabyte": "gb",
        "gigabytes": "gb",

        "terabyte": "tb",
        "terabytes": "tb"
    }

    from_unit = aliases.get(
        from_unit,
        from_unit
    )

    to_unit = aliases.get(
        to_unit,
        to_unit
    )


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

    if from_unit in length and to_unit in length:

        meters = value * length[from_unit]

        result = meters / length[to_unit]

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )


    # ========================================================
    # WEIGHT
    # ========================================================

    weight = {

        "mg": 0.000001,
        "g": 0.001,
        "kg": 1,
        "oz": 0.0283495,
        "lb": 0.45359237

    }

    if from_unit in weight and to_unit in weight:

        kilograms = value * weight[from_unit]

        result = kilograms / weight[to_unit]

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )


    # ========================================================
    # VOLUME
    # ========================================================

    volume = {

        "ml": 0.001,
        "l": 1,
        "gal": 3.785411784

    }

    if from_unit in volume and to_unit in volume:

        liters = value * volume[from_unit]

        result = liters / volume[to_unit]

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )


    # ========================================================
    # TIME
    # ========================================================

    time_units = {

        "s": 1,
        "min": 60,
        "h": 3600,
        "day": 86400

    }

    if from_unit in time_units and to_unit in time_units:

        seconds = value * time_units[from_unit]

        result = seconds / time_units[to_unit]

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )


    # ========================================================
    # SPEED
    # ========================================================

    speed = {

        "m/s": 1,
        "km/h": 1000 / 3600,
        "mph": 1609.344 / 3600

    }

    if from_unit in speed and to_unit in speed:

        mps = value * speed[from_unit]

        result = mps / speed[to_unit]

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )


    # ========================================================
    # AREA
    # ========================================================

    area = {

        "m2": 1,
        "km2": 1000000,
        "ft2": 0.09290304,
        "acre": 4046.8564224,
        "hectare": 10000

    }

    if from_unit in area and to_unit in area:

        square_meters = value * area[from_unit]

        result = square_meters / area[to_unit]

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )


    # ========================================================
    # DIGITAL STORAGE
    # ========================================================

    data = {

        "byte": 1,
        "kb": 1024,
        "mb": 1024 ** 2,
        "gb": 1024 ** 3,
        "tb": 1024 ** 4

    }

    if from_unit in data and to_unit in data:

        bytes_value = value * data[from_unit]

        result = bytes_value / data[to_unit]

        return (
            f"{value:g} {from_unit} = "
            f"{result:.4f} {to_unit}"
        )


    return (
        f"Sorry, conversion from "
        f"{from_unit} to {to_unit} "
        f"is not supported yet."
    )


# ============================================================
# TOOLS
# ============================================================

tools = [

    calculate_percentage,
    calculate_cgpa,
    calculate_attendance,
    unit_converter

]


# ============================================================
# GEMINI MODEL
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

    tools=tools,

    system_prompt="""

You are a Student Utility Agent.

You help students with:

1. Percentage
2. CGPA
3. Attendance
4. Unit conversions

Use the correct tool for calculations.

Give simple and clear answers.

For percentage:
percentage = obtained marks / total marks × 100

For attendance:
attendance = attended classes / total classes × 100

For CGPA:
calculate the average of the supplied grade points.

Do not invent numerical results.

"""

)


# ============================================================
# EXTRACT AGENT RESPONSE
# ============================================================

def extract_text_response(result):

    if not isinstance(result, dict):

        return str(result)


    messages = result.get(
        "messages",
        []
    )


    if not messages:

        return str(result)


    last_message = messages[-1]

    content = getattr(
        last_message,
        "content",
        ""
    )


    if isinstance(
        content,
        str
    ):

        return content.strip()


    if isinstance(
        content,
        list
    ):

        parts = []


        for item in content:

            if isinstance(
                item,
                dict
            ):

                if item.get(
                    "type"
                ) == "text":

                    text = item.get(
                        "text",
                        ""
                    )

                    if text:

                        parts.append(
                            text
                        )

            elif isinstance(
                item,
                str
            ):

                parts.append(item)


        return "".join(
            parts
        ).strip()


    return str(content)


# ============================================================
# LANGSERVE
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Message for the agent"
    )


def format_for_agent(x):

    user_input = (

        x["input"]

        if isinstance(
            x,
            dict
        )

        else x.input

    )

    return {

        "messages": [

            (
                "user",
                user_input
            )

        ]

    }


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
# FASTAPI
# ============================================================

app = FastAPI(

    title="Student Utility Agent",

    description=(
        "Student Utility Agent powered by "
        "Gemini and LangChain"
    )

)


# ============================================================
# CHAT REQUEST
# ============================================================

class ChatRequest(BaseModel):

    message: str


# ============================================================
# CONVERSION DETECTOR
# ============================================================

def detect_conversion(text):

    text = text.lower().strip()

    text = text.replace(
        "?",
        ""
    )


    # --------------------------------------------------------
    # 30 Celsius to Fahrenheit
    # 2 feet to centimeters
    # 5 km to miles
    # --------------------------------------------------------

    match = re.search(

        r"(-?\d+(?:\.\d+)?)\s*"
        r"(?:degrees?\s+)?"
        r"([a-zA-Z°²/ ]+?)"
        r"\s+(?:to|into|in)\s+"
        r"([a-zA-Z°²/ ]+?)$",

        text

    )


    if match:

        return (

            float(match.group(1)),

            match.group(2).strip(),

            match.group(3).strip()

        )


    # --------------------------------------------------------
    # 2 feet is how many cm
    # --------------------------------------------------------

    match = re.search(

        r"(-?\d+(?:\.\d+)?)\s*"
        r"([a-zA-Z°²/ ]+?)"
        r"\s+(?:is|are)\s+how\s+many\s+"
        r"([a-zA-Z°²/ ]+?)$",

        text

    )


    if match:

        return (

            float(match.group(1)),

            match.group(2).strip(),

            match.group(3).strip()

        )


    # --------------------------------------------------------
    # how many cm is 2 feet
    # --------------------------------------------------------

    match = re.search(

        r"how\s+many\s+"
        r"([a-zA-Z°²/ ]+?)"
        r"\s+(?:is|are)\s+"
        r"(-?\d+(?:\.\d+)?)\s*"
        r"([a-zA-Z°²/ ]+?)$",

        text

    )


    if match:

        return (

            float(match.group(2)),

            match.group(3).strip(),

            match.group(1).strip()

        )


    return None


# ============================================================
# CHAT API
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.strip()


    if not message:

        return JSONResponse(

            content={
                "response":
                    "Please enter a question."
            },

            status_code=400

        )


    # ========================================================
    # DIRECT CONVERSION
    # ========================================================

    conversion = detect_conversion(
        message
    )


    if conversion:

        value, from_unit, to_unit = conversion


        try:

            answer = unit_converter.invoke({

                "value": value,

                "from_unit": from_unit,

                "to_unit": to_unit

            })


            return {
                "response": answer
            }


        except Exception as e:

            return JSONResponse(

                content={
                    "response":
                        f"Conversion error: {str(e)}"
                },

                status_code=500

            )


    # ========================================================
    # AI AGENT
    # ========================================================

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

            content={
                "response":
                    f"Agent error: {str(e)}"
            },

            status_code=500

        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "message": "Student Utility Agent is running"
    }


# ============================================================
# WEB APPLICATION
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
}


body {

    margin: 0;

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

    padding: 30px;

    text-align: center;

}


.header h1 {

    margin: 0 0 8px 0;

    font-size: 30px;

}


.header p {

    margin: 0;

    font-size: 15px;

}


.features {

    display: flex;

    justify-content: center;

    flex-wrap: wrap;

    gap: 10px;

    padding: 16px;

    border-bottom:
        1px solid #e5e7eb;

}


.feature {

    padding:
        8px 14px;

    border-radius: 20px;

    background: #eef2ff;

    color: #3730a3;

    font-size: 13px;

    font-weight: bold;

}


.chat {

    height: 430px;

    overflow-y: auto;

    padding: 22px;

}


.message {

    max-width: 80%;

    padding:
        14px 17px;

    margin-bottom: 16px;

    border-radius: 15px;

    font-size: 15px;

    line-height: 1.55;

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


.input-area {

    display: flex;

    gap: 10px;

    padding: 16px;

    border-top:
        1px solid #e5e7eb;

}


#message {

    flex: 1;

    min-width: 0;

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


.examples {

    padding:
        0 18px 15px;

    color: #64748b;

    font-size: 13px;

}


.status {

    text-align: center;

    color: #94a3b8;

    font-size: 12px;

    padding-bottom: 18px;

}


@media(max-width:600px) {

    body {

        padding: 10px;

    }


    .container {

        border-radius: 15px;

    }


    .header h1 {

        font-size: 24px;

    }


    .chat {

        height: 400px;

        padding: 15px;

    }


    .message {

        max-width: 90%;

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


<!-- HEADER -->

<div class="header">

    <h1>
        🎓 Student Utility Agent
    </h1>

    <p>
        Your AI assistant for student calculations
    </p>

</div>


<!-- FEATURES -->

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


<!-- CHAT -->

<div
    id="chat"
    class="chat"
>

    <div class="message bot">

👋 Hi! I'm your Student Utility Agent.

I can help you with:

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


<!-- INPUT -->

<div class="input-area">

    <input
        id="message"
        type="text"
        placeholder="Ask your question..."
        autocomplete="off"
    >

    <button
        id="send"
        type="button"
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


// ============================================================
// WAIT UNTIL PAGE IS LOADED
// ============================================================

window.addEventListener(
    "DOMContentLoaded",
    function() {


        // ----------------------------------------------------
        // GET ELEMENTS
        // ----------------------------------------------------

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


        // ----------------------------------------------------
        // SEND FUNCTION
        // ----------------------------------------------------

        async function sendMessage() {


            const message =
                input.value.trim();


            // Nothing entered
            if (!message) {

                input.focus();

                return;

            }


            // ------------------------------------------------
            // USER MESSAGE
            // ------------------------------------------------

            const userBubble =
                document.createElement(
                    "div"
                );


            userBubble.className =
                "message user";


            userBubble.textContent =
                message;


            chat.appendChild(
                userBubble
            );


            // Clear input
            input.value = "";


            // ------------------------------------------------
            // BUTTON
            // ------------------------------------------------

            button.disabled = true;

            button.textContent =
                "Thinking...";


            // ------------------------------------------------
            // LOADING
            // ------------------------------------------------

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


                // ============================================
                // CALL BACKEND
                // ============================================

                const response =
                    await fetch(
                        "/chat",
                        {

                            method: "POST",

                            headers: {

                                "Content-Type":
                                    "application/json",

                                "Accept":
                                    "application/json"

                            },

                            body:
                                JSON.stringify({

                                    message:
                                        message

                                })

                        }
                    );


                // ============================================
                // READ RESPONSE
                // ============================================

                const data =
                    await response.json();


                loading.remove();


                // ============================================
                // BOT RESPONSE
                // ============================================

                const botBubble =
                    document.createElement(
                        "div"
                    );


                botBubble.className =
                    "message bot";


                botBubble.textContent =
                    data.response ||
                    "No response received.";


                chat.appendChild(
                    botBubble
                );


            }


            catch(error) {


                // ==========================================
                // ERROR
                // ==========================================

                loading.remove();


                const errorBubble =
                    document.createElement(
                        "div"
                    );


                errorBubble.className =
                    "message bot";


                errorBubble.textContent =
                    "❌ Unable to contact the server.\n\n" +
                    "Please refresh the page and try again.";


                chat.appendChild(
                    errorBubble
                );


                console.error(
                    "Chat error:",
                    error
                );

            }


            // ------------------------------------------------
            // RESET BUTTON
            // ------------------------------------------------

            button.disabled = false;

            button.textContent =
                "Send";


            input.focus();


            chat.scrollTop =
                chat.scrollHeight;

        }


        // ----------------------------------------------------
        // BUTTON CLICK
        // ----------------------------------------------------

        button.onclick =
            sendMessage;


        // ----------------------------------------------------
        // ENTER KEY
        // ----------------------------------------------------

        input.addEventListener(

            "keydown",

            function(event) {

                if (
                    event.key === "Enter"
                ) {

                    event.preventDefault();

                    sendMessage();

                }

            }

        );


    }

);

</script>


</body>

</html>

"""


# ============================================================
# LANGSERVE ROUTE
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/agent"

)


# ============================================================
# START SERVER
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
