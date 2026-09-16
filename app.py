
# your FastAPI + LangServe deployment code
import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


# =========================
# TOOLS
# =========================

@tool
def calculate_percentage(marks_obtained: float, total_marks: float) -> str:
    """Calculate percentage from marks obtained and total marks."""

    if total_marks <= 0:
        return "Total marks must be greater than zero."

    percentage = (marks_obtained / total_marks) * 100
    return f"Percentage = {percentage:.2f}%"


@tool
def calculate_cgpa(
    sem1: float,
    sem2: float,
    sem3: float,
    sem4: float
) -> str:
    """Calculate average CGPA from four semester CGPA values."""

    cgpa = (sem1 + sem2 + sem3 + sem4) / 4
    return f"Average CGPA = {cgpa:.2f}"


@tool
def calculate_attendance(
    attended_classes: int,
    total_classes: int
) -> str:
    """Calculate attendance percentage."""

    if total_classes <= 0:
        return "Total classes must be greater than zero."

    attendance = (attended_classes / total_classes) * 100
    return f"Attendance = {attendance:.2f}%"


@tool
def unit_converter(value: float, conversion: str) -> str:
    """Convert common units."""

    conversion = conversion.lower()

    if conversion == "km_to_miles":
        result = value * 0.621371
        return f"{value} km = {result:.2f} miles"

    elif conversion == "miles_to_km":
        result = value * 1.60934
        return f"{value} miles = {result:.2f} km"

    elif conversion == "kg_to_pounds":
        result = value * 2.20462
        return f"{value} kg = {result:.2f} pounds"

    elif conversion == "pounds_to_kg":
        result = value * 0.453592
        return f"{value} pounds = {result:.2f} kg"

    elif conversion == "celsius_to_fahrenheit":
        result = (value * 9 / 5) + 32
        return f"{value}°C = {result:.2f}°F"

    elif conversion == "fahrenheit_to_celsius":
        result = (value - 32) * 5 / 9
        return f"{value}°F = {result:.2f}°C"

    else:
        return "Unsupported conversion."


# =========================
# GEMINI MODEL
# =========================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# =========================
# AGENT
# =========================

tools = [
    calculate_percentage,
    calculate_cgpa,
    calculate_attendance,
    unit_converter
]

student_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a Student Utility AI Agent.

Your job is to help students with:

1. Percentage calculations
2. CGPA calculations
3. Attendance calculations
4. Unit conversions

Use the appropriate tool whenever a calculation is required.

Do not invent calculation results.

If the question is unrelated to these tasks, politely say:

"I am specialized in student calculations and unit conversions."
"""
)


# =========================
# LANGSERVE
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

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))

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
# FASTAPI APP
# =========================

app = FastAPI()

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
