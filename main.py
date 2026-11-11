import asyncio
import os
from typing import TypedDict, Annotated
import operator
import uuid
from pathlib import Path
import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

# from tools.tavily_tool import tavily_search
from mcp_client import extract_destination, forecast_mcp_search, run_mcp, tavily_mcp_search, aviation_mcp_call, weather_mcp_search

from tools.flight_tool import search_flights
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=1000,
)


# State
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str


# Flight Tool Router Prompt
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


# Flight Agent
def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:

        airports = run_mcp(aviation_mcp_call("list_airports"))

        airlines = run_mcp(aviation_mcp_call("list_airlines"))

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt),
            ]
        )

        flight_data = response.content

    except Exception as e:

        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

def weather_agent(state: TravelState):
    print("\nINSIDE WEATHER AGENT\n")
    try:
        city = extract_destination(state["user_query"])
        weather_data = run_mcp(weather_mcp_search(city))
        forecast_data = run_mcp(forecast_mcp_search(city))
        weather_results = f"""
        Destination: {city}
        Current Weather:
        {weather_data}

        Forecast:
        {forecast_data}
        """
        
        print(f"Weather results: {weather_results}")
    except Exception as e:
        weather_results = f"Weather information unavailable: {e}"

    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather information fetched")],
    }


# Hotel Agent
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = run_mcp(tavily_mcp_search(query))
    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information fetched")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Itinerary Agent
def itinerary_agent(state: TravelState):

    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
    
    Weather Information:
    {state['weather_results']}
    """

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner"),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Final Response Agent
def final_agent(state: TravelState):

    print("\nINSIDE FINAL AGENT\n")

    final_prompt = f"""
    User Request:
    {state['user_query']}

    Flight Information:
    {state.get('flight_results') or 'Not available'}

    Hotel Information:
    {state.get('hotel_results') or 'Not available'}

    Travel Itinerary:
    {state.get('itinerary') or 'Not available'}

    Weather Information:
    {state.get('weather_results') or 'Not available'}

    Instructions:
    1. Generate a complete travel response.
    2. Include both flight and hotel information.
    3. Include the itinerary.
    4. Include current weather and forecast information when available, with the destination clearly identified.
    5. Clearly mention unavailable information.
    6. Never invent flight availability, prices or bookings.
    7. Use clear headings and bullet points.
    8. Do not return an empty response.
    """

    response = llm.invoke(
        [
            SystemMessage(content="You are a professional travel planning assistant."),
            HumanMessage(content=final_prompt),
        ]
    )

    content = response.content
    if isinstance(content, str):
        final_content = content.strip()
    elif isinstance(content, list):
        # Some chat model integrations return text as content blocks.
        final_content = "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("text")
        ).strip()
    else:
        final_content = ""

    if not final_content:
        # Keep the useful gathered results if the final model call is empty.
        final_content = "\n".join(
            [
                f"Travel request: {state['user_query']}",
                f"\nFlights\n{state.get('flight_results') or 'Information unavailable.'}",
                f"\nHotels\n{state.get('hotel_results') or 'Information unavailable.'}",
                f"\nItinerary\n{state.get('itinerary') or 'Information unavailable.'}",
                f"\nWeather\n{state.get('weather_results') or 'Information unavailable.'}",
            ]
        )

    return {
        "messages": [AIMessage(content=final_content)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

# Use a synchronous checkpointer for LangGraph. FastMCP calls run on their own
# persistent loop through run_mcp(), keeping Windows' stdio and database loops apart.
_conn = psycopg.connect(DATABASE_URL, autocommit=True)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()
app = graph.compile(checkpointer=checkpointer)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


if __name__ == "__main__":
    user_input = input("Enter travel request: ")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "weather_results": "",
            "llm_calls": 0,
        },
        config=config,
    )

    final_response = result["messages"][-1].content
    output_path = OUTPUT_DIR / f"{uuid.uuid4()}.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_response, encoding="utf-8")

    print("\nFINAL RESPONSE:\n")
    print(final_response)
    print(f"\nSaved response to: {output_path}")
