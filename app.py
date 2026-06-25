import os
import requests
import sqlite3
import streamlit as st

from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_groq import ChatGroq

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
    add_messages
)

from langgraph.checkpoint.sqlite import SqliteSaver

from typing import TypedDict, Annotated

# ====================================================
# LOAD ENV
# ====================================================

load_dotenv()

flight_api_key = os.getenv("AVIATIONSTACK_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# ====================================================
# STREAMLIT CONFIG
# ====================================================

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

st.title("🌍 AI Travel Planner")
st.markdown(
    "Plan your complete trip using LangGraph + Groq + Tavily + AviationStack"
)

# ====================================================
# FLIGHT SEARCH
# ====================================================

def search_flights(query):

    url = "https://api.aviationstack.com/v1/flights"

    params = {
        "access_key": flight_api_key,
        "limit": 5
    }

    try:

        response = requests.get(url, params=params)

        data = response.json()

        flights = []

        if "data" in data:

            for flight in data["data"][:5]:

                airline = flight["airline"]["name"]

                flight_number = flight["flight"]["number"]

                departure_airport = flight["departure"]["airport"]

                departure_time = flight["departure"]["scheduled"]

                arrival_airport = flight["arrival"]["airport"]

                arrival_time = flight["arrival"]["scheduled"]

                flights.append(
                    f"""
Airline: {airline}
Flight Number: {flight_number}
Departure Airport: {departure_airport}
Departure Time: {departure_time}
Arrival Airport: {arrival_airport}
Arrival Time: {arrival_time}
                    """
                )

            return "\n\n".join(flights)

        return "No flights found."

    except Exception as e:
        return f"Flight API Error: {str(e)}"


# ====================================================
# TAVILY SEARCH
# ====================================================

client = TavilyClient(api_key=tavily_api_key)

def tavily_search(query):

    try:

        response = client.search(
            query=query,
            max_results=5
        )

        results = []

        for i, r in enumerate(response["results"], 1):

            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("content", "")

            results.append(
                f"""
{i}. {title}

{snippet}

{url}
                """
            )

        return "\n\n".join(results)

    except Exception as e:
        return str(e)


# ====================================================
# LLM
# ====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

# ====================================================
# SQLITE CHECKPOINTER
# ====================================================

conn = sqlite3.connect(
    "graph.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)

# ====================================================
# STATE
# ====================================================

class TravelState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

    user_query: str

    flight_results: str

    hotel_results: str

    itinerary: str

    llm_calls: int


# ====================================================
# AGENTS
# ====================================================

def flight_agent(state: TravelState):

    flight_data = search_flights(
        state["user_query"]
    )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight information collected."
            )
        ],
        "llm_calls": state["llm_calls"] + 1
    }


def hotel_agent(state: TravelState):

    hotel_data = tavily_search(
        f"Best hotels for {state['user_query']}"
    )

    return {
        "hotel_results": hotel_data,
        "messages": [
            AIMessage(
                content="Hotel information collected."
            )
        ],
        "llm_calls": state["llm_calls"] + 1
    }


def itinerary_agent(state: TravelState):

    prompt = f"""
Create a complete travel itinerary.

USER REQUEST:
{state['user_query']}

FLIGHTS:
{state['flight_results']}

HOTELS:
{state['hotel_results']}
"""

    response = llm.invoke([
        SystemMessage(
            content="You are a professional travel planner."
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state["llm_calls"] + 1
    }


def final_agent(state: TravelState):

    prompt = f"""
Create a final travel recommendation.

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return {
        "messages": [response],
        "llm_calls": state["llm_calls"] + 1
    }


# ====================================================
# GRAPH
# ====================================================

graph = StateGraph(TravelState)

graph.add_node(
    "flight_agent",
    flight_agent
)

graph.add_node(
    "hotel_agent",
    hotel_agent
)

graph.add_node(
    "itinerary_agent",
    itinerary_agent
)

graph.add_node(
    "final_agent",
    final_agent
)

graph.add_edge(
    START,
    "flight_agent"
)

graph.add_edge(
    "flight_agent",
    "hotel_agent"
)

graph.add_edge(
    "hotel_agent",
    "itinerary_agent"
)

graph.add_edge(
    "itinerary_agent",
    "final_agent"
)

graph.add_edge(
    "final_agent",
    END
)

travel_app = graph.compile(
    checkpointer=checkpointer
)

# ====================================================
# UI
# ====================================================

with st.sidebar:

    st.header("⚙️ Settings")

    thread_id = st.text_input(
        "Session ID",
        value="user_tathagata"
    )

user_query = st.text_area(
    "Enter Travel Request",
    placeholder="Plan a 5 day trip to Paris..."
)

if st.button("Generate Travel Plan"):

    with st.spinner("Planning your trip..."):

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        result = travel_app.invoke(

            {
                "messages": [
                    HumanMessage(
                        content=user_query
                    )
                ],
                "user_query": user_query,
                "flight_results": "",
                "hotel_results": "",
                "itinerary": "",
                "llm_calls": 0
            },

            config=config

        )

        st.success("Travel Plan Generated")

        st.subheader("📋 Final Response")

        final_message = result["messages"][-1].content

        st.markdown(final_message)

        st.divider()

        st.metric(
            "LLM Calls",
            result["llm_calls"]
        )