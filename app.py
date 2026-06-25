import os
import requests
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

from typing import TypedDict, Annotated

# =====================================================

# LOAD ENVIRONMENT VARIABLES

# =====================================================

load_dotenv()

AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# =====================================================

# STREAMLIT PAGE CONFIG

# =====================================================

st.set_page_config(
page_title="AI Travel Planner",
page_icon="✈️",
layout="wide"
)

st.title("✈️ AI Travel Planner")
st.markdown(
"Plan trips using LangGraph + Groq + Tavily + AviationStack"
)

# =====================================================

# FLIGHT SEARCH

# =====================================================

def search_flights(query):

```
url = "https://api.aviationstack.com/v1/flights"

params = {
    "access_key": AVIATIONSTACK_API_KEY,
    "limit": 5
}

try:
    response = requests.get(url, params=params)
    data = response.json()

    flights = []

    if "data" in data and len(data["data"]) > 0:

        for flight in data["data"][:5]:

            airline = (
                flight.get("airline", {})
                .get("name", "Unknown")
            )

            flight_number = (
                flight.get("flight", {})
                .get("number", "Unknown")
            )

            departure_airport = (
                flight.get("departure", {})
                .get("airport", "Unknown")
            )

            departure_time = (
                flight.get("departure", {})
                .get("scheduled", "Unknown")
            )

            arrival_airport = (
                flight.get("arrival", {})
                .get("airport", "Unknown")
            )

            arrival_time = (
                flight.get("arrival", {})
                .get("scheduled", "Unknown")
            )

            flights.append(
                f"""
```

Airline: {airline}
Flight Number: {flight_number}
Departure Airport: {departure_airport}
Departure Time: {departure_time}
Arrival Airport: {arrival_airport}
Arrival Time: {arrival_time}
"""
)

```
        return "\n\n".join(flights)

    return "No flight information found."

except Exception as e:
    return f"Flight Search Error: {str(e)}"
```

# =====================================================

# TAVILY SEARCH

# =====================================================

tavily_client = TavilyClient(
api_key=TAVILY_API_KEY
)

def tavily_search(query):

```
try:

    response = tavily_client.search(
        query=query,
        max_results=5
    )

    results = []

    for i, item in enumerate(
        response["results"], start=1
    ):

        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")

        results.append(
            f"""
```

{i}. {title}

{content}

{url}
"""
)

```
    return "\n\n".join(results)

except Exception as e:
    return f"Tavily Search Error: {str(e)}"
```

# =====================================================

# LLM

# =====================================================

llm = ChatGroq(
model="llama-3.3-70b-versatile",
api_key=GROQ_API_KEY
)

# =====================================================

# STATE

# =====================================================

class TravelState(TypedDict):

```
messages: Annotated[
    list[BaseMessage],
    add_messages
]

user_query: str

flight_results: str

hotel_results: str

itinerary: str

llm_calls: int
```

# =====================================================

# FLIGHT AGENT

# =====================================================

def flight_agent(state: TravelState):

```
flights = search_flights(
    state["user_query"]
)

return {
    "flight_results": flights,
    "messages": [
        AIMessage(
            content="Flight search completed."
        )
    ],
    "llm_calls": state["llm_calls"] + 1
}
```

# =====================================================

# HOTEL AGENT

# =====================================================

def hotel_agent(state: TravelState):

```
hotel_query = (
    f"Best hotels for "
    f"{state['user_query']}"
)

hotels = tavily_search(
    hotel_query
)

return {
    "hotel_results": hotels,
    "messages": [
        AIMessage(
            content="Hotel search completed."
        )
    ],
    "llm_calls": state["llm_calls"] + 1
}
```

# =====================================================

# ITINERARY AGENT

# =====================================================

def itinerary_agent(state: TravelState):

```
prompt = f"""
```

Create a detailed travel itinerary.

USER REQUEST:
{state['user_query']}

FLIGHT INFORMATION:
{state['flight_results']}

HOTEL INFORMATION:
{state['hotel_results']}
"""

```
response = llm.invoke([
    SystemMessage(
        content="You are an expert travel planner."
    ),
    HumanMessage(content=prompt)
])

return {
    "itinerary": response.content,
    "messages": [response],
    "llm_calls": state["llm_calls"] + 1
}
```

# =====================================================

# FINAL AGENT

# =====================================================

def final_agent(state: TravelState):

```
prompt = f"""
```

Generate the final travel recommendation.

Flight Information:
{state['flight_results']}

Hotel Information:
{state['hotel_results']}

Travel Itinerary:
{state['itinerary']}
"""

```
response = llm.invoke([
    HumanMessage(content=prompt)
])

return {
    "messages": [response],
    "llm_calls": state["llm_calls"] + 1
}
```

# =====================================================

# BUILD GRAPH

# =====================================================

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

travel_graph = graph.compile()

# =====================================================

# UI

# =====================================================

user_query = st.text_area(
"Enter Travel Request",
placeholder="Example: Plan a 5-day trip to Paris with flight and hotel suggestions."
)

if st.button("Generate Travel Plan"):

```
if not user_query.strip():

    st.warning(
        "Please enter a travel request."
    )

else:

    with st.spinner(
        "Generating travel plan..."
    ):

        result = travel_graph.invoke(
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
            }
        )

        st.success(
            "Travel Plan Generated"
        )

        final_response = (
            result["messages"][-1]
            .content
        )

        st.subheader(
            "📋 Final Travel Plan"
        )

        st.markdown(
            final_response
        )

        st.divider()

        st.metric(
            "LLM Calls",
            result["llm_calls"]
        )
```
