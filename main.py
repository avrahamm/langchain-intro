from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, AIMessage

from langchain_groq import ChatGroq

import os
import dotenv

dotenv.load_dotenv()

@tool("PlanetGeneralInfo")
def planet_general_info_tool(planet_name: str) ->str:
    """
    this tool should take the name of a planet as input (string)
    and handle general planet queries that are not about the planet's distance from the Sun
    or its revolution period. This tool should retrieve information about the planet
    by performing a similarity search
    over documents in the planets/ directory.
    :param planet_name:
    :return: str
    """
    if planet_name not in ["Earth", "Mars", "Jupiter", "Pluto",
                           "Mercury", "Venus", "Uranus", "Neptune",
                           "Saturn"]:
        return f"Additional information for {planet_name} is not available in this tool."

    # Load all text files from the 'planets/' directory
    loader = DirectoryLoader("planets", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Initialize HuggingFace embeddings model.
    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key=os.getenv("HUGGINGFACE_API_KEY")
    )

    # Create a Chroma vector store from the loaded documents and HuggingFace embeddings.
    db = Chroma.from_documents(documents, embeddings_model)

    # Perform a similarity search on the vector store with a sample query.
    # query = input()  #"What makes Saturn unique?"
    docs = db.similarity_search(planet_name)

    # Print the content of the most similar document.
    return docs[0].page_content


@tool("PlanetDistanceSun ")
def planet_distance_sun_tool(planet_name: str) ->str:
    """
    this tool should take the name of a planet as input (string)
    and return its approximate distance from the Sun
    in Astronomical Units (AU);
    :param planet_name:
    :return: str
    """
    answers_dict = {
        "Earth": "Earth is approximately 1 AU from the Sun.    ",
        "Mars": "Mars is approximately 1.5 AU from the Sun.   ",
        "Jupiter": "Jupiter is approximately 5.2 AU from the Sun.",
        "Pluto": "Pluto is approximately 39.5 AU from the Sun. ",
        "other": f"Information about the distance of {planet_name} from the Sun is not available in this tool."
    }

    return answers_dict.get(planet_name, answers_dict["other"])


@tool("PlanetRevolutionPeriod")
def planet_revolution_period_tool(planet_name: str) -> str:
    """
    this tool should take the name of a planet as input (string)
    and return its approximate revolution period around the Sun
    in Earth years;
    :param planet_name:
    :return: str
    """
    revolution_periods = {
        "Earth": "Earth takes approximately 1 Earth year to revolve around the Sun.",
        "Mars": "Mars takes approximately 1.88 Earth years to revolve around the Sun.",
        "Jupiter": "Jupiter takes approximately 11.86 Earth years to revolve around the Sun.",
        "Pluto": "Pluto takes approximately 248 Earth years to revolve around the Sun.",
        "other": f"Information about the revolution period of {planet_name} around the Sun is not available in this tool."
    }

    return revolution_periods.get(planet_name, revolution_periods["other"])


# Assume we have llm_response that contains identified_tools.tool_calls
# This would be a list of tool calls identified by the LLM

def execute_tool_call(tool_call):
    # Dictionary mapping tool names to their respective functions
    tools_dict = {
        "PlanetDistanceSun": planet_distance_sun_tool,
        "PlanetRevolutionPeriod": planet_revolution_period_tool,
        "PlanetGeneralInfo": planet_general_info_tool,
        # Add any other tools you have in your application
    }

    # Loop through each tool call
    tool_name = tool_call.get("name")
    # print(f"tool_name = {tool_name}")
    try:
        planet_name = tool_call["args"]["planet_name"]
        # print(f"planet_name = {planet_name}")
    except (KeyError, TypeError, AttributeError) as e:
        return str(e)

    # Check if the tool exists in our dictionary
    if tool_name not in tools_dict:
        return f"Tool {tool_name} is not available in this tool."
    try:
        # Execute the tool with the provided arguments
        if tool_name == "PlanetDistanceSun":
            return planet_distance_sun_tool.invoke(planet_name)
        if tool_name == "PlanetRevolutionPeriod":
            return planet_revolution_period_tool.invoke(planet_name)
        # if tool_name == "PlanetGeneralInfo":
        return planet_general_info_tool.invoke(planet_name)

    except Exception as e:
            return str(e)


def manage_tool_calls(identified_tools: BaseMessage | AIMessage):
    if not identified_tools or not hasattr(identified_tools, "tool_calls"):
        return "No tools found in the response."

    tool_call_result = execute_tool_call(identified_tools.tool_calls[0])
    return tool_call_result

def stage5():

    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant who answers questions
        users may have.
        You are asked: {question}."
    """
    )

    # Initialize the ChatGroq language model with specified parameters
    llm = ChatGroq(
        # model="llama-3.3-70b-versatile",
        model="llama-3.1-8b-instant",
        temperature=0.6,
        max_retries=4
    )

    # llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))
    # llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    tools_list = [
        planet_distance_sun_tool,
        planet_revolution_period_tool,
        planet_general_info_tool,
    ]
    model_with_tools = llm.bind_tools(tools_list)
    chain = prompt | model_with_tools | manage_tool_calls  # Create a chain by composing prompt and LLM
    user_query = input()
    chain_result = chain.invoke({"question": user_query})
    print(chain_result)
    print(chain)


if __name__ == "__main__":
    stage5()
