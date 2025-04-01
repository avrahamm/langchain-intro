# install the required packages:
# pip install langchain-groq langchain-core python-dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_groq import ChatGroq
import dotenv

dotenv.load_dotenv()


def main():
    """
    To run tests
    :return:
    """

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_retries=2
    )

    examples = [
        {"input": "Jupiter", "output": """
        Jupiter is the largest planet in the solar system.
        It is a gas giant primarily composed of hydrogen and helium.
        It has a Great Red Spot, a massive storm, and at least 79 known moons, including Ganymede, the largest moon in the solar system.
        """
         },
        {"input": "Mars", "output": """
        Mars is the fourth planet from the Sun.
         It has a thin atmosphere composed mainly of carbon dioxide and is known for its red appearance due to iron oxide on its surface.
    Notable features include Olympus Mons, the largest volcano in the solar system, and Valles Marineris, a vast canyon system.
        """},
    ]

    example_template = PromptTemplate.from_template("Q: {input}\nA: {output}")

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        suffix="{question}",
        input_variables=["question"],
    )
    planet = input()
    final_prompt = few_shot_prompt.format(question=f"Tell me about {planet} planet?")

    response = llm.invoke(final_prompt)

    print(response.content)

if __name__ == "__main__":
    main()
