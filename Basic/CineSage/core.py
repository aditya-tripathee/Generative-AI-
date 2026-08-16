import sys
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser

# Reconfigure stdout for UTF-8 compatibility on Windows
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv(find_dotenv())

# Initialize LLM Model
model = ChatMistralAI(model="open-mistral-7b")

# Pydantic Schema Definition
class MovieAnalysis(BaseModel):
    """Schema for structured movie analysis"""
    director: str = Field(description="Director of the movie")
    release_year: int = Field(description="Year the movie was released")
    genre: List[str] = Field(description="List of genres as strings")
    cast: List[str] = Field(description="Main cast members as strings")
    plot_summary: str = Field(description="Bulleted plot summary as a string")
    highlights: List[str] = Field(description="List of key highlights as plain text strings (do NOT use nested dicts/objects)")
    themes: List[str] = Field(description="List of underlying themes as strings")
    impact: str = Field(description="Cultural or commercial impact as a string")

# Instantiate Pydantic Output Parser
parser = PydanticOutputParser(pydantic_object=MovieAnalysis)

# PromptTemplate with format_instructions provided as a partial_variable
prompt = PromptTemplate(
    template="""You are CineSage, an expert movie analyst. Analyze the given movie accurately.

IMPORTANT: Ensure all list values (like highlights, genre, themes) are plain text strings, NOT nested dictionary objects.

{format_instructions}

Question: {question}
""",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


# Build LCEL Chain: Prompt -> Model -> Parser
chain = prompt | model | parser

# Invoke chain to receive a typed MovieAnalysis Pydantic object
response: MovieAnalysis = chain.invoke({
    "question": "Provide a detailed movie analysis for 'KGF: Chapter 1'"
})

# Display parsed Pydantic data
print("=== STRUCTURED MOVIE ANALYSIS ===")
print(f"Movie Director: {response.director}")
print(f"Release Year:   {response.release_year}")
print(f"Genre:          {', '.join(response.genre)}")
print(f"Cast:           {', '.join(response.cast)}")
print(f"\nPlot Summary:\n{response.plot_summary}")
print("\nHighlights:")
for item in response.highlights:
    print(f" - {item}")
print("\nThemes:")
for theme in response.themes:
    print(f" - {theme}")
print(f"\nCultural & Commercial Impact:\n{response.impact}")


