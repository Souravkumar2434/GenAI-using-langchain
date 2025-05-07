from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class Review(BaseModel):
    summary: str = Field(description="A short summary of the review")
    key_points: list[str] = Field(description= "A list of key points from the review")
    rating : float = Field(description="A rating from 1 to 5")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the review")
    movie_name : Optional[str]= Field(default=None,  description="The name of the movie being reviewed")

structured_model = model.with_structured_output(Review, method="function_calling")

result = structured_model.invoke("""Inception is a mind-bending science fiction thriller directed by Christopher Nolan that blends action, psychological suspense, and high-concept storytelling. The film centers on Dom Cobb, a skilled thief who specializes in stealing secrets from within the subconscious during the dream state. Cobb is offered a chance at redemption when he's tasked with performing inception—not extracting an idea, but planting one—in the mind of a corporate heir. As the team dives deeper into layered dreams, the lines between reality and illusion begin to blur, leading to an emotionally and intellectually gripping journey. With stunning visuals, a haunting score, and a complex narrative structure, Inception has earned widespread acclaim, holding an IMDb rating of 8.8, a Rotten Tomatoes score of 87%, and a Metacritic score of 74.""")

print(result)
print(result.movie_name)