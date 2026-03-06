from agent_framework import Agent, tool
from agent_framework.openai import  OpenAIChatClient
import os , logging
from typing import Annotated
from pydantic import Field
import asyncio
from app import rag
from rich import print
from rich.logging import RichHandler

# Setup logging
handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

client = OpenAIChatClient(
        base_url="https://models.github.ai/inference",
        api_key=os.environ["GITHUB_TOKEN"],
        model_id=os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini"),
    )


@tool
def search_met_artworks(
    query: Annotated[str, Field(description="The search query for Met artworks")]) -> str: 
    """Searches the Met Museum's artwork collection based on a user query and returns relevant information about the artworks."""
    print(f"[trace] tool:search_met_artworks called with query={query!r}")
    search_results = rag(query)
    preview = (search_results[:220] + "...") if len(search_results) > 220 else search_results
    print(f"[trace] tool:search_met_artworks result preview: {preview}")
    return search_results

@tool
def add_artwork_to_tour_csv(
    artwork_name: Annotated[str, Field(description="The name of the artwork to add to the tour CSV")],
    artwork_artist: Annotated[str, Field(description="The artist of the artwork")],
    artwork_gallery_link: Annotated[str, Field(description="The gallery link of the artwork in the MET museum")],
):
    """Adds an artwork to a tour CSV file with the artwork's name, artist, and gallery link."""
    print(
        "[trace] tool:add_artwork_to_tour_csv called with "
        f"artwork_name={artwork_name!r}, artwork_artist={artwork_artist!r}, "
        f"artwork_gallery_link={artwork_gallery_link!r}"
    )
    file_name = "met_tour.csv"
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write("artwork_name,artwork_artist,artwork_gallery_link\n")
        print(f"[trace] created {file_name}")

    with open(file_name, "a") as f:
        f.write(f"{artwork_name},{artwork_artist},{artwork_gallery_link}\n")
    print(f"[trace] appended artwork row to {file_name}")


agent = Agent(client=client, 
              instructions="You are an assistant that answers questions about artworks in the MET museum. If you don't know the answer, say you don't know, but try to use the tool to find out.",
              tools=[search_met_artworks, add_artwork_to_tour_csv]
              )

async def main():
    prompt = "Could you please prepare a tour of artworks by Van Gogh in the MET museum?"
    print(f"[trace] agent.run start with prompt={prompt!r}")
    response = await agent.run(prompt)
    print("[trace] agent.run completed")
    print(response.text)



if __name__ == "__main__":
    asyncio.run(main())