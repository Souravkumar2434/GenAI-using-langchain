from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


url = r"https://www.youtube.com/watch?v=1w5cCXlh7JQ"
loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
data = loader.load()
print("--------------------------------------------------")
print(data[0].metadata)
print("--------------------------------------------------")
print(data[0].page_content)