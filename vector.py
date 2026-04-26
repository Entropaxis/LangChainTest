from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

texts = [
    'I love apples',
    'I enjoy oranges',
    'I think pears taste very good',
    'I hate bananas',
    'I dislike raspberries',
    'I despise mangos',
    'I love Linux',
    'I hate Windows',
 ]

vectore_store = FAISS.from_texts(texts, embedding=embeddings)

print(vectore_store.similarity_search('What fruits do you like?', k=3))
print(vectore_store.similarity_search('What fruits do you hate?', k=3))

retreiver = vectore_store.as_retriever(search_kwargs={'k': 3})

retreiver_tool = create_retriever_tool(retreiver, name='kb_search', description='Search the small product / fruit knowladge base for information')

agent = create_agent(
    model = 'gpt-4.1-mini',
    tools = [retreiver_tool],
    system_prompt = ("You are a helpful assistant that answers questions about Macs, apple, or laptops, \
    first call the tool 'kb_search' to retreive context, then answer succinctly.\
    Maybe you have to use it multiple times before answering")

)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What three fruits do you like and what\
        three fruits do you dislike?"}]
})

print(response['messages'][-1].content)
