from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.vectorstores import Pinecone
import chainlit as cl
import pinecone

from dotenv import load_dotenv


model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
)

load_dotenv()

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    memorkey = "history"
    llm = ChatOpenAI(temperature = 0, model="gpt-3.5-turbo-0613")
    memory = AgentTokenBufferMemory(memory_key=memorkey, llm=llm)
    system_message = SystemMessage(
        content=(
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if neccessary"
        )
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memorkey)]
    )

    index = pinecone.Index('gpt-4-langchain-docs-fast')
    text_field = "text"
    vectorstore = Pinecone(
        index=index, embedding=embed.embed_query, text_key=text_field
    )

    retriever = vectorstore.as_retriever()

    tool = create_retriever_tool(
        retriever, 
        "search_langchain_docs",
        "Searches and returns documents regarding the the python package langchain."
    )
    tools = [tool]

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                                   return_intermediate_steps=True)

    # Store the chain in the user session
    cl.user_session.set("langchain_agent", agent_executor)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    agent_executor = cl.user_session.get("langchain_agent")  # type: LLMChain

    # Call the chain asynchronously
    res = await agent_executor.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    print(res)

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["output"]).send()
