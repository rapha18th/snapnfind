import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from crewai import Agent, Task, Crew
from crewai_tools import DirectoryReadTool, CSVSearchTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents.agent_types import AgentType
from langchain_community.tools import DuckDuckGoSearchRun
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent, create_csv_agent

llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
search = DuckDuckGoSearchRun()

csv_rag_tool = CSVSearchTool(csv="idata.csv")

        # Create agents with the instantiated tools
analyst_agent = Agent(
            role='CSV Analyst Agent',
            goal='Research the CSV file with the objective of answering the prompt fully',
            backstory="""You are an expert analyst who takes a user's prompt and answers it using csv files to aid your understanding and analysis, so your answers are accurate and comprehensive.""",
            allow_delegation=False,
            verbose=True,
            max_iter=10,
            tools=[csv_rag_tool],
            llm = ChatGoogleGenerativeAI(model="gemini-pro")
        )

# Define a function to process the image
def image_to_text(image):
    message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": image},
    ]
    )
    res = llm.invoke([message])
  # Replace this with your actual image processing logic
  # This example just returns a generic message
    res_text = res.content

    # Split the content string at the first occurrence of '.'
    extracted_text = res_text.split('of')[-1]

    # Create tasks for your agents
    #agent = create_csv_agent(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0),"idata.csv", verbose=True)
    #result = agent.invoke(f"what is the quantity available for {extracted_text}, what is the retail price and which stores have it?")
    task_1 = Task(
            description=f"what is the QuantityAvailable for {extracted_text}, what is the RetailPrice which StoreNames haves it",
            expected_output='Answer questions acccurately by searching  through the csv',
            agent=analyst_agent
        )
    crew = Crew(
            agents=[analyst_agent],
            tasks=[task_1],
            verbose=2
        )
    result1 = crew.kickoff()
    result = search.run(f"briefly describe the nutritional facts about {extracted_text} and what is the retail price and where can I find it?")
    return "Your Image:"+ str(res_text) + "\n\n Details on availability: \n " + str(result1) + "\n \n Facts About Product: \n "+ str(result)

# Create a Gradio interface
interface = gr.Interface(
  fn=image_to_text,
  inputs=gr.Image(type="pil"),
  outputs="text",
  title="Image to Text",
  description="Upload an image and get a description",
)

# Launch the Gradio interface
interface.launch(share=True)
