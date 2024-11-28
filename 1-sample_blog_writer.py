# from crewai import Agent

# agent = Agent(
#   role='Data Analyst',
#   goal='Extract actionable insights',
#   backstory="""You're a data analyst at a large company.
#     You're responsible for analyzing data and providing insights
#     to the business.
#     You're currently working on a project to analyze the
#     performance of our marketing campaigns.""",
#   tools=[my_tool1, my_tool2],  # Optional, defaults to an empty list
#   llm=my_llm,  # Optional
#   function_calling_llm=my_llm,  # Optional
#   max_iter=15,  # Optional
#   max_rpm=None, # Optional
#   max_execution_time=None, # Optional
#   verbose=True,  # Optional
#   allow_delegation=False,  # Optional
#   step_callback=my_intermediate_step_callback,  # Optional
#   cache=True,  # Optional
#   system_template=my_system_template,  # Optional
#   prompt_template=my_prompt_template,  # Optional
#   response_template=my_response_template,  # Optional
#   config=my_config,  # Optional
#   crew=my_crew,  # Optional
#   tools_handler=my_tools_handler,  # Optional
#   cache_handler=my_cache_handler,  # Optional
#   callbacks=[callback1, callback2],  # Optional
#   allow_code_execution=True,  # Optional
#   max_retry_limit=2,  # Optional
#   use_system_prompt=True,  # Optional
#   respect_context_window=True,  # Optional
#   code_execution_mode='safe',  # Optional, defaults to 'safe'
# )

# agent = Agent(
#         role="{topic} specialist",
#         goal="Figure {goal} out",
#         backstory="I am the master of {role}",
#         system_template="""<|start_header_id|>system<|end_header_id|>
#                         {{ .System }}<|eot_id|>""",
#         prompt_template="""<|start_header_id|>user<|end_header_id|>
#                         {{ .Prompt }}<|eot_id|>""",
#         response_template="""<|start_header_id|>assistant<|end_header_id|>
#                         {{ .Response }}<|eot_id|>""",
# )

# from crewai import Agent, Task, Crew
# from custom_agent import CustomAgent # You need to build and extend your own agent logic with the CrewAI BaseAgent class then import it here.

# from langchain.agents import load_tools

# langchain_tools = load_tools(["google-serper"], llm=llm)

# agent1 = CustomAgent(
#     role="agent role",
#     goal="who is {input}?",
#     backstory="agent backstory",
#     verbose=True,
# )

# task1 = Task(
#     expected_output="a short biography of {input}",
#     description="a short biography of {input}",
#     agent=agent1,
# )

# agent2 = Agent(
#     role="agent role",
#     goal="summarize the short bio for {input} and if needed do more research",
#     backstory="agent backstory",
#     verbose=True,
# )

# task2 = Task(
#     description="a tldr summary of the short biography",
#     expected_output="5 bullet point summary of the biography",
#     agent=agent2,
#     context=[task1],
# )

# my_crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
# crew = my_crew.kickoff(inputs={"input": "Mark Twain"})
# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
from crewai import LLM

ollama_llm=LLM(model="ollama/llama3.1", base_url="http://localhost:11434")

planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    llm=ollama_llm,
    allow_delegation=False,
	verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    llm=ollama_llm,
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    llm=ollama_llm,
    allow_delegation=False,
    verbose=True
)


editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    llm=ollama_llm,
    allow_delegation=False,
    verbose=True
)


### TASKS

plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.",
    agent=planner,
)


write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=editor
)


crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})