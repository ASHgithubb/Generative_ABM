# TITEL: Generalized ABM Performing a CPS-task with Teams Consisting of Different Personality Traits

#------------------------------
# Setup code space
#------------------------------

import openai # OpenAI API
from openai import OpenAI
client = OpenAI()
import os # Operating system 
import sys # System-specific parameters and functions
import io # Input and output operations
import numpy as np # Numerical operations
from datetime import datetime 
import shelve # Database for assistants
import pandas as pd 

#------------------------------
# Creating global variables and classes
#------------------------------

# Constants
TEAM_SIZE = 3  # Adjust as necessary
no_simulations = 10
turn_takings = 5
temperature = 0.5
defined_model = "gpt-4o-mini"

# OpenAI API Key
openai_api_key = os.getenv('OPENAI_API_KEY')  # Use environment variable
openai.api_key = openai_api_key # Set the API key
if openai_api_key:
    print("Breakpoint: API Key is set.")
else:
    print("Breakpoint: API Key is not set. Please check your environment variables.")
    try:
    # Make a simple API call to check connectivity
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print("API is reachable. Response:", response)
    
    except Exception as e:
        print("Error connecting to the API:", e)

# Output directories
output_file = ""

output_directory_conversation = "output_files_conversation"     
if not os.path.exists(output_directory_conversation):
    os.makedirs(output_directory_conversation)

output_directory_saved = "output_files_saved"     
if not os.path.exists(output_directory_saved):
    os.makedirs(output_directory_saved)

#Capturing print statements
output_capture = io.StringIO()  # Create a StringIO object to capture output
conversation_outputs = [] # Initialize a list to store individual outputs
saved_outputs = []
interactive_task_outputs = []
sys.stdout = output_capture  # Redirect stdout to the StringIO object
sys.stdout = sys.__stdout__  # Reset stdout to its default value

#Agents and assistants
assistants = {}
agents = {}
agent_names = ["High_stability", "High_plasticity", "High_both", "Low_stability", "Low_plasticity", "Low_both"]

# Personality trait descriptions 
high_neuro = "You easily feel panic, often doubt things, feel vulnerable to threats, get stressed quickly, fear the worst, and often worry. You seek to mitigate potential risks."
low_neuro = "You feel confident, are rarely irritated, remain unbothered, seldom get angry, and stay calm under pressure."
base_neuro = "You generally feel calm and secure but may experience mild stress in challenging situations. You approach potential risks with caution but do not dwell excessively on worst-case scenarios."

high_extra = "You are skilled at socializing, captivate people, start conversations easily, enjoy being the center of attention, and cheer people up."
low_extra = "You prefer to remain in the background, keep quiet, avoid attention, avoid engaging much, and find it difficult to approach others."
base_extra = "You are comfortable in social settings and enjoy interaction but do not seek the spotlight. You engage in conversation and connect with others, but refrain from dominating the conversations."

high_open = "You have a vivid imagination, elevate conversations, enjoy new ideas, and get excited by thinking deeply."
low_open = "You prefer practical over abstract ideas, rarely seek deeper meaning, and have difficulty with theoretical discussions."
base_open = "You have a moderate curiosity and interest in new ideas, but you also appreciate familiar routines and practical approaches. You’re open to some exploration and abstract thought, though you refrain from actively seeking novelty or deep theoretical discussions."

high_agree = "You have a kind word for everyone, assume good intentions, are very cooperative, respect others, trust people, and treat everyone equally."
low_agree = "You speak directly, hold grudges, feel superior, and often contradict others."
base_agree = "You are generally cooperative and considerate but assertive when needed. You trust others to a reasonable extent and value harmony but are not afraid to speak up when necessary."

high_consc = "You carefully consider all factors, pay attention to details, get tasks done immediately, plan thoroughly, and follow through on tasks."
low_consc = "You struggle to focus, do just enough to get by, lose interest quickly, and often leave tasks unfinished."
base_consc = "You are reasonably organized and disciplined. You follow through on tasks and strive for accuracy, although you can be flexible when necessary and may occasionally allow small details to slide in favor of efficiency."

#Personality types 
High_stability = high_consc + high_agree + low_neuro + base_extra + base_open
High_plasticity = base_consc + base_agree + base_neuro + high_extra + high_open
High_both = high_consc + high_agree + low_neuro + high_extra + high_open
Low_stability = low_consc + low_agree + high_neuro + base_extra + base_open
Low_plasticity = base_consc + base_agree + base_neuro + low_extra + low_open
Low_both = low_consc + low_agree + high_neuro + low_extra + low_open
Basic = base_consc + base_agree + base_neuro + base_extra + base_open

#Prompts
introduction = f"""Imagine you have crash-landed in the Atacama Desert in mid-July. 
It's around 10:00 am, and the temperature will reach 110°F (43°C), but at ground level, it will feel like 130°F (54°C). 
Your group of five non-injured survivors has salvaged 15 items from the wreckage. 
Survival will depend on making the right choices. The desert is vast, and rescue is uncertain. 
Every item’s utility could be the difference between life and death."""

items = f"""Torch with 4 battery-cells, Folding knife, Air map of the area, Plastic raincoat (large size), Magnetic compass, First-aid kit, 45 calibre pistol (loaded), Parachute (red & white), Bottle of 1000 salt tablets, 2 litres of water per person, A book entitled ‘Desert Animals That Can Be Eaten’, Sunglasses (for everyone), 2 litres of 180 proof liquor, Overcoat (for everyone), A cosmetic mirror."""

guidelines = f"""You can choose to reply immediately or wait for someone else to reply.  
Listen actively to your coworkers, considering their reasoning and respecting diverse perspectives. 
Stay focused on the survival task, and avoid off-topic discussions. You have {turn_takings} turn-takings in total to reach consensus.
Communicate clearly and persuasively: if you believe an item should be ranked higher or lower, according to you individual ranking explain why, considering survival priorities in the desert environment. 
Aim for a collaborative and respectful conversation. The group decision is finalized when all members agree on a ranked list of items."""
#If you choose to wait for someone else to reply, please just reply 'thinking...' and wait for your next turn.

print("Breakpoint: setup done")

#------------------------------
# Defining the agent class  
#------------------------------

class Agent():
    def __init__(self, agent_name): #constructor
        self.name = agent_name
        if self.name == "High_stability":
            self.traits = High_stability
        if self.name == "High_plasticity":
            self.traits = High_plasticity
        if self.name == "High_both":
            self.traits = High_both
        if self.name == "Low_stability":
            self.traits = Low_stability
        if self.name == "Low_plasticity":
            self.traits = Low_plasticity
        if self.name == "Low_both":
            self.traits = Low_both
        if self.name == "Basic":
            self.traits = Basic        

    def get_agent_name(self):
        return self.name
        
    def assign_ranking(self, output):
        self.ranking = []
        self.ranking.append(output) #.split(','))
        if len(self.ranking) != 15: # Ensure we have exactly 15 elements where elements are equal to columns
            note = print(f"Warning: {self.name}'s ranking does not contain exactly 15 items.")
            saved_outputs.append(note)

    def instructions_system(self):
        return f"""
        Your name is {self.name}, and you have the following personality profile: {self.traits}.
        Please use this personality profile to guide your approach to the task.
        You do not have specialized knowledge about survival in a desert.
        The task has the following introduction: {introduction}.
        The items available are: {items}.
        """

    def instructions_user(self):
        return f"""
        First, you need to complete the individual task, with the following instructions: 
        Using your assigned personality traits, individually rank the items in order of importance for the team’s survival, where ‘1’ is the most crucial, and ‘15’ is the least. 
        Focus on survival needs, the harsh desert environment, distance from help, and each item's potential use. 
        Complete the task without any input from your co-workers. 
        Desired output: The items in ranked order separated by commas, and end with the statement: 'ranking_complete'.
        """
    def start_task_system(self):
        return f"""
        Now that you have completed your individual ranking, it is time to collaborate. 
        You work in a company, and today you and your five co-workers are tasked to engage in a team-building task unrelated to your work.
        None of you have specialized knowledge about survival in a desert.
        For this discussion, use these guidelines for collaboration: {guidelines}.
        Your name and personality profile remain: {self.name}, {self.traits}.
        Please use this personality profile to guide your behavior, communication style, and approach to the task.
        The task introduction remains: {introduction}.
        The items remain: {items}.        """

    def start_task_user(self):
        return f"""
        Now, you move on to the collaborative task with the following instructions: 
        You work in a company, and today you and your five co-workers are tasked to engage in a team-building task unrelated to your work.
        None of you have specialized knowledge about survival in a desert.
        Your goal as a team is now to collaboratively RANK the 15 items in order of importance for survival of the team, with ‘1’ being the most important and ‘15’ the least. 
        Work together to discuss and finalize a ranked list of the 15 items. This may involve negotiating and persuading others to consider your reasoning. 
        Your decision should still focus on survival, considering the extreme desert environment, distance from help, and the potential uses of each item.
        Desired output: please state 'I understand' and will wait for further instructions."    
        """

    def interactive_system(self):
        return f"""
        Your name and personality profile remain: {self.name}, {self.traits}.
        The task introduction remains: {introduction}.
        The items remain: {items}.
        These guidelines for collaboration remain: {guidelines}.
        """

    def interactive_user(self):
        return f"""
        Continue the collaborative ranking task discussion based on the previous context. Be aware that you have a maximum of {turn_takings} replies all together.
        Still, use your personality profile to guide your behavior, communication style, and approach to the task. 
        Desired output: when you have decided on a list, please state ‘This is our final list’ followed by the items in ranked order separated by commas, and end with the statement: 'ranking_complete'. 
        If there have been {turn_takings} replies in the discussion, and you still haven’t reached consensus on a finalized list, please state ‘Consensus not reached.’ and stop conversing.  
        """

print("Breakpoint: class Agent setup done")

# --------------------------------------------------------------
# Create assistant class
# --------------------------------------------------------------

class Assistant():
    def __init__(self, agent_name): #constructor
        self.assistant = client.beta.assistants.create(
            name=agent_name,
            temperature= temperature,
            model= defined_model,
        )
        self.thread = self.create_thread()
    
    def get_assistant_id(self):
        self.assistant = openai.beta.assistants.retrieve(self.assistant.id) # Retrieve the Assistant
        return self.assistant.id
    
    def create_thread(self):
        self.thread = client.beta.threads.create()
        return self.thread
    
    def get_thread_id(self):
        self.thread = openai.beta.threads.retrieve(self.thread.id)
        return self.thread.id
    
    def message(self, message_content):
        client.beta.threads.messages.create( #create input message
            thread_id=self.thread.id,
            role="user", #betyder at beskeden kommer udefra
            content= message_content
        ) 
    def run_assistant(self, instructions):
        run = client.beta.threads.runs.create_and_poll( #create run
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions= instructions
        )

        if run.status == 'completed': 
            messages = client.beta.threads.messages.list(
                thread_id=self.thread.id,
                order="desc"
            )
            self.message_text = messages.data[0].content
            return self.message_text
        else:
            print(f"""{self.assistant.name} is creating output message. Status: {run.status}""")

    def delete_assistant(self):
        client.beta.assistants.delete(self.assistant.id)

    def delete_thread(self):
        client.beta.threads.delete(self.thread.id)

print("Breakpoint: class Assistant setup done")

#------------------------------ 
# Defining the world class
#------------------------------

class World():
    def __init__(self): #constructor
        self.group_agents = []
        self.group_assistants = []  # Create a dictionary to store assistants

    def create_group(self):
        for i in range(TEAM_SIZE-1):
            new_agent = Agent("Basic")
            new_agent.name = new_agent.name+f"_{i+1}"
            self.group_agents.append(new_agent)
            
            new_assistant = Assistant(new_agent.get_agent_name())
            self.group_assistants.append(new_assistant)
        
        new_personality_agent = Agent(np.random.choice(agent_names))
        new_personality_agent.name = new_personality_agent.name+f"_{TEAM_SIZE}"
        self.group_agents.append(new_personality_agent)
        
        new_personality_assistant = Assistant(new_personality_agent.get_agent_name())
        self.group_assistants.append(new_personality_assistant)
        
        for i in range(TEAM_SIZE):
            agent = self.group_agents[i]
            assistant = self.group_assistants[i]
            saved_outputs.append(f"""Name: {agent.get_agent_name()}""")
            saved_outputs.append(f"""Assistant: {assistant.get_assistant_id()}""")  
        print(f"""Breakpoint: Group created with the following agents: {self.group_agents[0].name}, {self.group_agents[1].name}, and {self.group_agents[2].name}""")

    def run_once(self):        
        print("Starting first part: instructions")
        conversation_outputs.append("Starting first part: instructions")
        for i in range(TEAM_SIZE): #first part
            assistant = self.group_assistants[i]
            agent = self.group_agents[i]
            print(f"""Sending message number {i+1}""")
            assistant.message(agent.instructions_system())
            print(f"""Doing run number {i+1}""")
            output = assistant.run_assistant(agent.instructions_user())
            conversation_outputs.append(f"""Agent {agent.name} has responded: {output}""")
            print(f"""Assigning ranking number {i+1}""")
            agent.assign_ranking(output)
            saved_outputs.append(f"""Ranking for agent {i+1}: {agent.ranking}""")
        print("Breakpoint: done with instructions")

        print("Starting second part: start_task")
        conversation_outputs.append("Starting second part: start_task")
        for i in range(TEAM_SIZE): #second part
            assistant = self.group_assistants[i]
            agent = self.group_agents[i]
            assistant.message(agent.start_task_system())
            output = assistant.run_assistant(agent.start_task_user())
            conversation_outputs.append(f"""Agent {agent.name} has responded: {output}""")
        print("Breakpoint: done with start_task")
        
        output = None

        print("Starting third part: interactive_task")
        conversation_outputs.append("Starting interactive task:")

        i=0
        while i<turn_takings: #third part
            print(f"Starting interactive_task no. {i+1}")
            number = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2] #30 cases
            assistant = self.group_assistants[number[i]]
            assistant_other1 = self.group_assistants[number[i+1]]
            assistant_other2 = self.group_assistants[number[i+2]]
            agent = self.group_agents[number[i]]
            assistant.message(agent.interactive_system())
            output = assistant.run_assistant(agent.interactive_user())
            final_output=(f"""Agent {agent.name} has responded: {output}""")
            conversation_outputs.append(final_output)
            assistant_other1.message(final_output)
            assistant_other2.message(final_output)
            i+=1
        print("Breakpoint: done with interactive_task")

    def delete_assistants(self):
        for i in range(TEAM_SIZE):
            assistant = self.group_assistants[i]
            assistant.delete_thread()
            assistant.delete_assistant()
        print("Breakpoint: assistants deleted")

    def save_outputs(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(f"run_{current_time}.csv")    # Creating file
        root = os.getcwd()

        csv_path_outputs = root+ '\\' +output_directory_conversation+ '\\'  +filename
        df_conversation = pd.DataFrame(conversation_outputs)  # Create DataFrame
        df_conversation.to_csv(csv_path_outputs, index=False)  # Save DataFrame to CSV

        csv_path_saved = root+ '\\' +output_directory_saved+ '\\'  +filename
        df_saved = pd.DataFrame(saved_outputs)  # Create DataFrame
        df_saved.to_csv(csv_path_saved, index=False)  # Save DataFrame to CSV


#------------------------------ 
# Running simulations
#------------------------------
model = World()
model.create_group()
model.run_once()
model.save_outputs()
model.delete_assistants()    

print("Model run complete.")