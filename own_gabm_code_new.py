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
TEAM_SIZE = 3  
no_simulations = 2 # Adjust as necessary
turn_takings = 30 # Adjust as necessary
temperature = 0.8 # Adjust as necessary
defined_model = "gpt-4o-mini" # Adjust as necessary

simulation_no = 0
personality = None
agent_1_list = None
agent_2_list = None
agent_3_list = None
group_list = None
turn_takings_count = 0

#Capturing print statements
conversation_outputs = [] # Initialize a list to store individual outputs
final_dataframe = pd.DataFrame()

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
            model="gpt-4o-mini",
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

#Agents and assistants
assistants = {}
agents = {}
agent_names = ["High_neuro", "High_extra", "High_open", "High_agree", "High_consc", "Low_neuro", "Low_extra", "Low_open", "Low_agree", "Low_consc", "Basic"]

# Personality trait descriptions 
high_neuro = "You panic easily, doubt yourself, fear the worst, and worry often."
low_neuro = "You stay calm, confident, and unbothered, even under pressure."
base_neuro = "You’re calm but can feel mild stress; cautious without dwelling on risks."

high_extra = "You socialize easily, captivate others, and enjoy attention."
low_extra = "You stay quiet, avoid attention, and find socializing difficult."
base_extra = "You interact comfortably but avoid the spotlight."

high_open = "You’re very imaginative, love new ideas, and enjoy deep thinking."
low_open = "You prefer practical ideas and avoid abstract or theoretical discussions."
base_open = "You balance curiosity with routine and practical thinking."

high_agree = "You trust others, are very cooperate, and treat everyone equally."
low_agree = "You speak bluntly, hold grudges, and feel superior."
base_agree = "You’re cooperative but assertive when needed, valuing harmony and fairness."

high_consc = "You plan thoroughly, focus on details, and finish tasks on time."
low_consc = "You struggle with focus, do the minimum, and leave tasks unfinished."
base_consc = "You’re moderately organized, follow through on tasks, but may occasionally overlook details."

#Personality types 
High_extra = high_extra + base_consc + base_agree + base_neuro + base_open
High_neuro = high_neuro + base_consc + base_agree + base_extra + base_open
High_open = high_open + base_consc + base_agree + base_neuro + base_extra
High_agree = high_agree + base_consc + base_neuro + base_extra + base_open
High_consc = high_consc + base_agree + base_neuro + base_extra + base_open
Low_extra = low_extra + base_consc + base_agree + base_neuro + base_open
Low_neuro = low_neuro + base_consc + base_agree + base_extra + base_open
Low_open = low_open + base_consc + base_agree + base_neuro + base_extra
Low_agree = low_agree + base_consc + base_neuro + base_extra + base_open
Low_consc = low_consc + base_agree + base_neuro + base_extra + base_open

Basic = base_consc + base_agree + base_neuro + base_extra + base_open

#Prompts
introduction = f"""Imagine you have crash-landed in the Atacama Desert in mid-July. 
It's around 10:00 am, and the temperature will reach 110°F (43°C), but at ground level, it will feel like 130°F (54°C). 
Your group of {TEAM_SIZE} non-injured survivors has salvaged 15 items from the wreckage. 
Survival will depend on making the right choices. The desert is vast, and rescue is uncertain. 
Every item’s utility could be the difference between life and death."""

items = f"""Torch with 4 battery-cells, Folding knife, Air map of the area, Plastic raincoat (large size), Magnetic compass, First-aid kit, 45 calibre pistol (loaded), Parachute (red & white), Bottle of 1000 salt tablets, 2 litres of water per person, A book entitled ‘Desert Animals That Can Be Eaten’, Sunglasses (for everyone), 2 litres of 180 proof liquor, Overcoat (for everyone), A cosmetic mirror."""

guidelines = f"""
- **Active Listening**: Take turns listening actively to your teammates, understanding their arguments, and respecting their distinct perspectives.
- **Focus**: Please stay on topic and avoid irrelevant conversations. Your goal is staying alive, each choice counts.
- **Consensus Decision**: You have {turn_takings} combined turn-takings to reach a consensus on the ranking. Ensure that all team members understand and compromise on the final order.
- **Clear Communication**: While discussing the utility of each item, provide reasoning for your choices, especially where you think an item should move up or down the ranking. Ground your arguments in survival priorities unique to the conditions of the desert.
"""

print("Breakpoint: setup done")

#------------------------------
# Defining the agent class  
#------------------------------

class Agent():
    def __init__(self, agent_name): #constructor
        self.name = agent_name
        if self.name == "High_extra":
            self.traits = High_extra
        if self.name == "High_neuro":
            self.traits = High_neuro
        if self.name == "High_open":
            self.traits = High_open
        if self.name == "High_agree":
            self.traits = High_agree
        if self.name == "High_consc":
            self.traits = High_consc
        if self.name == "Low_extra":
            self.traits = Low_extra
        if self.name == "Low_neuro":
            self.traits = Low_neuro
        if self.name == "Low_open":
            self.traits = Low_open
        if self.name == "Low_agree":
            self.traits = Low_agree
        if self.name == "Low_consc":
            self.traits = Low_consc
        if self.name == "Basic":
            self.traits = Basic        

    def get_agent_name(self):
        return self.name
    
    def get_agent_traits(self):
        return self.traits

    def instructions_system_basic(self):
        return f"""
        {introduction}
        The items available are: {items}. 
        Use your assigned personality traits to individually rank the salvaged items in order of importance for the team’s survival, with 1 being the most crucial, and 15 is the least. 
        Focus on survival needs, the harsh desert environment, distance from help, and each item's potential use. 
        Complete the task without any input from your co-workers. Refrain from stating anything else than the desired output.
        #Output format: A ranked list of items separated by commas, ending with the statement 'ranking_complete'. An example of the format of a list is: " ['1. Torch with 4 battery-cells, 2. Folding knife, 3. Air map of the area, 4. Plastic raincoat (large size), 5. Magnetic compass, 6. First-aid kit, 7. 45 calibre pistol (loaded), 8. Parachute (red & white), 9. Bottle of 1000 salt tablets, 10. 2 litres of water per person, 11. A book entitled ‘Desert Animals That Can Be Eaten’, 12. Sunglasses (for everyone), 13. 2 litres of 180 proof liquor, 14. Overcoat (for everyone), 15. A cosmetic mirror. ranking_complete']"
        """
    
    def instructions_system_personality(self):
        return f"""
        {introduction}
        The items available are: {items}.
        Use your assigned personality traits, especially the first sentence herin, to individually rank the salvaged items in order of importance for the team’s survival, with 1 being the most crucial, and 15 is the least. 
        Focus on survival needs, the harsh desert environment, distance from help, and each item's potential use. 
        Complete the task without any input from your co-workers. Refrain from stating anything else than the desired output.
        #Output format: A ranked list of items separated by commas, ending with the statement 'ranking_complete'. An example of the format of a list is: " ['1. Torch with 4 battery-cells, 2. Folding knife, 3. Air map of the area, 4. Plastic raincoat (large size), 5. Magnetic compass, 6. First-aid kit, 7. 45 calibre pistol (loaded), 8. Parachute (red & white), 9. Bottle of 1000 salt tablets, 10. 2 litres of water per person, 11. A book entitled ‘Desert Animals That Can Be Eaten’, 12. Sunglasses (for everyone), 13. 2 litres of 180 proof liquor, 14. Overcoat (for everyone), 15. A cosmetic mirror. ranking_complete']"
        """

    def start_task_system(self):
        return f"""
        You work in a company, and today you and your {TEAM_SIZE} co-workers are tasked to engage in a team-building task unrelated to your work.
        None of you have specialized knowledge about survival in a desert.
        Your goal is to collaboratively rank the 15 items in order of importance for survival of the team in an extreme desert environment.  
        The ranking should be made from most critical (1) to least critical (15). 
        Discuss and negotiate to persuade others to consider your reasoning about the placement of the items. 
        Your decisions should be rational and focused on maximizing survival potential with regards to the given challenging context.
        Eventually, create the final team ranking based on group agreement, ensuring that everyone involved is satisfied with each item's placement.
        Your decision should still focus on survival, considering the extreme desert environment, distance from help, and the potential uses of each item.
        """

    def interactive_system_basic(self):
        return f"""
        Continue the collaborative ranking task discussion based on the previous context. Be aware that you have a maximum of {turn_takings} replies all of you together.
        The task introduction remains: {introduction}.
        The items remain: {items}.
        These guidelines for collaboration remain: {guidelines}.
        Use your personality profile to guide your behavior, communication style, and approach to the task. 
        # Output format: When you all agree on a finalized list containing all 15 items, please state "This is our final list:" followed by the items in ranked order separated by commas, and end with the statement ‘ranking_complete.’. An example of the format of a list is: "This is our final list: ['1. Torch with 4 battery-cells, 2. Folding knife, 3. Air map of the area, 4. Plastic raincoat (large size), 5. Magnetic compass, 6. First-aid kit, 7. 45 calibre pistol (loaded), 8. Parachute (red & white), 9. Bottle of 1000 salt tablets, 10. 2 litres of water per person, 11. A book entitled ‘Desert Animals That Can Be Eaten’, 12. Sunglasses (for everyone), 13. 2 litres of 180 proof liquor, 14. Overcoat (for everyone), 15. A cosmetic mirror. ranking_complete']"  
        """
    
    def interactive_system_personality(self):
        return f"""
        Continue the collaborative ranking task discussion based on the previous context. Be aware that you have a maximum of {turn_takings} replies all of you together.
        The task introduction remains: {introduction}.
        The items remain: {items}.
        These guidelines for collaboration remain: {guidelines}.
        Use your personality profile, especially the first sentence herein, to guide your behavior, communication style, and approach to the task. 
        # Output format: When you all agree on a finalized list containing all 15 items, please state "This is our final list:" followed by the items in ranked order separated by commas, and end with the statement ‘ranking_complete.’. An example of the format of a list is: "This is our final list: ['1. Torch with 4 battery-cells, 2. Folding knife, 3. Air map of the area, 4. Plastic raincoat (large size), 5. Magnetic compass, 6. First-aid kit, 7. 45 calibre pistol (loaded), 8. Parachute (red & white), 9. Bottle of 1000 salt tablets, 10. 2 litres of water per person, 11. A book entitled ‘Desert Animals That Can Be Eaten’, 12. Sunglasses (for everyone), 13. 2 litres of 180 proof liquor, 14. Overcoat (for everyone), 15. A cosmetic mirror. ranking_complete']"  
        """

print("Breakpoint: class Agent setup done")

# --------------------------------------------------------------
# Create assistant class
# --------------------------------------------------------------

class Assistant():
    def __init__(self, agent_name, agent_traits): #constructor
        self.assistant = client.beta.assistants.create(
            name=agent_name,
            temperature= temperature,
            model= defined_model,
            description= f"""Your name is {agent_name}, and you have the following personality profile: {agent_traits}.
            You do NOT have specialized knowledge about deserts."""
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
            self.message_text = messages.data[0].content[0].__getattribute__('text').__getattribute__('value') 
            self.message_text= str( self.message_text)
            return self.message_text 
        else:
            print(f"{self.assistant.name} is creating output message. Status: {run.status}")

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
        #creating the personality agent
        global personality
        non_basic_agent_names = [name for name in agent_names if name != "Basic"] # Filter out "Basic" from agent_names
        new_personality_agent = Agent(np.random.choice(non_basic_agent_names))
        new_personality_agent.name = new_personality_agent.name+f"_{TEAM_SIZE}"
        self.group_agents.append(new_personality_agent)
        personality = new_personality_agent.name
        
        new_personality_assistant = Assistant(new_personality_agent.get_agent_name(), new_personality_agent.get_agent_traits())
        self.group_assistants.append(new_personality_assistant)

        #creating the two basic agents
        for i in range(TEAM_SIZE-1):
            new_agent = Agent("Basic")
            new_agent.name = new_agent.name+f"_{i+1}"
            self.group_agents.append(new_agent)
            
            new_assistant = Assistant(new_agent.get_agent_name(), new_agent.get_agent_traits())
            self.group_assistants.append(new_assistant)
        
        print(f"Breakpoint: Group created with the following agents: {self.group_agents[0].name}, {self.group_agents[1].name}, and {self.group_agents[2].name}")

    def run_once(self):        
        print("Starting first part: instructions")
        conversation_outputs.append("Starting first part: instructions")

        #for personality agent
        assistant = self.group_assistants[0]
        agent = self.group_agents[0]
        print(f"Assigning ranking number {1}")
        output = assistant.run_assistant(agent.instructions_system_personality())
        conversation_outputs.append(f"Agent {agent.name} has responded: {output}")
        global agent_1_list
        agent_1_list = output

        #for the two basic agents
        for i in range(TEAM_SIZE-1): #first part
            assistant = self.group_assistants[i+1]
            agent = self.group_agents[i+1]
            print(f"Assigning ranking number {i+2}")
            output = assistant.run_assistant(agent.instructions_system_basic())
            conversation_outputs.append(f"Agent {agent.name} has responded: {output}")
            if i == 0:
                global agent_2_list
                agent_2_list = output
            if i == 1:
                global agent_3_list
                agent_3_list = output

        print("Breakpoint: done with instructions")

        print("Starting second part: start_task")

        for i in range(TEAM_SIZE): #second part
            assistant = self.group_assistants[i]
            agent = self.group_agents[i]
            assistant.message(agent.start_task_system())
        print("Breakpoint: done with start_task")
        
        output = None

        print("Starting third part: interactive_task")
        conversation_outputs.append("Starting interactive task:")

        i=0
        while i<turn_takings: #third part
            global turn_takings_count
            global group_list
            turn_takings_count = i+1
            print(f"Starting interactive_task no. {i+1}")
            number = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2] #39 cases
            assistant = self.group_assistants[number[i]]
            assistant_other1 = self.group_assistants[number[i+1]]
            assistant_other2 = self.group_assistants[number[i+2]]
            agent = self.group_agents[number[i]]
            #for the personality agent
            if number[i] == 0:
                output = assistant.run_assistant(agent.interactive_system_personality())            
            #for the two basic agents
            else:
                output = assistant.run_assistant(agent.interactive_system_basic())
            final_output=(f"Agent {agent.name} has responded: {output}")
            conversation_outputs.append(f"(Turn {i+1}) {final_output}")
            if output is not None and isinstance(output, str):
                if "ranking_complete" in output:
                    start = output.find("This is our final list:")
                    end = output.find("ranking_complete.")
                    group_list = output[start:end]
                    conversation_outputs.append(f"Note: Task finished as concensus was reached withing {i+1} turntakings.")
                    break            
            assistant_other1.message(final_output)
            assistant_other2.message(final_output)
            i+=1
        if i == turn_takings:
            conversation_outputs.append(f"Note: Task finished as concensus was not reached within {turn_takings} turntakings.")
            group_list = None
        print("Breakpoint: done with interactive_task")

    def delete_assistants(self):
        for i in range(TEAM_SIZE):
            assistant = self.group_assistants[i]
            assistant.delete_thread()
            assistant.delete_assistant()
        print("Breakpoint: assistants deleted")

    def save_outputs_conversation(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(f"run_{current_time}.csv")    # Creating file
        root = os.getcwd()

        csv_path_outputs = root+ '\\' +output_directory_conversation+ '\\'  +filename
        df_conversation = pd.DataFrame(conversation_outputs)  # Create DataFrame
        df_conversation.to_csv(csv_path_outputs, index=False)  # Save DataFrame to CSV

    def save_outputs_final(self):    
        global final_dataframe
        global simulation_no
        global personality
        global agent_1_list
        global agent_2_list
        global agent_3_list
        global group_list
        global turn_takings_count
        list ={
            "simulation_no.": sim_no,
            "personality": personality,
            "agent_1_list": agent_1_list,
            "agent_2_list": agent_2_list,
            "agent_3_list": agent_3_list,
            "group_list": group_list,
            "turn_takings": turn_takings_count
            }
        new_df = pd.DataFrame([list], columns=[
            "simulation_no.",
            "personality",
            "agent_1_list",
            "agent_2_list",
            "agent_3_list",
            "group_list",
            "turn_takings"
        ])
        final_dataframe = pd.concat([final_dataframe, new_df], ignore_index=True)

#------------------------------ 
# Running simulations
#------------------------------
for i in range(no_simulations):
    print(f"Starting model run {i+1} of {no_simulations}.")
    global sim_no
    sim_no = i+1
    conversation_outputs = [] # resetting output documents
    model = World()
    model.create_group()
    model.run_once()
    model.save_outputs_conversation()
    model.save_outputs_final()
    model.delete_assistants()
    print(f"Model run {i+1} of {no_simulations} complete.")   

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = os.path.join(f"run_{current_time}.csv")    # Creating file
root = os.getcwd()
csv_path_saved = root+ '\\' +output_directory_saved+ '\\'  +filename
final_dataframe.to_csv(csv_path_saved, index=False)  # Save DataFrame to CSV 