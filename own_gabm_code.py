# -*- coding: utf-8 -*-


# TITEL: Generalized ABM Performing a CPS-task with Teams Consisting of Different Personality Traits

# Setup code space
import pandas as pd
import numpy as np
import os
import shutil
import pprint
import json
import sys
from datetime import datetime
import time
from multiprocessing import Pool, Manager
from functools import partial

import random
from random import choice
import csv

import openai
from openai import OpenAI
from openai.types import ChatModel
import io


# """ Creating global variables and classes"""

# Constants
TEAM_SIZE = 5  # Adjust as necessary
no_simulations = 10
i=0

# Global Variables
openai_api_key = os.getenv('OPENAI_API_KEY')  # Use environment variable
openai.api_key = openai_api_key # Set the API key
if openai_api_key:
    print("API Key is set.")
else:
    print("API Key is not set. Please check your environment variables.")
    try:
    # Make a simple API call to check connectivity
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print("API is reachable. Response:", response)
    
    except Exception as e:
        print("Error connecting to the API:", e)

agent_names = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
output_directory = "output_files_gabm"
output_file = ""

# Personality trait descriptions
high_neuro = "Easily feel panic, often doubt things, feel vulnerable to threats, get stressed quickly, fear the worst, and often worry. I take active steps to mitigate risks."
low_neuro = "Feel confident, am rarely irritated, remain unbothered, seldom get angry, and stay calm under pressure."

high_extra = "Am skilled at socializing, captivate people, start conversations easily, enjoy being the center of attention, and cheer people up."
low_extra = "Prefer to remain in the background, keep quiet, avoid attention, don’t engage much, and find it difficult to approach others."

high_open = "Have a vivid imagination, elevate conversations, enjoy new ideas, and get excited by thinking deeply."
low_open = "Prefer practical over abstract ideas, rarely seek deeper meaning, and have difficulty with theoretical discussions."

high_agree = "Have a kind word for everyone, assume good intentions, respect others, trust people, and treat everyone equally."
low_agree = "Speak directly, hold grudges, feel superior, and often contradict others."

high_consc = "Carefully consider all factors, pay attention to details, get tasks done immediately, plan thoroughly, and follow through on tasks."
low_consc = "Struggle to focus, do just enough to get by, lose interest quickly, and often leave tasks unfinished."


#Personality types
type1 = high_neuro+low_extra+high_open+low_agree+high_consc
type2 = low_neuro+high_extra+high_open+low_agree+high_consc
type3 = high_neuro+high_extra+low_open+low_agree+high_consc
type4 = high_neuro+low_extra+low_open+high_agree+low_consc
type5 = high_neuro+high_extra+high_open+low_agree+low_consc
type6 = low_neuro+high_extra+low_open+high_agree+low_consc
type7 = high_neuro+low_extra+low_open+high_agree+low_consc
type8 = low_neuro+low_extra+high_open+high_agree+low_consc
type9 = low_neuro+high_extra+high_open+low_agree+high_consc
type10 = high_neuro+low_extra+low_open+high_agree+high_consc

#Prompts
introduction = """Imagine you have crash-landed in the Atacama Desert in mid-July. 
It's around 10:00 am, and the temperature will reach 110°F (43°C), but at ground level, it will feel like 130°F (54°C). 
Your group of five non-injured survivors has salvaged 15 items from the wreckage. 
Survival will depend on making the right choices. The desert is vast, and rescue is uncertain. 
Every item’s utility could be the difference between life and death."""
items = """Torch with 4 battery-cells, Folding knife, Air map of the area, Plastic raincoat (large size), Magnetic compass, First-aid kit, 45 calibre pistol (loaded), Parachute (red & white), Bottle of 1000 salt tablets, 2 litres of water per person, A book entitled ‘Desert Animals That Can Be Eaten’, Sunglasses (for everyone), 2 litres of 180 proof liquor, Overcoat (for everyone), A cosmetic mirror."""
instruction_individual = """Individually, rank the items in order of importance for the team’s survival, where ‘1’ is the most crucial, and ‘15’ is the least. 
Focus on survival needs, the harsh desert environment, distance from help, and each item's potential use. 
Rank them without any input from your co-workers. 
Once finished, state ‘This is my individual list:’ followed by the ranked items, separated by commas."""
instruction_collaborative = """Your goal as a team is now to collaboratively RANK the 15 items in order of importance for survival of the team, with ‘1’ being the most important and ‘15’ the least. 
Work together to discuss and finalize a ranked list of the 15 items. This may involve negotiating and persuading others to consider your reasoning. 
Your decision should still focus on survival, considering the extreme desert environment, distance from help, and the potential uses of each item."""
guidelines = """There are no turn-taking rules: if you have something to say, speak up. 
Listen actively to others, considering their reasoning and respecting diverse perspectives. 
Stay focused on the survival task, and avoid off-topic discussions. 
Communicate clearly and persuasively: if you believe an item should be ranked higher or lower, explain why, considering survival priorities in the desert environment. 
Aim for a collaborative and respectful conversation. The group decision is finalized when all members agree on a ranked list of items.
When you have decided on a list, please state ‘This is our final list’ followed by the items in ranked order seperated by commas, and stop conversing. 
If there have been 30 turn-takings in the discussion, and you still haven’t reached consensus on a finalized list, please state ‘Consensus not reached.’ and stop conversing."""

# Ensure the directory exists   
output_directory = "output_files_gabm"     
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

#Capturing print statements
output_capture = io.StringIO()  # Create a StringIO object to capture output
printed_outputs = [] # Initialize a list to store individual outputs
sys.stdout = output_capture  # Redirect stdout to the StringIO object
sys.stdout = sys.__stdout__  # Reset stdout to its default value

print("Breakpoint: setup done")

# Defining the agent class  
class Agent():
    def __init__(self): #constructor
        self.ranking = []
        self.name = np.random.choice(agent_names)
        if self.name == "T1":
            self.traits = type1
        if self.name == "T2":
            self.traits = type2
        if self.name == "T3":
            self.traits = type3
        if self.name == "T4":
            self.traits = type4
        if self.name == "T5":
            self.traits = type5
        if self.name == "T6":
            self.traits = type6
        if self.name == "T7":
            self.traits = type7
        if self.name == "T8":
            self.traits = type8
        if self.name == "T9":
            self.traits = type9
        if self.name == "T10":
            self.traits = type10


    def get_output_from_chatgpt(self, messages, 
                            model ="gpt-4o-mini", temperature=0): #method
        # temperature is the degree of randomness of the model's output
        success = False
        retry = 0
        max_retries = 10
        response = None  # Initialize response to None

        while retry < max_retries and not success:
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                success = True
            except Exception as e:
                print(f"Error: {e}\nRetrying ...")
                retry += 1
                time.sleep(0.5)

        if response is not None:  # Check if response was successfully obtained
            sys.stdout = output_capture  # Redirect stdout to the StringIO object
            return response.choices[0].message.content #The first answer generated
        else:
            print("Failed to get a response after multiple retries.")
            return None  # Return None or handle the error as needed

    
    def assign_ranking(self, output):
        # Find the list of ranked items in the output
        start_phrase = "This is my individual list:"
        if start_phrase in output:
            output.replace(start_phrase, '')
            self.ranking.append(output.split(','))
        else:
            sys.stdout = output_capture  # Redirect stdout to the StringIO object
            print(f"Error: Could not find the ranking list in {self.name}'s output.")
            self.ranking.append('no presented ranking')
            sys.stdout = sys.__stdout__  # Reset stdout to its default value
        if len(self.ranking[0]) != 15: # Ensure we have exactly 15 elements where elements are equal to columns
            sys.stdout = output_capture  # Redirect stdout to the StringIO object
            print(f"Warning: {self.name}'s ranking does not contain exactly 15 items.")
            sys.stdout = sys.__stdout__  # Reset stdout to its default value
    
    def instructions(self):
        # Define the system message (personality and scenario setup)
        instruct_system_prompt = f"""
        Your name is {self.name}, and you have the following personality profile: {self.traits}.
        You work in a company, and today you and your five co-workers are tasked to engage in a team-building task unrelated to your work.
        None of you have specialized knowledge about survival in a desert.
        The task has the following introduction: {introduction}.
        The items available are: {items}.
        """
    
        # Define the user message (individual ranking instructions)
        instruct_user_prompt = f"""
        Based on the personality trait descriptions, use your assigned personality profile to guide your approach to the following tasks.
        First, you need to complete the individual task, with the following instructions: {instruction_individual}.
        When you have completed this task, wait for new instructions.
        """

        messages = [
            {'role': 'system', 'content': instruct_system_prompt},
            {'role': 'user', 'content': instruct_user_prompt}]
        try:
            output = self.get_output_from_chatgpt(messages)            
            self.assign_ranking(output) #Saves the ranked items in a list
        except Exception as e: #If the above "try" fails
            print(f"{e}\nProgram paused . Retrying after 60 s ...")
            time.sleep(60)
            output = self.get_output_from_chatgpt(messages)
        sys.stdout = output_capture  # Redirect stdout to the StringIO object
        print(f"{self.name}s personal list: {output}") #Prints answer
        sys.stdout = sys.__stdout__  # Reset stdout to its default value

    def start_task(self) : #method
        # System setup (scenario and task context)
        start_task_system_prompt = f"""
        Now that you have completed your individual ranking, it is time to collaborate.
        Your name and personality profile remain: {self.name}, {self.traits}.
        The task introduction remains: {introduction}.
        The items remain: {items}.
        You can choose to reply immediately or wait for someone else to reply. 
        If you choose to wait for someone else to reply, please do not reply anything.
        """
    
        # User instructions (collaborative task and guidelines)
        start_task_user_prompt = f"""
        Now you move on to the collaborative task with the following instructions: {instruction_collaborative}.
        For this discussion, use these guidelines for collaboration: {guidelines}.
        Here is a resume of the discussion so far: "{printed_outputs}".
        """

        messages = [
            {'role': 'system', 'content': start_task_system_prompt},
            {'role': 'user', 'content': start_task_user_prompt}]
        try:
            output = self.get_output_from_chatgpt(messages)  # Get messages from agent
            sys.stdout = sys.__stdout__  # Reset stdout to its default value
        except Exception as e:
            print(f"{e}\nProgram paused . Retrying after 60 s ...")
            time.sleep(60)
            output = self.get_output_from_chatgpt(messages)
            sys.stdout = sys.__stdout__  # Reset stdout to its default value
        sys.stdout = output_capture  # Redirect stdout to the StringIO object
        print(f"{self.name}s response: {output}") #Prints answer
        sys.stdout = sys.__stdout__  # Reset stdout to its default value

    def interactive_task(self) : #method
        global i
        # System setup (discussion context)
        interactive_system_prompt = f"""
        Continue the collaborative ranking task discussion based on the previous context.
        Your name and personality profile remain: {self.name}, {self.traits}.
        The task introduction remains: {introduction}.
        The items remain: {items}.
        You can choose to reply immediately or wait for someone else to reply. 
        If you choose to wait for someone else to reply, please do not reply anything.
        """

        # User instructions (discussion guidelines)
        interactive_user_prompt = f"""
        Here is a resume of the discussion so far: "{printed_outputs}".       
        """
        messages = [
            {'role': 'system', 'content': interactive_system_prompt},
            {'role': 'user', 'content': interactive_user_prompt}
        ]
        try:
            output = self.get_output_from_chatgpt(messages)  # Get messages from agent
            sys.stdout = sys.__stdout__  # Reset stdout to its default value
        except Exception as e:
            print(f"{e}\nProgram paused . Retrying after 60 s ...")
            time.sleep(60)
            output = self.get_output_from_chatgpt(messages)
            sys.stdout = sys.__stdout__  # Reset stdout to its default value
        sys.stdout = output_capture  # Redirect stdout to the StringIO object
        if output is not None: 
            print(f"{self.name}s response: {output}") #Prints answer
            final_phrase = "This is our final list:"
            if final_phrase in output:
                print(f"This is the groups final list: {output.split(final_phrase)[1].split(',')}") #Prints answer
                print("Terminating as task finished.")
                i = 100
        i = i + 1
        sys.stdout = sys.__stdout__  # Reset stdout to its default value

print("Breakpoint: class Agent setup done")

# Defining the world class
class World():
    def __init__(self): #constructor
        self.agent_list = []
        self.agent_names = []
        global i
    def run_once(self):
        # Creating file 
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(f"run_{current_time}.csv")        
        for t in range(TEAM_SIZE):
            this_agent = Agent()
            this_agent.name = this_agent.name+f"_{t+1}"
            self.agent_list.append(this_agent)
            self.agent_names.append(this_agent.name)
            self.agent_names.append(this_agent.ranking)
        sys.stdout = output_capture  # Redirect stdout to the StringIO object
        print(f"Simulation with personalities: {self.agent_names}.")
        sys.stdout = sys.__stdout__  # Reset stdout to its default value
        for agent in self.agent_list:
            agent.instructions()  # Get messages from agent
            printed_outputs.append([output_capture.getvalue().strip()])  # Append the captured output to the list
            printed_outputs.append([agent.ranking])
            # Reset the output capture for the next print
            output_capture.truncate(0)  # Clear the StringIO object
            output_capture.seek(0)  # Move to the start of the StringIO object
        print("done with instructions")
        for agent in self.agent_list:
            agent.start_task()  # Get messages from agent
            printed_outputs.append([output_capture.getvalue().strip()])  # Append the captured output to the list
            # Reset the output capture for the next print
            output_capture.truncate(0)  # Clear the StringIO object
            output_capture.seek(0)  # Move to the start of the StringIO object   
        print("done with start_task")
        while i<20:
            for agent in self.agent_list:      
                print(f"Starting interactive_task no. {i+1}")
                agent.interactive_task()  # Get messages from agent
                printed_outputs.append([output_capture.getvalue().strip()])  # Append the captured output to the list
                # Reset the output capture for the next print
                output_capture.truncate(0)  # Clear the StringIO object
                output_capture.seek(0)  # Move to the start of the StringIO object
                if i > 20:
                    print("Terminating as they agreed on list")
                    return  # Use return to exit the function if i > 10
        root = os.getcwd()
        csv_path = root+ '\\' +output_directory+ '\\'  +filename
        df_outputs = pd.DataFrame(printed_outputs)  # Create DataFrame
        df_outputs.to_csv(csv_path, index=False)  # Save DataFrame to CSV
        
         
    def run_many_times(self):
        for i in range (no_simulations+1):
            for i in range(TEAM_SIZE):
                this_agent = Agent()  
                self.agent_list.append(this_agent)
            print(f"Simulation {i} with personalities: {self.agent_list}.")
            if i == no_simulations:
                break
            for agent in self.agent_list:
                message_ranking = agent.instructions()  # Get messages from agent
                agent.get_output_from_chatgpt(message_ranking) #method to call chat
            for agent in self.agent_list:
                message_go = agent.start_task()  # Get messages from agent
                agent.get_output_from_chatgpt(message_go) #method to call chat

print("Breakpoint: class World setup done")

#"""## Running simulations"""
model = World()
model.run_once()

print("Model run complete.")






