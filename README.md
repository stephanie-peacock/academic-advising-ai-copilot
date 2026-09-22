# Academic Advising AI Assistant

## Project Overview

This project explores how generative AI and a multi-agent architecture can be used to make complex academic advising information easier for students to access and understand.

The application combines a custom SQL data source with four specialized AI agents to interpret natural-language questions, determine the appropriate academic information to retrieve, generate SQL queries, and return clear, conversational answers.

Rather than relying on a single source of information, the project required building a structured academic advising database from multiple university documents and data sources.

## Data Preparation

A significant part of the project involved creating the SQL data source used by the AI application.

Academic information was collected, cleaned, standardized, and organized from multiple university resources, including:

- University course catalog information
- Academic advising resources and program requirements
- Academic-year class schedules

Information from these separate sources was integrated into a structured SQL database so that the AI agents could query academic information consistently.

This data preparation step was necessary because the information students need for academic planning is often distributed across multiple documents and resources rather than contained in a single structured dataset.

## Multi-Agent Architecture

The application uses four specialized AI agents rather than relying on a single model to perform the entire task.

The agents divide the workflow into distinct responsibilities, including:

### Data Selection and Routing
Interprets the student's question and determines which source of academic data — or combination of sources — is most appropriate for answering it.

### SQL Generation
Translates the interpreted request into SQL queries that retrieve the relevant information from the academic database.

### Response Generation
Transforms the retrieved information into a clear, conversational response appropriate for the student's question.

### Additional Agent
A fourth specialized agent supports the application workflow. [Description to be added.]

This architecture separates interpretation, data retrieval, and response generation into specialized tasks rather than asking a single AI model to perform the entire process.

## Tools & Skills

- Python
- SQL
- Generative AI / Large Language Models
- Multi-Agent AI Architecture
- Prompt Engineering
- Data Cleaning and Integration
- Relational Data Design
- Streamlit
- Natural-Language Interfaces

## Application

A deployed version of the Academic Advising AI Assistant is available through Streamlit:

[Launch the Academic Advising AI Assistant](https://academic-advising-ai-copilot-mvvkzubuqmehmwqoskzlu4.streamlit.app/)

> **Please note:** The application is hosted using Streamlit's shared hosting environment. If the application has been inactive, it may need to restart before opening. The initial load can take up to approximately five minutes.

## Project Repository

The repository contains the Python application, agent components, supporting data, environment configuration, and application dependencies.

## Project Purpose

The project demonstrates how unstructured and distributed organizational information can be transformed into a structured data source and paired with generative AI to create a more accessible user interface.

The goal was not simply to build a chatbot, but to create a workflow in which natural-language questions are interpreted, translated into structured database queries, and converted into useful responses.
