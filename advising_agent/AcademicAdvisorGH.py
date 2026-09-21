# ─────────────────────────────────────────────────────────────────────────────
# Academic Advising Multi-Agent App
# BANA 6910 Capstone Project
# Presented May 14th, 2026
# Team Members: Angela Crabtree, Tafadzwa Mutandiko, Briana Palencia, Stephanie Peacock
# ─────────────────────────────────────────────────────────────────────────────

# ── ABOUT THIS APP ────────────────────────────────────────────────────────────
# Academic Advising AI Copilot
# A multi-agent conversational assistant for Business School students.
# Powered by LangGraph and GPT-4o, the app routes student questions across
# three specialized SQL agents:
#   - Course Catalog: course details, prerequisites, credits, and history
#   - Class Schedule: times, locations, instructors, and modality by semester
#   - Degree Requirements: major and minor program requirements and policies
# Built with Streamlit. Requires a connection to AdvisingData_6910.db.
# ─────────────────────────────────────────────────────────────────────────────

# streamlit run advising_agent/AcademicAdvisorGH.py

# Sample Catalog questions to try:
# "What departments are in the Business School?"
# "What courses does the Computer Science department offer?"
# "Which courses are repeatable?"
# "What are all the 1-credit courses in the Business School?"
# "What are the prerequisites for ACCT 2200?"
# "How many courses does each college offer? Show me the top 5."

# Sample schedule questions:
# "What location is class 2200"
# "What courses are offered on the subject business"
# "Which courses are combined?"
# "What is the name of the latest class at night"

# Sample degree requirements questions:
# Sample major requirements questions:
# "What are the main courses for Accounting?"
# "What are the main courses for Finance?"
# "How many courses are linked to the Marketing program?"
# "Which courses belong to multiple majors?"
# "What core courses are in Finance?"
# "Which majors include ACCT 2200?"
# "Show me the courses linked to Sports Business."

# Sample minor requirements questions:
# "What are the requirements for Business Analytics Minor?"
# "What electives are in the Finance Minor?"
# "How many electives are required for the RMI Minor?"
# "What prerequisites are needed for the Business Analytics Minor?"
# "What is the total credit requirement for Business Fundamentals Minor?"


# ── IMPORTS ──────────────────────────────────────────────────────────────────

import streamlit as st                                          # ADDED: Streamlit UI framework

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langgraph.graph import END, StateGraph

import re
import os
import yaml
import re
from pprint import pprint
from typing import List, TypedDict, Literal

import pandas as pd
import sqlalchemy as sql

# Available academic years in the database
CURRENT_YEAR      = "2024-2025"
HISTORICAL_YEARS  = ["2021-2022", "2022-2023", "2023-2024"]

_COURSE_COLUMN_DOCS = """
    Key columns in the 'courses' table:
    - college           : the college or school (e.g. "Business School")
    - department        : the department name (e.g. "Accounting") NOT A CODE
    - dept_code         : short department code (e.g. "ACCT")
    - course_code       : full course code (e.g. "ACCT 2200")
    - course_number     : just the number (e.g. "2200")
    - course_name       : full course title
    - credits_min       : minimum credits
    - credits_max       : maximum credits
    - description       : full course description text
    - prerequisites     : prerequisite text (free text)
    - grading_basis     : e.g. "Letter Grade"
    - typically_offered : when the course is usually offered
    - restrictions      : enrollment restrictions
    - cross_listed      : cross-listed course codes
    - repeatable        : 1 if the course can be taken multiple times, 0 if not
    - max_credits       : max total credits if repeatable
    - academic_year     : the catalog year for this row (e.g. "2024-2025")
"""


# ── AI SETUP ─────────────────────────────────────────────────────────────────

# os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('credentials.yml'))['openai']

from dotenv import load_dotenv
load_dotenv()  # loads variables from a local .env file, if present (does nothing on Streamlit Cloud)

api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error("OpenAI API key not found. Set OPENAI_API_KEY as an environment variable, "
              "in a local .env file, or in Streamlit secrets.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(model="gpt-4o")


# ── STREAMLIT PAGE CONFIG ─────────────────────────────────────────────────────
# ADDED: Must be the first Streamlit call in the script

st.set_page_config(page_title="Academic Advising AI Copilot")
# st.session_state.clear() # Use to update text on the streamlit intro view
st.title("🎓 Academic Advising AI Copilot")
st.markdown("""
    I’m here to help guide you through your academic program by bringing together course catalog details, class schedules, 
    and degree requirements in one place so you can easily plan, track, and make informed decisions about your business school journey.
""")

# Sidebar placeholders — updated after each question
agent_label  = st.sidebar.empty()
sql_label    = st.sidebar.empty()
debug_label  = st.sidebar.empty()

# ── AGENTS ───────────────────────────────────────────────────────────────────

routing_preprocessor_prompt = PromptTemplate(
    template="""
    You are an expert in routing decisions for three main SQL database agent areas:
    a class schedule advising agent, a course catalog advising agent, and a degree requirements advising router.

    Your job is to take the user's initial question, format it for the downstream SQL agents,
    and determine which main area should handle the question.

    If the question is a follow-up (e.g. "What are they?", "Tell me more", "Which ones?"),
    use the recent conversation history below to understand what the user is referring to,
    and rewrite the question as a fully self-contained question before routing it.

    Recent conversation history:
    {chat_history}    

    The three main areas are:

    1. Class Schedule Advising Agent:
       Specializes in class schedules for a specific semester (e.g. Fall 2025).
       Use this when the question asks about:
       - class times, locations, buildings, rooms
       - instructor names or emails
       - modality (in person, online, hybrid, remote)
       - which classes are offered in a specific term or semester
       - filtering courses by time of day (e.g. "after 12pm", "before 9am", "evening classes")
       - combined sections, session length (8-week, 16-week)
       Code=schedule

    2. Course Catalog Router:
       Specializes in course descriptions, prerequisites, credits, and general course information
       that is NOT tied to a specific semester or schedule.
       Use this when the question asks about:
       - course descriptions or details
       - prerequisites or restrictions
       - how many credits a course is worth
       - whether a course is repeatable
       - what courses a department offers in general
       Code=catalog

    3. Degree Requirements Router:
       Handles questions about major requirements, minor requirements, electives,
       prerequisites, GPA rules, total credits, and academic policies.
       Code=degreereq

Important routing rules:
    - If the question mentions a specific semester (Fall 2025, Spring 2026, etc.) OR asks about
      times, locations, or instructors → ALWAYS route to schedule.
    - If the question asks about a major → route to degreereq.
    - If the question asks about a minor → route to degreereq.
    - Do NOT decide major vs minor here. That will happen inside the degree requirements router.
    - Do NOT decide current vs historical courses here. That will happen inside the catalog router.

    INITIAL_USER_QUESTION: {initial_question}

    Respond ONLY with a valid JSON object in this exact format, with no explanation or markdown:
    {{
        "routing_preprocessor_decision": "<schedule|catalog|degreereq>",
        "formatted_user_question_sql_only": "<rewritten or original question>"
    }}
    """,
    input_variables=["initial_question", "chat_history"]
)

routing_preprocessor = routing_preprocessor_prompt | llm | JsonOutputParser()
response_prompt = PromptTemplate(
    template="""
    You are a helpful academic advising assistant. A student asked a question and you ran a SQL query to get the data.

    The question was handled by the {agent_type} agent.

    Your job is to answer the student's question clearly using the data provided.
    - If the data is a list of courses or records, present it in a clean,
      readable way using a short table or bullet list if helpful.
    - If the data contains a single value or simple fact, answer conversationally.
    - Always be concise and directly answer what was asked.
    - If the data came from a historical year, make that clear in your response.
    - Never expose raw column names or SQL jargon to the student.

    Use this to guide your response format:
    - catalog: focus on course details like credits, prerequisites, and descriptions.
      If the data spans multiple academic years, note which year each row is from.
    - schedule: focus on time, location, modality, and instructor details.
      Present times in a readable format (e.g. "Mon/Wed 10:00–11:15 AM").
    - major: present courses grouped by category (Core, Business Core, Major Requirements).
      Use a structured list or table.
    - minor: present courses grouped by requirement type (Prerequisite, Required, Elective, Capstone).
      Always include how many electives/capstones must be chosen if selection_rule is present.  

    STUDENT QUESTION: {question}

    DATA FROM DATABASE:
    {data}

    Your response:
    """,
    input_variables=["question", "data"]
)

response_synthesizer = response_prompt | llm | StrOutputParser()

def synthesize_response(state):
    question   = state.get("user_question")
    data       = state.get("data")
    decision   = state.get("routing_preprocessor_decision", "catalog")
    data_str   = pd.DataFrame(data).to_string(index=False)

    if decision == "degreereq":
        agent_type = "minor" if state.get("sql_query_minor") else "major"
    else:
        agent_type = decision  # "catalog" or "schedule"

    response = response_synthesizer.invoke({
        "question":   question,
        "data":       data_str,
        "agent_type": agent_type
    })
    return {"response": response}

# ── DATABASE ──────────────────────────────────────────────────────────────────

#PATH_DB = "sqlite:///AdvisingMultiAgentV4/data/AdvisingData_6910.db"
PATH_DB = "sqlite:///data/AdvisingData_6910.db"

db = SQLDatabase.from_uri(PATH_DB)
sql_engine = sql.create_engine(PATH_DB)
conn = sql_engine.connect()

# ── SQL HELPERS ───────────────────────────────────────────────────────────────

def extract_sql_code(text):
    """Pull SQL out of markdown code fences if the LLM wraps it that way."""
    sql_code_match = re.search(r'```sql(.*?)```', text, re.DOTALL)
    if sql_code_match:
        return sql_code_match.group(1).strip()
    else:
        sql_code_match = re.search(r"sql(.*?)'", text, re.DOTALL)
        if sql_code_match:
            return sql_code_match.group(1).strip()
        return None


class SQLOutputParser(BaseOutputParser):
    """Strips markdown fences from LLM SQL output."""
    def parse(self, text: str):
        sql_code = extract_sql_code(text)
        return sql_code if sql_code is not None else text


# ── LANGGRAPH STATE ───────────────────────────────────────────────────────────

class Graph_State(TypedDict):
    user_question: str
    formatted_user_question_sql_only: str
    routing_preprocessor_decision: str
    query_scope: str          # "current" | "historical"
    prompt_sqlite_catalog_current: str
    prompt_sqlite_catalog_historical: str
    prompt_sqlite_schedule: str
    prompt_sqlite_degreereq: str
    sql_query_catalog: str
    sql_query_schedule: str
    sql_query_major: str
    sql_query_minor: str
    data: dict
    response: str
    num_steps: int


# ── GRAPH NODES ───────────────────────────────────────────────────────────────

def preprocess_routing(state):
    print("---ROUTER---")
    question = state.get("user_question")
    num_steps = state.get("num_steps")
    num_steps += 1

    # If the question contains a YYYY-YYYY academic year pattern, it's always catalog
    if re.search(r'20\d{2}-20\d{2}', question) and not any(
        kw in question.lower() for kw in ["time", "location", "instructor", "building", "room"]
    ):
        print("Decision: catalog (academic year pattern detected)")
        return {
            "formatted_user_question_sql_only": question,
            "routing_preprocessor_decision": "catalog",
            "num_steps": num_steps
        }

    chat_history = ""
    response = routing_preprocessor.invoke({"initial_question": question, "chat_history": chat_history})
    formatted_user_question_sql_only = response['formatted_user_question_sql_only']
    routing_preprocessor_decision = response['routing_preprocessor_decision']

    print("Decision: " + str(routing_preprocessor_decision))

    return {
        "formatted_user_question_sql_only": formatted_user_question_sql_only,
        "routing_preprocessor_decision": routing_preprocessor_decision,
        "num_steps": num_steps
    }


def catalog_current_agent(state):
    print("---CATALOG CURRENT AGENT---")
    question = state.get("formatted_user_question_sql_only")

    prompt_sqlite_catalog_current = PromptTemplate(
        input_variables=['input', 'table_info', 'top_k'],
        template=f"""
        You are a SQLite expert working with an academic database table named course_catalog.
        Given an input question, create a syntactically correct SQLite query to answer it.

          IMPORTANT: This question is about the CURRENT course catalog. You MUST
        filter every query by academic_year. The value to filter on is: {CURRENT_YEAR}
        Use a standard single-quoted string literal in your SQL, for example:
        WHERE "academic_year" = '2024-2025'
        Do not include rows from any other academic year.

        Do not use a LIMIT clause with {{top_k}} unless the user specifies a limit.
        Return SQL in ```sql ``` format.
        Only return a single query if possible.
        Never query for all columns unless the user asks. Only select the columns
        needed to answer the question. Wrap each column name in double quotes (") as delimited identifiers.
        Pay attention to use only the column names visible in the tables below.
        Do not query for columns that do not exist.

        Key columns in the 'course_catalog' table:
        {_COURSE_COLUMN_DOCS}

        Only use the following tables:
        {{table_info}}

        Question: {{input}}
        """
    )

    if question is None:
        question = state.get("user_question")

    sql_generator_catalog_current = (
        create_sql_query_chain(
            llm=llm,
            db=db,
            k=int(1e7),
            prompt=prompt_sqlite_catalog_current
        )
        | SQLOutputParser()
    )

    sql_query_catalog_current = sql_generator_catalog_current.invoke({"question": question,"routing_descision":"current"})
    print(f"  Generated SQL:\n{sql_query_catalog_current}")
    return {"sql_query_catalog": sql_query_catalog_current}



def catalog_historical_agent(state):
    print("---CATALOG HISTORICAL AGENT---")
    question = state.get("formatted_user_question_sql_only")
    if question is None:
        question = state.get("user_question")

    # Extract the year the user mentioned, checking both formatted and original question
    year_match = re.search(r'20\d{2}-20\d{2}', question)
    if not year_match:
        year_match = re.search(r'20\d{2}-20\d{2}', state.get("user_question", ""))

    if year_match:
        year_instruction = f"""
        CRITICAL: The user explicitly asked about the academic year {year_match.group()}.
        You MUST filter by: WHERE "academic_year" = '{year_match.group()}'
        Do NOT use any other academic year.
        """
    else:
        year_instruction = f"""
        No specific year was mentioned. Query all relevant historical years or compare
        across years as needed. Available years: {HISTORICAL_YEARS + [CURRENT_YEAR]}.
        """

    prompt_sqlite_catalog_historical = PromptTemplate(
        input_variables=['input', 'table_info', 'top_k'],
        template=f"""
        You are a SQLite expert working with an academic course catalog database.
        Given an input question, create a syntactically correct SQLite query to answer it.

        {year_instruction}

        The database contains data for these academic years: {HISTORICAL_YEARS + [CURRENT_YEAR]}.
        The current/default academic year is {CURRENT_YEAR}.
        Always use standard single-quoted string literals in your SQL, for example:
            WHERE "academic_year" = '2023-2024'

        Do not use a LIMIT clause with {{top_k}} unless the user specifies a limit.
        Return SQL in ```sql ``` format.
        Only return a single query if possible.
        Never query for all columns unless the user asks. Only select the columns
        needed to answer the question. Wrap each column name in double quotes (")
        as delimited identifiers.
        Pay attention to use only the column names visible in the tables below.
        Do not query for columns that do not exist.

        {_COURSE_COLUMN_DOCS}

        Only use the following tables:
        {{table_info}}

        Question: {{input}}
        """
    )

    sql_generator_catalog_historical = (
        create_sql_query_chain(
            llm=llm,
            db=db,
            k=int(1e7),
            prompt=prompt_sqlite_catalog_historical
        )
        | SQLOutputParser()
    )

    sql_query_catalog_historical = sql_generator_catalog_historical.invoke({"question": question})
    print(f"  Generated SQL:\n{sql_query_catalog_historical}")
    return {"sql_query_catalog": sql_query_catalog_historical}



def schedule_agent(state):
    print("---SCHEDULE AGENT---")
    question = state.get("formatted_user_question_sql_only")

    prompt_sqlite_schedule = PromptTemplate(
        input_variables=['input', 'table_info', 'top_k'],
        template="""
        You are a SQLite expert working with the AdvisingData database view vw_class_schedule
        Given an input question, create a syntactically correct SQLite query to answer it.
        Only return a SQLite Query or empty string. Do not return anything else.

        Do not use a LIMIT clause with {top_k} unless the user specifies a limit.
        Return SQL in ```sql ``` format.
        Only return a single query if possible.
        Never query for all columns unless the user asks. Only select the columns
        needed to answer the question. Wrap each column name in double quotes (") as delimited identifiers.
        Pay attention to use only the column names visible in the view below.
        Do not query for columns that do not exist.

        Exact column names in the 'vw_class_schedule' view — use these exactly as written:
        - term          : semester (e.g. '2025 Fall')
        - subject       : subject name (e.g. 'Accounting')
        - subject_code  : 4-character code (e.g. 'BANA', 'ACCT')
        - catalog_number: course number (e.g. 2200)
        - class_section, class_number, course_title
        - campus        : DC=Denver Campus; EXSTD=Extended Studies
        - academic_level: UGRD=Undergraduate; GRAD=Graduate
        - session       : DMR=16-week; DMH=First 8-week; DMI=Second 8-week
        - component     : LEC=Lecture; INT=Internship; IND=Independent Study; SEM=Seminar
        - combined_section (C=Yes Combined), combined_sections, consent
        - start, end    : date (YYYY-MM-DD)
        - time_start     : display format AM/PM string (e.g. '12:30PM') — use for SELECT only
        - time_end       : display format AM/PM string (e.g. '01:45PM') — use for SELECT only
        - time_start_raw : 24-hour time string HH:MM:SS — use for filtering and comparisons
        - time_end_raw   : 24-hour time string HH:MM:SS — use for filtering and comparisons
        - pattern       : days of week (M, T, W, TH, F)
        - location, building, room
        - instructor    : stored as LastName,FirstName
        - instructor_email
        - modality      : P=In Person; OL=Online; IS=Independent Study; HY=Hybrid; R=Remote/ZOOM
        - academic_level_description, campus_description
        - IsCombined_description, component_description
        - modality_description, session_description

        Time filtering rules:
        - ALWAYS use time_start_raw and time_end_raw for WHERE clauses and comparisons.
        - ALWAYS use time_start and time_end for SELECT (display to user in AM/PM format).
        - Compare time_start_raw directly to a 24-hour time string.
        - Examples:
            After 12pm  : WHERE "time_start_raw" > '12:00:00'
            Before 9am  : WHERE "time_start_raw" < '09:00:00'
            At 6:30pm   : WHERE "time_start_raw" = '18:30:00'
        - Classes with time_start_raw = '00:00:00' have no fixed time (online/independent study).
          Exclude them from time-based queries: AND "time_start_raw" != '00:00:00'

        - Classes with time_start_raw = '00:00:00' have no fixed time (online/independent study).
          Exclude them from time-based queries: AND "time_start_raw" != '00:00:00'

        Note: When a student asks about "core" courses for a major, they typically mean 
        "Major Requirements" in the database, not the "Core" normalized_category which 
        contains general education courses like English and Math.

        Example: To find core Marketing courses in Fall 2025:
        SELECT s."term", s."subject", s."catalog_number", s."course_title"
        FROM vw_class_schedule s
        WHERE s."term" = '2025 Fall'
        AND CAST(s."subject_code" || ' ' || s."catalog_number" AS TEXT) IN (
            SELECT "course_code"
            FROM vw_major_course_detail
            WHERE "major_name" = 'Marketing'
            AND "normalized_category" = 'Major Requirements'
        )

        Only use the following tables:
        {table_info}

        Question: {input}
        """
    )

    if question is None:
        question = state.get("user_question")

    sql_generator_schedule = (
        create_sql_query_chain(
            llm=llm,
            db=db,
            k=int(1e7),
            prompt=prompt_sqlite_schedule
        )
        | SQLOutputParser()
    )

    sql_query_schedule = sql_generator_schedule.invoke({"question": question})
    print(f"  Generated SQL:\n{sql_query_schedule}")
    return {"sql_query_schedule": sql_query_schedule}


def major_agent(state):
    print("---MAJOR AGENT---")
    question = state.get("formatted_user_question_sql_only")

    prompt_sqlite_major = PromptTemplate(
        input_variables=["input", "table_info", "top_k"],
        template="""
        You are a SQLite expert working with a database about university majors, courses,
        and requirement categories.

        Given a user question, create one syntactically correct SQLite query that answers it.

        Important rules:
        - Return SQL in ```sql``` format.
        - Only return one query.
        - Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or PRAGMA.
        - Do not use LIMIT {top_k} unless the user explicitly asks for a limit.
        - Only select the columns needed to answer the question.
        - Wrap each column name in double quotes.
        - Use DISTINCT when needed to avoid duplicate course rows.

        Best tables/views to use:
        1. vw_major_course_detail
           Key columns: major_name, course_code, course_title, category_name,
           normalized_category, relationship_type, group_label,
           min_courses, max_courses, recommended_semester,
           fall_mode, spring_mode, summer_mode
        2. vw_course_major_summary
           Key columns: course_code, course_title, major_count, majors

        Other tables: majors, categories, courses, requirement_groups, major_course_map

        Selection rule guidance:
        - Use min_courses and max_courses from vw_major_course_detail or requirement_groups
          to determine how many courses are required within a group.
        - If min_courses == max_courses, all courses in that group are required.
        - If max_courses > min_courses, include both values so the response can state
          e.g. "choose 2 of the following 4 courses".
        - Always include group_label and normalized_category so results can be
          grouped clearly by section (Core, Business Core, Major Requirements, etc.).

        Only use the following schema information:
        {table_info}

        Question: {input}
        """,
    )

    if question is None:
        question = state.get("user_question")

    sql_generator_major = (
        create_sql_query_chain(
            llm=llm,
            db=db,
            k=int(1e7),
            prompt=prompt_sqlite_major
        )
        | SQLOutputParser()
    )

    sql_query_major = sql_generator_major.invoke({"question": question})
    print(f"  Generated SQL:\n{sql_query_major}")
    return {"sql_query_major": sql_query_major}


def minor_agent(state):
    print("---MINOR AGENT---")
    question = state.get("formatted_user_question_sql_only")

    prompt_sqlite_minor = PromptTemplate(
        input_variables=["input", "table_info", "top_k"],
        template="""
You are a SQLite expert working with a database about university minors, courses,
and requirement rules.

Given a user question, create one syntactically correct SQLite query that answers it.

IMPORTANT:
- This database also contains tables about majors, but you MUST ignore all major-related tables and views.
- Only use tables and views related to minors.
- Do NOT use:
  - vw_major_course_detail
  - vw_course_major_summary
  - majors
  - categories
  - courses
  - requirement_groups
  - major_course_map

Important rules:
- Return SQL in ```sql``` format.
- Only return one query.
- Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or PRAGMA.
- Do not use LIMIT {top_k} unless the user explicitly asks for a limit.
- Only select the columns needed to answer the question.
- Wrap each column name in double quotes.
- Use DISTINCT when needed to avoid duplicate course rows.
- Prefer the views below when they make the query simpler.

Best tables/views to use:
1. vw_minor_course_detail
   Helpful for:
   - requirements for a minor
   - prerequisite courses for a minor
   - required courses in a minor
   - elective options in a minor
   - counts of courses in a minor
   Main columns:
   - program_id
   - program_name
   - program_type
   - catalog_year
   - requirement_group
   - requirement_type
   - course_code
   - course_title
   - selection_rule
   - term_note
   - requirement_note
   - credits

2. minor_degree_programs
   Helpful for:
   - program-level details for a minor
   - catalog year
   - program description
   - GPA requirements
   - residency / eligibility / declaration notes
   - total credits

3. minor_degree_requirements
   Helpful for:
   - underlying requirement rows for each minor
   - custom filtering when needed

4. program_lookup
   Helpful for:
   - checking whether something is a major or a minor
   - matching a program name to its source table and id


Interpretation guidance:
- "program" here usually means a minor.
#DEFINITIONS FIRST (highest priority)
- If the user asks what a minor is, what it is about, or what it focuses on, use "program_description" from minor_degree_programs.
- For short definitional questions such as "What is the Information Systems Minor?" or "What is the Business Analytics Minor?", prefer "program_description" from minor_degree_programs instead of returning course requirements.
# REQUIREMENTS LOGIC
- If the user asks for requirements for a minor, prefer vw_minor_course_detail and include
  "requirement_group", "selection_rule", "course_code", and "course_title".
- For requirement list questions, do not return only "course_code" and "course_title" when
  "requirement_group" or "selection_rule" would help explain the requirement.
# BROAD / EVERYTHING QUESTIONS
- If the user asks broad questions such as "what are all the requirements", "tell me everything I need for this minor", "give me the full requirements", or similar, include both:
  - program-level information from minor_degree_programs
  - course requirement information from vw_minor_course_detail
- For broad requirement questions, do not return only course rows if program-level fields are available and useful.
- If the user asks for a full overview or complete information, include both:
  - "program_description" and other useful program-level fields from minor_degree_programs
  - course requirements from vw_minor_course_detail

Requirement categories:
- If the user asks for prerequisites, filter "requirement_group" = 'Prerequisite'.
- If the user asks for required courses, filter "requirement_group" = 'Required'.
- If the user asks for electives, filter "requirement_group" = 'Elective'.
- If the user asks for capstone requirements, filter "requirement_group" = 'Capstone'.
- If the user asks a generic question such as "tell me all the classes for this minor", "what are all the courses for this minor", "show me the full requirements", or similar, treat it as a full requirements breakdown.
- For full requirements breakdown questions, return "requirement_group", "selection_rule", "course_code", and "course_title".
- For full requirements breakdown questions, organize results so prerequisites appear first, then required courses, then electives, then capstone if applicable.
- For full requirements breakdown questions, do not return only a flat list of course names.

Counting vs rules:
- If the user asks how many courses are needed for a minor, count DISTINCT "course_code"
  unless they ask for a narrower category.
- If the user asks how many electives they are required to take, use the "selection_rule" field
  instead of counting all elective options.
- For elective requirements such as "choose 1", "choose 2", or "choose 3", return the required
  number from "selection_rule".
- If "selection_rule" is repeated across multiple rows, use DISTINCT so the rule is returned only once.
- Do not infer requirements from the number of rows returned when a rule field such as
  "selection_rule" is available.
- When "selection_rule" is 'all', it means the course is required within that requirement group.
- For Elective and Capstone requirements, always include "selection_rule" so the result clearly shows how many courses must be taken (e.g., "choose 1", "choose 2", or "all").
- Do not list only course names for Elective or Capstone groups; include the rule that indicates how many must be selected.


Program-level information (minor_degree_programs):
- If the user asks about GPA requirements, use "min_gpa_admission" or "min_gpa_graduation".
- If the user asks about declaration instructions, use "declaration_note".
- If the user asks about eligibility, prerequisite restrictions, or registration restrictions,
  use "eligibility_note".
- If the user asks about residency requirements, transfer credit, courses from other institutions,
  completing hours elsewhere, or how many hours may be completed at another institution,
  use "residency_note".
- If the user asks about total credits for a minor, use "total_credits".
- If the user asks about a general description of a minor, use "program_description".

Naming and shorthand:
- Users may refer to Risk Management and Insurance Minor as "RMI Minor".
- Common shorthand:
  - "RMI Minor" = "Risk Management and Insurance Minor"
- Do not assume abbreviations match "program_name" exactly; use full program names when querying.

Query construction guidance:
- For list questions, include enough columns to make the result understandable.
- For requirement questions, prefer returning structured fields instead of only a count when the
  rules matter.
- When returning grouped requirements, structure the query so each requirement_group retains its associated selection_rule.
- For Elective and Capstone groups, if multiple rows share the same "selection_rule", use DISTINCT where appropriate so the rule is not redundantly repeated.
- Ensure the query result allows the user to understand both the available course options and how many must be chosen for Elective and Capstone groups.
- For full requirements breakdown questions, order results by requirement group in this sequence when applicable:
  Prerequisite, Required, Elective, Capstone.
- After ordering by requirement group, order by course_code when appropriate.

Routing decision:
{routing_decision}

Use the routing decision to guide the query:
- full_requirements:
  return both:
  - program-level information from minor_degree_programs when available, including:
    "program_name",
    "catalog_year",
    "min_gpa_admission",
    "min_gpa_graduation",
    "residency_note",
    "eligibility_note",
    "declaration_note",
    "total_credits"
  - a structured requirements breakdown from vw_minor_course_detail with:
    "requirement_group",
    "selection_rule",
    "course_code",
    "course_title"
  - if possible, use a single query that joins minor_degree_programs to vw_minor_course_detail by "program_id"
- program_overview:
  return program-level information from minor_degree_programs, including when available:
  - "program_name"
  - "program_description"
  - "catalog_year"
  - "min_gpa_admission"
  - "min_gpa_graduation"
  - "residency_note"
  - "eligibility_note"
  - "declaration_note"
  - "total_credits"
  and include course requirements from vw_minor_course_detail when the user asks for full details or everything about the minor
- electives_rule:
  focus on elective rows and "selection_rule"
- capstone_rule:
  focus on capstone rows and "selection_rule"
- policy_note:
  prefer relevant note fields from minor_degree_programs such as "residency_note", "eligibility_note", or "declaration_note"
- simple_fact:
  return the smallest set of columns needed to answer directly
- general:
  use the best relevant minor tables or views

Only use the following schema information:
{table_info}

Question: {input}
"""
    )

    if question is None:
        question = state.get("user_question")

    sql_generator_minor = (
        create_sql_query_chain(
            llm=llm,
            db=db,
            k=int(1e7),
            prompt=prompt_sqlite_minor,
        )
        | SQLOutputParser()
    )

    sql_query_minor = sql_generator_minor.invoke({
        "question": question,
        "routing_decision": "general"
    })

    print(f"  Generated SQL:\n{sql_query_minor}")
    return {"sql_query_minor": sql_query_minor}


def degree_router(state):
    print("---DEGREE ROUTER---")

    question = state.get("formatted_user_question_sql_only") or state.get("user_question")

    if "minor" in question.lower():
        return "minor"
    else:
        return "major"
    
def catalog_router(state):
    print("---CATALOG ROUTER---")

    question = state.get("formatted_user_question_sql_only") or state.get("user_question")
    # -----------------------------------------------------------------------------
    # TEMPORAL ROUTER
    # A lightweight LLM call that classifies the question as current vs historical.
    # --
    router_prompt = PromptTemplate(
        input_variables=["question", "historical_years", "current_year"],
        template="""
        You are classifying a student's question about a university course catalog.

        The database contains course data for these past academic years:
        {historical_years}

        The current/default academic year is: {current_year}

        Classify the question below as one of:
        - "current"    : The question is about courses in general, right now, or
                        does not mention a specific past year. The answer should
                        come from the current catalog only.
        - "historical" : The question explicitly refers to a past academic year
                        (e.g. "2022-2023", "last year", "previously", "used to",
                        "was offered", "in the past", "dropped", "removed",
                        "before 2024", "compared to previous years", etc.)

        When in doubt, choose "current". Only choose "historical" if the question
        clearly and specifically relates to past catalog data.

        Respond with ONLY the single word: current   or   historical

        Question: {question}
        """
    )

    router_chain = router_prompt| llm | StrOutputParser()

    raw = router_chain.invoke({
            "question":        question,
            "historical_years": ", ".join(HISTORICAL_YEARS),
            "current_year":    CURRENT_YEAR,
        })

    scope = "historical" if "historical" in raw.strip().lower() else "current"

    print(f"  Question scope : {scope.upper()}")
    if scope == "historical":
        print(f"  → Will query all years ({', '.join(HISTORICAL_YEARS + [CURRENT_YEAR])})")
    else:
        print(f"  → Will query current year only ({CURRENT_YEAR})")
 
    return scope


def convert_dataframe(state):
    print("---CONVERT DATAFRAME---")

    subject = state.get("routing_preprocessor_decision")
    catalogscope = state.get("query_scope")
    print("Main Subject: " + str(subject))
    print("Main Scope: " + str(catalogscope))
    if subject == 'catalog':
        sql_query = state.get("sql_query_catalog")
        #if catalogscope =='current':
        #    sql_query = state.get("sql_query_catalog_current")
        #    print("Catalog Branch: Current")
        #else:
        #    sql_query = state.get("sql_query_catalog_historical")
        #    print("Catalog Branch: Historical")

    elif subject == 'schedule':
        sql_query = state.get("sql_query_schedule")

    else:
        # Degree requirements path:
        # use whichever degree branch produced SQL
        if state.get("sql_query_minor"):
            sql_query = state.get("sql_query_minor")
            print("Degree Branch: minor")
        else:
            sql_query = state.get("sql_query_major")
            print("Degree Branch: major")

    print("SQL Query: " + str(sql_query))

    num_steps = state.get("num_steps")
    num_steps += 1

    df = pd.read_sql(sql_query, conn)
    print(df.head())

    return {"data": dict(df), "num_steps": num_steps}


def decide_question_subject(state):
    print("---DECIDE QUESTION SUBJECT---")
    decision = state.get('routing_preprocessor_decision')
    if decision == "catalog":
        return "catalog_router"
    elif decision == "schedule":
        return "schedule"
    else:
        return "degree_router"


def print_state(state):
    """Terminal debug output — unchanged from original."""
    print("---PRINT STATE---")
    print(f"User Question: {state['user_question']}")
    print(f"Question Type: {state['routing_preprocessor_decision']}")
    print(f"Formatted Question (SQL): {state['formatted_user_question_sql_only']}")
    print(f"Data: \n{pd.DataFrame(state['data'])}\n")
    print(f"Num Steps: {state['num_steps']}")


# ── DEGREE ROUTER NODE ────────────────────────────────────────────────────────
# Placeholder node used to route degree-related questions (degreereq)
# to either the major_agent or minor_agent using the degree_router function
#AC comment:   I dont think we need these and the router code, maybe merge

def degree_router_node(state):
    print("---DEGREE ROUTER NODE---")
    return {}

def catalog_router_node(state):
    print("---CATALOG ROUTER NODE---")
    return {}


# ── WORKFLOW DAG ──────────────────────────────────────────────────────────────

workflow = StateGraph(Graph_State)

workflow.add_node("preprocess_routing", preprocess_routing)
workflow.add_node("catalog_router_node", catalog_router_node)
workflow.add_node("catalog_current_agent", catalog_current_agent)
workflow.add_node("catalog_historical_agent", catalog_historical_agent)
workflow.add_node("schedule_agent", schedule_agent)
workflow.add_node("degree_router_node", degree_router_node)
workflow.add_node("major_agent", major_agent)
workflow.add_node("minor_agent", minor_agent)
workflow.add_node("convert_dataframe", convert_dataframe)
workflow.add_node("synthesize_response", synthesize_response)
workflow.add_node("print_state", print_state)

workflow.set_entry_point("preprocess_routing")

workflow.add_conditional_edges(
    "preprocess_routing",
    decide_question_subject,
    {
        "schedule": "schedule_agent",
        "degree_router": "degree_router_node",
        "catalog_router": "catalog_router_node" 
    }
)

workflow.add_conditional_edges(
    "degree_router_node",
    degree_router,
    {
        "major": "major_agent",
        "minor": "minor_agent"
    }
)

workflow.add_conditional_edges(
    "catalog_router_node",
    catalog_router,
    {
        "current": "catalog_current_agent",
        "historical": "catalog_historical_agent"
    }
)

workflow.add_edge("catalog_current_agent", "convert_dataframe")
workflow.add_edge("catalog_historical_agent", "convert_dataframe")
workflow.add_edge("schedule_agent", "convert_dataframe")
workflow.add_edge("major_agent", "convert_dataframe")
workflow.add_edge("minor_agent", "convert_dataframe")
workflow.add_edge("convert_dataframe",   "synthesize_response")
workflow.add_edge("synthesize_response", "print_state")
workflow.add_edge("print_state", END)

app = workflow.compile()

# ── STREAMLIT CHAT UI ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "ai", "content": "Ask me about course schedules, the course catalog, or degree requirements."}]

if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

# Friendly labels for the sidebar routing indicator
AGENT_LABELS = {
    "catalog": "📚 Course Catalog Agent",
    "schedule": "📅 Class Schedule Agent",
    "degreereq": "🎓 Degree Requirements Agent"
}

# ADDED: Frequently asked questions for the dropdown
FAQ_PLACEHOLDER = "— Choose a frequently asked question —"
FAQS = [
    FAQ_PLACEHOLDER,
    "What are the core courses in Marketing?",
    "What are the graduation requirements for Accounting?",
    "What is the course information for BANA 4950 in 2024-2025?",
    "What is the course information for BANA 4120 in 2023-2024?",
    "What are the time and location details for BANA 6610 in Fall 2025?",
    "What courses does Ziyi Wang teach in Fall 2025, and what is his email address?",
    "How many courses does Ziyi Wang teach in Fall 2025?",
    "What BANA courses are offered after 12 pm in Fall 2025?",
    "How many BANA undergraduate courses are in 2024-2025?",
    "What is the prerequisite for ACCT 2200?",
    "What courses should a freshman in Marketing take in Semester 1?",
    "Which core courses in Marketing are offered in Fall 2025?",
    "What are the requirements for Business Analytics Minor?",
    "How many electives are required for the RMI Minor?",
    "What is the total credit requirement for Business Fundamentals Minor?"
]


def display_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "DATAFRAME_INDEX:" in msg["content"]:
                df_index = int(msg["content"].split("DATAFRAME_INDEX:")[1])
                st.dataframe(st.session_state.dataframes[df_index])
            else:
                st.write(msg["content"])


def run_question(question):
    """Invoke the LangGraph app and display the result."""
    st.chat_message("human").write(question)
    st.session_state.messages.append({"role": "human", "content": question})

    inputs = {"user_question": question, "num_steps": 0}

    error_occurred = False
    try:
        result = app.invoke(inputs)
    except Exception as e:
        error_occurred = True
        print(e)

    if not error_occurred:
        decision = result.get('routing_preprocessor_decision', 'unknown')
        catalogscope = catalogscope = result.get('query_scope')
        print("Main Subject: " + str(decision))
        print("Main Scope: " + str(catalogscope))

        # Update sidebar to show which main area answered
        agent_label.markdown(f"**Last routed to:**\n\n{AGENT_LABELS.get(decision, decision)}")

        if decision == "catalog":
            sql_query = result.get("sql_query_catalog", "")

            #if catalogscope =='current':
            #    sql_query = result.get("sql_query_catalog_current", "")
            #else:
            #    sql_query = result.get("sql_query_catalog_historical", "")
        elif decision == "schedule":
            sql_query = result.get("sql_query_schedule", "")

        else:
            # Degree requirements path: show either major or minor SQL
            if result.get("sql_query_minor"):
                sql_query = result.get("sql_query_minor", "")
            else:
                sql_query = result.get("sql_query_major", "")

        synthesized = result.get("response", "")
        response_text = f"{synthesized}"

        # Push SQL query to sidebar
        sql_label.markdown(f"**Last SQL query:**\n\n```sql\n{sql_query}\n```")

        # Debug sidebar
        debug_label.markdown("**Debug — message history:**")
        with st.sidebar.expander("View message history", expanded=False):
            st.json(st.session_state.messages)

        response_df = pd.DataFrame(result['data'])

        df_index = len(st.session_state.dataframes)
        st.session_state.dataframes.append(response_df)

        st.session_state.messages.append({"role": "ai", "content": response_text})
        st.session_state.messages.append({"role": "ai", "content": f"DATAFRAME_INDEX:{df_index}"})

        st.chat_message("ai").write(response_text)
        st.dataframe(response_df)

    else:
        response_text = (
            "An error occurred while processing your question. "
            "Please try rephrasing it and I'll do my best to help."
        )
        st.session_state.messages.append({"role": "ai", "content": response_text})  # in the error handler
        st.chat_message("ai").write(response_text)

# Render existing chat history on every rerun
display_chat_history()

# ADDED: Dropdown for frequently asked questions
st.markdown("**Choose a frequently asked question or enter your own below:**")
selected_faq = st.selectbox(
    label="Frequently asked questions",
    options=FAQS,
    label_visibility="collapsed"   # hides the label since we put it above as markdown
)

# ADDED: Submit button for the dropdown selection
if st.button("Ask selected question", disabled=(selected_faq == FAQ_PLACEHOLDER)):
    with st.spinner("Thinking..."):
        run_question(selected_faq)

st.divider()

# Free-text chat input (unchanged behavior)
if question := st.chat_input("Or type your own question here:"):
    with st.spinner("Thinking..."):
        run_question(question)


