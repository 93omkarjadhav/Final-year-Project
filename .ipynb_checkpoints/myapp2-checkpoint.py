
# import os
# import re
# import json
# import time
# import pandas as pd
# import streamlit as st
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_community.utilities import SQLDatabase
# from langchain.chains import create_sql_query_chain
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# # --- Page Configuration ---
# st.set_page_config(page_title="Retail Sales AI", page_icon="🤖", layout="wide")

# # --- Constants & File Paths ---
# HISTORY_FILE = "chat_history.json"

# # --- Helper Functions ---
# @st.cache_resource
# def get_engine():
#     """Create and cache the SQLAlchemy engine."""
#     DB_USER, DB_PASSWORD, DB_HOST, DB_NAME = "root", "root123", "localhost", "retail_sales_db"
#     try:
#         engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
#         return engine
#     except Exception as e:
#         st.error(f"🔥 Database connection failed: {e}")
#         st.stop()

# @st.cache_resource
# def get_llm_and_chain(_engine):
#     """Create and cache the LLM and LangChain components."""
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#     if not GOOGLE_API_KEY:
#         st.error("🚨 Google API Key not found!")
#         st.stop()
        
#     db = SQLDatabase(_engine, sample_rows_in_table_info=3)
#     llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
#     chain = create_sql_query_chain(llm, db)
#     return llm, chain

# def load_chat_history():
#     try:
#         with open(HISTORY_FILE, "r") as f: return json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError): return {}

# def save_chat_history(history):
#     with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)

# def extract_sql_from_response(response: str) -> str:
#     match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
#     if match: return match.group(1).strip()
#     match = re.search(r"(?:SQLQuery:|SQL Query:|SELECT)\s*(.*)", response, re.DOTALL | re.IGNORECASE)
#     if match:
#         query = match.group(1).strip()
#         if not query.upper().startswith('SELECT'): query = 'SELECT ' + query
#         return query.split(';')[0].strip() + ';'
#     return response.strip()

# # --- NEW: Helper function to clean the Python code response ---
# def extract_python_code(response: str) -> str:
#     """Extracts pure Python code from a string, removing markdown fences."""
#     match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
#     if match:
#         return match.group(1).strip()
#     # Fallback for when no markdown fence is found
#     return response.strip()

# def get_user_intent(llm, user_prompt):
#     prompt = f"""
#     Analyze the user's prompt. Is the intent a `data_query` for new information from the database, or a `visualization_request` to create a chart from recent data?
#     User Prompt: '{user_prompt}'
#     Respond with only the intent name.
#     """
#     try:
#         response = llm.invoke(prompt)
#         intent = response.strip().lower()
#         if "data_query" in intent: return "data_query"
#         if "visualization_request" in intent: return "visualization_request"
#         return "unknown"
#     except Exception:
#         return "data_query"

# def generate_chart_code(llm, question, df):
#     prompt = f"""
#     Given the user's request: '{question}' and the following data:
#     {df.to_string()}
#     Generate a single block of Python code for a Streamlit chart. The data is in a DataFrame named `df`.
#     Only output the code, no explanations or markdown.
#     """
#     try:
#         response = llm.invoke(prompt)
#         # --- FIX IS APPLIED HERE ---
#         return extract_python_code(response)
#     except Exception: return None

# # --- Main Application ---
# load_dotenv()
# engine = get_engine()
# llm, chain = get_llm_and_chain(engine)

# if "all_chats" not in st.session_state: st.session_state.all_chats = load_chat_history()
# if "active_chat_id" not in st.session_state:
#     st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0] if st.session_state.all_chats else None

# # --- Sidebar UI (Unchanged) ---
# with st.sidebar:
#     st.title("🗂️ Chat History")

#     #new
#     st.header("📂 Data Source")
#     source_type = st.radio("Choose Source:", ["MySQL Database", "Upload File"])
    
#     uploaded_file = None
#     if source_type == "Upload File":
#         uploaded_file = st.file_uploader(
#             "Upload CSV, Excel, or PDF", 
#             type=["csv", "xlsx", "pdf"]
#         )
#         def load_data_file(file):
#             if file.name.endswith('.csv'):
#                 return pd.read_csv(file)
        
#     elif file.name.endswith('.xlsx'):
#         return pd.read_excel(file)
#     elif file.name.endswith('.pdf'):
#         # For PDFs, we extract tables as DataFrames
#         import pdfplumber
#         with pdfplumber.open(file) as pdf:
#             all_tables = []
#             for page in pdf.pages:
#                 table = page.extract_table()
#                 if table:
#                     all_tables.append(pd.DataFrame(table[1:], columns=table[0]))
#             return pd.concat(all_tables) if all_tables else None
#             if uploaded_file:
#     df = load_data_file(uploaded_file)
#     # This agent talks to your uploaded file instead of the database
#     agent = create_pandas_dataframe_agent(
#         llm, 
#         df, 
#         verbose=True, 
#         allow_dangerous_code=True
#     )
#     # The agent will now answer questions, show tables, and create charts
#     response = agent.invoke(user_prompt)

#     #old
#     if st.button("➕ New Chat", use_container_width=True):
#         chat_id = f"chat_{int(time.time())}"
#         st.session_state.all_chats[chat_id] = {"title": "New Chat", "messages": [{"role": "assistant", "content": "Hello! How can I help you today?"}]}
#         st.session_state.active_chat_id = chat_id
#         save_chat_history(st.session_state.all_chats)
#         st.rerun()
#     st.write("---")
#     chat_ids = list(st.session_state.all_chats.keys())
#     for chat_id in reversed(chat_ids):
#         chat = st.session_state.all_chats[chat_id]
#         col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
#         with col1:
#             if st.button(chat['title'], key=f"select_{chat_id}", use_container_width=True, help=chat['title']):
#                 st.session_state.active_chat_id = chat_id
#                 st.rerun()
#         with col2:
#             with st.popover("✏️", use_container_width=True):
#                 new_title = st.text_input("New chat name", value=chat['title'], key=f"rename_{chat_id}")
#                 if st.button("Save", key=f"save_{chat_id}"):
#                     st.session_state.all_chats[chat_id]['title'] = new_title
#                     save_chat_history(st.session_state.all_chats)
#                     st.rerun()
#         with col3:
#             if st.button("🗑️", key=f"delete_{chat_id}", use_container_width=True):
#                 del st.session_state.all_chats[chat_id]
#                 if st.session_state.active_chat_id == chat_id: st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0] if st.session_state.all_chats else None
#                 save_chat_history(st.session_state.all_chats)
#                 st.rerun()

# # --- Main Chat UI ---
# st.title("🤖 AI SQL AGENT")
# st.caption("Ask a question to get data, then ask me to visualize it!")

# if st.session_state.active_chat_id:
#     active_chat = st.session_state.all_chats[st.session_state.active_chat_id]

#     for msg in active_chat["messages"]:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
#             if msg.get("dataframe"): st.dataframe(pd.read_json(msg["dataframe"], orient="split"))
#             if msg.get("chart_code"):
#                 try:
#                     df = pd.read_json(msg["dataframe_for_chart"], orient="split")
#                     exec(msg["chart_code"])
#                 except Exception as e: st.warning(f"Could not re-display chart: {e}")

#     if prompt := st.chat_input("Ask a question..."):
#         active_chat["messages"].append({"role": "user", "content": prompt})
#         if active_chat["title"] == "New Chat": active_chat["title"] = prompt[:30] + "..."
#         with st.chat_message("user"): st.markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 intent = get_user_intent(llm, prompt)

#                 if intent == "data_query":
#                     raw_llm_output = st.write_stream(chain.stream({"question": prompt}))
#                     final_sql_query = extract_sql_from_response(raw_llm_output)
#                     st.markdown(f"**Generated SQL Query:**\n```sql\n{final_sql_query}\n```")
#                     df = pd.read_sql_query(final_sql_query, engine)
#                     st.dataframe(df)
#                     active_chat["messages"].append({
#                         "role": "assistant", "content": f"**Generated SQL Query:**\n```sql\n{final_sql_query}\n```",
#                         "dataframe": df.to_json(orient="split")
#                     })

#                 elif intent == "visualization_request":
#                     last_df_json = None
#                     for msg in reversed(active_chat["messages"]):
#                         if msg.get("dataframe"):
#                             last_df_json = msg["dataframe"]
#                             break
                    
#                     if last_df_json:
#                         df = pd.read_json(last_df_json, orient="split")
#                         chart_code = generate_chart_code(llm, prompt, df)
#                         if chart_code:
#                             st.markdown("Here is the visualization you requested:")
#                             # --- FIX IS APPLIED HERE: We execute the cleaned code ---
#                             exec(chart_code)
#                             active_chat["messages"].append({
#                                 "role": "assistant", "content": "Here is the visualization you requested:",
#                                 "chart_code": chart_code, "dataframe_for_chart": last_df_json
#                             })
#                         else:
#                             st.warning("I was unable to generate a visualization for that.")
#                     else:
#                         st.warning("I need some data to visualize first. Please ask a question to get data from the database.")
        
#         save_chat_history(st.session_state.all_chats)
#         st.rerun()
# else:
#     st.info("Click '➕ New Chat' in the sidebar to begin!")


#2222222
# import os
# import re
# import json
# import time
# import pandas as pd
# import streamlit as st
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_community.utilities import SQLDatabase
# from langchain.chains import create_sql_query_chain
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# # --- Page Configuration ---
# st.set_page_config(page_title="Retail Sales AI Agent", page_icon="🤖", layout="wide")

# # --- Constants & File Paths ---
# HISTORY_FILE = "chat_history.json"

# # --- Helper Functions ---
# @st.cache_resource
# def get_engine():
#     """Create and cache the SQLAlchemy engine."""
#     DB_USER, DB_PASSWORD, DB_HOST, DB_NAME = "root", "root123", "localhost", "retail_sales_db"
#     try:
#         engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
#         return engine
#     except Exception as e:
#         st.error(f"🔥 Database connection failed: {e}")
#         st.stop()

# @st.cache_resource
# def get_llm_and_chain(_engine):
#     """Create and cache the LLM and LangChain components."""
#     load_dotenv()
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#     if not GOOGLE_API_KEY:
#         st.error("🚨 Google API Key not found! Check your .env file.")
#         st.stop()
        
#     db = SQLDatabase(_engine, sample_rows_in_table_info=3)
#     llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
#     chain = create_sql_query_chain(llm, db)
#     return llm, chain

# def load_chat_history():
#     try:
#         with open(HISTORY_FILE, "r") as f: return json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError): return {}

# def save_chat_history(history):
#     with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)

# def load_data_file(file):
#     """Loads CSV, Excel, or PDF into a Pandas DataFrame."""
#     try:
#         if file.name.endswith('.csv'):
#             return pd.read_csv(file)
#         elif file.name.endswith('.xlsx'):
#             return pd.read_excel(file)
#         elif file.name.endswith('.pdf'):
#             import pdfplumber
#             with pdfplumber.open(file) as pdf:
#                 all_tables = []
#                 for page in pdf.pages:
#                     table = page.extract_table()
#                     if table:
#                         all_tables.append(pd.DataFrame(table[1:], columns=table[0]))
#                 return pd.concat(all_tables) if all_tables else None
#     except Exception as e:
#         st.error(f"Error reading file: {e}")
#         return None

# def extract_sql_from_response(response: str) -> str:
#     match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
#     if match: return match.group(1).strip()
#     return response.strip()

# def extract_python_code(response: str) -> str:
#     match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
#     return match.group(1).strip() if match else response.strip()

# def get_user_intent(llm, user_prompt):
#     prompt = f"Analyze intent: 'data_query' or 'visualization_request'. User: '{user_prompt}'. Reply with one word."
#     try:
#         response = llm.invoke(prompt)
#         intent = response.strip().lower()
#         return "visualization_request" if "visualization" in intent else "data_query"
#     except: return "data_query"

# def generate_chart_code(llm, question, df):
#     prompt = f"Write Python code using Streamlit to visualize this data based on: '{question}'. DataFrame name is `df`. No explanation, just code."
#     try:
#         response = llm.invoke(prompt)
#         return extract_python_code(response)
#     except: return None

# # --- Main Application Logic ---
# engine = get_engine()
# llm, chain = get_llm_and_chain(engine)

# if "all_chats" not in st.session_state: st.session_state.all_chats = load_chat_history()
# if "active_chat_id" not in st.session_state:
#     st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0] if st.session_state.all_chats else None

# # --- Sidebar UI ---
# with st.sidebar:
#     st.title("🗂️ Chat History")
    
#     st.header("📂 Data Source")
#     source_type = st.radio("Choose Source:", ["MySQL Database", "Upload File"])
    
#     uploaded_file = None
#     if source_type == "Upload File":
#         uploaded_file = st.file_uploader("Upload CSV, Excel, or PDF", type=["csv", "xlsx", "pdf"])

#     if st.button("➕ New Chat", use_container_width=True):
#         chat_id = f"chat_{int(time.time())}"
#         st.session_state.all_chats[chat_id] = {"title": "New Chat", "messages": [{"role": "assistant", "content": "Hello! I'm ready. What should we analyze?"}]}
#         st.session_state.active_chat_id = chat_id
#         save_chat_history(st.session_state.all_chats)
#         st.rerun()

#     st.write("---")
#     for cid in reversed(list(st.session_state.all_chats.keys())):
#         chat = st.session_state.all_chats[cid]
#         col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
#         with col1:
#             if st.button(chat['title'], key=f"sel_{cid}", use_container_width=True):
#                 st.session_state.active_chat_id = cid
#                 st.rerun()
#         with col2:
#             with st.popover("✏️"):
#                 new_t = st.text_input("Rename", value=chat['title'], key=f"ren_{cid}")
#                 if st.button("Ok", key=f"ok_{cid}"):
#                     st.session_state.all_chats[cid]['title'] = new_t
#                     save_chat_history(st.session_state.all_chats)
#                     st.rerun()
#         with col3:
#             if st.button("🗑️", key=f"del_{cid}"):
#                 del st.session_state.all_chats[cid]
#                 save_chat_history(st.session_state.all_chats)
#                 st.rerun()

# # --- Main Chat UI ---
# st.title("🤖 AI SQL & Data Agent")

# if st.session_state.active_chat_id:
#     active_chat = st.session_state.all_chats[st.session_state.active_chat_id]

#     # Display History
#     for msg in active_chat["messages"]:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
#             if msg.get("dataframe"): st.dataframe(pd.read_json(msg["dataframe"], orient="split"))
#             if msg.get("chart_code"):
#                 try:
#                     df = pd.read_json(msg["dataframe_for_chart"], orient="split")
#                     exec(msg["chart_code"])
#                 except: st.warning("Chart could not be rendered.")

#     # User Input
#     if prompt := st.chat_input("Ask about your data..."):
#         active_chat["messages"].append({"role": "user", "content": prompt})
#         with st.chat_message("user"): st.markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Processing..."):
#                 intent = get_user_intent(llm, prompt)

#                 # --- HYBRID LOGIC: DATABASE OR FILE ---
#                 if intent == "data_query":
#                     if source_type == "Upload File" and uploaded_file:
#                         df = load_data_file(uploaded_file)
#                         agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)
#                         response = agent.invoke(prompt)
#                         st.write(response["output"])
#                         active_chat["messages"].append({"role": "assistant", "content": response["output"]})
#                     else:
#                         raw_output = st.write_stream(chain.stream({"question": prompt}))
#                         sql = extract_sql_from_response(raw_output)
#                         st.code(sql, language="sql")
#                         df = pd.read_sql_query(sql, engine)
#                         st.dataframe(df)
#                         active_chat["messages"].append({
#                             "role": "assistant", "content": f"Query executed successfully.",
#                             "dataframe": df.to_json(orient="split")
#                         })

#                 elif intent == "visualization_request":
#                     last_df_json = next((m["dataframe"] for m in reversed(active_chat["messages"]) if m.get("dataframe")), None)
#                     if last_df_json:
#                         df = pd.read_json(last_df_json, orient="split")
#                         code = generate_chart_code(llm, prompt, df)
#                         if code:
#                             exec(code)
#                             active_chat["messages"].append({
#                                 "role": "assistant", "content": "Chart generated.",
#                                 "chart_code": code, "dataframe_for_chart": last_df_json
#                             })
#                     else: st.warning("Please get some data first!")

#         save_chat_history(st.session_state.all_chats)
#         st.rerun()

#333333333
# import os
# import re
# import json
# import time
# import pandas as pd
# import streamlit as st
# from redis_config import init_redis_cache  # 1. Import your new config
# from pathlib import Path
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from langchain_google_genai import GoogleGenerativeAI,HarmCategory,HarmBlockThreshold
# from langchain_community.utilities import SQLDatabase
# from langchain.chains import create_sql_query_chain
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_core.globals import get_llm_cache
# from langchain_google_genai import ChatGoogleGenerativeAI # Better for caching

# # --- Page Configuration ---
# st.set_page_config(page_title="Retail AI: DB & Files", page_icon="🤖", layout="wide")
# # 2. Initialize the cache before everything else starts
# redis_status = init_redis_cache()
# # --- Constants & Directories ---
# HISTORY_FILE = "chat_history.json"
# UPLOAD_DIR = Path("temp_data")
# UPLOAD_DIR.mkdir(exist_ok=True)

# # --- Helper Functions ---
# @st.cache_resource
# def get_engine():
#     DB_USER, DB_PASSWORD, DB_HOST, DB_NAME = "root", "root123", "localhost", "retail_sales_db"
#     try:
#         return create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
#     except Exception as e:
#         st.error(f"🔥 DB Connection Failed: {e}")
#         st.stop()

# @st.cache_resource
# def get_llm_and_chain(_engine):
#     load_dotenv()
#     key = os.getenv("GOOGLE_API_KEY")
#     if not key:
#         st.error("🚨 API Key missing in .env!")
#         st.stop()

#     # --- ADD THIS SAFETY SETTING BLOCK ---
#     safety_settings = {
#         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     }
#     db = SQLDatabase(_engine, sample_rows_in_table_info=3)
#     llm = GoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=key, temperature=0, safety_settings=safety_settings, cache=get_llm_cache() )
#     chain = create_sql_query_chain(llm, db)
#     return llm, chain

# def load_chat_history():
#     try:
#         with open(HISTORY_FILE, "r") as f: return json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError): return {}

# def save_chat_history(history):
#     with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)

# def load_data_file(file_path):
#     """Helper to load data from a physical path."""
#     ext = Path(file_path).suffix.lower()
#     if ext == '.csv': return pd.read_csv(file_path)
#     if ext == '.xlsx': return pd.read_excel(file_path)
#     if ext == '.pdf':
#         import pdfplumber
#         with pdfplumber.open(file_path) as pdf:
#             tbls = [pd.DataFrame(p.extract_table()[1:], columns=p.extract_table()[0]) for p in pdf.pages if p.extract_table()]
#             return pd.concat(tbls) if tbls else None
#     return None

# def get_user_intent(llm, user_prompt):
#     prompt = f"Categorize intent: 'data_query' or 'visualization_request'. Input: '{user_prompt}'. Output one word."
#     try: return llm.invoke(prompt).strip().lower()
#     except: return "data_query"

# # --- Initialization ---
# engine = get_engine()
# llm, chain = get_llm_and_chain(engine)

# if "all_chats" not in st.session_state: st.session_state.all_chats = load_chat_history()
# if "active_chat_id" not in st.session_state:
#     st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0] if st.session_state.all_chats else None

# # --- Sidebar UI ---
# with st.sidebar:
#     st.title("🗂️ Chat Sessions")
    
#     # ➕ New Chat
#     if st.button("➕ New Chat", use_container_width=True):
#         cid = f"chat_{int(time.time())}"
#         st.session_state.all_chats[cid] = {
#             "title": "New Session", 
#             "messages": [{"role": "assistant", "content": "Hello! Choose a source and let's analyze."}],
#             "file_path": None,
#             "source": "MySQL Database"
#         }
#         st.session_state.active_chat_id = cid
#         save_chat_history(st.session_state.all_chats)
#         st.rerun()

#     active_id = st.session_state.active_chat_id
#     if active_id:
#         chat_data = st.session_state.all_chats[active_id]
        
#         st.write("---")
#         st.header("📂 Data Source")
#         # Sync source from history
#         source_type = st.radio("Current Source:", ["MySQL Database", "Upload File"], 
#                                index=0 if chat_data.get("source") == "MySQL Database" else 1)
#         chat_data["source"] = source_type
        
#         if source_type == "Upload File":
#             u_file = st.file_uploader("Drop CSV/Excel/PDF", type=["csv", "xlsx", "pdf"])
#             if u_file:
#                 # Save file to disk to persist across refreshes
#                 path = UPLOAD_DIR / u_file.name
#                 with open(path, "wb") as f: f.write(u_file.getbuffer())
#                 chat_data["file_path"] = str(path)
#                 st.success(f"Loaded: {u_file.name}")
            
#             if chat_data.get("file_path"):
#                 st.info(f"Using: {Path(chat_data['file_path']).name}")
        
#         st.write("---")
#         # Session Management
#         for cid in reversed(list(st.session_state.all_chats.keys())):
#             c = st.session_state.all_chats[cid]
#             col1, col2 = st.columns([0.8, 0.2])
#             if col1.button(c['title'], key=f"s_{cid}", use_container_width=True):
#                 st.session_state.active_chat_id = cid
#                 st.rerun()
#             if col2.button("🗑️", key=f"d_{cid}"):
#                 del st.session_state.all_chats[cid]
#                 save_chat_history(st.session_state.all_chats)
#                 st.rerun()
#     # Optional: Add a small status indicator in the sidebar
#     if redis_status:
#         st.sidebar.success("⚡ Cache: Online (Upstash)")
#     else:
#         st.sidebar.warning("🕒 Cache: Offline")

# # --- Main Interface ---
# st.title("🤖 AI Data Agent")

# if active_id:
#     active_chat = st.session_state.all_chats[active_id]

#     # Re-display History
#     for m in active_chat["messages"]:
#         with st.chat_message(m["role"]):
#             st.markdown(m["content"])
#             if "dataframe" in m: st.dataframe(pd.read_json(m["dataframe"], orient="split"))
#             if "chart_code" in m:
#                 df = pd.read_json(m["dataframe_for_chart"], orient="split")
#                 exec(m["chart_code"])

#     # Chat Input
#     if prompt := st.chat_input("Analyze my data..."):
#         active_chat["messages"].append({"role": "user", "content": prompt})
#         if active_chat["title"] == "New Session": active_chat["title"] = prompt[:25] + "..."
#         st.chat_message("user").markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Analyzing..."):
#                 intent = get_user_intent(llm, prompt)

#                 if intent == "data_query":
#                     if active_chat["source"] == "Upload File" and active_chat.get("file_path"):
#                         # --- FILE PROCESSING ---
#                         #df = load_data_file(active_chat["file_path"])
#                         full_df = load_data_file(active_chat["file_path"])
#                         agent = create_pandas_dataframe_agent(llm, full_df, verbose=False, allow_dangerous_code=True,agent_executor_kwargs={"handle_parsing_errors": True}) #,agent_executor_kwargs={"handle_parsing_errors": True}
#                         # We prompt the agent to strictly provide the filtered subset
#                         # query_prompt = (
#                         #     f"Answer this: {prompt}. "
#                         #     "Give only a natural language explanation. "
#                         #     "Do NOT print tables or code here."
#                         # )

#                         query_prompt = (
#                            f"Return a textual answer AND the specific python code used for the user question: {prompt}"
#                         )
#                         # Get steps + result
#                         agent_response = agent.invoke(query_prompt)
#                         answer = agent_response["output"]
#                         # --- CLEANING STEP: Removes common table-like text patterns if they leak through ---
#                         # answer = re.sub(r'\[\d+ rows x \d+ columns\]', '', answer) # Removes row/column count
#                         # answer = re.sub(r'\|\s*[-:]+\s*\|', '', answer)           # Removes markdown table separators
#                         code_prompt = f"Based on the result '{answer}', write the one-line pandas code to get this exact filtered dataframe from a df named'df'. Code only."
#                         filter_code = llm.invoke(code_prompt).replace("```python", "").replace("```", "").strip()

#                         try:
#                             # We apply the filter code to the full_df to get ONLY the requested rows
#                             # For example: df = full_df[full_df['Airline'] == 'IndiGo']
#                             df = full_df # fallback
#                             local_vars = {'df': full_df, 'pd': pd}
#                             # We execute the filter logic to get the subset
#                             exec(f"result_df = {filter_code}", {}, local_vars)
#                             filtered_df = local_vars['result_df']
#                         except:
#                             filtered_df = full_df.head(10)
                        
#                         st.markdown(answer)
#                         st.dataframe(filtered_df)
#                         #st.dataframe(df.head(10)) # Preview top results
#                         # Add a direct download button for ONLY the filtered data
#                         st.download_button(
#                             label="📥 Download Filtered Results",
#                             data=filtered_df.to_csv(index=False).encode('utf-8'),
#                             file_name="filtered_data.csv",
#                             mime="text/csv",
#                             key=f"dl_{int(time.time())}"
#                         )
                        
#                         active_chat["messages"].append({
#                             "role": "assistant", "content": answer,
#                             "dataframe": filtered_df.to_json(orient="split")
#                         })
#                     else:
#                         # --- SQL PROCESSING ---
#                         raw_sql = st.write_stream(chain.stream({"question": prompt}))
#                         sql = raw_sql.replace("```sql", "").replace("```", "").strip()
#                         df = pd.read_sql_query(sql, engine)
#                         st.dataframe(df)
#                         active_chat["messages"].append({
#                             "role": "assistant", "content": f"**SQL Generated:**\n```sql\n{sql}\n```",
#                             "dataframe": df.to_json(orient="split")
#                         })

#                 elif intent == "visualization_request":
#                     # Find last data
#                     last_df = next((m["dataframe"] for m in reversed(active_chat["messages"]) if "dataframe" in m), None)
#                     if last_df:
#                         df = pd.read_json(last_df, orient="split")
#                         viz_prompt = f"Generate ONLY Streamlit Python code to visualize this data based on: {prompt}. DF name is `df`."
#                         code = llm.invoke(viz_prompt).replace("```python", "").replace("```", "").strip()
#                         exec(code)
#                         active_chat["messages"].append({
#                             "role": "assistant", "content": "Visualizing your data now.",
#                             "chart_code": code, "dataframe_for_chart": last_df
#                         })
#                     else: st.warning("No data found to visualize!")

#         save_chat_history(st.session_state.all_chats)
#         st.rerun()

#4444444 updateing in upstash 
# import os
# import re
# import json
# import time
# import pandas as pd
# import streamlit as st
# from pathlib import Path
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from redis_config import init_redis_cache  
# from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
# from langchain_community.utilities import SQLDatabase
# from langchain.chains import create_sql_query_chain
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_core.globals import get_llm_cache
# from langchain_mongodb.agent_toolkit import MongoDBDatabase, MongoDBDatabaseToolkit

# # --- Page Configuration ---
# st.set_page_config(page_title="Retail AI: DB & Files", page_icon="🤖", layout="wide")

# # 1. Initialize Cache and Force-Test the Connection
# # 1. Initialize Cache and Force-Test the Connection
# load_dotenv()
# redis_client = init_redis_cache()

# # --- FORCE PING TEST ---
# # This will show up in your Upstash Data Browser as "connection_test"
# if redis_client:
#     try:
#         redis_client.set("connection_test", f"Last active: {time.ctime()}")
#         st.sidebar.success("⚡ Cache: Online & Verified")
#     except Exception as e:
#         st.sidebar.error(f"❌ Redis Write Failed: {e}")
# else:
#     st.sidebar.warning("⚠️ Cache: Offline - Check your .env file for UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN")

# # --- Constants & Directories ---
# HISTORY_FILE = "chat_history.json"
# UPLOAD_DIR = Path("temp_data")
# UPLOAD_DIR.mkdir(exist_ok=True)

# # --- Helper Functions ---
# @st.cache_resource
# def get_engine():
#     DB_USER, DB_PASSWORD, DB_HOST, DB_NAME = "root", "root123", "localhost", "retail_sales_db"
#     try:
#         return create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
#     except Exception as e:
#         st.error(f"🔥 DB Connection Failed: {e}")
#         st.stop()

# @st.cache_resource
# def get_llm_and_chain(_engine):
#     key = os.getenv("GOOGLE_API_KEY")
#     if not key:
#         st.error("🚨 API Key missing in .env!")
#         st.stop()

#     safety_settings = {
#         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     }
    
#     db = SQLDatabase(_engine, sample_rows_in_table_info=3)
    
#     # FIX: Get cache AFTER it's been initialized globally
#     from langchain_core.globals import get_llm_cache
#     current_cache = get_llm_cache()
    
#     # FIX: Only pass cache if it's not None
#     llm_kwargs = {
#         "model": "gemini-2.5-flash-lite",
#         "google_api_key": key, 
#         "temperature": 0, 
#         "safety_settings": safety_settings,
#     }
    
#     # Only add cache if it exists
#     if current_cache is not None:
#         llm_kwargs["cache"] = current_cache
    
#     llm = ChatGoogleGenerativeAI(**llm_kwargs)
    
#     chain = create_sql_query_chain(llm, db)
#     return llm, chain

# # ... (Rest of your helper functions: load_chat_history, save_chat_history, load_data_file, get_user_intent stay exactly the same) ...

# def load_chat_history():
#     try:
#         with open(HISTORY_FILE, "r") as f: return json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError): return {}

# def save_chat_history(history):
#     with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)

# def load_data_file(file_path):
#     ext = Path(file_path).suffix.lower()
#     if ext == '.csv': return pd.read_csv(file_path)
#     if ext == '.xlsx': return pd.read_excel(file_path)
#     if ext == '.pdf':
#         import pdfplumber
#         with pdfplumber.open(file_path) as pdf:
#             tbls = [pd.DataFrame(p.extract_table()[1:], columns=p.extract_table()[0]) for p in pdf.pages if p.extract_table()]
#             return pd.concat(tbls) if tbls else None
#     return None

# def get_user_intent(llm, user_prompt):
#     prompt = f"Categorize intent: 'data_query' or 'visualization_request'. Input: '{user_prompt}'. Output one word."
#     try: 
#         # For ChatGoogleGenerativeAI, we access .content
#         return llm.invoke(prompt).content.strip().lower()
#     except: return "data_query"

# # --- Initialization ---
# engine = get_engine()
# llm, chain = get_llm_and_chain(engine)

# if "all_chats" not in st.session_state: st.session_state.all_chats = load_chat_history()
# if "active_chat_id" not in st.session_state:
#     st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0] if st.session_state.all_chats else None

# # --- Sidebar UI (Kept exactly as yours) ---
# with st.sidebar:
#     st.title("🗂️ Chat Sessions")
    
#     if st.button("➕ New Chat", use_container_width=True):
#         cid = f"chat_{int(time.time())}"
#         st.session_state.all_chats[cid] = {
#             "title": "New Session", 
#             "messages": [{"role": "assistant", "content": "Hello! Choose a source and let's analyze."}],
#             "file_path": None,
#             "source": "MySQL Database"
#         }
#         st.session_state.active_chat_id = cid
#         save_chat_history(st.session_state.all_chats)
#         st.rerun()

#     active_id = st.session_state.active_chat_id
#     if active_id:
#         chat_data = st.session_state.all_chats[active_id]
#         st.write("---")
#         st.header("📂 Data Source")
#         source_type = st.radio("Current Source:", ["MySQL Database", "Upload File"], 
#                                index=0 if chat_data.get("source") == "MySQL Database" else 1)
#         chat_data["source"] = source_type
        
#         if source_type == "Upload File":
#             u_file = st.file_uploader("Drop CSV/Excel/PDF", type=["csv", "xlsx", "pdf"])
#             if u_file:
#                 path = UPLOAD_DIR / u_file.name
#                 with open(path, "wb") as f: f.write(u_file.getbuffer())
#                 chat_data["file_path"] = str(path)
#                 st.success(f"Loaded: {u_file.name}")
            
#             if chat_data.get("file_path"):
#                 st.info(f"Using: {Path(chat_data['file_path']).name}")
        
#         st.write("---")
#         for cid in reversed(list(st.session_state.all_chats.keys())):
#             c = st.session_state.all_chats[cid]
#             col1, col2 = st.columns([0.8, 0.2])
#             if col1.button(c['title'], key=f"s_{cid}", use_container_width=True):
#                 st.session_state.active_chat_id = cid
#                 st.rerun()
#             if col2.button("🗑️", key=f"d_{cid}"):
#                 del st.session_state.all_chats[cid]
#                 save_chat_history(st.session_state.all_chats)
#                 st.rerun()

# # --- Main Interface ---
# st.title("🤖 AI Data Agent")

# if active_id:
#     active_chat = st.session_state.all_chats[active_id]

#     # Re-display History
#     for m in active_chat["messages"]:
#         with st.chat_message(m["role"]):
#             st.markdown(m["content"])
#             if "dataframe" in m: st.dataframe(pd.read_json(m["dataframe"], orient="split"))
#             if "chart_code" in m:
#                 df = pd.read_json(m["dataframe_for_chart"], orient="split")
#                 exec(m["chart_code"])

#     if prompt := st.chat_input("Analyze my data..."):
#         active_chat["messages"].append({"role": "user", "content": prompt})
#         if active_chat["title"] == "New Session": active_chat["title"] = prompt[:25] + "..."
#         st.chat_message("user").markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Analyzing..."):
#                 intent = get_user_intent(llm, prompt)

#                 if intent == "data_query":
#                     if active_chat["source"] == "Upload File" and active_chat.get("file_path"):
#                         full_df = load_data_file(active_chat["file_path"])
#                         # Pass the explicitly cached LLM to the agent
#                         agent = create_pandas_dataframe_agent(llm, full_df, verbose=False, allow_dangerous_code=True, agent_executor_kwargs={"handle_parsing_errors": True})
                        
#                         query_prompt = f"Answer clearly: {prompt}"
#                         agent_response = agent.invoke(query_prompt)
#                         answer = agent_response["output"]
                        
#                         # Use LLM to get filter code
#                         code_prompt = f"Write one-line pandas code to get '{prompt}' from a df named 'df'. Code only."
#                         filter_code = llm.invoke(code_prompt).content.replace("```python", "").replace("```", "").strip()

#                         try:
#                             local_vars = {'df': full_df, 'pd': pd}
#                             exec(f"result_df = {filter_code}", {}, local_vars)
#                             filtered_df = local_vars['result_df']
#                         except:
#                             filtered_df = full_df.head(10)
                        
#                         st.markdown(answer)
#                         st.dataframe(filtered_df)
                        
#                         active_chat["messages"].append({
#                             "role": "assistant", "content": answer,
#                             "dataframe": filtered_df.to_json(orient="split")
#                         })
#                     else:
#                         # --- SQL PROCESSING ---
#                         raw_sql = st.write_stream(chain.stream({"question": prompt}))
#                         sql = raw_sql.replace("```sql", "").replace("```", "").strip()
#                         df = pd.read_sql_query(sql, engine)
#                         st.dataframe(df)
#                         active_chat["messages"].append({
#                             "role": "assistant", "content": f"**SQL Generated:**\n```sql\n{sql}\n```",
#                             "dataframe": df.to_json(orient="split")
#                         })

#                 elif intent == "visualization_request":
#                     last_df_data = next((m["dataframe"] for m in reversed(active_chat["messages"]) if "dataframe" in m), None)
#                     if last_df_data:
#                         df = pd.read_json(last_df_data, orient="split")
#                         viz_prompt = f"Generate ONLY Streamlit Python code to visualize this data based on: {prompt}. DF name is `df`."
#                         code = llm.invoke(viz_prompt).content.replace("```python", "").replace("```", "").strip()
#                         exec(code)
#                         active_chat["messages"].append({
#                             "role": "assistant", "content": "Visualizing your data now.",
#                             "chart_code": code, "dataframe_for_chart": last_df_data
#                         })
#                     else: st.warning("No data found to visualize!")

#         save_chat_history(st.session_state.all_chats)
#         st.rerun()

#Multiple dbs support
import os
import re
import json
import time
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from redis_config import init_redis_cache  
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_community.utilities import SQLDatabase
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.globals import get_llm_cache
from langchain_mongodb.agent_toolkit import MongoDBDatabase, MongoDBDatabaseToolkit

# --- Page Configuration ---
st.set_page_config(page_title="Retail AI: Multi-DB & Files", page_icon="🤖", layout="wide")

# 1. Initialize Cache and Force-Test the Connection
load_dotenv()
redis_client = init_redis_cache()

# --- FORCE PING TEST ---
if redis_client:
    try:
        redis_client.set("connection_test", f"Last active: {time.ctime()}")
        st.sidebar.success("⚡ Cache: Online & Verified")
    except Exception as e:
        st.sidebar.error(f"❌ Redis Write Failed: {e}")
else:
    st.sidebar.warning("⚠️ Cache: Offline")

# --- Constants & Directories ---
HISTORY_FILE = "chat_history.json"
UPLOAD_DIR = Path("temp_data")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Helper Functions ---
def get_db_engine(db_type):
    """Dynamically creates an engine based on the selected database type."""
    try:
        if db_type == "MySQL Database":
            return create_engine(f"mysql+pymysql://root:root123@localhost/retail_sales_db")
        elif db_type == "PostgreSQL":
            return create_engine(os.getenv("POSTGRES_URL"))
        elif db_type == "Oracle":
            return create_engine(os.getenv("ORACLE_URL"))
        return None
    except Exception as e:
        st.error(f"🔥 {db_type} Connection Failed: {e}")
        return None

@st.cache_resource
def get_llm_and_chain(db_type):
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        st.error("🚨 API Key missing in .env!")
        st.stop()

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    from langchain_core.globals import get_llm_cache
    current_cache = get_llm_cache()
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=key, 
        temperature=0, 
        safety_settings=safety_settings,
        cache=current_cache if current_cache else None
    )
    
    # Handle SQL Databases
    if db_type in ["MySQL Database", "PostgreSQL", "Oracle"]:
        _engine = get_db_engine(db_type)
        if _engine:
            db = SQLDatabase(_engine, sample_rows_in_table_info=3)
            chain = create_sql_query_chain(llm, db)
            return llm, chain, _engine
    
    # Handle MongoDB separately
    return llm, None, None

def load_chat_history():
    try:
        with open(HISTORY_FILE, "r") as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return {}

def save_chat_history(history):
    with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)

def load_data_file(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == '.csv': return pd.read_csv(file_path)
    if ext == '.xlsx': return pd.read_excel(file_path)
    if ext == '.pdf':
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            tbls = [pd.DataFrame(p.extract_table()[1:], columns=p.extract_table()[0]) for p in pdf.pages if p.extract_table()]
            return pd.concat(tbls) if tbls else None
    return None

def get_user_intent(llm, user_prompt):
    prompt = f"Categorize intent: 'data_query' or 'visualization_request'. Input: '{user_prompt}'. Output one word."
    try: return llm.invoke(prompt).content.strip().lower()
    except: return "data_query"

# --- Initialization ---
if "all_chats" not in st.session_state: st.session_state.all_chats = load_chat_history()
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0] if st.session_state.all_chats else None

# --- Sidebar UI ---
with st.sidebar:
    st.title("🗂️ Chat Sessions")
    
    if st.button("➕ New Chat", use_container_width=True):
        cid = f"chat_{int(time.time())}"
        st.session_state.all_chats[cid] = {
            "title": "New Session", 
            "messages": [{"role": "assistant", "content": "Hello! Choose a source and let's analyze."}],
            "file_path": None,
            "source": "MySQL Database"
        }
        st.session_state.active_chat_id = cid
        save_chat_history(st.session_state.all_chats)
        st.rerun()

    active_id = st.session_state.active_chat_id
    if active_id:
        chat_data = st.session_state.all_chats[active_id]
        st.write("---")
        st.header("📂 Data Source")
        
        # Extended Source Options
        source_options = ["MySQL Database", "PostgreSQL", "Oracle", "MongoDB", "Upload File"]
        current_idx = source_options.index(chat_data.get("source", "MySQL Database"))
        source_type = st.radio("Current Source:", source_options, index=current_idx)
        chat_data["source"] = source_type
        
        if source_type == "Upload File":
            u_file = st.file_uploader("Drop CSV/Excel/PDF", type=["csv", "xlsx", "pdf"])
            if u_file:
                path = UPLOAD_DIR / u_file.name
                with open(path, "wb") as f: f.write(u_file.getbuffer())
                chat_data["file_path"] = str(path)
                st.success(f"Loaded: {u_file.name}")
        
        st.write("---")
        for cid in reversed(list(st.session_state.all_chats.keys())):
            c = st.session_state.all_chats[cid]
            col1, col2 = st.columns([0.8, 0.2])
            if col1.button(c['title'], key=f"s_{cid}", use_container_width=True):
                st.session_state.active_chat_id = cid
                st.rerun()
            if col2.button("🗑️", key=f"d_{cid}"):
                del st.session_state.all_chats[cid]
                save_chat_history(st.session_state.all_chats)
                st.rerun()

# Get LLM and Chain for active source
llm, chain, engine = get_llm_and_chain(chat_data["source"])

# --- Main Interface ---
st.title("🤖 AI Data Agent")

if active_id:
    active_chat = st.session_state.all_chats[active_id]

    for m in active_chat["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if "dataframe" in m: st.dataframe(pd.read_json(m["dataframe"], orient="split"))
            if "chart_code" in m:
                df = pd.read_json(m["dataframe_for_chart"], orient="split")
                exec(m["chart_code"])

    if prompt := st.chat_input("Analyze my data..."):
        active_chat["messages"].append({"role": "user", "content": prompt})
        if active_chat["title"] == "New Session": active_chat["title"] = prompt[:25] + "..."
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                intent = get_user_intent(llm, prompt)

                if intent == "data_query":
                    if active_chat["source"] == "Upload File" and active_chat.get("file_path"):
                        full_df = load_data_file(active_chat["file_path"])
                        agent = create_pandas_dataframe_agent(llm, full_df, verbose=False, allow_dangerous_code=True, agent_executor_kwargs={"handle_parsing_errors": True})
                        agent_response = agent.invoke(f"Answer: {prompt}")
                        answer = agent_response["output"]
                        
                        code_prompt = f"Write one-line pandas code to get '{prompt}' from a df named 'df'. Code only."
                        filter_code = llm.invoke(code_prompt).content.replace("```python", "").replace("```", "").strip()

                        try:
                            local_vars = {'df': full_df, 'pd': pd}
                            exec(f"result_df = {filter_code}", {}, local_vars)
                            filtered_df = local_vars['result_df']
                        except:
                            filtered_df = full_df.head(10)
                        
                        st.markdown(answer)
                        st.dataframe(filtered_df)
                        active_chat["messages"].append({"role": "assistant", "content": answer, "dataframe": filtered_df.to_json(orient="split")})

                    elif active_chat["source"] == "MongoDB":
                        # Mongo requires specialized toolkit
                        st.info("🔄 MongoDB Tooling is being initialized...")
                        answer = "MongoDB functionality requires the database to be up. Check terminal for MQL."
                        st.markdown(answer)
                    
                    else:
                        # SQL PROCESSING (MySQL, Postgres, Oracle)
                        raw_sql = st.write_stream(chain.stream({"question": prompt}))
                        sql = raw_sql.replace("```sql", "").replace("```", "").strip()
                        df = pd.read_sql_query(sql, engine)
                        st.dataframe(df)
                        active_chat["messages"].append({"role": "assistant", "content": f"**SQL Generated:**\n```sql\n{sql}\n```", "dataframe": df.to_json(orient="split")})

                elif intent == "visualization_request":
                    last_df_data = next((m["dataframe"] for m in reversed(active_chat["messages"]) if "dataframe" in m), None)
                    if last_df_data:
                        df = pd.read_json(last_df_data, orient="split")
                        viz_prompt = f"Generate Streamlit code to visualize: {prompt}. DF name is `df`."
                        code = llm.invoke(viz_prompt).content.replace("```python", "").replace("```", "").strip()
                        exec(code)
                        active_chat["messages"].append({"role": "assistant", "content": "Visualizing your data now.", "chart_code": code, "dataframe_for_chart": last_df_data})

        save_chat_history(st.session_state.all_chats)
        st.rerun()