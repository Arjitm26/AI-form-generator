import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import firebase_admin as fa
from firebase_admin import auth
import requests
import pandas as pd
import time
import os
import jwt
import datetime
# Load environment variables and initialize Firebase once
load_dotenv()

FIREBASE_API_KEY = os.environ["FIREBASE_API_KEY"]  # Move to secrets management
jwtkey = os.environ["jwt_key"]
cred = os.environ["cred"]

# Initialize Firebase only once
if not fa._apps:
    cred = fa.credentials.Certificate(cred)
    fa.initialize_app(cred)

# Page config
st.set_page_config(
    page_title="Generate your form",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Cache the LLM instance
@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-specdec", temperature=0.5)

# Cache the prompt templates
@st.cache_resource
def get_prompt_templates():
    system_prompt = """
        You are a form generation expert. Generate a survey form in JSON format based on the provided specifications.

        Input Parameters:-

        form_description: description about form,
        goal: goal of the form,
        target_audience: audience addressing,
        tone: tonality,
        question_count: 1-8 or 8-16 or 17-24 or 24+,
        question_types: miltiple choice or plain-text or slider,
        additional_context: additional requirements or NULL,

        Instructions:
        - Generate questions that directly align with the stated goals
        - Match the specified tone and style
        - Use only the requested question types
        - Create logical question flow from general to specific topics
        - Ensure questions are clear, unambiguous, and free from bias
        - Return response in valid JSON format

        Expected Output Format:
        {{
        "form_metadata": {{
            "title": string,
            "description": string,
            "target_audience": string
        }},
        "questions": [
            {{
            "id": integer,
            "question_text": string,
            "question_type": string,
            "options": [string] | null
            }}
        ]
        }}

        Return a complete, valid JSON object following this exact structure. 
        Ensure all question types match the input specifications and the total number of questions matches the requested question_count.
        """  
    examples = [
        {
            "input" :   """
                        form_description:   A survey to evaluate customer satisfaction for our mobile app. The app focuses on health tracking 
                                            and meal planning. We want to understand user satisfaction with the features and usability.,
                        goal: Gather feedback on app usability, key pain points, and feature suggestions.,
                        target_audience: Active users who have used the app for at least one month.,
                        tone: engaging,
                        question_count: 1-8,
                        question_types: slider,multiple choice,open-ended text,
                        additional_context: Include a question about How would you rate the app's ease of use. 
                                            use slider as question input. End the form with a thank-you note. """,
            "output" : """ 
                {
                "form_metadata": {
                    "description": A survey to evaluate customer satisfaction for our mobile app. The app focuses on health tracking 
                                and meal planning. We want to understand user satisfaction with the features and usability.,
                    "target_audience": Active users who have used the app for at least one month.
                },
                "questions": [
                    {
                    "id": 1,
                    "question_text": "How would you rate the app's ease of use on a scale of 1-10?",
                    "question_type": "Slider",
                    "options": null
                    },
                    {
                    "id": 2,
                    "question_text": ""Which feature do you use most often?",
                    "question_type": "Multiple-choice",
                    "options": ["Health Tracking", "Meal Planning", "Reminders"]
                    }
                    ....upto 8
                    ]
                }
                    """
            },
            {"input" :   """
                        form_description: An anonymous feedback form for employees to share opinions on workplace culture, 
                                        team collaboration, and management effectiveness.,
                        goal: Gather feedback on app usability, key pain points, and feature suggestions.,
                        target_audience: All employees, across departments and experience levels.,
                        tone: formal,
                        question_count: 1-8,
                        question_types: Use mostly Likert scale and open-ended text.,
                        additional_context: Null
            """,
            "output" : """ 
                {
                "form_metadata": {
                    "title" : "Internal Feedback Form",
                    "description": "An anonymous feedback form for employees to share opinions on workplace culture, 
                                    team collaboration, and management effectiveness.",
                    "target_audience": "All employees, across departments and experience levels."
                },
                "questions": [
                    {
                    "id": 1,
                    "question_text": "On a scale of 1-10, how would you rate the company's overall culture?",
                    "question_type": "Slider",
                    "options": null
                    },
                    {
                    "id": 2,
                    "question_text": "What changes would you recommend to improve collaboration?",
                    "question_type": "open-ended text",
                    "options": Null
                    },
                    {
                    "id": 2,
                    "question_text": "How often do you feel supported by your team members?",
                    "question_type": "slider",
                    "options": Null
                    }
                    ]
                }
                """
     }
     ]  # Your examples here
    
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    fewshot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=base_prompt
    )
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful form generator"),
        fewshot_prompt,
        ("human", "{input}")
    ])
    
    return final_prompt

# Cache Firebase authentication
@st.cache_data(ttl=3600)  
def authenticate_user(email, password):
    firebase_auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }
    response = requests.post(firebase_auth_url, json=payload)
    if response.status_code == 200:
        # Create a session token
        auth_data = response.json()
        session_token = jwt.encode(
            {
                'email': email,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            },
            jwtkey,  # Store this in environment variables
            algorithm='HS256'
        )
        return response, session_token
    return response, None

# Cache form generation
@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_form(user_input):
    llm = get_llm()
    prompt = get_prompt_templates()
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({"input": user_input})

def convert_to_csv(input_data):
    return pd.DataFrame(input_data['questions']).to_csv().encode("utf-8")

def display_form(form_data):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(form_data["form_metadata"]['title'])
        st.write(form_data["form_metadata"]['description'])
        
        for i, ques in enumerate(form_data['questions'], 1):
            st.subheader(f"Q{i}. {ques['question_text']}")
            st.caption(f"Type: {ques['question_type']}")
            
            if ques['options']:
                st.write("Options:")
                for opt in ques['options']:
                    st.write(f"- {opt}")

def verify_token():
    if 'token' in st.session_state:
        try:
            token = st.session_state.token
            decoded = jwt.decode(token, jwtkey, algorithms=['HS256'])
            # Check if token is expired
            exp = datetime.datetime.fromtimestamp(decoded['exp'])
            if exp > datetime.datetime.utcnow():
                return True
        except:
            return False
    return False


def login_page():
    st.header("Login")
    
    # First check if user is already authenticated
    if verify_token():
        st.session_state.authenticated = True
        st.session_state.page = "main_page"
        st.rerun()
        
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit and email and password:
            with st.spinner("Authenticating..."):
                response, token = authenticate_user(email, password)
                if response.status_code == 200 and token:
                    st.session_state.token = token  # Store token in session state
                    st.session_state.authenticated = True
                    st.session_state.page = "main_page"
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

def signup_page():
    st.header("Sign Up")
    with st.form("signup_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        repassword = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")
        
        if submit:
            if password != repassword:
                st.error("Passwords don't match")
            elif all([username, email, password]):
                try:
                    user = auth.create_user(
                        email=email,
                        password=password,
                        uid=username
                    )
                    st.success("Account created successfully!")
                    st.session_state.page = "login"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def main_page():
    st.title("Form Generator")
    
    result = None  # Initialize result variable outside form
    
    with st.form("form_generator"):
        form_description = st.text_area("Form Description")
        goal = st.text_area("Goals")
        target = st.text_input("Target Audience")
        tone = st.selectbox("Tone", ["formal", "engaging", "casual"])
        questions = st.select_slider(
            "Number of Questions",
            options=["1-8", "8-16", "17-24", "24+"]
        )
        type = st.multiselect(
            "Question Types",
            ["Multiple choice", "Likert scale", "open-ended text", "Multiselect"]
        )
        additional = st.text_area("Additional Instructions", value="None")
        
        submit = st.form_submit_button("Generate Form")
        
        if submit:
            if all([form_description, goal, target, tone, questions, type]):
                with st.spinner("Generating form..."):
                    user_input = f"""
                    form_description: {form_description},
                    goal: {goal},
                    target_audience: {target},
                    tone: {tone},
                    question_count: {questions},
                    question_types: {type},
                    additional_context: {additional}
                    """
                    
                    result = generate_form(user_input)
            else:
                st.error("Please fill all required fields")
    
    if result:
        display_form(result)

        st.write('-'*100)
        
        csv_data = convert_to_csv(result)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="form.csv",
            mime="text/csv"
        )

# Main app logic
def init_session_state():
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "token" not in st.session_state:
        st.session_state.token = None


def main():
    # Initialize session state
    init_session_state()

    # Verify authentication token (if exists) on every refresh
    if st.session_state.token and not st.session_state.authenticated:
        if verify_token():
            st.session_state.authenticated = True
            st.session_state.page = "main_page"
        else:
            st.session_state.authenticated = False
            st.session_state.token = None
            st.session_state.page = "login"

    # Sidebar navigation
    if st.session_state.authenticated:
        if st.sidebar.button("Logout"):
            st.session_state.clear()  # Clear all session state
            st.session_state.page = "login"
            st.rerun()

    # Page routing
    if st.session_state.page == "login" and not st.session_state.authenticated:
        login_page()
        if st.button("Create Account"):
            st.session_state.page = "signup"
            st.rerun()
    elif st.session_state.page == "signup":
        signup_page()
        if st.button("Back to Login"):
            st.session_state.page = "login"
            st.rerun()
    elif st.session_state.authenticated:
        main_page()
    else:
        st.session_state.page = "login"
        st.rerun()

if __name__ == "__main__":
    main()