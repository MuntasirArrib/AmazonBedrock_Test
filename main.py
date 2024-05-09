from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st
import json

# Set AWS credentials
os.environ["AWS_PROFILE"] = "ArribIAM"

# Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

modelID = "anthropic.claude-v2"

# Initialize Bedrock LLM
llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_tokens_to_sample": 1000, "temperature": 0.9}
)

def my_chatbot(role, job_description):
    prompt = PromptTemplate(
        input_variables=["role", "job_description"],
        template=f"You are a Career Coach. Consider the role '{role}' selected by the user and the job description provided:\n\n{job_description}\n\nGenerate 3 relevant bullet points for their resume:"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    response = bedrock_chain({'role': role, 'job_description': job_description})
    bullet_points = response['text'].split('\n')
    bullet_points = [point for point in bullet_points if point]  # Remove empty lines
    return bullet_points

# Streamlit interface
st.title("Resume Bullet Points Generator for IT Professionals")

# Define IT roles
it_roles = [
    "Systems Analyst",
    "Software Engineer",
    "Network Administrator",
    "Database Administrator",
    "Web Developer",
    "Cybersecurity Analyst",
    "IT Project Manager",
    "Cloud Consultant",
    "IT Support Engineer",
    "Data Scientist",
    "DevOps Engineer",
    "Quality Assurance (QA) Engineer",
    "Business Analyst",
    "IT Consultant",
    "Technical Writer"
]

# Sidebar for user input
selected_role = st.sidebar.selectbox("Select IT Role", it_roles)

# Text area for job description
if selected_role:
    st.sidebar.header("Enter Job Description")
    job_description = st.sidebar.text_area("Describe the job in detail", max_chars=1000)

# Button to generate bullet points
if job_description:
    st.sidebar.write("")  # Add space
    st.sidebar.write("")  # Add space
    if st.sidebar.button("Generate Bullet Points", key="generate_button"):
        bullet_points = my_chatbot(selected_role, job_description)
        st.header("Generated Bullet Points:")
        for point in bullet_points:
            st.write(point)
