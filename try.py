from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

os.environ["AWS_PROFILE"] = "ArribIAM"

# Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

modelID = "anthropic.claude-v2"

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
    return response


st.title("IT Job Description Bullet Points Generator")

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

selected_role = st.sidebar.selectbox("Select IT Role", it_roles)

if selected_role:
    job_description = st.sidebar.text_area("Enter Job Description", max_chars=1000)

if job_description:
    response = my_chatbot(selected_role, job_description)
    bullet_points = response['text'].split('\n')
    st.write("Generated Bullet Points:")
    for point in bullet_points:
        st.write(f"- {point}")
