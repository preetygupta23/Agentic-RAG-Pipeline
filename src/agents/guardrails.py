import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.utils.schemas import GuardrailResult

# 1. Load the variables from the .env file
load_dotenv()

# 2. Initialize the Groq LLM
llm = ChatGroq(
    model_name=os.environ.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant"),
    temperature=0
)


def guardrail_node(state: dict):
    """
    Evaluates the generated response against system policies.
    """
    user_query = state.get("question")
    generated_response = state.get("generation")

    guardrail_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict compliance evaluator for an AI system. 
        Your job is to review the AI's generated response to the user's query.

        POLICY: The AI MUST NOT discuss the weather, current events, or anything outside the provided PDF context.

        Analyze the response. If it violates the POLICY, flag it as non-compliant."""),
        ("user", "USER QUERY: {query}\n\nAI RESPONSE: {response}")
    ])

    # Bind the Pydantic schema to force structured JSON output
    evaluator_chain = guardrail_prompt | llm.with_structured_output(GuardrailResult)

    result = evaluator_chain.invoke({"query": user_query, "response": generated_response})

    # Convert the string "yes"/"no" back to a Python boolean (True/False)
    return {
        "is_safe": result.is_compliant.lower() == "yes",
        "violation_reason": result.violation_reason
    }