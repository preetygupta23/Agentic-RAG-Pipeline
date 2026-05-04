from pydantic import BaseModel, Field

class GuardrailResult(BaseModel):
    is_compliant: str = Field(
        description="Output the exact string 'yes' if the response adheres to policies, or 'no' if it violates them."
    )
    violation_reason: str = Field(
        description="If is_compliant is 'no', briefly explain why. If 'yes', you MUST output the exact string 'N/A'."
    )