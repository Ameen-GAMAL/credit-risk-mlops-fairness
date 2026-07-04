"""Pydantic request/response contracts for the credit-risk API.

The request contains ONLY model features — protected attributes
(`personal_status`/`age` and the derived `sex`/`age_binary`) are deliberately
absent: the model never sees them, at training or inference time.
"""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "checking_status": "<0",
                "duration": 24,
                "credit_history": "existing paid",
                "purpose": "radio/tv",
                "credit_amount": 3500,
                "savings_status": "<100",
                "employment": "1<=X<4",
                "installment_commitment": 3,
                "other_parties": "none",
                "residence_since": 2,
                "property_magnitude": "car",
                "other_payment_plans": "none",
                "housing": "own",
                "existing_credits": 1,
                "job": "skilled",
                "num_dependents": 1,
                "own_telephone": "yes",
                "foreign_worker": "yes",
            }
        }
    )

    checking_status: str
    duration: float = Field(ge=0)
    credit_history: str
    purpose: str
    credit_amount: float = Field(ge=0)
    savings_status: str
    employment: str
    installment_commitment: float
    other_parties: str
    residence_since: float
    property_magnitude: str
    other_payment_plans: str
    housing: str
    existing_credits: float
    job: str
    num_dependents: float
    own_telephone: str
    foreign_worker: str

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probability_good: float
    model_run_id: str | None
