from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# =====================================================
# LOAD MODELS
# =====================================================

binary_model = joblib.load("healthcare_Binary_pipeline.joblib")

multi_model = joblib.load("Healthcare_Multiclass_pipeline.joblib")

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI()

# =====================================================
# INPUT SCHEMA
# =====================================================

class PatientData(BaseModel):

    Hospital_code: int
    Hospital_type_code: str
    City_Code_Hospital: int
    Hospital_region_code: str
    Available_Extra_Rooms_in_Hospital: int
    Department: str
    Ward_Type: str
    Ward_Facility_Code: str
    Bed_Grade: float
    City_Code_Patient: float
    Type_of_Admission: str
    Severity_of_Illness: str
    Visitors_with_Patient: int
    Age: str
    Admission_Deposit: int

# =====================================================
# PREDICTION API
# =====================================================

@app.post("/predict/{model_type}")

def predict(model_type: str, data: PatientData):

    input_dict = data.dict()

    df = pd.DataFrame([input_dict])

    # =================================================
    # BINARY MODEL
    # =================================================

    if model_type == "binary":

        prediction = binary_model.predict(df)[0]

        probabilities = binary_model.predict_proba(df)[0]

        confidence = np.max(probabilities)

        return {

            "prediction": int(prediction),

            "confidence_score": round(
                float(confidence * 100), 2
            ),

            "probabilities": {

                "Class_0": round(
                    float(probabilities[0] * 100), 2
                ),

                "Class_1": round(
                    float(probabilities[1] * 100), 2
                )
            }
        }

    # =================================================
    # MULTI CLASS MODEL
    # =================================================

    elif model_type == "multiclass":

        prediction = multi_model.predict(df)[0]

        probabilities = multi_model.predict_proba(df)[0]

        confidence = np.max(probabilities)

        classes = multi_model.classes_

        probability_dict = {

            str(classes[i]): round(
                float(probabilities[i] * 100), 2
            )

            for i in range(len(classes))
        }

        return {

            "prediction": int(prediction),

            "confidence_score": round(
                float(confidence * 100), 2
            ),

            "all_class_probabilities": probability_dict
        }

    # =================================================
    # INVALID MODEL
    # =================================================

    else:

        return {
            "error": "Invalid model type"
        }