import sys
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging
from src.my_project.pipelines.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Credit Card Customer Intelligence Platform")

app.mount("/static", StaticFiles(directory="fastapi_app/static"), name="static")
templates = Jinja2Templates(directory="fastapi_app/templates")

# Loaded once at startup, reused across requests
prediction_pipeline = PredictionPipeline()


class CustomerInput(BaseModel):
    Customer_Age: int
    Dependent_count: int
    Months_on_book: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: float
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: float
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float
    Gender: str
    Education_Level: str
    Marital_Status: str
    Income_Category: str
    Card_Category: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    try:
        form = await request.form()
        customer_data = {
            "Customer_Age": int(form["Customer_Age"]),
            "Dependent_count": int(form["Dependent_count"]),
            "Months_on_book": int(form["Months_on_book"]),
            "Total_Relationship_Count": int(form["Total_Relationship_Count"]),
            "Months_Inactive_12_mon": int(form["Months_Inactive_12_mon"]),
            "Contacts_Count_12_mon": int(form["Contacts_Count_12_mon"]),
            "Credit_Limit": float(form["Credit_Limit"]),
            "Total_Revolving_Bal": float(form["Total_Revolving_Bal"]),
            "Avg_Open_To_Buy": float(form["Avg_Open_To_Buy"]),
            "Total_Amt_Chng_Q4_Q1": float(form["Total_Amt_Chng_Q4_Q1"]),
            "Total_Trans_Amt": float(form["Total_Trans_Amt"]),
            "Total_Trans_Ct": int(form["Total_Trans_Ct"]),
            "Total_Ct_Chng_Q4_Q1": float(form["Total_Ct_Chng_Q4_Q1"]),
            "Avg_Utilization_Ratio": float(form["Avg_Utilization_Ratio"]),
            "Gender": form["Gender"],
            "Education_Level": form["Education_Level"],
            "Marital_Status": form["Marital_Status"],
            "Income_Category": form["Income_Category"],
            "Card_Category": form["Card_Category"],
        }

        result = prediction_pipeline.predict(customer_data)
        return templates.TemplateResponse(request, "index.html", {"result": result})

    except Exception as e:
        logging.error(f"Prediction request failed: {e}")
        error_result = {"error": str(e)}
        return templates.TemplateResponse(request, "index.html", {"result": error_result})


# JSON API, for programmatic access (Power BI,testing,future frontend)
@app.post("/api/predict")
async def predict_api(customer: CustomerInput):
    try:
        result = prediction_pipeline.predict(customer.model_dump())
        return result
    except Exception as e:
        raise CustomException(e, sys)


@app.get("/health")
async def health():
    return {"status": "ok"}