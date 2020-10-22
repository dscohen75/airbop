import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import pandas as pd
from pydantic import BaseModel, conint

description = """
AirBOP deploys a linear regression model fit on the US Cities AirBnB data.
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title = "AirBOP",
    docs_url='/'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*']
)

model = load('linregmodel.joblib')

class Airbnb(BaseModel):
    """Parse & validate airbnb features """
    bedrooms: conint(gt=-1, lt=11)
    bathrooms: conint(gt=-1, lt=9)

    def to_df(self):
        """Convert to pandas dataframe with 1 row """
        return pd.DataFrame([dict(self)])


@app.post('/predict')
def predict_price(airbnb: Airbnb):
    """Predict airbnb price"""
    df = airbnb.to_df()
    price = model.predict(df)
    return price[0]

@app.get('/random')
def random_price():
    """Return a random integer price """
    price = random.randint(50,2000)
    return price