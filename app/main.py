#!/usr/bin/env python
# coding: utf-8

# In[9]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import re
import scipy


import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from io import BytesIO, StringIO
import cv2
import numpy as np
import logging
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw
import os
import cv2
import nest_asyncio
import uvicorn
import base64




# In[10]:


pip install aiofiles


# In[11]:


pip install fastapi uvicorn


# In[12]:


pip install openai


# In[13]:


from fastapi import FastAPI
import openai


# In[14]:


pip install openai==0.28


# In[15]:


openai.api_key = 'e88a6209068a430a8cba0fede006b220'
openai.api_base =  'https://imaginemvp-zenflo.openai.azure.com/' 
openai.api_type = 'azure' # Necessary for using the OpenAI library with Azure OpenAI
openai.api_version = '2023-12-01-preview' # Latest / target version of the API

deployment_name = 'Trial'
prompt="What is Azure OpenAI?"

response = openai.ChatCompletion.create(
    engine=deployment_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
)

print(type(response['choices'][0]['message']['content']),response['choices'][0]['message']['content'],response,type(response))


# In[16]:


from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory=r"C:\Users\envy\OneDrive\Desktop\Templates")

class SymptomData(BaseModel):
    symptoms: str

class LocationData(BaseModel):
    location: str

openai.api_key = 'e88a6209068a430a8cba0fede006b220'
openai.api_base =  'https://imaginemvp-zenflo.openai.azure.com/' 
openai.api_type = 'azure' 
openai.api_version = '2023-12-01-preview' 

deployment_name = 'Trial'

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    

@app.post("/get_asana")
async def get_asana(symptom_data: SymptomData):
    prompt = symptom_data.symptoms
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[
            {"role": "system", "content": '''You are a Yoga AI assistant that gives answers always in 2 or 3 bullet points.
            If the user enters a symptom, body part or a specific area of body where the user is suffering pain, then give the user ⅔ recommendations for yoga asanas, in bullet points for separate headings of the descriptions, benefits, contradictions if any.
            If the user enters a location  north/east/west/south/central, he means  yoga studios in north/east/west/south/central Singapore so if the user enters central it means central Singapore, north means north Singapore, west means west Singapore, east means east Singapore, south means south Singapore. Refer to the dataset column N/S/E/W then to give the user ⅔ yoga studio recommendations in bullet points with address. Do not recommend the user to use a search engine.
            For example, if the user enters knee pain, search data sources and tell them top 2 yoga asanas for it and their entire details from the data source. If they enter Jurong, you tell them that Jurong is in West singapore and according to dataset these are the top 2 yoga studios in west singapore. If the user asks something not in the dataset, don't tell the user that they've asked something out of data sources, just tell them the answer you would've otherwise if the data sources were not there, again in 2/3 pointers.
            STRICTLY, do not ever display ' The requested information is not available in the retrieved data. Please try another query or topic', if there is such a case then just give the answer that you would have just using your LLM training'''}, 
            {"role": "user", "content": prompt}
        ]
    )
    asana = response['choices'][0]['message']['content']
    return {"asana": asana}

@app.post("/find_studio")
async def find_studio(location_data: LocationData):
    prompt = location_data.location
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[
            {"role": "system", "content": '''You are a Yoga AI assistant that gives answers always in 2 or 3 bullet points.
            If the user enters a symptom, body part or a specific area of body where the user is suffering pain, then give the user ⅔ recommendations for yoga asanas, in bullet points for separate headings of the descriptions, benefits, contradictions if any.
            If the user enters a location  north/east/west/south/central, he means  yoga studios in north/east/west/south/central Singapore so if the user enters central it means central Singapore, north means north Singapore, west means west Singapore, east means east Singapore, south means south Singapore. Refer to the dataset column N/S/E/W then to give the user ⅔ yoga studio recommendations in bullet points with address. Do not recommend the user to use a search engine.
            For example, if the user enters knee pain, search data sources and tell them top 2 yoga asanas for it and their entire details from the data source. If they enter Jurong, you tell them that Jurong is in West singapore and according to dataset these are the top 2 yoga studios in west singapore. If the user asks something not in the dataset, don't tell the user that they've asked something out of data sources, just tell them the answer you would've otherwise if the data sources were not there, again in 2/3 pointers.
            STRICTLY, do not ever display ' The requested information is not available in the retrieved data. Please try another query or topic', if there is such a case then just give the answer that you would have just using your LLM training'''}, 
            {"role": "user", "content": prompt}
        ]
    )
    location = response['choices'][0]['message']['content']
    return {"location": location}



if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app)
