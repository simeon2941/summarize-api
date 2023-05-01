from transformers import pipeline
import urllib.request
from bs4 import BeautifulSoup
from fastapi import FastAPI,Response
from pydantic import BaseModel

app = FastAPI()

class SummarizeRequest(BaseModel):
    url: str


def extract_text(url):
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, "html.parser")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return text

def process(text):
    sumarizer = pipeline("summarization", model='t5-base', tokenizer='t5-base',truncation=True, framework="tf")
    result = sumarizer(text, min_length=30, truncation=True)
    return result[0]['summary_text']

import validators

@app.post("/summarize")
def summarize(request: SummarizeRequest):
    url = request.url

    # Validate the URL
    if not validators.url(url):
        return Response("Invalid URL", status_code=400)

    text = extract_text(url)
    summary = process(text)
    return Response(summary)



@app.get("/")
def root():
    return Response("<h1>Summarizer API</h1><p>Go to /docs to see the documentation.</p>", media_type="text/html")

