from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deep_translator import GoogleTranslator

app = FastAPI()

# Supported language mapping 
LANG_CODE = {
    "English": "en",
    "Spanish": "es", 
    "French":  "fr",
    "Italian": "it",
}
# Defines the expected JSON body
class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.post("/translate")
def translate(req: TranslateRequest):
    # If source and target are the same, skip translation
    if req.source_lang == req.target_lang:
        return {"translated_text": req.text}
    src = LANG_CODE.get(req.source_lang)
    dest = LANG_CODE.get(req.target_lang)
    if src is None or dest is None:
        raise HTTPException(status_code=400, detail=f"supported languages: {list(LANG_CODE)}")
    result = GoogleTranslator(source=src, target=dest).translate(req.text)
    return {"translated_text": result}

@app.get("/ok")
def ok():
    return {"status": "ok"}
