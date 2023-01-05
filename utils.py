import numpy as np
import pandas as pd
from gtts import gTTS
from gradio import Audio, Interface, Textbox
from mtranslate import translate
import speech_recognition as sr
from transformers import BlenderbotSmallForConditionalGeneration,BlenderbotSmallTokenizer
from base64 import b64encode
from io import BytesIO


def stt(audio, language):
    r = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data, language= language)
    return text

def to_english(text, language):
    return translate(text, "en", language)

def to_language(text, language):
    return translate(text, language, "en")

def tts(text, language):
    return gTTS(text=text, lang=language, slow = False)
    
def tts_to_bytesio(tts_object):
    bytes_object = BytesIO()
    tts_object.write_to_fp(bytes_object)
    bytes_object.seek(0)
    return bytes_object.getvalue()

def html_audio_autoplay(bytes):
    b64 = b64encode(bytes).decode()
    html = f"""
    <audio controls autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    return html



class TextGen:

    tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
    model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")


    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def preprocess(self, text):
        return self.tokenizer(text, return_tensors = "pt")
    
    def postprocess(self,outputs):
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def __call__(self, text: str) -> str:
        tokenized_text = self.preprocess(text)
        output = self.model.generate(**tokenized_text, **self.__dict__)
        return self.postprocess(output)





    
