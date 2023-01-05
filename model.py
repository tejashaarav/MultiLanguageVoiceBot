from gradio import Audio, Interface, Textbox
from utils import TextGen, to_english,stt, to_language,  tts, tts_to_bytesio, html_audio_autoplay

max_answer_length = 100
desired_language = "hi"
response_generator_pipe = TextGen(max_length=max_answer_length)

def main(audio):
    user_speech_text = stt(audio,desired_language )
    translated_text = to_english(user_speech_text, desired_language)
    bot_response_english = response_generator_pipe(translated_text)
    bot_response_translated = to_language(bot_response_english, desired_language)
    bot_audio_languge = tts(bot_response_translated, desired_language)
    bot_audio_in_bytes = tts_to_bytesio(bot_audio_languge)
    html = html_audio_autoplay(bot_audio_in_bytes)
    return user_speech_text, bot_response_translated, bot_response_english, html

Interface(
    fn=main,
    inputs=[
        Audio(
            source="microphone",
            type="filepath",
        ),
    ],
    outputs=[
        Textbox(label="You said: "),
        Textbox(label="AI said: "),
        Textbox(label="AI said (English): "),
        "html",
    ],
    live=True,
    allow_flagging="never",
).launch(share = True)
