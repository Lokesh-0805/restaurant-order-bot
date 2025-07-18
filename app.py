import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gtts import gTTS
from openai import OpenAI
import openai 

load_dotenv()

# Setup environment variables
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


# Setup LangChain
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-4o")
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant that listens to messy restaurant orders and formats them clearly.\n"
     "Use only the following menu items:\n"
     "- Paneer Tikka\n"
     "- Veg Biryani\n"
     "- Butter Naan\n"
     "- Dal Makhani\n"
     "- Masala Dosa\n"
     "- Cold Coffee\n"
     "- Brownie with Ice Cream\n\n"
     "Your job is to:\n"
     "1. Extract items from the user's message that match the menu.\n"
     "2. Ignore irrelevant or unlisted items.\n"
     "3. Format the output as:\n"
     "   Your order:\n"
     "   <quantity>  <Menu Item>\n"
     "   ... (one per line)\n"
     "   Spice level: <value> (default to 'Normal' if not specified)\n\n"
     "If no menu items are found, respond with:\n"
     "   No recognizable items found. Please repeat your order clearly.\n"
     "Only include items that appear in the menu above."
    ),
    ("user", "{input}")
])




output_parser = StrOutputParser()
chain = prompt | llm | output_parser

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    response_text = ""
    if request.method == "POST":
        import base64

        audio_data = request.form.get("recordedAudio")
        if audio_data:
            header, encoded = audio_data.split(",", 1)
            audio_bytes = base64.b64decode(encoded)
            filepath = os.path.join("audio", "audio1.mp3")
            with open(filepath, "wb") as f:
                f.write(audio_bytes)

            with open(filepath, "rb") as audio_file:
                transcript = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file
                )       

            transcription = transcript.text


            # Clean the order
            response = chain.invoke({"input": transcription})
            response_text = response

            # Convert to speech
            tts = gTTS(response_text, lang='en', slow=False)
            tts.save("static/output.mp3")

    return render_template("index.html", order=response_text)

if __name__ == "__main__":
    app.run(debug=True)
