import os
import librosa
import numpy as np
import json  # To parse Vosk results
from flask import Flask, render_template, request, redirect, flash
from vosk import Model, KaldiRecognizer
from textblob import TextBlob  # For sentiment analysis
import pyaudio  # To capture live audio
import pyttsx3  # For text-to-speech (TTS)
import wave  # To save audio files
from flask import jsonify
import requests
import spacy
import language_tool_python
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Replace with a strong secret key
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER1 = 'uploads1/'
app.config['UPLOAD_FOLDER1'] = UPLOAD_FOLDER1

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER1):
    os.makedirs(UPLOAD_FOLDER1)

# Load the Vosk model for transcription
model_path = r"D:\ai_eduhacks\vosk-model-en-us-0.42-gigaspeech"  # Update this to your model path
if not os.path.exists(model_path):
    raise RuntimeError(f"Model not found at {model_path}")
model = Model(model_path)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load and process the audio file for analysis
def load_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    return y, sr

# Analyze speech characteristics
def analyze_pace(y, sr):
    y_trimmed, _ = librosa.effects.trim(y)
    active_duration = librosa.get_duration(y=y_trimmed, sr=sr)
    pace = len(y_trimmed) / active_duration if active_duration > 0 else 0
    return pace

def analyze_volume(y):
    rms = librosa.feature.rms(y=y)
    average_volume = rms.mean()
    return average_volume

def analyze_pitch(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    if len(pitches) > 0:
        pitch = np.mean(pitches)
    else:
        pitch = 0
    return pitch

def analyze_clarity(y, sr):
    non_silent_intervals = librosa.effects.split(y, top_db=20)
    total_duration = librosa.get_duration(y=y, sr=sr)
    non_silent_duration = sum((end - start) for start, end in non_silent_intervals) / sr
    clarity_score = non_silent_duration / total_duration if total_duration > 0 else 0
    return clarity_score

def transcribe_speech(y, sr):
    recognizer = KaldiRecognizer(model, sr)
    y_int16 = (y * 32767).astype(np.int16)
    bytes_data = y_int16.tobytes()
    recognizer.AcceptWaveform(bytes_data)
    result = recognizer.Result()
    result_dict = json.loads(result)
    transcription = result_dict.get('text', '')
    return transcription

# Provide feedback based on analyzed characteristics
def give_feedback(pace, volume, pitch, clarity):
    feedback = []

    if pace < 120:
        feedback.append("Your speech is too slow. Try speeding up.")
    elif pace > 150:
        feedback.append("Your speech is too fast. Try slowing down.")
    else:
        feedback.append("Your speech pace is optimal.")

    if volume < 0.05:
        feedback.append("Your volume is too low. Speak louder for better clarity.")
    elif volume > 0.2:
        feedback.append("Your volume is too high. Try lowering your voice slightly.")
    else:
        feedback.append("Your volume is good.")

    if pitch < 100:
        feedback.append("Your pitch is too low. Try to raise your voice a little.")
    elif pitch > 300:
        feedback.append("Your pitch is too high. Try to lower your voice slightly.")
    else:
        feedback.append("Your pitch is in a good range.")

    if clarity < 0.5:
        feedback.append("Your speech clarity could be improved. Avoid unnecessary pauses.")
    else:
        feedback.append("Your speech clarity is good.")

    return feedback

def analyze_sentiment(transcription):
    analysis = TextBlob(transcription)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Analyze the audio file and return transcription and feedback
def analyze_audio(file_path):
    y, sr = load_audio(file_path)
    pace = analyze_pace(y, sr)
    volume = analyze_volume(y)
    pitch = analyze_pitch(y, sr)
    clarity = analyze_clarity(y, sr)
    transcription = transcribe_speech(y, sr)
    feedback = give_feedback(pace, volume, pitch, clarity)
    return transcription, feedback

# Live speech recognition and audio analysis
def speech_recognition():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)

    engine.say("Hola!, I am Jarvis, ready to listen.")
    engine.runAndWait()

    p = pyaudio.PyAudio()
    
    # Increase frames_per_buffer size to prevent overflow
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=16000)
    stream.start_stream()

    recognizer = KaldiRecognizer(model, 16000)

    print("Listening... Start speaking.")
    engine.say("I'm listening")
    engine.runAndWait()

    frames = []
    while True:
        try:
            data = stream.read(4000, exception_on_overflow=False)  # Added exception_on_overflow=False
            frames.append(data)
            if recognizer.AcceptWaveform(data):
                break
        except OSError as e:
            print(f"Input Overflow Error: {e}")
            break  # Exit loop on error

    stream.stop_stream()
    stream.close()
    p.terminate()

    result = recognizer.Result()
    result_dict = json.loads(result)
    transcription = result_dict.get('text', '')

    if transcription:
        # Save the recorded audio to a file for analysis
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        live_audio_path = os.path.join(UPLOAD_FOLDER, "live_speech.wav")
        wf = wave.open(live_audio_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Analyze the saved audio file
        transcription_analysis, feedback = analyze_audio(live_audio_path)
        sentiment = analyze_sentiment(transcription_analysis)
        return transcription_analysis, feedback, sentiment
    else:
        return None, None, None

# Routes


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        try:
            transcription, feedback = analyze_audio(filename)
            sentiment = analyze_sentiment(transcription)
            return render_template('feedback.html', transcription=transcription, feedback=feedback, sentiment=sentiment)
        except Exception as e:
            return f"Error: {str(e)}"
    else:
        flash('Invalid file type. Only WAV, MP3, and OGG are allowed.')
        return redirect(request.url)

@app.route('/live', methods=['GET'])
def live_speech_recognition():
    transcription, feedback, sentiment = speech_recognition()
    if transcription:
        return render_template('live_feedback.html', transcription=transcription, feedback=feedback, sentiment=sentiment)
    else:
        flash('Unable to recognize the speech.')
        return redirect('/')


def generate_response(user_input):
    api_key = 'AIzaSyDDCW5QvaFRgN9UyxvLLFLW_L6FhiK91Io'  # Replace with your actual API key
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=' + api_key

    headers = {
        'Content-Type': 'application/json',
    }

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": user_input
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            return response_data['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Error: No candidates found in response."
    else:
        return f"Error: {response.status_code} - {response.text}"

def analyze_essay(essay_text):
    doc = nlp(essay_text)
    feedback = []

    # Check grammar (simplistic)
    for sent in doc.sents:
        if len(sent) < 3:
            feedback.append("Sentence too short: '{}'".format(sent.text))

    # Check coherence (basic check)
    if len(doc.ents) == 0:
        feedback.append("Your essay seems to lack clear arguments or points.")

    # Provide feedback on overall length
    word_count = len(doc)
    if word_count < 250:
        feedback.append("Consider expanding your essay. It's quite short ({} words).".format(word_count))
    
    if feedback:
        return "Feedback:\n" + "\n".join(feedback)
    else:
        return "Your essay is well-written and clear!"

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/chat1', methods=['POST'])
def chat1():
    user_input = request.get_json().get("input", "")
    
    if user_input.lower() == "bye":
        return jsonify({"response": "Goodbye!"})

    # Generate a response from the Gemini API
    bot_response = generate_response(user_input)
    return jsonify({"response": bot_response})

@app.route('/upload1', methods=['POST'])
def upload_file1():
    file = request.files['file']
    if file and file.filename.endswith('.txt'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER1'], file.filename)
        file.save(filepath)
        
        with open(filepath, 'r') as f:
            essay_text = f.read()
        
        feedback = analyze_essay(essay_text)
        return jsonify({"feedback": feedback})

    return jsonify({"error": "Invalid file format. Please upload a .txt file."}), 400


nlp1 = spacy.load("en_core_web_sm")
language_tool = language_tool_python.LanguageTool('en-US')
summarizer = pipeline("summarization")
vader_analyzer = SentimentIntensityAnalyzer()

transition_words = ["however", "therefore", "furthermore", "moreover", "consequently", "in addition"]

@app.route('/essay')
def essay():
    return render_template('essay.html')

@app.route('/essay_result', methods=['POST'])
def essay_result():
    essay = request.form['essay']
    feedback = analyze_essay(essay)
    return render_template('essay_result.html', feedback=feedback)

def analyze_essay(essay):
    feedback = []

    # Grammar Check
    matches = language_tool.check(essay)
    if matches:
        feedback.append("Grammar and style issues:")
        for match in matches:
            feedback.append(f"  - {match.message} at position {match.offset}")
    else:
        feedback.append("No grammar issues found.")

    # Sentiment Analysis (VADER)
    vader_score = vader_analyzer.polarity_scores(essay)
    tone = "Neutral"
    if vader_score['compound'] >= 0.5:
        tone = "Positive"
    elif vader_score['compound'] <= -0.5:
        tone = "Negative"
    feedback.append(f"Sentiment analysis: The essay's overall tone is {tone}.")

    # Summarization
    try:
        summary = summarizer(essay, max_length=130, min_length=30, do_sample=False)
        feedback.append("Summary of your essay:")
        feedback.append(summary[0]['summary_text'])
    except Exception as e:
        feedback.append(f"Error during summarization: {str(e)}")

    # Readability Score
    readability = textstat.flesch_reading_ease(essay)
    feedback.append(f"Readability score (Flesch Reading Ease): {readability:.2f}")
    if readability > 60:
        feedback.append("The essay is fairly easy to read.")
    elif readability > 30:
        feedback.append("The essay is somewhat difficult to read.")
    else:
        feedback.append("The essay is very difficult to read.")

    # Vocabulary Enrichment
    doc = nlp1(essay)
    word_count = {}
    for token in doc:
        if token.is_alpha:
            word = token.text.lower()
            word_count[word] = word_count.get(word, 0) + 1
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    feedback.append("Most frequently used words:")
    for word, count in sorted_word_count[:5]:
        feedback.append(f"  - '{word}' used {count} times")

    # Transition Analysis
    transition_feedback = []
    sentences = list(doc.sents)
    for i, sentence in enumerate(sentences):
        if any(word in sentence.text.lower() for word in transition_words):
            transition_feedback.append(f"  - Transitional word used in sentence {i + 1}: '{sentence.text.strip()}'")
    if transition_feedback:
        feedback.append("Logical transitions detected in your essay:")
        feedback.extend(transition_feedback)
    else:
        feedback.append("Consider improving the logical flow between sentences and paragraphs.")

    # Word Count and Structure
    word_count = len(doc)
    feedback.append(f"Word count: {word_count}")
    if word_count < 300:
        feedback.append("The essay might be too short for formal writing. Consider adding more content.")
    elif word_count > 1000:
        feedback.append("The essay might be too long. Consider making it more concise.")

    # Check paragraph lengths
    paragraphs = essay.split("\n\n")
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph.split()) > 150:
            feedback.append(f"Paragraph {i + 1} is quite long. Consider splitting it.")

    # Essay Organization Feedback
    intro_words = ["introduction", "begin", "introduce"]
    conclusion_words = ["conclusion", "summarize", "in conclusion", "to sum up"]
    essay_lower = essay.lower()

    if any(word in essay_lower for word in intro_words):
        feedback.append("The essay has a clear introduction.")
    else:
        feedback.append("The essay could benefit from a clearer introduction.")

    if any(word in essay_lower for word in conclusion_words):
        feedback.append("The essay has a clear conclusion.")
    else:
        feedback.append("The essay could benefit from a clearer conclusion.")

    return feedback

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
