from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import json
import re
import base64
import io
from PIL import Image, ImageDraw, ImageFilter
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Gemini API configuration - Using Vercel environment variable
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

# Generate questions with appropriate answer options based on life area
@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json()
        life_area = data.get('lifeArea', 'general life')

        # For testing without API key, use fallback questions
        if not GEMINI_API_KEY:
            return jsonify({"questions": get_fallback_questions(life_area)})

        prompt = {
            "contents": [{
                "parts": [{
                    "text": f"""
                    As an emotional wellness AI, generate exactly 5 questions to help someone analyze their {life_area}.
                    For EACH question, also provide 5 appropriate answer options that make sense for that specific question.

                    Return the questions and options as a JSON array with this exact format:
                    [
                      {{
                        "question": "Question text here",
                        "options": ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
                      }},
                      // ... repeat for 5 questions
                    ]

                    Make sure the answer options are relevant to each specific question.
                    For example, if the question is about frequency, options could be about time periods.
                    If the question is about intensity, options could be about strength of feeling.
                    """
                }]
            }]
        }

        headers = {'Content-Type': 'application/json'}
        response = requests.post(GEMINI_API_URL, headers=headers, json=prompt)

        if response.status_code == 200:
            response_data = response.json()
            questions_text = response_data['candidates'][0]['content']['parts'][0]['text']
            questions_data = parse_questions_with_options(questions_text)
        else:
            raise Exception(f"API error: {response.status_code}")

        return jsonify({"questions": questions_data})

    except Exception as e:
        print(f"Error generating questions: {e}")
        return jsonify({"questions": get_fallback_questions(life_area)})


# Generate analysis based on responses
@app.route('/api/generate-analysis', methods=['POST'])
def generate_analysis():
    try:
        data = request.get_json()
        life_area = data.get('lifeArea', 'general life')
        responses = data.get('responses', {})
        questions_data = data.get('questionsData', [])

        # For testing without API key, use fallback analysis
        if not GEMINI_API_KEY:
            return jsonify({"analysis": get_fallback_analysis(life_area, responses, questions_data)})

        # Format responses for the prompt
        response_text = "QUESTIONS AND ANSWERS:\n\n"
        for key, value in responses.items():
            if key.startswith('question'):
                q_index = int(key.replace('question', ''))
                if q_index < len(questions_data):
                    question_text = questions_data[q_index]['question']
                    response_text += f"Q: {question_text}\nA: {value['answer']}\n\n"

        prompt = {
            "contents": [{
                "parts": [{
                    "text": f"""
                    Based on the following responses about {life_area}, provide a comprehensive emotional analysis:

                    {response_text}

                    Please provide a personalized analysis that:
                    1. Summarizes their emotional state specifically based on their answers
                    2. Identifies 3 key challenges or blockers they're facing
                    3. Notes any signs of emotional fatigue or burnout
                    4. Provides 4-5 actionable suggestions tailored to their specific responses

                    Make the analysis personal and specific to their answers, not generic.
                    Format your response in HTML with headings (h4 for section titles) and paragraphs/lists.
                    Do not include any introductory or concluding text - just the analysis.
                    """
                }]
            }]
        }

        headers = {'Content-Type': 'application/json'}
        response = requests.post(GEMINI_API_URL, headers=headers, json=prompt)

        if response.status_code == 200:
            response_data = response.json()
            analysis = response_data['candidates'][0]['content']['parts'][0]['text']
        else:
            raise Exception(f"API error: {response.status_code}")

        return jsonify({"analysis": analysis})

    except Exception as e:
        print(f"Error generating analysis: {e}")
        return jsonify({"analysis": get_fallback_analysis(life_area, responses, questions_data)})


# Generate emotional visualization image
@app.route('/api/generate-emotion-image', methods=['POST'])
def generate_emotion_image():
    try:
        data = request.get_json()
        analysis = data.get('analysis', '')
        life_area = data.get('lifeArea', 'general life')

        # Extract emotional cues from the analysis
        emotional_cues = extract_emotional_cues(analysis)

        # Generate an abstract image based on emotional cues
        image_data = generate_abstract_visualization(emotional_cues)

        return jsonify({
            "imageData": image_data,
            "description": f"Abstract visualization of your emotional state regarding {life_area}"
        })

    except Exception as e:
        print(f"Error generating emotion image: {e}")
        # Return a fallback abstract image
        return jsonify({
            "imageData": generate_fallback_image(),
            "description": "Abstract representation of emotional state"
        })


def get_fallback_questions(life_area):
    """Return fallback questions if API fails"""
    return [
        {
            "question": f"How satisfied are you with your {life_area} currently?",
            "options": ["Very dissatisfied", "Somewhat dissatisfied", "Neutral", "Somewhat satisfied", "Very satisfied"]
        },
        {
            "question": f"What emotions arise when you think about your {life_area}?",
            "options": ["Mostly negative", "More negative than positive", "Mixed emotions",
                        "More positive than negative", "Mostly positive"]
        },
        {
            "question": f"How supported do you feel in improving your {life_area}?",
            "options": ["Not supported at all", "Minimally supported", "Somewhat supported", "Well supported",
                        "Extremely supported"]
        },
        {
            "question": f"What would you like to change about your {life_area}?",
            "options": ["Everything needs change", "Many aspects need change", "Some aspects need change",
                        "A few things need change", "Very little needs change"]
        },
        {
            "question": f"How does your {life_area} affect your overall wellbeing?",
            "options": ["Extremely negatively", "Somewhat negatively", "No significant effect", "Somewhat positively",
                        "Extremely positively"]
        }
    ]


def get_fallback_analysis(life_area, responses, questions_data):
    """Return fallback analysis if API fails"""
    # Create a slightly more personalized fallback based on responses
    positive_count = 0
    total_answers = 0

    for key, value in responses.items():
        if key.startswith('question'):
            total_answers += 1
            # Check if answer indicates positive sentiment (options 4-5 on 1-5 scale)
            if value['answer'].isdigit() and int(value['answer']) >= 4:
                positive_count += 1

    positivity_ratio = positive_count / total_answers if total_answers > 0 else 0.5

    if positivity_ratio > 0.7:
        sentiment = "generally positive"
    elif positivity_ratio > 0.4:
        sentiment = "mixed"
    else:
        sentiment = "generally challenging"

    return f"""
    <h4>Emotional State Summary</h4>
    <p>Based on your responses about your {life_area}, you're experiencing a {sentiment} period. Your answers suggest you're aware of areas needing attention and have motivation for positive change.</p>

    <h4>Key Blockers</h4>
    <ul>
        <li>Uncertainty about next steps in your {life_area}</li>
        <li>Difficulty prioritizing needs amidst competing demands</li>
        <li>External pressures influencing your decisions about {life_area}</li>
    </ul>

    <h4>Emotional Fatigue Signs</h4>
    <p>Your responses suggest some emotional fatigue related to your {life_area}, which is common when evaluating important life areas. Be mindful of burnout symptoms like decreased motivation or irritability.</p>

    <h4>Growth Suggestions</h4>
    <ul>
        <li>Set small, achievable goals specifically for your {life_area} to build momentum</li>
        <li>Practice mindfulness to stay connected with your emotions about this area</li>
        <li>Seek support from trusted friends or mentors regarding your {life_area}</li>
        <li>Celebrate small victories along your journey with {life_area}</li>
    </ul>
    """


def parse_questions_with_options(text):
    """Parse the response from Gemini to extract questions with options"""
    try:
        # Try to find JSON array in the response
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            # Clean up any markdown code formatting
            json_str = re.sub(r'```json\s*|\s*```', '', json_str)
            questions_data = json.loads(json_str)

            # Validate structure
            if (isinstance(questions_data, list) and
                    len(questions_data) >= 5 and
                    all('question' in q and 'options' in q for q in questions_data)):
                return questions_data[:5]
    except Exception as e:
        print(f"Error parsing JSON: {e}")

    # Fallback if parsing fails
    return get_fallback_questions("selected area")


def extract_emotional_cues(analysis):
    """Extract emotional cues from the analysis text"""
    emotional_cues = {
        'positive': 0,
        'negative': 0,
        'energetic': 0,
        'calm': 0,
        'chaotic': 0,
        'focused': 0
    }

    # Simple text analysis to detect emotional tones
    text = analysis.lower()

    # Positive emotions
    positive_terms = ['positive', 'happy', 'joy', 'content', 'satisfied', 'growth', 'improve', 'good', 'well', 'better']
    for term in positive_terms:
        if term in text:
            emotional_cues['positive'] += text.count(term)

    # Negative emotions
    negative_terms = ['negative', 'sad', 'challenge', 'difficult', 'blocker', 'fatigue', 'stress', 'anxious',
                      'overwhelm', 'hard']
    for term in negative_terms:
        if term in text:
            emotional_cues['negative'] += text.count(term)

    # Energetic emotions
    energetic_terms = ['energy', 'active', 'dynamic', 'vibrant', 'excite', 'motivate', 'drive']
    for term in energetic_terms:
        if term in text:
            emotional_cues['energetic'] += text.count(term)

    # Calm emotions
    calm_terms = ['calm', 'peace', 'serene', 'tranquil', 'relax', 'balance', 'centered']
    for term in calm_terms:
        if term in text:
            emotional_cues['calm'] += text.count(term)

    # Normalize values
    total = sum(emotional_cues.values())
    if total > 0:
        for key in emotional_cues:
            emotional_cues[key] = emotional_cues[key] / total

    return emotional_cues


def generate_abstract_visualization(emotional_cues):
    """Generate an abstract image based on emotional cues"""
    # Create a color palette based on emotions
    width, height = 400, 400
    image = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Determine base colors based on emotional cues
    base_color = calculate_base_color(emotional_cues)

    # Draw abstract shapes based on emotions
    draw_emotional_shapes(draw, width, height, emotional_cues, base_color)

    # Apply filters based on emotional intensity
    apply_emotional_filters(image, emotional_cues)

    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


def calculate_base_color(emotional_cues):
    """Calculate a base color based on emotional cues"""
    # Positive emotions -> Warm colors (yellows, oranges)
    # Negative emotions -> Cool colors (blues, purples)
    # Energetic emotions -> Bright, saturated colors
    # Calm emotions -> Muted, desaturated colors

    # Calculate color components based on emotions
    r = int(255 * (emotional_cues['positive'] + emotional_cues['energetic'] * 0.5))
    g = int(255 * (emotional_cues['positive'] * 0.8 + emotional_cues['calm'] * 0.5))
    b = int(255 * (emotional_cues['negative'] + emotional_cues['calm']))

    # Ensure values are within 0-255 range
    r = min(255, max(0, r))
    g = min(255, max(0, g))
    b = min(255, max(0, b))

    return (r, g, b)


def draw_emotional_shapes(draw, width, height, emotional_cues, base_color):
    """Draw abstract shapes based on emotional cues"""
    # Number of shapes based on energy level
    num_shapes = int(10 + 40 * emotional_cues['energetic'])

    for _ in range(num_shapes):
        # Shape type based on emotional cues
        if emotional_cues['chaotic'] > 0.3:
            shape_type = 'chaos'
        elif emotional_cues['focused'] > 0.3:
            shape_type = 'focused'
        else:
            shape_type = random.choice(['circle', 'rectangle', 'polygon'])

        # Position and size
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(10, 100)

        # Color variation from base color
        color_variation = 50
        r = min(255, max(0, base_color[0] + random.randint(-color_variation, color_variation)))
        g = min(255, max(0, base_color[1] + random.randint(-color_variation, color_variation)))
        b = min(255, max(0, base_color[2] + random.randint(-color_variation, color_variation)))
        color = (r, g, b, random.randint(100, 200))  # Add alpha

        # Draw shape
        if shape_type == 'circle':
            draw.ellipse([x, y, x + size, y + size], fill=color)
        elif shape_type == 'rectangle':
            draw.rectangle([x, y, x + size, y + size], fill=color)
        elif shape_type == 'polygon':
            points = [(x, y), (x + size, y), (x + size // 2, y + size)]
            draw.polygon(points, fill=color)
        elif shape_type == 'chaos':
            # Chaotic lines
            for i in range(5):
                draw.line([(x, y), (x + random.randint(-50, 50), y + random.randint(-50, 50))],
                          fill=color, width=random.randint(1, 5))
        elif shape_type == 'focused':
            # Concentric circles
            for i in range(3):
                draw.ellipse([x - i * 10, y - i * 10, x + size + i * 10, y + size + i * 10],
                             outline=color, width=2)


def apply_emotional_filters(image, emotional_cues):
    """Apply filters based on emotional intensity"""
    if emotional_cues['chaotic'] > 0.4:
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
    if emotional_cues['calm'] > 0.6:
        image = image.filter(ImageFilter.SMOOTH)


def generate_fallback_image():
    """Generate a fallback abstract image"""
    width, height = 400, 400
    image = Image.new('RGB', (width, height), (30, 30, 50))
    draw = ImageDraw.Draw(image)

    # Draw some default shapes
    for _ in range(20):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(20, 80)
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200), 150)

        if random.choice([True, False]):
            draw.ellipse([x, y, x + size, y + size], fill=color)
        else:
            draw.rectangle([x, y, x + size, y + size], fill=color)

    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


# This is needed for Vercel serverless functions
if __name__ == '__main__':
    app.run(debug=True, port=5000)