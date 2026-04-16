from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pickle
import numpy as np
import random
import requests

app = Flask(__name__)

# ==========================================
# 1. SYSTEM INITIALIZATION & API KEYS
# ==========================================

# Load your custom Machine Learning Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'flight_model.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

AVIATIONSTACK_KEY = 'Bcb16080d02214f0f7fcbda1bb4c4f4a'
SERPAPI_KEY = "4e128895a92ee39c16907fa5fea0a8214cc21ccc75392d557f6e6da2e2b27753"

# ==========================================
# 2. USER INTERFACE ROUTES (The Frontend)
# ==========================================

@app.route('/')
def home():
    # The default home page is the Search screen
    return render_template('search.html')


@app.route('/results')
def results():
    origin = request.args.get('origin', 'LHR').upper()
    destination = request.args.get('destination', 'JFK').upper()
    date = request.args.get('date', '2026-05-07')
    currency = request.args.get('currency', 'USD').upper()
    
    # Map the currency code to its actual symbol for the UI
    symbols = {'USD': '$', 'GBP': '£', 'EUR': '€', 'AUD': 'A$'}
    sym = symbols.get(currency, '$')

    SERPAPI_KEY = "4e128895a92ee39c16907fa5fea0a8214cc21ccc75392d557f6e6da2e2b27753" 
    real_flights = []

    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": date,
            "currency": currency, # Dynamically passing user's currency to Google!
            "hl": "en",
            "api_key": SERPAPI_KEY
        }
        
        response = requests.get(url, params=params, timeout=8)
        data = response.json()

        if 'best_flights' in data:
            for item in data['best_flights'][:4]: 
                flight_info = item['flights'][0]
                price = f"{sym}{item['price']}" # Injecting the correct symbol
                airline = flight_info['airline']
                flight_num = flight_info['flight_number']
                
                dep_time = flight_info['departure_airport']['time'].split(' ')[1]

                real_flights.append({
                    "id": flight_num,
                    "airline": airline,
                    "time": f"{dep_time} Local",
                    "price": price
                })
    except Exception as e:
        pass 

    # THE FAILSAFE (Now adapts to your chosen currency!)
    if not real_flights:
        real_flights = [
            {"id": f"BA{random.randint(100, 999)}", "airline": "British Airways", "time": "14:30 Local", "price": f"{sym}{random.randint(300, 600)}"},
            {"id": f"VS{random.randint(10, 99)}", "airline": "Virgin Atlantic", "time": "16:00 Local", "price": f"{sym}{random.randint(250, 550)}"},
            {"id": f"AA{random.randint(100, 999)}", "airline": "American Airlines", "time": "18:15 Local", "price": f"{sym}{random.randint(350, 650)}"}
        ]

    return render_template('results.html', origin=origin, destination=destination, date=date, flights=real_flights)

    

@app.route('/autopilot')
def autopilot():
    # The flagship AI prediction page
    return render_template('autopilot.html')


# ==========================================
# 3. AI AUTOPILOT ENGINE (The Backend Brain)
# ==========================================

@app.route('/api/predict', methods=['POST'])
def predict_delay():
    data = request.get_json()
    flight_iata = data.get('pnr', '').upper().replace(" ", "")

    if not flight_iata:
        return jsonify({"error": "Please enter a valid Flight Number (e.g., AA100)"}), 400

    # 1. FETCH LIVE FLIGHT DATA FROM AVIATIONSTACK
    try:
        api_url = f"http://api.aviationstack.com/v1/flights?access_key={AVIATIONSTACK_KEY}&flight_iata={flight_iata}"
        api_response = requests.get(api_url, timeout=5)
        api_data = api_response.json()

        if 'data' not in api_data or len(api_data['data']) == 0:
            route_str = f"{flight_iata} Route (Simulated Fetch)"
            time_formatted = "14:30 GMT"
        else:
            flight_info = api_data['data'][0]
            dep = flight_info['departure']['iata'] or "LHR"
            arr = flight_info['arrival']['iata'] or "JFK"
            scheduled_time = flight_info['departure']['scheduled']
            route_str = f"{dep} ➔ {arr} (Live Data)"
            time_formatted = scheduled_time.split('T')[1][:5] + " GMT" if scheduled_time else "Unknown"

    except Exception as e:
        route_str = f"{flight_iata} Route (Offline Failsafe)"
        time_formatted = "14:30 GMT"

    # 2. MOCK WEATHER & LOGIC DATA FOR ML MODEL
    weather_severity = random.randint(1, 5)
    inbound_delayed = random.choice([0, 1])

    # 3. RUN THE AI PREDICTION
    features = np.array([[weather_severity, inbound_delayed]])
    delay_prob = model.predict_proba(features)[0][1] * 100
    is_triggered = delay_prob > 75

    backup_flights = [
        {"airline": "Virgin Atlantic", "time": "16:00 GMT", "price": "$120"},
        {"airline": "Delta Airlines", "time": "17:15 GMT", "price": "$150"}
    ] if is_triggered else []

    response = {
        "status": "success",
        "pnr": flight_iata,
        "flight_number": flight_iata,
        "route": route_str,
        "scheduled_time": time_formatted,
        "delay_probability": round(delay_prob, 1),
        "autopilot_triggered": bool(is_triggered),
        "message": f"Data - Weather: {weather_severity}/5 | Inbound Late: {bool(inbound_delayed)}. Generating backups..." if is_triggered else "Flight looks good! Have a great trip.",
        "backups": backup_flights
    }

    return jsonify(response)


# ==========================================
# 4. STATIC PWA FILES
# ==========================================

@app.route('/sw.js')
def serve_sw():
    return send_from_directory('static', 'sw.js')

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory('static', 'manifest.json')


if __name__ == '__main__':
    # Using host='0.0.0.0' is required for Render server deployment
    app.run(host='0.0.0.0', port=5000, debug=False)
