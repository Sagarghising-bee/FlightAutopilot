from flask import Flask, request, jsonify, render_template_string
import requests
import os

app = Flask(__name__)

# Your FlightLabs JWT Token
API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI0IiwianRpIjoiY2YyOTQ3NGU0NDc2NzA3ZjI3NDYyNjI5NTZhZDk2NDBmZmQ2NjdlMzdkNDQyMWNjNDRlZmMzNTU0NzdkZGFhM2RhOGFiNmVhMmIxZmJiOWQiLCJpYXQiOjE3NzkwMzA4NTcsIm5iZiI6MTc3OTAzMDg1NywiZXhwIjoxODEwNTY2ODU3LCJzdWIiOiIyODA1OSIsInNjb3BlcyI6W119.Z_dXcYA4Nl2CJjyUhF-WDZW40IWc7fxrDDgKAXw99UDbIYEWKIPYeProVHp6b5PUtwDywoyVkgbGwxQI-UtIfw"

# A simple HTML template to show a search form and results
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlightLabs Search</title>
    <style>
        body { font-family: sans-serif; padding: 20px; max-width: 600px; margin: auto; }
        .flight-card { border: 1px solid #ccc; padding: 15px; margin-bottom: 10px; border-radius: 8px; background-color: #f9f9f9;}
        .error { color: #d9534f; font-weight: bold; padding: 10px; border: 1px solid #d9534f; background: #fdf7f7;}
    </style>
</head>
<body>
    <h2>✈️ Flight Search (LGW ➔ JFK)</h2>
    <form method="GET" action="/search">
        Origin (IATA): <input type="text" name="origin" value="LGW" required><br><br>
        Destination (IATA): <input type="text" name="destination" value="JFK" required><br><br>
        Date (YYYY-MM-DD): <input type="text" name="date" value="2026-05-18" required><br><br>
        <button type="submit" style="padding: 8px 16px; cursor: pointer;">Search Flights</button>
    </form>
    <hr>
    <div id="results">
        {{ results|safe }}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, results="<p>Enter your details and hit search.</p>")

@app.route('/search')
def search():
    origin = request.args.get('origin', 'LGW').upper()
    destination = request.args.get('destination', 'JFK').upper()
    date = request.args.get('date', '2026-05-18')

    # FlightLabs Pricing Endpoint
    url = "https://www.goflightlabs.com/retrieveFlights"
    params = {
        'access_key': API_KEY,
        'originIATACode': origin,
        'destinationIATACode': destination,
        'date': date
    }

    try:
        # Added a timeout so the server doesn't hang indefinitely
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        flights = data.get('flights', [])
        if not flights:
            return render_template_string(HTML_TEMPLATE, results="<p>✅ Status Code: 200, but no flights found.</p>")
            
        html_output = f"<h3>✅ Found {len(flights)} flight options</h3>"
        
        # Loop through the flights and format them like your Colab output
        for i, flight in enumerate(flights):
            price = flight.get('price', 'N/A')
            currency = flight.get('currency', 'USD')
            # Extracting nested data (adjust if FlightLabs' JSON structure changes)
            dep_time = flight.get('departure', {}).get('time', 'N/A')
            arr_time = flight.get('arrival', {}).get('time', 'N/A')
            duration = flight.get('duration', 'N/A')
            stops = flight.get('stops', 'N/A')

            html_output += f"""
            <div class="flight-card">
                <b>✈️ Flight {i+1}: {origin} ➔ {destination}</b><br>
                💰 Price: {currency} ${price}<br>
                🕘 Departure: {dep_time}<br>
                🕘 Arrival: {arr_time}<br>
                ⏱ Duration: {duration}<br>
                🔁 Stops: {stops}
            </div>
            """
        return render_template_string(HTML_TEMPLATE, results=html_output)

    except requests.exceptions.RequestException as e:
        error_msg = f"""
        <div class='error'>
            <p>❌ <b>Error connecting to FlightLabs:</b></p>
            <p><code>{str(e)}</code></p>
        </div>
        """
        return render_template_string(HTML_TEMPLATE, results=error_msg)

if __name__ == '__main__':
    # Render will ignore this block because it uses Gunicorn, but it's good to leave it here 
    # in case you ever want to test it locally on your own computer!
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    
