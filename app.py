from flask import Flask, request, render_template_string
import requests
import os
import time

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
        .warning { color: #8a6d3b; font-weight: bold; padding: 15px; border: 1px solid #faebcc; background: #fcf8e3; border-radius: 8px;}
        .debug { background-color: #eee; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: monospace; }
    </style>
</head>
<body>
    <h2>✈️ Flight Search (Roundtrip)</h2>
    <form method="GET" action="/search">
        Origin (IATA): <input type="text" name="origin" value="{{ request.args.get('origin', 'LGW') }}" required><br><br>
        Destination (IATA): <input type="text" name="destination" value="{{ request.args.get('destination', 'JFK') }}" required><br><br>
        Departure Date: <input type="text" name="date" value="{{ request.args.get('date', '2026-05-18') }}" required><br><br>
        Return Date: <input type="text" name="returnDate" value="{{ request.args.get('returnDate', '2026-05-25') }}" required><br><br>
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
    return_date = request.args.get('returnDate', '2026-05-25')

    url = "https://www.goflightlabs.com/retrieveFlights"
    params = {
        "access_key": API_KEY,
        "originIATACode": origin,
        "destinationIATACode": destination,
        "date": date,
        "returnDate": return_date,
        "sortBy": "best",
        "mode": "roundtrip"
    }

    try:
        # We will check 6 times, waiting 4 seconds each time (24 seconds total)
        max_retries = 6  
        flights = None

        for attempt in range(max_retries):
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, dict) and data.get('status') == 'processing':
                if attempt < max_retries - 1:
                    time.sleep(4)  
                    continue
                else:
                    # AUTO-REFRESH FIX: If it takes too long, tell the browser to reload automatically in 4 seconds!
                    timeout_msg = f"""
                    <div class='warning'>
                        <p>⏳ <b>This is a complex route! Compiling prices...</b></p>
                        <p>FlightLabs is taking a bit longer to find all the connections for this trip. The page will automatically refresh in a few seconds to grab your results.</p>
                    </div>
                    <script>
                        setTimeout(function() {{
                            window.location.reload(true);
                        }}, 4000);
                    </script>
                    """
                    return render_template_string(HTML_TEMPLATE, results=timeout_msg)
            else:
                flights = data
                break
                
        if not flights:
            return render_template_string(HTML_TEMPLATE, results="<p>✅ Status Code: 200. No flights found for these dates.</p>")
            
        if isinstance(flights, dict):
            debug_msg = f"<p>❌ API returned an error message:</p><div class='debug'>{flights}</div>"
            return render_template_string(HTML_TEMPLATE, results=debug_msg)

        html_output = f"<h3>✅ Found {len(flights)} flight options</h3>"
        
        for i, flight in enumerate(flights):
            price = flight.get('price', 'N/A')
            currency = flight.get('currency', 'USD')
            flight_origin = flight.get('origin', {}).get('code', origin)
            flight_dest = flight.get('destination', {}).get('code', destination)
            departure = flight.get('departure', 'N/A')
            arrival = flight.get('arrival', 'N/A')
            
            duration_mins = flight.get('durationInMinutes', 0)
            hours = duration_mins // 60 if isinstance(duration_mins, int) else 0
            minutes = duration_mins % 60 if isinstance(duration_mins, int) else 0
            
            stops = flight.get('stopCount', 0)

            html_output += f"""
            <div class="flight-card">
                <b>✈️ Flight {i+1}: {flight_origin} ➔ {flight_dest}</b><br>
                💰 Price: {currency} ${price}<br>
                🕘 Departure: {departure}<br>
                🕘 Arrival: {arrival}<br>
                ⏱ Duration: {hours}h {minutes}m<br>
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
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    
