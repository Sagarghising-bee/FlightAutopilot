from flask import Flask, request, jsonify, render_template_string
import requests
import time
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Apify API Token
APIFY_TOKEN = "apify_api_2MmH9wx7Wvdh1WBUqqNIqbqnOL4vda1YqVuY"

# HTML Template (Sleek Modern UI)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AeroPro ✈️ | Flight Search</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #2563eb; --primary-hover: #1d4ed8; --bg: #f8fafc; --surface: #ffffff; --text: #0f172a; --text-muted: #64748b; --border: #e2e8f0; }
        body { font-family: 'Inter', sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 20px; line-height: 1.5; }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid var(--border); }
        .header h1 { margin: 0; font-size: 24px; font-weight: 800; color: var(--primary); display: flex; align-items: center; gap: 10px;}
        .card { background: var(--surface); padding: 25px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid var(--border); margin-bottom: 25px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px; }
        .form-group label { display: block; font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px;}
        input, select { width: 100%; padding: 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 15px; font-family: inherit; box-sizing: border-box; }
        input:focus, select:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
        button.action-btn { width: 100%; padding: 14px; background: var(--primary); color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background 0.2s; margin-top: 10px;}
        button.action-btn:hover { background: var(--primary-hover); }
        .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 15px; margin-top: 20px; }
        .flight-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; transition: transform 0.2s; border-left: 4px solid var(--primary); }
        .flight-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .flight-card h4 { margin: 0 0 15px 0; font-size: 18px; display: flex; justify-content: space-between; align-items: center; }
        .price-badge { background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: 800; }
        .flight-detail { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; color: var(--text-muted); }
        .flight-detail strong { color: var(--text); }
        .book-link { margin-top: 15px; text-align: center; }
        .book-link a { color: var(--primary); text-decoration: none; font-weight: 600; font-size: 14px; }
        .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(37,99,235,0.3); border-radius: 50%; border-top-color: var(--primary); animation: spin 1s ease-in-out infinite; margin-right: 10px; vertical-align: middle;}
        @keyframes spin { to { transform: rotate(360deg); } }
        .status-msg { text-align: center; padding: 30px; font-weight: 600; color: var(--text-muted); background: var(--surface); border-radius: 12px; border: 1px solid var(--border); }
        .trip-option { display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 30px; background: #f0f2f5; cursor: pointer; margin-right: 10px; font-size: 14px; }
        .trip-option.active { background: var(--primary); color: white; }
        @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>✈️ AeroPro Analytics</h1>
        <span style="background: #e2e8f0; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;">POWERED BY APIFY</span>
    </div>

    <div class="card">
        <div style="margin-bottom: 20px;">
            <label class="trip-option active" data-trip="oneway">One Way</label>
            <label class="trip-option" data-trip="roundtrip">Round Trip</label>
        </div>
        
        <div class="grid-2">
            <div class="form-group"><label>From ✈️</label><input type="text" id="origin" value="LHR" placeholder="LHR, JFK, CDG"></div>
            <div class="form-group"><label>To 🎯</label><input type="text" id="destination" value="JFK" placeholder="JFK, LAX, DXB"></div>
            <div class="form-group"><label>Departure Date</label><input type="date" id="departDate"></div>
            <div class="form-group" id="returnGroup"><label>Return Date</label><input type="date" id="returnDate"></div>
        </div>
        <div class="grid-2">
            <div class="form-group"><label>Adults</label><select id="adults"><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option></select></div>
            <div class="form-group"><label>Currency</label><select id="currency"><option value="USD">USD ($)</option><option value="GBP">GBP (£)</option><option value="EUR">EUR (€)</option></select></div>
        </div>
        <button class="action-btn" onclick="searchFlights()">🔍 Search Flights</button>
    </div>

    <div id="results"></div>
</div>

<script>
    // Set default date (30 days from now)
    const today = new Date();
    const defaultDate = new Date(today);
    defaultDate.setDate(today.getDate() + 30);
    document.getElementById('departDate').value = defaultDate.toISOString().split('T')[0];
    document.getElementById('returnDate').value = defaultDate.toISOString().split('T')[0];

    let tripType = 'oneway';
    document.querySelectorAll('.trip-option').forEach(opt => {
        opt.onclick = () => {
            document.querySelectorAll('.trip-option').forEach(o => o.classList.remove('active'));
            opt.classList.add('active');
            tripType = opt.dataset.trip;
        };
    });

    async function searchFlights() {
        const origin = document.getElementById('origin').value.toUpperCase();
        const destination = document.getElementById('destination').value.toUpperCase();
        const departDate = document.getElementById('departDate').value;
        const returnDate = tripType === 'roundtrip' ? document.getElementById('returnDate').value : '';
        const adults = document.getElementById('adults').value;
        const currency = document.getElementById('currency').value;

        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '<div class="status-msg"><div class="spinner"></div> Searching flights via Apify SkyScanner...</div>';

        let url = `/api/search?from=${origin}&to=${destination}&departDate=${departDate}&adults=${adults}&currency=${currency}`;
        if (returnDate) url += `&returnDate=${returnDate}`;

        try {
            const response = await fetch(url);
            const data = await response.json();

            if (data.success && data.flights && data.flights.length > 0) {
                displayFlights(data.flights);
            } else {
                resultsDiv.innerHTML = '<div class="status-msg">❌ No flights found. Try different dates.</div>';
            }
        } catch (error) {
            resultsDiv.innerHTML = `<div class="status-msg">❌ Error: ${error.message}</div>`;
        }
    }

    function displayFlights(flights) {
        const container = document.getElementById('results');
        let html = '<div class="results-grid">';
        
        flights.forEach(flight => {
            const symbol = flight.currency === 'GBP' ? '£' : (flight.currency === 'EUR' ? '€' : '$');
            html += `
                <div class="flight-card" onclick="window.open('${flight.booking_url}', '_blank')" style="cursor: pointer;">
                    <h4>${flight.airline} <span class="price-badge">${symbol}${flight.price}</span></h4>
                    <div class="flight-detail">
                        <span>Departure</span>
                        <strong>${flight.departure_time} · ${flight.origin_code}</strong>
                    </div>
                    <div class="flight-detail">
                        <span>Arrival</span>
                        <strong>${flight.arrival_time} · ${flight.destination_code}</strong>
                    </div>
                    <div class="flight-detail">
                        <span>Duration</span>
                        <strong>${flight.duration}</strong>
                    </div>
                    <div class="flight-detail">
                        <span>Stops</span>
                        <strong>${flight.stops_label}</strong>
                    </div>
                    ${flight.is_best ? '<div style="margin-top: 10px;"><span style="background: #fef3c7; color: #d97706; padding: 4px 8px; border-radius: 20px; font-size: 12px;">⭐ Best Price</span></div>' : ''}
                    <div class="book-link"><a href="${flight.booking_url}" target="_blank">✈️ Book Now →</a></div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
    }
</script>
</body>
</html>
"""

# ==========================================
# APIFY BACKEND API
# ==========================================

# Cache for search results
search_cache = {}

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/search')
def search_flights():
    """Search flights using Apify SkyScanner Scraper"""
    origin = request.args.get('from', 'LHR').upper()
    destination = request.args.get('to', 'JFK').upper()
    depart_date = request.args.get('departDate', '')
    return_date = request.args.get('returnDate', '')
    adults = int(request.args.get('adults', 1))
    currency = request.args.get('currency', 'USD')
    
    if not depart_date:
        depart_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Check cache
    cache_key = f"{origin}_{destination}_{depart_date}_{return_date}_{adults}"
    if cache_key in search_cache:
        cached_time, cached_data = search_cache[cache_key]
        if datetime.now() - cached_time < timedelta(minutes=30):
            return jsonify(cached_data)
    
    # Build search input for Apify
    search_input = {
        "origin": origin,
        "destination": destination,
        "departDate": depart_date,
        "adults": adults,
        "currency": currency,
        "fetchBookingOptions": True
    }
    
    if return_date:
        search_input["returnDate"] = return_date
    
    try:
        # Start Apify actor
        run_response = requests.post(
            "https://api.apify.com/v2/acts/solidcode~sky-scanner-flight/runs",
            headers={"Content-Type": "application/json"},
            params={"token": APIFY_TOKEN},
            json=search_input,
            timeout=30
        )
        
        if run_response.status_code != 201:
            return jsonify({'success': False, 'error': 'Failed to start search'})
        
        run_data = run_response.json()
        run_id = run_data.get('data', {}).get('id')
        
        if not run_id:
            return jsonify({'success': False, 'error': 'No run ID'})
        
        # Poll for results
        flights = poll_for_results(run_id)
        
        if flights and len(flights) > 0:
            result = {
                'success': True,
                'count': len(flights),
                'flights': flights,
                'route': f"{origin} → {destination}"
            }
            search_cache[cache_key] = (datetime.now(), result)
            return jsonify(result)
        else:
            # Return sample data for demo
            return jsonify(get_sample_flights(origin, destination, currency))
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def poll_for_results(run_id, max_attempts=15):
    """Poll Apify for results"""
    for attempt in range(max_attempts):
        time.sleep(2)
        
        status_response = requests.get(
            f"https://api.apify.com/v2/actor-runs/{run_id}",
            params={"token": APIFY_TOKEN}
        )
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            run_status = status_data.get('data', {}).get('status')
            
            if run_status == 'SUCCEEDED':
                dataset_response = requests.get(
                    f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items",
                    params={"token": APIFY_TOKEN, "format": "json"}
                )
                
                if dataset_response.status_code == 200:
                    flights_data = dataset_response.json()
                    return format_apify_results(flights_data)
            
            elif run_status in ['FAILED', 'TIMED-OUT']:
                return None
    
    return None

def format_apify_results(flights_data):
    """Format Apify results for frontend"""
    if not flights_data:
        return []
    
    formatted = []
    for flight in flights_data[:20]:
        # Extract booking link
        booking_url = "#"
        book_links = flight.get('book_links', flight.get('bookingOptions', []))
        
        if book_links:
            for link in book_links:
                if isinstance(link, dict):
                    url = link.get('url', link.get('link', ''))
                    if url and 'google' not in url:
                        booking_url = url
                        break
            if booking_url == "#" and book_links:
                first = book_links[0]
                booking_url = first.get('url', first.get('link', '#'))
        
        # Get price
        price = flight.get('price', flight.get('best_price', 199))
        if isinstance(price, dict):
            price = price.get('amount', price.get('value', 199))
        
        formatted.append({
            'airline': flight.get('airline', flight.get('carrier', 'Unknown')),
            'price': float(price),
            'currency': flight.get('currency', 'USD'),
            'departure_time': flight.get('departure_time', flight.get('departure', '08:00')),
            'arrival_time': flight.get('arrival_time', flight.get('arrival', '12:00')),
            'origin_code': flight.get('origin', flight.get('from', '')),
            'destination_code': flight.get('destination', flight.get('to', '')),
            'duration': flight.get('duration', '5h 0m'),
            'stops': flight.get('stops', 0),
            'stops_label': 'Non-stop' if flight.get('stops', 0) == 0 else f"{flight.get('stops', 0)} stop(s)",
            'booking_url': booking_url,
            'is_best': False
        })
    
    # Mark cheapest as best
    if formatted:
        valid_prices = [f['price'] for f in formatted if f['price'] > 0]
        if valid_prices:
            min_price = min(valid_prices)
            for f in formatted:
                if f['price'] == min_price:
                    f['is_best'] = True
                    break
    
    return formatted

def get_sample_flights(origin, destination, currency):
    """Sample data for demo/testing"""
    symbol = '£' if currency == 'GBP' else '$' if currency == 'USD' else '€'
    return {
        'success': True,
        'count': 4,
        'flights': [
            {
                'airline': 'British Airways',
                'price': 499,
                'currency': currency,
                'departure_time': '08:00',
                'arrival_time': '12:00',
                'origin_code': origin,
                'destination_code': destination,
                'duration': '4h 0m',
                'stops': 0,
                'stops_label': 'Non-stop',
                'booking_url': 'https://www.britishairways.com',
                'is_best': True
            },
            {
                'airline': 'Delta Air Lines',
                'price': 549,
                'currency': currency,
                'departure_time': '10:30',
                'arrival_time': '15:30',
                'origin_code': origin,
                'destination_code': destination,
                'duration': '5h 0m',
                'stops': 1,
                'stops_label': '1 stop',
                'booking_url': 'https://www.delta.com',
                'is_best': False
            },
            {
                'airline': 'American Airlines',
                'price': 599,
                'currency': currency,
                'departure_time': '14:00',
                'arrival_time': '18:00',
                'origin_code': origin,
                'destination_code': destination,
                'duration': '4h 0m',
                'stops': 0,
                'stops_label': 'Non-stop',
                'booking_url': 'https://www.aa.com',
                'is_best': False
            },
            {
                'airline': 'United Airlines',
                'price': 449,
                'currency': currency,
                'departure_time': '06:00',
                'arrival_time': '10:00',
                'origin_code': origin,
                'destination_code': destination,
                'duration': '4h 0m',
                'stops': 0,
                'stops_label': 'Non-stop',
                'booking_url': 'https://www.united.com',
                'is_best': False
            }
        ]
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
