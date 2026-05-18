from flask import Flask, request, jsonify, render_template_string
import requests
import os
from datetime import datetime

app = Flask(__name__)

# Your working Apify dataset URL and token
DATASET_URL = "https://api.apify.com/v2/datasets/dzLveggWIEFJKbre8/items"
APIFY_TOKEN = "apify_api_2MmH9wx7Wvdh1WBUqqNIqbqnOL4vda1YqVuY"

# HTML Template with Modern UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AeroPro ✈️ | Flight Search</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: #f0f2f5; min-height: 100vh; }
        
        /* Header */
        .header { background: linear-gradient(135deg, #0055cc, #003399); color: white; padding: 20px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .header-content { max-width: 1200px; margin: 0 auto; padding: 0 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px; }
        .logo { font-size: 24px; font-weight: 800; display: flex; align-items: center; gap: 10px; }
        .logo i { font-size: 28px; }
        .badge { background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-size: 12px; }
        
        /* Container */
        .container { max-width: 1200px; margin: 0 auto; padding: 30px 20px; }
        
        /* Search Card */
        .search-card { background: white; border-radius: 24px; padding: 30px; box-shadow: 0 8px 30px rgba(0,0,0,0.08); margin-bottom: 30px; }
        .form-row { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
        .form-group { flex: 1; min-width: 150px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: 600; font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
        .form-group label i { margin-right: 6px; color: #0055cc; }
        input, select { width: 100%; padding: 14px 16px; border: 2px solid #e0e0e0; border-radius: 12px; font-size: 14px; font-family: inherit; transition: all 0.3s; }
        input:focus, select:focus { outline: none; border-color: #0055cc; box-shadow: 0 0 0 3px rgba(0,85,204,0.1); }
        
        /* Trip Type Toggle */
        .trip-options { display: flex; gap: 15px; margin-bottom: 25px; }
        .trip-option { display: flex; align-items: center; gap: 8px; padding: 10px 24px; border-radius: 40px; background: #f0f2f5; cursor: pointer; font-weight: 600; transition: all 0.3s; }
        .trip-option.active { background: #0055cc; color: white; }
        .trip-option i { font-size: 14px; }
        
        /* Search Button */
        .search-btn { background: linear-gradient(135deg, #0055cc, #003399); color: white; border: none; padding: 16px 32px; border-radius: 14px; font-size: 16px; font-weight: 700; cursor: pointer; width: 100%; transition: all 0.3s; }
        .search-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,85,204,0.3); }
        
        /* Flight Cards */
        .results-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 15px; }
        .results-count { font-size: 18px; font-weight: 700; color: #0055cc; }
        .sort-select { padding: 8px 16px; border: 2px solid #e0e0e0; border-radius: 10px; font-size: 14px; background: white; }
        
        .flights-grid { display: flex; flex-direction: column; gap: 16px; }
        .flight-card { background: white; border-radius: 20px; padding: 20px; transition: all 0.3s; border: 1px solid #e0e0e0; cursor: pointer; }
        .flight-card:hover { transform: translateY(-3px); border-color: #0055cc; box-shadow: 0 8px 25px rgba(0,0,0,0.1); }
        .flight-card.best { border-left: 5px solid #ff6b00; background: linear-gradient(135deg, #fffbf5, white); }
        
        .flight-main { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px; }
        .flight-airline { display: flex; align-items: center; gap: 12px; min-width: 140px; }
        .airline-icon { width: 48px; height: 48px; background: linear-gradient(135deg, #0055cc, #003399); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 18px; }
        .airline-name { font-weight: 700; font-size: 16px; }
        .flight-number { font-size: 12px; color: #666; }
        
        .flight-times { display: flex; align-items: center; gap: 30px; flex: 1; justify-content: center; }
        .time-hour { font-size: 22px; font-weight: 700; color: #1a1a2e; }
        .time-airport { font-size: 12px; color: #666; }
        .flight-duration { text-align: center; }
        
        .flight-price { text-align: right; min-width: 130px; }
        .price-amount { font-size: 26px; font-weight: 800; color: #0055cc; }
        .price-label { font-size: 11px; color: #666; }
        .best-badge { background: #ff6b00; color: white; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; display: inline-block; margin-top: 5px; }
        
        .book-link { margin-top: 15px; text-align: right; padding-top: 12px; border-top: 1px solid #e0e0e0; }
        .book-link a { color: #0055cc; text-decoration: none; font-weight: 600; font-size: 14px; transition: all 0.3s; }
        .book-link a:hover { color: #003399; text-decoration: underline; }
        
        /* Loading */
        .loading { text-align: center; padding: 60px; background: white; border-radius: 20px; }
        .spinner { width: 50px; height: 50px; border: 4px solid #e0e0e0; border-top-color: #0055cc; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .empty-state { text-align: center; padding: 60px; background: white; border-radius: 20px; color: #666; }
        
        /* Footer */
        .footer { text-align: center; padding: 30px; color: #666; font-size: 12px; border-top: 1px solid #e0e0e0; margin-top: 30px; }
        
        @media (max-width: 768px) {
            .flight-main { flex-direction: column; text-align: center; }
            .flight-times { flex-direction: column; gap: 10px; }
            .flight-price { text-align: center; }
            .form-row { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="logo">
                <i class="fas fa-plane-departure"></i>
                <span>AeroPro</span>
            </div>
            <div class="badge">POWERED BY APIFY</div>
        </div>
    </div>

    <div class="container">
        <div class="search-card">
            <div class="trip-options">
                <div class="trip-option active" data-trip="oneway">
                    <i class="fas fa-arrow-right"></i> One Way
                </div>
                <div class="trip-option" data-trip="roundtrip">
                    <i class="fas fa-exchange-alt"></i> Round Trip
                </div>
            </div>
            
            <form id="searchForm">
                <div class="form-row">
                    <div class="form-group">
                        <label><i class="fas fa-plane-departure"></i> From</label>
                        <input type="text" id="origin" value="LAX" placeholder="LAX, LHR, JFK">
                    </div>
                    <div class="form-group">
                        <label><i class="fas fa-plane-arrival"></i> To</label>
                        <input type="text" id="destination" value="JFK" placeholder="JFK, CDG, DXB">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label><i class="fas fa-calendar"></i> Departure Date</label>
                        <input type="date" id="departDate">
                    </div>
                    <div class="form-group" id="returnGroup">
                        <label><i class="fas fa-calendar-return"></i> Return Date</label>
                        <input type="date" id="returnDate">
                    </div>
                    <div class="form-group">
                        <label><i class="fas fa-user"></i> Adults</label>
                        <select id="adults">
                            <option value="1">1 Adult</option>
                            <option value="2">2 Adults</option>
                            <option value="3">3 Adults</option>
                            <option value="4">4 Adults</option>
                        </select>
                    </div>
                </div>
                <button type="submit" class="search-btn">
                    <i class="fas fa-search"></i> Search Flights
                </button>
            </form>
        </div>

        <div id="resultsSection" style="display: none;">
            <div class="results-header">
                <div class="results-count" id="resultsCount"></div>
                <select id="sortBy" class="sort-select">
                    <option value="price">Sort by: Price (Lowest)</option>
                    <option value="duration">Sort by: Duration (Shortest)</option>
                    <option value="airline">Sort by: Airline</option>
                </select>
            </div>
            <div id="flightsList" class="flights-grid"></div>
        </div>
    </div>

    <div class="footer">
        <p>✈️ Flight data provided by Apify SkyScanner | Direct booking links to airlines and OTAs</p>
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
                const returnGroup = document.getElementById('returnGroup');
                if (tripType === 'roundtrip') {
                    returnGroup.style.opacity = '1';
                } else {
                    returnGroup.style.opacity = '0.5';
                }
            };
        });

        let currentFlights = [];

        document.getElementById('searchForm').onsubmit = async (e) => {
            e.preventDefault();
            
            const origin = document.getElementById('origin').value.toUpperCase();
            const destination = document.getElementById('destination').value.toUpperCase();
            const departDate = document.getElementById('departDate').value;
            const returnDate = tripType === 'roundtrip' ? document.getElementById('returnDate').value : '';
            const adults = document.getElementById('adults').value;
            
            const resultsDiv = document.getElementById('resultsSection');
            const flightsList = document.getElementById('flightsList');
            
            resultsDiv.style.display = 'block';
            flightsList.innerHTML = '<div class="loading"><div class="spinner"></div><p>Searching flights...</p></div>';
            
            let url = `/api/flights?from=${origin}&to=${destination}`;
            
            try {
                const response = await fetch(url);
                const data = await response.json();
                
                if (data.success && data.flights && data.flights.length > 0) {
                    currentFlights = data.flights;
                    document.getElementById('resultsCount').innerHTML = `✈️ ${currentFlights.length} flights found`;
                    displayFlights(currentFlights);
                } else {
                    flightsList.innerHTML = '<div class="empty-state"><i class="fas fa-plane-slash fa-3x" style="color: #ccc; margin-bottom: 15px;"></i><p>No flights found. Try different airports or dates.</p></div>';
                }
            } catch (error) {
                flightsList.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-triangle fa-3x" style="color: #ff6b00; margin-bottom: 15px;"></i><p>Error: ${error.message}</p></div>`;
            }
        };
        
        function displayFlights(flights) {
            const container = document.getElementById('flightsList');
            
            container.innerHTML = flights.map(flight => {
                const symbol = flight.currency === 'GBP' ? '£' : (flight.currency === 'EUR' ? '€' : '$');
                return `
                    <div class="flight-card ${flight.is_best ? 'best' : ''}" onclick="window.open('${flight.booking_url}', '_blank')">
                        <div class="flight-main">
                            <div class="flight-airline">
                                <div class="airline-icon">${flight.airline.substring(0, 2)}</div>
                                <div>
                                    <div class="airline-name">${flight.airline}</div>
                                    <div class="flight-number">${flight.flight_number || 'Flight'}</div>
                                </div>
                            </div>
                            <div class="flight-times">
                                <div style="text-align: center;">
                                    <div class="time-hour">${flight.departure_time}</div>
                                    <div class="time-airport">${flight.origin_code}</div>
                                </div>
                                <div class="flight-duration">
                                    <i class="fas fa-arrow-right" style="color: #0055cc;"></i>
                                    <div style="font-size: 12px; color: #666;">${flight.duration}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div class="time-hour">${flight.arrival_time}</div>
                                    <div class="time-airport">${flight.destination_code}</div>
                                </div>
                            </div>
                            <div class="flight-price">
                                <div class="price-amount">${symbol}${flight.price}</div>
                                <div class="price-label">per adult</div>
                                ${flight.is_best ? '<div class="best-badge">⭐ Best Deal</div>' : ''}
                            </div>
                        </div>
                        <div class="book-link">
                            <a href="${flight.booking_url}" target="_blank" onclick="event.stopPropagation()">
                                <i class="fas fa-ticket-alt"></i> Book Now on ${flight.airline} →
                            </a>
                        </div>
                    </div>
                `;
            }).join('');
            
            // Setup sorting
            const sortSelect = document.getElementById('sortBy');
            sortSelect.onchange = () => {
                const sorted = [...currentFlights];
                const sortBy = sortSelect.value;
                if (sortBy === 'price') sorted.sort((a, b) => a.price - b.price);
                if (sortBy === 'duration') {
                    sorted.sort((a, b) => {
                        const durA = parseInt(a.duration) || 0;
                        const durB = parseInt(b.duration) || 0;
                        return durA - durB;
                    });
                }
                if (sortBy === 'airline') sorted.sort((a, b) => a.airline.localeCompare(b.airline));
                displayFlights(sorted);
            };
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/flights')
def get_flights():
    """Fetch flights from Apify dataset and return JSON"""
    origin = request.args.get('from', 'LAX').upper()
    destination = request.args.get('to', 'JFK').upper()
    
    try:
        response = requests.get(DATASET_URL, params={"token": APIFY_TOKEN}, timeout=30)
        
        if response.status_code != 200:
            return jsonify({'success': False, 'error': 'Failed to fetch flight data'})
        
        flights_data = response.json()
        
        formatted_flights = []
        for flight in flights_data:
            # Get departure time (extract time from ISO string)
            depart_time = flight.get('departTime', '')
            if 'T' in depart_time:
                depart_time = depart_time.split('T')[1][:5]
            else:
                depart_time = 'N/A'
            
            # Extract booking link - THE GOLD!
            booking_url = flight.get('links', {}).get('book', '#')
            if not booking_url or booking_url == '#':
                booking_url = flight.get('links', {}).get('googleFlights', '#')
            
            formatted_flights.append({
                'airline': flight.get('airline', 'Unknown'),
                'flight_number': flight.get('flight_number', ''),
                'price': flight.get('bestPrice', 0),
                'currency': flight.get('currency', 'USD'),
                'departure_time': depart_time,
                'arrival_time': 'N/A',
                'origin_code': flight.get('origin', origin),
                'destination_code': flight.get('destination', destination),
                'duration': flight.get('duration', 'N/A'),
                'stops': flight.get('stops', 0),
                'stops_label': 'Non-stop' if flight.get('stops', 0) == 0 else f"{flight.get('stops', 0)} stop(s)",
                'booking_url': booking_url,
                'is_best': False
            })
        
        # Mark cheapest as best
        if formatted_flights:
            min_price = min(f['price'] for f in formatted_flights if f['price'] > 0)
            for f in formatted_flights:
                if f['price'] == min_price:
                    f['is_best'] = True
                    break
        
        return jsonify({
            'success': True,
            'count': len(formatted_flights),
            'flights': formatted_flights
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
