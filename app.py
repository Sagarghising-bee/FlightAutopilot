from flask import Flask, request, jsonify, render_template_string
import requests
import os

app = Flask(__name__)

# 🛑 SECURE BACKEND: Your API key is now hidden on the server!
API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI0IiwianRpIjoiY2YyOTQ3NGU0NDc2NzA3ZjI3NDYyNjI5NTZhZDk2NDBmZmQ2NjdlMzdkNDQyMWNjNDRlZmMzNTU0NzdkZGFhM2RhOGFiNmVhMmIxZmJiOWQiLCJpYXQiOjE3NzkwMzA4NTcsIm5iZiI6MTc3OTAzMDg1NywiZXhwIjoxODEwNTY2ODU3LCJzdWIiOiIyODA1OSIsInNjb3BlcyI6W119.Z_dXcYA4Nl2CJjyUhF-WDZW40IWc7fxrDDgKAXw99UDbIYEWKIPYeProVHp6b5PUtwDywoyVkgbGwxQI-UtIfw"

# ==========================================
# 🖥️ THE FRONTEND (Sleek Modern UI)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AeroPro Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #2563eb; --primary-hover: #1d4ed8; --bg: #f8fafc; --surface: #ffffff; --text: #0f172a; --text-muted: #64748b; --border: #e2e8f0; }
        body { font-family: 'Inter', sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 20px; line-height: 1.5; }
        .container { max-width: 900px; margin: 0 auto; }
        
        /* Header */
        .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid var(--border); }
        .header h1 { margin: 0; font-size: 24px; font-weight: 800; color: var(--primary); display: flex; align-items: center; gap: 10px;}
        
        /* Navigation Tabs */
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; overflow-x: auto; padding-bottom: 5px;}
        .tab-btn { flex: 1; padding: 12px 20px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; font-weight: 600; color: var(--text-muted); cursor: pointer; transition: all 0.2s; white-space: nowrap; }
        .tab-btn:hover { border-color: var(--primary); color: var(--primary); }
        .tab-btn.active { background: var(--primary); color: white; border-color: var(--primary); box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2); }
        
        /* Form Cards */
        .tab-content { display: none; background: var(--surface); padding: 25px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid var(--border); margin-bottom: 25px;}
        .tab-content.active { display: block; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
        
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px; }
        .form-group label { display: block; font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px;}
        input, select { width: 100%; padding: 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 15px; font-family: inherit; box-sizing: border-box; transition: border-color 0.2s;}
        input:focus, select:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
        
        button.action-btn { width: 100%; padding: 14px; background: var(--primary); color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background 0.2s; margin-top: 10px;}
        button.action-btn:hover { background: var(--primary-hover); }

        /* Results Area */
        #results { margin-top: 20px; }
        .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }
        .flight-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition: transform 0.2s; border-top: 4px solid var(--primary);}
        .flight-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .flight-card h4 { margin: 0 0 15px 0; font-size: 18px; display: flex; justify-content: space-between; align-items: center;}
        .price-badge { background: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 20px; font-size: 14px; font-weight: 800;}
        .flight-detail { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; color: var(--text-muted);}
        .flight-detail strong { color: var(--text); }
        
        /* Status & Terminal */
        .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(37,99,235,0.3); border-radius: 50%; border-top-color: var(--primary); animation: spin 1s ease-in-out infinite; margin-right: 10px; vertical-align: middle;}
        @keyframes spin { to { transform: rotate(360deg); } }
        .status-msg { text-align: center; padding: 30px; font-weight: 600; color: var(--text-muted); background: var(--surface); border-radius: 12px; border: 1px solid var(--border);}
        .terminal { background: #0f172a; color: #38bdf8; padding: 20px; border-radius: 12px; font-family: monospace; overflow-x: auto; font-size: 13px; line-height: 1.4; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);}
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>✈️ AeroPro Analytics</h1>
        <span style="background: #e2e8f0; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; color: #475569;">PRO TIER</span>
    </div>

    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('prices')">💰 Flight Prices</button>
        <button class="tab-btn" onclick="switchTab('schedules')">📅 Schedules</button>
        <button class="tab-btn" onclick="switchTab('realtime')">📡 Real-Time</button>
    </div>

    <div id="prices" class="tab-content active">
        <div class="grid-2">
            <div class="form-group"><label>Origin</label><input type="text" id="price_orig" value="LGW" placeholder="LHR"></div>
            <div class="form-group"><label>Destination</label><input type="text" id="price_dest" value="JFK" placeholder="JFK"></div>
            <div class="form-group"><label>Departure</label><input type="date" id="price_dep" value="2026-05-18"></div>
            <div class="form-group"><label>Return</label><input type="date" id="price_ret" value="2026-05-25"></div>
        </div>
        <button class="action-btn" onclick="searchPrices()">Search Best Prices</button>
    </div>

    <div id="schedules" class="tab-content">
        <div class="grid-2">
            <div class="form-group"><label>Airport Code</label><input type="text" id="sch_iata" value="JFK"></div>
            <div class="form-group">
                <label>Flight Type</label>
                <select id="sch_type">
                    <option value="departure">Departures</option>
                    <option value="arrival">Arrivals</option>
                </select>
            </div>
        </div>
        <button class="action-btn" onclick="searchSchedules()">Pull Live Schedule</button>
    </div>

    <div id="realtime" class="tab-content">
        <div class="form-group"><label>Airline Code (IATA)</label><input type="text" id="rt_airline" value="AA" placeholder="e.g., AA, BA, DL"></div>
        <button class="action-btn" onclick="searchRealTime()">Track Live Fleet</button>
    </div>

    <div id="results"></div>
</div>

<script>
    // Tab Switcher
    function switchTab(tabId) {
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        event.target.classList.add('active');
    }

    // --- 1. SEARCH PRICES (With Frontend Polling) ---
    async function searchPrices() {
        const resDiv = document.getElementById('results');
        resDiv.innerHTML = '<div class="status-msg"><div class="spinner"></div> Scanning global prices...</div>';

        const params = new URLSearchParams({
            origin: document.getElementById('price_orig').value.toUpperCase(),
            dest: document.getElementById('price_dest').value.toUpperCase(),
            date: document.getElementById('price_dep').value,
            retDate: document.getElementById('price_ret').value
        });

        let attempts = 0;
        const maxAttempts = 6;

        const poll = async () => {
            attempts++;
            try {
                // Notice we are calling OUR backend (/api/prices), not FlightLabs directly!
                const response = await fetch(`/api/prices?${params.toString()}`);
                const data = await response.json();

                if (data.status === 'processing') {
                    if (attempts < maxAttempts) {
                        resDiv.innerHTML = `<div class="status-msg"><div class="spinner"></div> Server compiling complex routes (Attempt ${attempts}/${maxAttempts})...</div>`;
                        setTimeout(poll, 4000);
                    } else {
                        resDiv.innerHTML = '<div class="status-msg">⚠️ Search timed out. Please try again.</div>';
                    }
                } else if (data.flights && data.flights.length > 0) {
                    let html = `<div class="results-grid">`;
                    data.flights.forEach(f => {
                        html += `
                        <div class="flight-card">
                            <h4>${f.origin?.code} ➔ ${f.destination?.code} <span class="price-badge">$${f.price} ${f.currency}</span></h4>
                            <div class="flight-detail"><span>Departure</span> <strong>${f.departure}</strong></div>
                            <div class="flight-detail"><span>Duration</span> <strong>${Math.floor(f.durationInMinutes/60)}h ${f.durationInMinutes%60}m</strong></div>
                            <div class="flight-detail"><span>Stops</span> <strong>${f.stopCount}</strong></div>
                        </div>`;
                    });
                    html += `</div>`;
                    resDiv.innerHTML = html;
                } else {
                    resDiv.innerHTML = '<div class="status-msg">No flights found for these dates.</div>';
                }
            } catch (err) {
                resDiv.innerHTML = `<div class="status-msg">❌ Error connecting to server: ${err}</div>`;
            }
        };
        poll();
    }

    // --- 2. SEARCH SCHEDULES ---
    async function searchSchedules() {
        const resDiv = document.getElementById('results');
        resDiv.innerHTML = '<div class="status-msg"><div class="spinner"></div> Accessing control tower...</div>';

        const iata = document.getElementById('sch_iata').value.toUpperCase();
        const type = document.getElementById('sch_type').value;

        try {
            const response = await fetch(`/api/schedules?iata=${iata}&type=${type}`);
            const data = await response.json();

            if (data.success && data.data) {
                let html = `<div class="results-grid">`;
                data.data.forEach(f => {
                    html += `
                    <div class="flight-card" style="border-top-color: #10b981;">
                        <h4>Flight ${f.flight?.iataNumber || 'N/A'} <span style="font-size:14px; color:#64748b;">${f.airline?.name || 'Unknown'}</span></h4>
                        <div class="flight-detail"><span>To/From</span> <strong>${f.movement?.airport?.name || 'N/A'}</strong></div>
                        <div class="flight-detail"><span>Time</span> <strong>${f.movement?.scheduledTime?.local || 'N/A'}</strong></div>
                        <div class="flight-detail"><span>Terminal/Gate</span> <strong>T${f.movement?.terminal || '-'} / G${f.movement?.gate || '-'}</strong></div>
                    </div>`;
                });
                html += `</div>`;
                resDiv.innerHTML = html;
            } else {
                resDiv.innerHTML = '<div class="status-msg">No schedule data available.</div>';
            }
        } catch (err) {
            resDiv.innerHTML = `<div class="status-msg">❌ Error: ${err}</div>`;
        }
    }

    // --- 3. SEARCH REAL-TIME (Outputs to Pro Terminal) ---
    async function searchRealTime() {
        const resDiv = document.getElementById('results');
        resDiv.innerHTML = '<div class="status-msg"><div class="spinner"></div> Tracking fleet coordinates...</div>';
        
        const airline = document.getElementById('rt_airline').value.toUpperCase();

        try {
            const response = await fetch(`/api/realtime?airline=${airline}`);
            const data = await response.json();
            // Since Real-Time data has massive amounts of info, dumping it to a sleek terminal looks highly professional
            resDiv.innerHTML = `<div class="terminal">// REAL-TIME DATA STREAM RECEIVED\n\n${JSON.stringify(data, null, 2)}</div>`;
        } catch (err) {
            resDiv.innerHTML = `<div class="status-msg">❌ Error: ${err}</div>`;
        }
    }
</script>
</body>
</html>
"""

# ==========================================
# ⚙️ THE BACKEND API (Talks to FlightLabs)
# ==========================================

@app.route('/')
def home():
    """Serves the main HTML interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/prices')
def api_prices():
    """Backend proxy for fetching prices."""
    url = "https://www.goflightlabs.com/retrieveFlights"
    params = {
        "access_key": API_KEY,
        "originIATACode": request.args.get('origin'),
        "destinationIATACode": request.args.get('dest'),
        "date": request.args.get('date'),
        "returnDate": request.args.get('retDate'),
        "sortBy": "best",
        "mode": "roundtrip"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        # If processing, pass that status to the frontend
        if isinstance(data, dict) and data.get('status') == 'processing':
            return jsonify({"status": "processing"})
        # Otherwise, pass the flight list
        return jsonify({"status": "success", "flights": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/schedules')
def api_schedules():
    """Backend proxy for fetching schedules."""
    url = "https://www.goflightlabs.com/advanced-flights-schedules"
    params = {
        "access_key": API_KEY,
        "iataCode": request.args.get('iata'),
        "type": request.args.get('type')
    }
    response = requests.get(url, params=params)
    return jsonify(response.json())

@app.route('/api/realtime')
def api_realtime():
    """Backend proxy for fetching live fleet data."""
    url = "https://www.goflightlabs.com/flights"
    params = {
        "access_key": API_KEY,
        "airlineIata": request.args.get('airline')
    }
    response = requests.get(url, params=params)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    
