import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import base64
from datetime import datetime

# ── Load models ───────────────────────────
with open('solar_model.pkl', 'rb') as f:
    models = pickle.load(f)
with open('city_map.pkl', 'rb') as f:
    city_to_code = pickle.load(f)

st.set_page_config(
    page_title="Solar Sphere AI — Pakistan",
    page_icon="☀️",
    layout="wide"
)

# ── CSS ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}
.stButton>button {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #1a1a1a;
    font-size: 16px;
    padding: 10px 20px;
    border-radius: 25px;
    border: none;
    width: 100%;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(247,151,30,0.4);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(247,151,30,0.6);
}
.result-box {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: #1a1a1a;
    margin-top: 10px;
    box-shadow: 0 8px 32px rgba(247,151,30,0.3);
}
.feature-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 10px 0;
    border-left: 4px solid #f7971e;
}
.section-header {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #1a1a1a;
    padding: 12px 20px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: bold;
    margin: 15px 0;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-top: 4px solid #f7971e;
}
.chat-msg-user {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #1a1a1a;
    padding: 12px 18px;
    border-radius: 18px 18px 5px 18px;
    margin: 8px 0;
    max-width: 80%;
    float: right;
    clear: both;
}
.chat-msg-bot {
    background: white;
    color: #333;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 5px;
    margin: 8px 0;
    max-width: 80%;
    float: left;
    clear: both;
    border-left: 3px solid #f7971e;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ── Data & Constants ──────────────────────
city_sun = {
    # Punjab
    'Lahore': 5.0, 'Faisalabad': 5.1,
    'Rawalpindi': 4.8, 'Gujranwala': 5.0,
    'Multan': 6.0, 'Sialkot': 4.9,
    'Bahawalpur': 6.1, 'Sargodha': 5.2,
    'Sheikhupura': 5.0, 'Jhang': 5.3,
    'Gujrat': 4.9, 'Kasur': 5.0,
    'Rahim Yar Khan': 6.0, 'Sahiwal': 5.1,
    'Okara': 5.1, 'Wah Cantt': 4.8,
    'Attock': 4.8, 'Khanewal': 5.5,
    'Hafizabad': 5.0, 'Chiniot': 5.1,
    'Pakpattan': 5.3, 'Mandi Bahauddin': 4.9,
    'Jhelum': 4.9, 'Narowal': 4.9,
    'Toba Tek Singh': 5.2, 'Vehari': 5.6,
    'Bahawalnagar': 5.9, 'Layyah': 5.8,
    'Lodhran': 5.7, 'Muzaffargarh': 5.7,
    'Nankana Sahib': 5.0, 'Chakwal': 4.8,
    # Sindh
    'Karachi': 5.5, 'Hyderabad': 5.4,
    'Sukkur': 6.2, 'Larkana': 5.9,
    'Nawabshah': 5.8, 'Mirpur Khas': 5.6,
    'Khairpur': 6.0, 'Jacobabad': 6.1,
    'Shikarpur': 5.9, 'Dadu': 5.8,
    'Tando Adam': 5.6, 'Badin': 5.5,
    'Thatta': 5.5, 'Tando Allahyar': 5.6,
    'Sanghar': 5.7, 'Matiari': 5.6,
    'Ghotki': 6.1, 'Kamber': 5.8,
    # KPK
    'Peshawar': 5.2, 'Abbottabad': 4.5,
    'Swat': 4.6, 'Mardan': 5.0,
    'Kohat': 5.1, 'Mansehra': 4.5,
    'Dera Ismail Khan': 5.5,
    'Nowshera': 5.0, 'Charsadda': 5.0,
    'Bannu': 5.3, 'Swabi': 5.0,
    'Haripur': 4.6, 'Battagram': 4.4,
    'Buner': 4.6, 'Shangla': 4.5,
    # Balochistan
    'Quetta': 6.5, 'Turbat': 6.3,
    'Khuzdar': 6.2, 'Chaman': 6.4,
    'Hub': 5.8, 'Sibi': 6.3,
    'Zhob': 6.0, 'Gwadar': 6.0,
    'Dera Bugti': 6.1, 'Nushki': 6.4,
    'Kharan': 6.5, 'Panjgur': 6.2,
    'Kalat': 6.3, 'Mastung': 6.3,
    # AJK & GB
    'Muzaffarabad': 4.3,
    'Mirpur AJK': 4.6,
    'Rawalakot': 4.4, 'Kotli': 4.5,
    'Gilgit': 5.5, 'Skardu': 5.8,
    'Hunza': 5.6, 'Chilas': 5.7,
    # Islamabad
    'Islamabad': 4.8,
}

cities = sorted(city_sun.keys())

COST_PER_KW          = 200000
UNIT_RATE            = 62
PANEL_WATT           = 550
PANEL_COST           = 35000
INVERTER_COST_PER_KW = 45000
BATTERY_LITHIUM_COST = 65000
BATTERY_AGM_COST     = 28000
BATTERY_LEADACID_COST = 16000
NEPRA_BUYBACK_RATE   = 19.32

appliance_watts = {
    'Fan': 75, 'LED Light': 15,
    'Fridge': 150, 'AC 1 Ton': 1500,
    'AC 1.5 Ton': 2000, 'TV': 100,
    'Washing Machine': 500,
    'Water Pump': 750, 'Computer/Laptop': 200,
    'Iron': 1000, 'Microwave': 1200,
    'Geyser': 2000, 'Cooler': 200
}

if 'history' not in st.session_state:
    st.session_state.history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ── Prediction Function ───────────────────
def predict_solar(city, fans, lights, fridges,
                  ac1, ac15, tvs, washing,
                  pumps, computers, hours,
                  roof_area=500):
    total_w = (fans*75 + lights*15 +
               fridges*150 + ac1*1500 +
               ac15*2000 + tvs*100 +
               washing*500 + pumps*750 +
               computers*200)
    daily_kwh = (total_w / 1000) * hours
    city_code = city_to_code.get(city, 0)

    inp = pd.DataFrame([[
        city_code, fans, lights, fridges,
        ac1, ac15, tvs, washing, pumps,
        computers, hours, roof_area,
        total_w, daily_kwh
    ]], columns=[
        'city_encoded', 'fans', 'lights', 'fridges',
        'ac_1ton', 'ac_15ton', 'tvs',
        'washing_machine', 'water_pump',
        'computers', 'hours_use', 'roof_area',
        'total_watts', 'daily_kwh'
    ])

    kw          = round(float(models['kw_needed'].predict(inp)[0]), 1)
    panels      = int(models['panels'].predict(inp)[0])
    cost        = int(models['cost'].predict(inp)[0])
    saving      = int(models['saving'].predict(inp)[0])
    monthly_bill = int(daily_kwh * 30 * UNIT_RATE * 1.3)
    payback     = round(cost / (saving * 12), 1)
    sun_h       = city_sun.get(city, 5.0)
    daily_gen   = round(kw * sun_h, 1)
    monthly_gen = round(daily_gen * 30, 0)

    return {
        'kw': kw, 'panels': panels,
        'cost': cost, 'saving': saving,
        'monthly_bill': monthly_bill,
        'payback': payback,
        'daily_kwh': round(daily_kwh, 1),
        'daily_gen': daily_gen,
        'monthly_gen': monthly_gen,
        'total_watts': total_w,
        'sun_hours': sun_h
    }

# ── BANNER ────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(
    135deg, #1a1a2e, #f7971e, #ffd200);
    padding:50px 40px;
    border-radius:25px;
    text-align:center;
    color:white;
    margin-bottom:30px;
    box-shadow:0 20px 60px rgba(0,0,0,0.3);">
    <div style="font-size:60px;">☀️</div>
    <h1 style="font-size:42px;margin:10px 0;
               font-weight:700;color:white;">
        Solar Sphere AI
    </h1>
    <p style="font-size:18px;opacity:0.95;margin:5px 0;">
        Pakistan ka Pehla AI-Powered Solar Energy Intelligence Platform
    </p>
    <div style="display:flex;justify-content:center;
                gap:15px;margin-top:15px;flex-wrap:wrap;">
        <span style="background:rgba(255,255,255,0.2);
                     padding:8px 20px;border-radius:20px;font-size:14px;">
            🤖 AI Powered</span>
        <span style="background:rgba(255,255,255,0.2);
                     padding:8px 20px;border-radius:20px;font-size:14px;">
            ⚡ Accurate Results</span>
        <span style="background:rgba(255,255,255,0.2);
                     padding:8px 20px;border-radius:20px;font-size:14px;">
            🏙️ 100+ Cities</span>
        <span style="background:rgba(255,255,255,0.2);
                     padding:8px 20px;border-radius:20px;font-size:14px;">
            💰 PKR Format</span>
    </div>
</div>
""", unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
for col, num, label in zip(
    [s1, s2, s3, s4],
    [f"{len(cities)}+", "95%", "2000+", "FREE"],
    ["Cities", "Accuracy", "Trained On", "To Use"]
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color:#f7971e;margin:0;">{num}</h2>
            <p style="margin:0;color:grey;">{label}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────
tabs = st.tabs([
    "⚡ Solar Calculator",
    "🏠 Appliance Guide",
    "📊 City Comparison",
    "💰 ROI Analysis",
    "🔋 Battery Guide",
    "🌐 Net Metering",
    "🏡 Roof Layout",
    "📅 Seasonal Graph",
    "🔌 Optimizer",
    "🛒 Marketplace",
    "🧑‍💼 Find Installer",
    "🔔 Smart Alerts",
    "👤 My Profiles",
    "💬 AI Assistant",
    "📋 History"
])
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12,tab13,tab14,tab15 = tabs
# ══════════════════════════════════════════
# TAB 1 — SOLAR CALCULATOR
# ══════════════════════════════════════════
with tab1:
    st.markdown("## ⚡ Solar System Calculator")
    st.info("💡 Apni city aur appliances enter karein — AI batayega kitna solar system chahiye!")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-header">📍 Location</div>', unsafe_allow_html=True)
        city = st.selectbox("City", cities, key='city1')
        sun_h = city_sun.get(city, 5.0)
        st.markdown(f"""
        <div class="feature-card">
            <small>☀️ Sun hours in {city}</small>
            <h4 style="color:#f7971e;margin:5px 0;">{sun_h} hours/day</h4>
            <small>{'🌟 Excellent solar city!' if sun_h >= 5.5 else '✅ Good solar potential'}</small>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="section-header">⏰ Usage</div>', unsafe_allow_html=True)
        hours     = st.selectbox("Daily Usage Hours", list(range(4, 20)), index=4, key='hours1')
        roof_area = st.number_input("Roof Area (sqft)", 200, 3000, 600, key='roof1')

    with c2:
        st.markdown('<div class="section-header">🏠 Appliances</div>', unsafe_allow_html=True)
        fans      = st.selectbox("🌀 Fans",             list(range(0, 11)), index=3, key='fans1')
        lights    = st.selectbox("💡 LED Lights",        list(range(0, 21)), index=8, key='lights1')
        fridges   = st.selectbox("🧊 Fridges",           list(range(0, 5)),  index=1, key='fridges1')
        tvs       = st.selectbox("📺 TVs",               list(range(0, 6)),  index=1, key='tvs1')
        computers = st.selectbox("💻 Computers/Laptops", list(range(0, 6)),  index=1, key='comp1')

    with c3:
        st.markdown('<div class="section-header">❄️ Heavy Appliances</div>', unsafe_allow_html=True)
        ac1     = st.selectbox("❄️ AC 1 Ton",         list(range(0, 5)), key='ac1')
        ac15    = st.selectbox("❄️ AC 1.5 Ton",       list(range(0, 5)), key='ac15')
        washing = st.selectbox("🫧 Washing Machine",   list(range(0, 4)), key='wash1')
        pumps   = st.selectbox("💧 Water Pump",        list(range(0, 3)), key='pump1')
        total_w = (fans*75 + lights*15 + fridges*150 +
                   ac1*1500 + ac15*2000 + tvs*100 +
                   washing*500 + pumps*750 + computers*200)
        st.markdown(f"""
        <div class="feature-card">
            <small>⚡ Total Load</small>
            <h4 style="color:#f7971e;margin:5px 0;">{total_w:,} Watts</h4>
            <small>{total_w/1000:.1f} kW</small>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    _, cb, _ = st.columns([1, 2, 1])
    with cb:
        calc_btn = st.button("☀️ Calculate Solar System", key='calc1')

    if calc_btn:
        prog   = st.progress(0)
        status = st.empty()
        for i in range(100):
            prog.progress(i + 1)
            if i < 30:   status.text("🔄 Analyzing your load...")
            elif i < 60: status.text("🤖 Running AI model...")
            elif i < 90: status.text("☀️ Calculating solar data...")
            else:         status.text("✅ Almost done...")
            time.sleep(0.015)
        prog.empty(); status.empty()

        r = predict_solar(city, fans, lights, fridges,
                          ac1, ac15, tvs, washing,
                          pumps, computers, hours, roof_area)

        st.markdown("### ☀️ Your Solar System Results:")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
            <div class="result-box">
                <h3>⚡ System Size</h3><h2>{r['kw']} KW</h2><p>Recommended</p>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="result-box">
                <h3>🔆 Solar Panels</h3><h2>{r['panels']} Panels</h2><p>{PANEL_WATT}W each</p>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="result-box">
                <h3>💰 System Cost</h3><h2>PKR {r['cost']/100_000:.1f}L</h2><p>Approx</p>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="result-box">
                <h3>💵 Monthly Saving</h3><h2>PKR {r['saving']:,}</h2><p>Per month</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(f"""
            <div class="feature-card">
                <h4>⚡ Energy Details:</h4>
                <p>🔌 Daily Load: <b>{r['daily_kwh']} kWh</b></p>
                <p>☀️ Daily Generation: <b>{r['daily_gen']} kWh</b></p>
                <p>📅 Monthly Generation: <b>{r['monthly_gen']:.0f} kWh</b></p>
                <p>🌞 Sun Hours: <b>{r['sun_hours']} hrs/day</b></p>
            </div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""
            <div class="feature-card">
                <h4>💰 Financial Details:</h4>
                <p>📋 Current Bill: <b>PKR {r['monthly_bill']:,}/month</b></p>
                <p>💵 After Solar: <b>PKR {max(0, r['monthly_bill']-r['saving']):,}/month</b></p>
                <p>✅ Monthly Saving: <b>PKR {r['saving']:,}</b></p>
                <p>📅 Payback Period: <b>{r['payback']} years</b></p>
            </div>""", unsafe_allow_html=True)
        with d3:
            coverage = min(100, int(r['daily_gen'] / max(r['daily_kwh'], 0.1) * 100))
            st.markdown(f"""
            <div class="feature-card">
                <h4>🏠 System Details:</h4>
                <p>⚡ Total Load: <b>{r['total_watts']:,}W</b></p>
                <p>🔆 Panels: <b>{r['panels']} × {PANEL_WATT}W</b></p>
                <p>📐 Roof Needed: <b>{r['panels']*20} sqft</b></p>
                <p>✅ Coverage: <b>{coverage}%</b></p>
            </div>""", unsafe_allow_html=True)

        st.markdown("### 📊 Bill Comparison:")
        fig, ax = plt.subplots(figsize=(8, 4))
        months = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec']
        bills_before = [r['monthly_bill']] * 12
        bills_after  = [max(0, r['monthly_bill'] - r['saving'])] * 12
        x = range(12)
        ax.bar([i-0.2 for i in x], bills_before, 0.4, label='Before Solar', color='#ff6b6b')
        ax.bar([i+0.2 for i in x], bills_after,  0.4, label='After Solar',  color='#f7971e')
        ax.set_xticks(x); ax.set_xticklabels(months)
        ax.set_ylabel('Bill (PKR)')
        ax.set_title('Monthly Bill Before vs After Solar')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig)

        st.markdown("### 📈 Investment Recovery:")
        years = list(range(1, 26))
        cumulative_saving = [r['saving'] * 12 * y for y in years]
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(years, [r['cost']]*25, '--', color='red',
                 label=f'System Cost: PKR {r["cost"]/100_000:.1f}L', linewidth=2)
        ax2.plot(years, cumulative_saving, color='#f7971e', linewidth=2.5,
                 label='Cumulative Savings')
        ax2.fill_between(years, cumulative_saving, alpha=0.2, color='#f7971e')
        ax2.axvline(x=r['payback'], color='green', linestyle='--',
                    label=f'Payback: {r["payback"]} years')
        ax2.set_xlabel('Years'); ax2.set_ylabel('PKR')
        ax2.set_title('Solar Investment Recovery')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig2)

        wa_text = (f"☀️ Solar System Result!%0ACity: {city}%0A"
                   f"System: {r['kw']} KW%0APanels: {r['panels']} panels%0A"
                   f"Cost: PKR {r['cost']/100_000:.1f} Lakh%0A"
                   f"Saving: PKR {r['saving']:,}/month%0A"
                   f"Payback: {r['payback']} years%0AVia: Solar Finder AI ☀️")
        st.markdown(
            f'<a href="https://wa.me/?text={wa_text}" target="_blank">'
            f'<button style="background:linear-gradient(135deg,#25D366,#128C7E);'
            f'color:white;padding:12px;border-radius:25px;border:none;'
            f'width:100%;font-size:16px;font-weight:bold;cursor:pointer;">'
            f'📱 Share on WhatsApp</button></a>', unsafe_allow_html=True)

        st.session_state.history.append({
            'City': city, 'System KW': r['kw'], 'Panels': r['panels'],
            'Cost (Lakh)': f"{r['cost']/100_000:.1f}",
            'Saving/Month': f"PKR {r['saving']:,}",
            'Payback': f"{r['payback']} yrs",
            'Date': datetime.now().strftime("%d/%m %H:%M")
        })
        st.balloons()
# ══════════════════════════════════════════
# TAB 2 — APPLIANCE GUIDE
# ══════════════════════════════════════════
with tab2:
    st.markdown("## 🏠 Appliance Power Guide")
    st.info("💡 Har appliance kitni bijli khata hai!")

    col1, col2 = st.columns(2)
    appliances_data = [
        ("🌀 Fan",             75,   "Low",            "green"),
        ("💡 LED Light",       15,   "Very Low",       "green"),
        ("🧊 Fridge",          150,  "Medium",         "orange"),
        ("❄️ AC 1 Ton",       1500, "Very High",      "red"),
        ("❄️ AC 1.5 Ton",    2000, "Extremely High", "red"),
        ("📺 TV",              100,  "Low",            "green"),
        ("🫧 Washing Machine", 500,  "High",           "orange"),
        ("💧 Water Pump",      750,  "High",           "orange"),
        ("💻 Computer",        200,  "Medium",         "orange"),
        ("🔥 Iron",           1000, "Very High",      "red"),
        ("🍳 Microwave",      1200, "Very High",      "red"),
        ("🚿 Geyser",         2000, "Extremely High", "red"),
        ("❄️ Cooler",         200,  "Medium",         "orange"),
    ]
    for i, (name, watts, level, color) in enumerate(appliances_data):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div class="feature-card" style="margin:5px 0;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:16px;"><b>{name}</b></span>
                    <span style="color:{color};font-weight:bold;">{watts}W</span>
                </div>
                <small style="color:grey;">Power Level: {level}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("### 💡 Energy Saving Tips:")
    tips = [
        "❄️ AC 1 ton = 20 fans ki bijli — AC kam chalao!",
        "🌀 Ceiling fan AC se 20 guna sasta hai",
        "💡 LED bulbs normal bulbs se 80% sasta",
        "🧊 Fridge door band rakho — 30% energy bachti hai",
        "🚿 Geyser timer pe lagao — bohot bijli bachegi",
        "☀️ Multan aur Quetta mein solar best invest hai!",
        "🔌 Standby appliances off karo — 10% saving",
        "⏰ Heavy appliances raat ko chalao — bill kam",
    ]
    for tip in tips:
        st.markdown(f"""
        <div class="feature-card" style="padding:10px 15px;margin:5px 0;">{tip}</div>""",
        unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 3 — CITY COMPARISON
# ══════════════════════════════════════════
# ============================================================
# TAB 3 – CITY COMPARISON
# ============================================================
with tab3:
    st.markdown("## 📊 City Solar Comparison")

    # City selectboxes MUST be inside with tab3:
    city_list = sorted(city_sun.keys())
    col_a, col_b = st.columns(2)
    with col_a:
        city_a = st.selectbox("🏙️ Select City A", city_list, index=0, key="city_a")
    with col_b:
        city_b = st.selectbox("🏙️ Select City B", city_list, index=1, key="city_b")

    # Cities chart (all cities)
    cities_data = sorted(city_sun.items(), key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(12, max(8, len(cities_data) * 0.28)))
    colors_bar = ['#ffd200' if h >= 6 else '#f7971e' if h >= 5 else '#ff6b6b'
                  for _, h in cities_data]
    bars = ax.barh([c[0] for c in cities_data], [c[1] for c in cities_data], color=colors_bar)
    ax.set_xlabel('Sun Hours Per Day')
    ax.set_title('🌞 City-wise Solar Potential in Pakistan')
    ax.grid(axis='x', alpha=0.3)
    for bar, (_, h) in zip(bars, cities_data):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{h}h', va='center', fontweight='bold', fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Compare button
    if st.button("🔄 Compare Cities"):
        if city_a == city_b:
            st.warning("Same city select ki hai! Alag cities choose karo!")
        else:
            sun_a = city_sun.get(city_a, 5.0)
            sun_b = city_sun.get(city_b, 5.0)

            # Fixed 5 marla standard house
            std_fans=4; std_lights=8
            std_fridge=1; std_ac=1
            std_tv=2; std_wash=1
            std_pump=1; std_comp=1
            std_hours=10

            total_w = (std_fans*75 + std_lights*15 +
                       std_fridge*150 + std_ac*1500 +
                       std_tv*100 + std_wash*500 +
                       std_pump*750 + std_comp*200)
            daily_kwh = (total_w / 1000) * std_hours

            # City A calculation
            kw_a = round(daily_kwh / sun_a * 1.25, 1)
            panels_a = int(np.ceil(kw_a * 1000 / PANEL_WATT))
            cost_a = int(kw_a * COST_PER_KW)
            saving_a = int(daily_kwh * 30 * UNIT_RATE)
            gen_a = round(kw_a * sun_a * 30, 0)
            payback_a = round(cost_a / (saving_a * 12), 1)

            # City B calculation
            kw_b = round(daily_kwh / sun_b * 1.25, 1)
            panels_b = int(np.ceil(kw_b * 1000 / PANEL_WATT))
            cost_b = int(kw_b * COST_PER_KW)
            saving_b = int(daily_kwh * 30 * UNIT_RATE)
            gen_b = round(kw_b * sun_b * 30, 0)
            payback_b = round(cost_b / (saving_b * 12), 1)

            ca1, ca2 = st.columns(2)
            with ca1:
                better = "🏆 BETTER CHOICE!" if sun_a > sun_b else ""
                st.markdown(f"""
                <div class="result-box">
                    <h3>🏙️ {city_a} {better}</h3>
                    <p>☀️ {sun_a} hrs sun/day</p>
                    <p>⚡ {kw_a} KW system</p>
                    <p>🔆 {panels_a} panels</p>
                    <p>☀️ {gen_a:.0f} kWh/month gen</p>
                    <p>💰 PKR {cost_a/100_000:.1f}L cost</p>
                    <p>💵 PKR {saving_a:,}/month saving</p>
                    <p>📅 {payback_a} years payback</p>
                </div>""", unsafe_allow_html=True)

            with ca2:
                better2 = "🏆 BETTER CHOICE!" if sun_b > sun_a else ""
                st.markdown(f"""
                <div class="result-box">
                    <h3>🏙️ {city_b} {better2}</h3>
                    <p>☀️ {sun_b} hrs sun/day</p>
                    <p>⚡ {kw_b} KW system</p>
                    <p>🔆 {panels_b} panels</p>
                    <p>☀️ {gen_b:.0f} kWh/month gen</p>
                    <p>💰 PKR {cost_b/100_000:.1f}L cost</p>
                    <p>💵 PKR {saving_b:,}/month saving</p>
                    <p>📅 {payback_b} years payback</p>
                </div>""", unsafe_allow_html=True)

            # Comparison chart
            fig_c, axes_c = plt.subplots(1, 3, figsize=(12, 4))

            axes_c[0].bar([city_a, city_b], [sun_a, sun_b], color=['#ffd200', '#f7971e'])
            axes_c[0].set_title('☀️ Sun Hours/Day')
            axes_c[0].set_ylabel('Hours')
            for i, (c, v) in enumerate([(city_a, sun_a), (city_b, sun_b)]):
                axes_c[0].text(i, v + 0.05, f'{v}h', ha='center', fontweight='bold')

            axes_c[1].bar([city_a, city_b], [cost_a/100_000, cost_b/100_000],
                          color=['#2e86de', '#26de81'])
            axes_c[1].set_title('💰 System Cost (Lakh)')
            axes_c[1].set_ylabel('PKR Lakh')

            axes_c[2].bar([city_a, city_b], [payback_a, payback_b],
                          color=['#ff6b6b', '#a29bfe'])
            axes_c[2].set_title('📅 Payback (Years)')
            axes_c[2].set_ylabel('Years')

            plt.tight_layout()
            st.pyplot(fig_c)

            # Winner announcement
            if sun_a > sun_b:
                diff_cost = cost_b - cost_a
                st.success(
                    f"✅ {city_a} is BETTER! "
                    f"{sun_a - sun_b:.1f} more sun hours. "
                    f"PKR {diff_cost/1000:.0f}K cheaper system. "
                    f"{payback_b - payback_a:.1f} years faster payback!")
            else:
                diff_cost = cost_a - cost_b
                st.success(
                    f"✅ {city_b} is BETTER! "
                    f"{sun_b - sun_a:.1f} more sun hours. "
                    f"PKR {diff_cost/1000:.0f}K cheaper system. "
                    f"{payback_a - payback_b:.1f} years faster payback!")
   # ══════════════════════════════════════════
# TAB 4 — SAVINGS ANALYSIS
# ══════════════════════════════════════════
with tab4:
    st.markdown("## 💰 Deep ROI & Tariff Analysis")
    st.info(
        "💡 Pakistan electricity tariff "
        "har saal barhta hai — dekhein "
        "kaise solar investment grow karta hai!")

    roi_tabs = st.tabs([
        "📊 Electricity Slabs",
        "📈 Tariff Scenarios",
        "🎯 Best/Worst Case ROI"
    ])

    # Pakistan Electricity Slabs 2026
    with roi_tabs[0]:
        st.markdown(
            "### 📊 Pakistan Electricity "
            "Tariff Slabs 2026")
        st.warning(
            "⚠️ Prices updated as per "
            "NEPRA 2025-26 tariff!")

        slabs = [
            ["Residential 1-100 units",
             "PKR 24.93/unit", "Low"],
            ["Residential 101-200 units",
             "PKR 31.57/unit", "Medium"],
            ["Residential 201-300 units",
             "PKR 38.45/unit", "Medium-High"],
            ["Residential 301-700 units",
             "PKR 47.02/unit", "High"],
            ["Residential 700+ units",
             "PKR 62.00/unit", "Very High"],
            ["Commercial",
             "PKR 55-75/unit", "Commercial"],
            ["Industrial",
             "PKR 40-60/unit", "Industrial"],
        ]

        slab_df = pd.DataFrame(
            slabs,
            columns=[
                "Category",
                "Rate 2026",
                "Level"])
        st.dataframe(
            slab_df,
            use_container_width=True)

        # Slab calculator
        st.markdown(
            "### 🧮 Your Slab Calculator:")
        monthly_units = st.number_input(
            "⚡ Monthly Units Used",
            50, 2000, 300,
            key='slab_units')

        if monthly_units <= 100:
            rate = 24.93
            slab = "Slab 1 (0-100 units)"
        elif monthly_units <= 200:
            rate = 31.57
            slab = "Slab 2 (101-200)"
        elif monthly_units <= 300:
            rate = 38.45
            slab = "Slab 3 (201-300)"
        elif monthly_units <= 700:
            rate = 47.02
            slab = "Slab 4 (301-700)"
        else:
            rate = 62.00
            slab = "Slab 5 (700+)"

        est_bill = monthly_units * rate
        fc_tax = est_bill * 0.17  # GST
        total_bill = est_bill + fc_tax + 350

        sb1,sb2,sb3 = st.columns(3)
        with sb1:
            st.markdown(f"""
            <div class="result-box">
                <h3>📊 Your Slab</h3>
                <h2>PKR {rate}/unit</h2>
                <p>{slab}</p>
            </div>""", unsafe_allow_html=True)
        with sb2:
            st.markdown(f"""
            <div class="result-box">
                <h3>📋 Est. Bill</h3>
                <h2>PKR {total_bill:,.0f}</h2>
                <p>incl GST + FC charges</p>
            </div>""", unsafe_allow_html=True)
        with sb3:
            if monthly_units > 200:
                solar_saving = (
                    monthly_units * 0.7 * rate)
                st.markdown(f"""
                <div class="result-box">
                    <h3>☀️ Solar Saving</h3>
                    <h2>PKR {solar_saving:,.0f}</h2>
                    <p>with 70% solar coverage</p>
                </div>""",
                    unsafe_allow_html=True)

    # Tariff Scenarios
    with roi_tabs[1]:
        st.markdown(
            "### 📈 Tariff Increase Scenarios")

        ts1, ts2 = st.columns(2)
        with ts1:
            t_city = st.selectbox(
                "📍 City", cities,
                key='t_city')
            t_kw = st.number_input(
                "⚡ System Size (KW)",
                1.0, 50.0, 5.0,
                key='t_kw')
            t_cost = st.number_input(
                "💰 System Cost (PKR)",
                100000, 10000000,
                1000000, 50000,
                key='t_cost')
        with ts2:
            t_units = st.number_input(
                "📊 Monthly Units",
                50, 2000, 400,
                key='t_units')
            t_years = st.selectbox(
                "📅 Analysis Years",
                [10, 15, 20, 25],
                key='t_years')
            current_rate = st.number_input(
                "💵 Current Rate (PKR/unit)",
                10.0, 150.0, 47.0, 1.0,
                key='t_rate')

        if st.button(
            "📈 Show All Scenarios",
            key='t_btn'):

            sun_h = city_sun.get(t_city, 5.0)
            solar_units = (
                t_kw * sun_h * 30 * 0.8)
            grid_units = max(
                0, t_units - solar_units)

            years = list(range(1, t_years+1))

            scenarios = {
                "Conservative (8%/yr)": 0.08,
                "Moderate (15%/yr)": 0.15,
                "Aggressive (25%/yr)": 0.25,
            }
            colors_s = [
                '#26de81','#f7971e','#ff6b6b']

            fig, ax = plt.subplots(
                figsize=(12, 5))

            all_savings = {}
            for (name, rate_inc), color in zip(
                scenarios.items(), colors_s):
                cum_saving = []
                total = 0
                for y in years:
                    tariff = (
                        current_rate *
                        (1+rate_inc)**y)
                    annual = (
                        solar_units * 12 *
                        tariff)
                    total += annual
                    cum_saving.append(total)
                all_savings[name] = cum_saving
                ax.plot(
                    years, cum_saving,
                    'o-', color=color,
                    linewidth=2.5,
                    label=name)

            ax.axhline(
                y=t_cost,
                color='red', linestyle='--',
                linewidth=2,
                label=f'System Cost: '
                      f'PKR {t_cost/100_000:.1f}L')
            ax.set_xlabel('Years')
            ax.set_ylabel('Cumulative Saving (PKR)')
            ax.set_title(
                f'{t_city} — Tariff Scenario '
                f'Analysis ({t_kw} KW)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format y axis in Lakh
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x,p:
                    f'PKR {x/100_000:.0f}L'))
            plt.tight_layout()
            st.pyplot(fig)

            # Payback per scenario
            st.markdown(
                "### 📅 Payback Per Scenario:")
            sc1,sc2,sc3 = st.columns(3)
            for (name,rate_inc),col,clr in zip(
                scenarios.items(),
                [sc1,sc2,sc3],
                ['#26de81','#f7971e','#ff6b6b']
            ):
                savings = all_savings[name]
                payback_y = None
                for i, s in enumerate(savings):
                    if s >= t_cost:
                        payback_y = i+1
                        break
                pb_text = (
                    f"{payback_y} years"
                    if payback_y
                    else f"{t_years}+ years")
                final_profit = (
                    savings[-1] - t_cost)
                col.markdown(f"""
                <div style="background:{clr};
                    padding:15px;
                    border-radius:12px;
                    color:white;
                    text-align:center;">
                    <b>{name}</b><br>
                    Payback: {pb_text}<br>
                    Profit: PKR
                    {max(0,final_profit)/100_000:.1f}L
                </div>""",
                    unsafe_allow_html=True)

    # Best/Worst Case ROI
    with roi_tabs[2]:
        st.markdown(
            "### 🎯 Best Case / Worst Case ROI")

        bw1, bw2 = st.columns(2)
        with bw1:
            bw_city = st.selectbox(
                "📍 City", cities,
                key='bw_city')
            bw_kw = st.number_input(
                "⚡ System (KW)",
                1.0, 50.0, 5.0, key='bw_kw')
        with bw2:
            bw_cost = st.number_input(
                "💰 System Cost (PKR)",
                100000, 10000000,
                1000000, 50000,
                key='bw_cost')
            bw_monthly = st.number_input(
                "📋 Current Bill (PKR)",
                500, 200000, 10000,
                key='bw_bill')

        if st.button(
            "🎯 Calculate Best/Worst ROI",
            key='bw_btn'):

            sun_h = city_sun.get(bw_city, 5.0)

            cases = {
                "🟢 Best Case": {
                    "efficiency": 1.0,
                    "tariff_inc": 0.25,
                    "maintenance": 0.005,
                    "panel_deg": 0.003,
                    "color": "#26de81"
                },
                "🟡 Base Case": {
                    "efficiency": 0.85,
                    "tariff_inc": 0.15,
                    "maintenance": 0.01,
                    "panel_deg": 0.005,
                    "color": "#f7971e"
                },
                "🔴 Worst Case": {
                    "efficiency": 0.70,
                    "tariff_inc": 0.08,
                    "maintenance": 0.02,
                    "panel_deg": 0.008,
                    "color": "#ff6b6b"
                }
            }

            years = list(range(1, 26))
            fig, ax = plt.subplots(
                figsize=(12, 5))

            results = {}
            for case_name, params in \
                    cases.items():
                cum_savings = []
                total = 0
                monthly_gen = (
                    bw_kw *
                    sun_h *
                    params['efficiency'] * 30)
                for y in years:
                    # Panel degrades yearly
                    degraded_gen = (
                        monthly_gen *
                        (1-params['panel_deg'])**y)
                    tariff = (
                        62 *
                        (1+params['tariff_inc'])**y)
                    annual_saving = (
                        degraded_gen *
                        12 * tariff)
                    maintenance = (
                        bw_cost *
                        params['maintenance'])
                    net = annual_saving - maintenance
                    total += net
                    cum_savings.append(total)
                results[case_name] = cum_savings

                ax.plot(
                    years, cum_savings,
                    linewidth=2.5,
                    label=case_name,
                    color=params['color'])
                ax.fill_between(
                    years, cum_savings,
                    alpha=0.1,
                    color=params['color'])

            ax.axhline(
                y=bw_cost,
                color='navy',
                linestyle='--',
                linewidth=2,
                label=f'Investment: PKR '
                      f'{bw_cost/100_000:.1f}L')

            ax.set_xlabel('Years')
            ax.set_ylabel('Net Savings (PKR)')
            ax.set_title(
                f'{bw_city} — {bw_kw} KW '
                f'ROI Analysis (25 Years)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x,p:
                    f'PKR {x/100_000:.0f}L'))
            plt.tight_layout()
            st.pyplot(fig)

            # Summary cards
            rc1,rc2,rc3 = st.columns(3)
            for (cname, params), col in zip(
                cases.items(), [rc1,rc2,rc3]):
                sav = results[cname]
                pb = None
                for i,s in enumerate(sav):
                    if s >= bw_cost:
                        pb = i+1; break
                profit_25 = sav[-1]-bw_cost
                col.markdown(f"""
                <div style="background:{params['color']};
                    padding:15px;
                    border-radius:12px;
                    color:white;
                    text-align:center;">
                    <b>{cname}</b><br>
                    Payback: {pb or '25+'} yrs<br>
                    25yr Profit: PKR
                    {max(0,profit_25)/100_000:.1f}L
                </div>""",
                    unsafe_allow_html=True)
# ══════════════════════════════════════════
# TAB 5 — BATTERY GUIDE
# ══════════════════════════════════════════
with tab5:
    st.markdown("## 🔋 Battery & Backup Guide")
    st.info("💡 Kitni batteries chahiye load shedding ke liye!")

    bb1, bb2 = st.columns(2)
    with bb1:
        backup_hours  = st.selectbox("⏰ Backup Hours Needed", [2, 4, 6, 8, 10, 12])
        critical_load = st.number_input(
            "⚡ Critical Load (Watts)\n(fans + lights + fridge only)", 100, 5000, 500, 100)
    with bb2:
        battery_type = st.selectbox("🔋 Battery Type",
            ["Lithium (Recommended)", "AGM/Gel", "Lead Acid (Conventional)"])
        battery_volt = st.selectbox("🔌 Battery Voltage", [12, 24, 48])

    if st.button("🔋 Calculate Batteries"):
        total_wh          = critical_load * backup_hours
        dod               = 0.8 if "Lithium" in battery_type else 0.5
        battery_wh_needed = total_wh / (dod * 0.9)
        battery_ah        = 200
        battery_wh        = battery_volt * battery_ah
        num_batteries     = int(np.ceil(battery_wh_needed / battery_wh))
        costs = {
            "Lithium (Recommended)":    BATTERY_LITHIUM_COST,
            "AGM/Gel":                  BATTERY_AGM_COST,
            "Lead Acid (Conventional)": BATTERY_LEADACID_COST
        }
        battery_cost = num_batteries * costs[battery_type]

        btr1, btr2, btr3 = st.columns(3)
        with btr1:
            st.markdown(f"""<div class="result-box">
                <h3>🔋 Batteries</h3><h2>{num_batteries}</h2>
                <p>{battery_volt}V {battery_ah}Ah each</p></div>""", unsafe_allow_html=True)
        with btr2:
            st.markdown(f"""<div class="result-box">
                <h3>💰 Battery Cost</h3><h2>PKR {battery_cost/1000:.0f}K</h2>
                <p>Approx total</p></div>""", unsafe_allow_html=True)
        with btr3:
            st.markdown(f"""<div class="result-box">
                <h3>⏰ Backup</h3><h2>{backup_hours} Hours</h2>
                <p>For {critical_load}W load</p></div>""", unsafe_allow_html=True)

        st.markdown("### 📋 Battery Comparison:")
        bcomp_df = pd.DataFrame(
            [["Lithium",   "3000–5000 cycles", "10–15 years", "80%", "Best",  "High"],
             ["AGM/Gel",   "500–1000 cycles",  "3–5 years",   "50%", "Good",  "Medium"],
             ["Lead Acid", "200–400 cycles",   "2–3 years",   "50%", "Basic", "Low"]],
            columns=["Type","Cycles","Life","DoD","Quality","Price"])
        st.dataframe(bcomp_df, use_container_width=True)

# ══════════════════════════════════════════
# TAB 6 — NET METERING
# ══════════════════════════════════════════
with tab6:
    st.markdown("## 🌐 Net Metering Simulator")
    st.info(f"💡 Pakistan mein net metering se extra bijli bech sakte hain! "
            f"NEPRA buyback rate: PKR {NEPRA_BUYBACK_RATE}/unit")

    st.markdown(f"""
    <div class="feature-card">
        <h4>ℹ️ Net Metering kya hai?</h4>
        <p>Jab aapka solar system zyada bijli banata hai to woh grid mein jati hai
        aur WAPDA/LESCO aapko PKR {NEPRA_BUYBACK_RATE} per unit deta hai!</p>
    </div>""", unsafe_allow_html=True)

    nm1, nm2 = st.columns(2)
    with nm1:
        nm_city      = st.selectbox("📍 City", cities, key='nm_city')
        nm_kw        = st.number_input("⚡ System Size (KW)", 1.0, 50.0, 5.0, 0.5, key='nm_kw')
        nm_daily_use = st.number_input("🔌 Daily Usage (kWh)", 1.0, 100.0, 15.0, 0.5, key='nm_use')
    with nm2:
        nm_bill   = st.number_input("📋 Current Monthly Bill (PKR)", 500, 200000, 10000, 500, key='nm_bill')
        nm_tariff = st.selectbox("🏠 Consumer Category",
            ["Residential (0–100 units)", "Residential (101–200 units)",
             "Residential (200+ units)", "Commercial", "Industrial"])
        disc = st.selectbox("🏢 DISCO",
            ["LESCO (Lahore)", "KESC/K-Electric (Karachi)", "MEPCO (Multan)",
             "IESCO (Islamabad)", "FESCO (Faisalabad)", "PESCO (Peshawar)",
             "QESCO (Quetta)", "SEPCO (Sukkur)", "HESCO (Hyderabad)"])

    if st.button("🌐 Simulate Net Metering", key='nm_btn'):
        sun_h_nm       = city_sun.get(nm_city, 5.0)
        daily_gen      = nm_kw * sun_h_nm
        monthly_gen    = daily_gen * 30
        monthly_use    = nm_daily_use * 30
        units_exported = max(0, monthly_gen - monthly_use)
        units_imported = max(0, monthly_use - monthly_gen)
        export_earning = units_exported * NEPRA_BUYBACK_RATE
        bill_with_solar = max(0, units_imported * UNIT_RATE - export_earning)
        net_saving     = nm_bill - bill_with_solar

        nr1, nr2, nr3, nr4 = st.columns(4)
        with nr1:
            st.markdown(f"""<div class="result-box">
                <h3>☀️ Monthly Gen</h3><h2>{monthly_gen:.0f} kWh</h2>
                </div>""", unsafe_allow_html=True)
        with nr2:
            st.markdown(f"""<div class="result-box">
                <h3>📤 Units Exported</h3><h2>{units_exported:.0f} kWh</h2>
                <p>Sold to grid</p></div>""", unsafe_allow_html=True)
        with nr3:
            st.markdown(f"""<div class="result-box">
                <h3>💰 Export Earning</h3><h2>PKR {export_earning:,.0f}</h2>
                <p>@ PKR {NEPRA_BUYBACK_RATE}/unit</p></div>""", unsafe_allow_html=True)
        with nr4:
            st.markdown(f"""<div class="result-box">
                <h3>📋 New Bill</h3><h2>PKR {bill_with_solar:,.0f}</h2>
                <p>vs PKR {nm_bill:,} before</p></div>""", unsafe_allow_html=True)

        st.markdown("### 📊 Monthly Energy Flow:")
        fig, ax = plt.subplots(figsize=(10, 4))
        months_nm     = ['Jan','Feb','Mar','Apr','May','Jun',
                         'Jul','Aug','Sep','Oct','Nov','Dec']
        season_factor = [0.75,0.80,0.90,1.0,1.1,1.15,1.1,1.1,1.0,0.9,0.80,0.75]
        gen_monthly   = [monthly_gen * f for f in season_factor]
        use_monthly   = [monthly_use] * 12
        ax.bar(months_nm, gen_monthly, label='Generation ☀️', color='#ffd200', alpha=0.8)
        ax.bar(months_nm, use_monthly, label='Usage ⚡',       color='#ff6b6b', alpha=0.6, width=0.4)
        ax.set_ylabel('kWh'); ax.set_title('Monthly Generation vs Usage')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig)

        st.markdown(f"""
        <div class="feature-card">
            <h4>📋 Net Metering Summary:</h4>
            <p>🏢 DISCO: <b>{disc}</b></p>
            <p>📤 Units exported: <b>{units_exported:.0f} kWh/month</b></p>
            <p>📥 Units imported: <b>{units_imported:.0f} kWh/month</b></p>
            <p>💰 Export earnings: <b>PKR {export_earning:,.0f}/month</b></p>
            <p>📋 Old bill: <b>PKR {nm_bill:,}</b></p>
            <p>📋 New bill: <b>PKR {bill_with_solar:,.0f}</b></p>
            <p>✅ Net saving: <b>PKR {net_saving:,.0f}/month</b></p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 7 — ROOF LAYOUT
# ══════════════════════════════════════════
with tab7:
    st.markdown("## 🏡 Roof Layout Visualizer")
    st.info("💡 Apni chhat pe panels ka layout dekho visually!")

    rl1, rl2 = st.columns(2)
    with rl1:
        roof_w    = st.number_input("📐 Roof Width (feet)",  10, 100, 30, key='roof_w')
        roof_l    = st.number_input("📐 Roof Length (feet)", 10, 100, 40, key='roof_l')
        roof_type = st.selectbox("🏠 Roof Type", ["Flat Roof", "Slanted Roof", "Mixed"])
    with rl2:
        panel_kw    = st.selectbox("⚡ System Size (KW)",
                                   [1,2,3,4,5,6,8,10,12,15,20], index=4)
        orientation = st.selectbox("🧭 Panel Orientation",
            ["South Facing (Best ✅)", "East-West", "North (Avoid ❌)"])

    if st.button("🏡 Generate Roof Layout", key='roof_btn'):
        num_panels      = int(np.ceil(panel_kw * 1000 / PANEL_WATT))
        panel_area      = num_panels * 20
        roof_area_total = roof_w * roof_l
        coverage_pct    = min(100, int(panel_area / roof_area_total * 100))

        svg_w    = 600; svg_h = 400
        scale_x  = (svg_w - 80) / roof_w
        scale_y  = (svg_h - 80) / roof_l
        p_w      = min(scale_x * 3.5, 45)
        p_h      = min(scale_y * 5.5, 55)
        cols_fit = int((svg_w - 100) / (p_w + 4))
        rows_fit = int((svg_h - 100) / (p_h + 4))

        panels_svg = ""
        count = 0
        for row in range(rows_fit):
            for col in range(cols_fit):
                if count >= num_panels: break
                x = 40 + col * (p_w + 4)
                y = 40 + row * (p_h + 4)
                panels_svg += (
                    f'<rect x="{x}" y="{y}" width="{p_w}" height="{p_h}" '
                    f'fill="#1a3a6b" stroke="#ffd200" stroke-width="1.5" rx="2"/>'
                    f'<line x1="{x+p_w/2}" y1="{y+3}" x2="{x+p_w/2}" y2="{y+p_h-3}" '
                    f'stroke="#ffd200" stroke-width="0.5" opacity="0.5"/>'
                    f'<line x1="{x+3}" y1="{y+p_h/2}" x2="{x+p_w-3}" y2="{y+p_h/2}" '
                    f'stroke="#ffd200" stroke-width="0.5" opacity="0.5"/>')
                count += 1
            if count >= num_panels: break

        svg = f"""
        <svg width="{svg_w}" height="{svg_h+60}" xmlns="http://www.w3.org/2000/svg"
             style="border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,0.2);background:#f0f0f0;">
          <rect x="20" y="20" width="{svg_w-40}" height="{svg_h-40}"
                fill="#d4a853" stroke="#8b6914" stroke-width="3" rx="5"/>
          {panels_svg}
          <text x="{svg_w/2}" y="{svg_h+20}" text-anchor="middle"
                font-size="14" font-weight="bold" fill="#333">
            🏡 {roof_w}ft × {roof_l}ft Roof | {num_panels} Panels | {coverage_pct}% Coverage
          </text>
          <text x="{svg_w/2}" y="{svg_h+45}" text-anchor="middle" font-size="12" fill="#666">
            ⚡ {panel_kw} KW System | ☀️ {orientation.split('(')[0].strip()}
          </text>
          <text x="{svg_w-40}" y="50" font-size="20">🧭</text>
          <text x="{svg_w-45}" y="70" font-size="11" fill="#333">N</text>
        </svg>"""
        st.markdown(svg, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        rl_r1, rl_r2, rl_r3 = st.columns(3)
        with rl_r1:
            st.markdown(f"""<div class="result-box">
                <h3>🔆 Panels</h3><h2>{num_panels}</h2>
                <p>× {PANEL_WATT}W each</p></div>""", unsafe_allow_html=True)
        with rl_r2:
            st.markdown(f"""<div class="result-box">
                <h3>📐 Area Used</h3><h2>{panel_area} sqft</h2>
                <p>{coverage_pct}% of roof</p></div>""", unsafe_allow_html=True)
        with rl_r3:
            remaining = roof_area_total - panel_area
            st.markdown(f"""<div class="result-box">
                <h3>✅ Free Space</h3><h2>{max(0,remaining):.0f} sqft</h2>
                <p>Remaining roof</p></div>""", unsafe_allow_html=True)

        if coverage_pct > 80:
            st.warning("⚠️ Roof almost full! Consider reducing system size or using higher watt panels.")
        elif coverage_pct < 40:
            st.success("✅ Plenty of roof space! You can even expand system in future.")
        else:
            st.info("📊 Good roof utilization! Optimal layout.")
# ══════════════════════════════════════════
# TAB 8 — SEASONAL GRAPH
# ══════════════════════════════════════════
with tab8:
    st.markdown("## 📅 Seasonal Performance")
    st.info("💡 Garmi mein solar zyada bijli banata hai — sardi mein kam!")

    sg1, sg2 = st.columns(2)
    with sg1:
        sg_city = st.selectbox("📍 City", cities, key='sg_city')
        sg_kw   = st.number_input("⚡ System Size (KW)", 1.0, 50.0, 5.0, key='sg_kw')
    with sg2:
        sg_bill = st.number_input("💰 Monthly Bill (PKR)", 500, 200000, 8000, key='sg_bill')
        sg_use  = st.number_input("⚡ Daily Usage (kWh)", 1.0, 100.0, 15.0, key='sg_use')

    if st.button("📅 Show Seasonal Analysis", key='sg_btn'):
        sun_h_sg    = city_sun.get(sg_city, 5.0)
        months_sg   = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        seasonal_sg = [0.72,0.78,0.88,0.98,1.05,1.08,1.02,1.05,1.02,0.95,0.82,0.72]
        temp_factor = [0.98,0.97,0.96,0.94,0.92,0.90,0.91,0.91,0.93,0.95,0.97,0.98]

        monthly_gen_sg   = [sg_kw * sun_h_sg * seasonal_sg[i] * 30 for i in range(12)]
        monthly_use_sg   = [sg_use * 30] * 12
        monthly_gen_real = [monthly_gen_sg[i] * temp_factor[i] for i in range(12)]
        monthly_surplus  = [max(0, monthly_gen_real[i] - monthly_use_sg[i]) for i in range(12)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'☀️ {sg_city} — {sg_kw} KW Seasonal Performance',
                     fontsize=14, fontweight='bold')

        ax1 = axes[0, 0]
        x = range(12)
        ax1.plot(x, monthly_gen_real, 'o-',  color='#ffd200', linewidth=2.5, markersize=8, label='Generation ☀️')
        ax1.plot(x, monthly_use_sg,   'o--', color='#ff6b6b', linewidth=2,   markersize=6, label='Usage ⚡')
        ax1.fill_between(x, monthly_gen_real, monthly_use_sg,
            where=[g >= u for g,u in zip(monthly_gen_real, monthly_use_sg)],
            alpha=0.3, color='#26de81', label='Surplus ✅')
        ax1.fill_between(x, monthly_gen_real, monthly_use_sg,
            where=[g < u for g,u in zip(monthly_gen_real, monthly_use_sg)],
            alpha=0.3, color='#ff6b6b', label='Deficit ❌')
        ax1.set_xticks(x); ax1.set_xticklabels(months_sg, rotation=45)
        ax1.set_ylabel('kWh'); ax1.set_title('Generation vs Usage')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        monthly_saving_sg = [min(monthly_gen_real[i], monthly_use_sg[i]) * UNIT_RATE for i in range(12)]
        ax2.bar(months_sg, monthly_saving_sg, color='#f7971e', alpha=0.8)
        ax2.set_ylabel('PKR'); ax2.set_title('Monthly Bill Saving (PKR)')
        ax2.tick_params(axis='x', rotation=45); ax2.grid(axis='y', alpha=0.3)
        for i, s in enumerate(monthly_saving_sg):
            ax2.text(i, s+100, f'{s/1000:.1f}K', ha='center', fontsize=8)

        ax3 = axes[1, 0]
        sun_monthly_sg = [sun_h_sg * seasonal_sg[i] for i in range(12)]
        ax3.bar(months_sg, sun_monthly_sg, color='#ffd200', alpha=0.9)
        ax3.set_ylabel('Hours/Day'); ax3.set_title('Monthly Sun Hours')
        ax3.tick_params(axis='x', rotation=45); ax3.grid(axis='y', alpha=0.3)
        for i, s in enumerate(sun_monthly_sg):
            ax3.text(i, s+0.05, f'{s:.1f}h', ha='center', fontsize=8)

        ax4 = axes[1, 1]
        export_earn = [monthly_surplus[i] * NEPRA_BUYBACK_RATE for i in range(12)]
        ax4.bar(months_sg, export_earn, color='#26de81', alpha=0.8)
        ax4.set_ylabel('PKR'); ax4.set_title('Net Metering Earnings (PKR)')
        ax4.tick_params(axis='x', rotation=45); ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout(); st.pyplot(fig)

        sr1, sr2, sr3 = st.columns(3)
        with sr1:
            st.markdown(f"""<div class="result-box">
                <h3>☀️ Annual Generation</h3>
                <h2>{sum(monthly_gen_real):.0f} kWh</h2></div>""", unsafe_allow_html=True)
        with sr2:
            st.markdown(f"""<div class="result-box">
                <h3>💰 Annual Saving</h3>
                <h2>PKR {sum(monthly_saving_sg)/1000:.0f}K</h2></div>""", unsafe_allow_html=True)
        with sr3:
            st.markdown(f"""<div class="result-box">
                <h3>📤 Export Earnings</h3>
                <h2>PKR {sum(export_earn)/1000:.0f}K</h2></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 9 — APPLIANCE OPTIMIZER
# ══════════════════════════════════════════
with tab9:
    st.markdown("## 🔌 Home Optimizer")
    st.info("💡 AI suggest karega kaise apna ghar solar ke liye optimize karo!")
    st.markdown("### Apne current appliances enter karo:")

    op1, op2 = st.columns(2)
    with op1:
        op_fans    = st.number_input("🌀 Fans",            0, 20, 4,  key='op_f')
        op_ac1     = st.number_input("❄️ AC 1 Ton",        0, 10, 2,  key='op_a1')
        op_ac15    = st.number_input("❄️ AC 1.5 Ton",      0, 10, 0,  key='op_a15')
        op_fridges = st.number_input("🧊 Fridges",          0,  5, 1,  key='op_fr')
        op_lights  = st.number_input("💡 Lights",           0, 30, 10, key='op_li')
    with op2:
        op_tvs     = st.number_input("📺 TVs",             0, 10, 2,  key='op_tv')
        op_washing = st.number_input("🫧 Washing Machine", 0,  5, 1,  key='op_w')
        op_pump    = st.number_input("💧 Water Pump",       0,  3, 1,  key='op_p')
        op_comp    = st.number_input("💻 Computers",        0, 10, 1,  key='op_c')
        op_hours   = st.selectbox("⏰ Usage Hours/Day", list(range(4, 20)), index=4, key='op_h')

    if st.button("🔌 Optimize My Home", key='op_btn'):
        cur_watts     = (op_fans*75 + op_lights*15 + op_fridges*150 +
                         op_ac1*1500 + op_ac15*2000 + op_tvs*100 +
                         op_washing*500 + op_pump*750 + op_comp*200)
        cur_kwh       = (cur_watts / 1000) * op_hours
        cur_kw_needed = round(cur_kwh / 5.0 * 1.25, 1)
        cur_cost      = cur_kw_needed * COST_PER_KW
        suggestions   = []

        if op_ac1 > 0 or op_ac15 > 0:
            total_ac  = op_ac1 + op_ac15
            new_watts = cur_watts - (op_ac1*1500 + op_ac15*2000) + (total_ac*900)
            new_kw    = round((new_watts/1000) * op_hours / 5.0 * 1.25, 1)
            saving_kw = cur_kw_needed - new_kw
            suggestions.append({
                'title': '❄️ Inverter AC lagao',
                'detail': f'{total_ac} AC(s) ko inverter AC se replace karo',
                'saving_kw': round(saving_kw, 1),
                'saving_cost': int(saving_kw * COST_PER_KW), 'priority': 'HIGH'})

        if op_lights > 8:
            saving_w = int((op_lights - 8) * 50)
            suggestions.append({
                'title': '💡 LED lights use karo',
                'detail': f'{op_lights} lights mein se zyada LED use karo',
                'saving_kw': round(saving_w * op_hours / 1000 / 5.0 * 1.25, 1),
                'saving_cost': int(saving_w * op_hours / 1000 / 5.0 * 1.25 * COST_PER_KW),
                'priority': 'MEDIUM'})

        if op_washing > 0:
            suggestions.append({
                'title': '🫧 Washing Machine daytime chalao',
                'detail': 'Solar peak hours (10am–3pm) mein chalao',
                'saving_kw': 0.5, 'saving_cost': int(0.5 * COST_PER_KW), 'priority': 'EASY'})

        if op_pump > 0:
            suggestions.append({
                'title': '💧 Water Pump timer lagao',
                'detail': 'Tank fill hone ke baad automatically band',
                'saving_kw': 0.3, 'saving_cost': int(0.3 * COST_PER_KW), 'priority': 'EASY'})

        if op_fridges > 1:
            suggestions.append({
                'title': '🧊 Extra fridge band karo',
                'detail': f'{op_fridges} fridges mein se {op_fridges-1} band karo',
                'saving_kw': 0.4, 'saving_cost': int(0.4 * COST_PER_KW), 'priority': 'MEDIUM'})

        total_opt_saving = sum(s['saving_kw'] for s in suggestions)
        opt_kw   = max(1, cur_kw_needed - total_opt_saving)
        opt_cost = int(opt_kw * COST_PER_KW)

        st.markdown("### 📊 Before vs After Optimization:")
        ba1, ba2 = st.columns(2)
        with ba1:
            st.markdown(f"""
            <div style="background:#ff6b6b;padding:20px;border-radius:15px;
                        color:white;text-align:center;">
                <h3>❌ Before</h3><h2>{cur_kw_needed} KW System</h2>
                <p>Cost: PKR {cur_cost/100_000:.1f} Lakh</p>
                <p>Load: {cur_watts:,}W</p></div>""", unsafe_allow_html=True)
        with ba2:
            st.markdown(f"""
            <div style="background:#26de81;padding:20px;border-radius:15px;
                        color:white;text-align:center;">
                <h3>✅ After Optimization</h3><h2>{opt_kw:.1f} KW System</h2>
                <p>Cost: PKR {opt_cost/100_000:.1f} Lakh</p>
                <p>Saving: PKR {(cur_cost-opt_cost)/100_000:.1f} Lakh!</p></div>""",
            unsafe_allow_html=True)

        st.markdown("### 💡 AI Recommendations:")
        for s in suggestions:
            pc = ('#ff6b6b' if s['priority']=='HIGH' else
                  '#f7971e' if s['priority']=='MEDIUM' else '#26de81')
            st.markdown(f"""
            <div class="feature-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <h4>{s['title']}</h4>
                    <span style="background:{pc};color:white;
                                 padding:3px 10px;border-radius:10px;font-size:12px;">
                        {s['priority']}</span>
                </div>
                <p>{s['detail']}</p>
                <p>💰 System size saving: <b>{s['saving_kw']} KW</b>
                   (PKR {s['saving_cost']:,} saved on system cost)</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("### ⏰ Optimal Daily Schedule:")
        schedule = [
            ("6:00 AM – 7:00 AM",  "☀️ Solar starts — light loads OK",                   "#ffd200"),
            ("8:00 AM – 10:00 AM", "💧 Water pump + washing machine start",               "#26de81"),
            ("10:00 AM – 3:00 PM", "🌟 PEAK SOLAR — all heavy loads OK! ACs, irons etc", "#26de81"),
            ("3:00 PM – 5:00 PM",  "📺 TVs, computers, fans — normal use",               "#f7971e"),
            ("5:00 PM – 7:00 PM",  "🔋 Battery power — reduce heavy loads",              "#ff6b6b"),
            ("7:00 PM – 6:00 AM",  "🌙 Grid power — only essentials!",                   "#ff6b6b"),
        ]
        for time_slot, activity, color in schedule:
            st.markdown(f"""
            <div style="border-left:4px solid {color};padding:8px 15px;
                        margin:5px 0;background:white;border-radius:5px;">
                <b>{time_slot}</b><br>{activity}</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 10 — MARKETPLACE
# ══════════════════════════════════════════
with tab10:
    st.markdown("## 🛒 Solar Marketplace")
    st.info("💡 Pakistan mein best solar products — updated 2025 prices!")

    mp_tabs = st.tabs(["🔆 Solar Panels", "🔌 Inverters", "🔋 Batteries", "📦 Complete Packages"])

    with mp_tabs[0]:
        st.markdown("### 🔆 Solar Panels")
        panels_mp = [
            {"brand":"Longi Solar",    "origin":"🇨🇳 China",  "watt":550,"efficiency":"21.3%","warranty":"25 years","price":35000,"rating":"⭐⭐⭐⭐⭐","recommended":True, "desc":"Pakistan mein #1 selling panel!"},
            {"brand":"Jinko Solar",    "origin":"🇨🇳 China",  "watt":540,"efficiency":"20.8%","warranty":"25 years","price":32000,"rating":"⭐⭐⭐⭐⭐","recommended":True, "desc":"World's largest manufacturer"},
            {"brand":"Canadian Solar", "origin":"🇨🇦 Canada", "watt":530,"efficiency":"20.2%","warranty":"25 years","price":38000,"rating":"⭐⭐⭐⭐⭐","recommended":False,"desc":"Premium quality panels"},
            {"brand":"Risen Energy",   "origin":"🇨🇳 China",  "watt":545,"efficiency":"20.5%","warranty":"25 years","price":30000,"rating":"⭐⭐⭐⭐", "recommended":False,"desc":"Good budget option"},
            {"brand":"Trina Solar",    "origin":"🇨🇳 China",  "watt":550,"efficiency":"21.0%","warranty":"25 years","price":34000,"rating":"⭐⭐⭐⭐⭐","recommended":True, "desc":"Tier 1 manufacturer"},
        ]
        pc1, pc2 = st.columns(2)
        for i, p in enumerate(panels_mp):
            col   = pc1 if i%2==0 else pc2
            badge = "🏆 RECOMMENDED" if p['recommended'] else ""
            with col:
                st.markdown(f"""
                <div class="feature-card">
                    <div style="display:flex;justify-content:space-between;">
                        <h4>{p['brand']}</h4>
                        <span style="color:#f7971e;font-size:12px;">{badge}</span></div>
                    <p>{p['origin']} | {p['rating']}</p>
                    <p>⚡ <b>{p['watt']}W</b> | 📊 Efficiency: {p['efficiency']}</p>
                    <p>🛡️ Warranty: {p['warranty']}</p>
                    <p>💰 <b>PKR {p['price']:,}/panel</b></p>
                    <small style="color:#666;">{p['desc']}</small>
                </div>""", unsafe_allow_html=True)

    with mp_tabs[1]:
        st.markdown("### 🔌 Inverters")
        inverters_mp = [
            {"brand":"Huawei SUN2000", "type":"String Inverter", "sizes":"3/5/8/10/15/20 KW","price_5kw":180000,"warranty":"10 years","rating":"⭐⭐⭐⭐⭐","recommended":True, "desc":"Best efficiency, WiFi monitoring included"},
            {"brand":"Solis",          "type":"String Inverter", "sizes":"3/5/8/10 KW",       "price_5kw":130000,"warranty":"5 years", "rating":"⭐⭐⭐⭐", "recommended":True, "desc":"Budget friendly, reliable"},
            {"brand":"Growatt",        "type":"Hybrid Inverter", "sizes":"3/5/8/10 KW",       "price_5kw":160000,"warranty":"10 years","rating":"⭐⭐⭐⭐⭐","recommended":True, "desc":"Battery ready, great for load shedding"},
            {"brand":"Sungrow",        "type":"String Inverter", "sizes":"5/8/10/15 KW",      "price_5kw":170000,"warranty":"10 years","rating":"⭐⭐⭐⭐⭐","recommended":False,"desc":"World #1 inverter manufacturer"},
            {"brand":"Victron Energy", "type":"Hybrid Inverter", "sizes":"3/5/8/10 KW",       "price_5kw":250000,"warranty":"5 years", "rating":"⭐⭐⭐⭐⭐","recommended":False,"desc":"Premium European brand"},
        ]
        ic1, ic2 = st.columns(2)
        for i, inv in enumerate(inverters_mp):
            col   = ic1 if i%2==0 else ic2
            badge = "🏆 RECOMMENDED" if inv['recommended'] else ""
            with col:
                st.markdown(f"""
                <div class="feature-card">
                    <div style="display:flex;justify-content:space-between;">
                        <h4>{inv['brand']}</h4>
                        <span style="color:#f7971e;font-size:12px;">{badge}</span></div>
                    <p>🔌 {inv['type']} | {inv['rating']}</p>
                    <p>⚡ Sizes: {inv['sizes']}</p>
                    <p>🛡️ Warranty: {inv['warranty']}</p>
                    <p>💰 5KW: <b>PKR {inv['price_5kw']:,}</b></p>
                    <small style="color:#666;">{inv['desc']}</small>
                </div>""", unsafe_allow_html=True)

    with mp_tabs[2]:
        st.markdown("### 🔋 Batteries")
        batteries_mp = [
            {"brand":"Pylontech",       "type":"Lithium LFP","capacity":"74Ah / 3.55 kWh","price":85000, "life":"6000+ cycles / 15 yrs","warranty":"10 years","rating":"⭐⭐⭐⭐⭐","recommended":True, "desc":"Best lithium battery in Pakistan!"},
            {"brand":"BYD Battery-Box", "type":"Lithium LFP","capacity":"100Ah / 5.1 kWh", "price":120000,"life":"6000+ cycles / 15 yrs","warranty":"10 years","rating":"⭐⭐⭐⭐⭐","recommended":True, "desc":"Premium lithium, scalable"},
            {"brand":"AGS Battery",     "type":"Lead Acid",  "capacity":"200Ah",            "price":18000, "life":"300–500 cycles / 3 yrs","warranty":"1 year", "rating":"⭐⭐⭐",  "recommended":False,"desc":"Budget option, short life"},
            {"brand":"Phoenix Contact", "type":"AGM/Gel",    "capacity":"200Ah",            "price":32000, "life":"800–1000 cycles / 5 yrs","warranty":"2 years","rating":"⭐⭐⭐⭐", "recommended":False,"desc":"Better than lead acid"},
        ]
        btc1, btc2 = st.columns(2)
        for i, bat in enumerate(batteries_mp):
            col   = btc1 if i%2==0 else btc2
            badge = "🏆 RECOMMENDED" if bat['recommended'] else ""
            with col:
                st.markdown(f"""
                <div class="feature-card">
                    <div style="display:flex;justify-content:space-between;">
                        <h4>{bat['brand']}</h4>
                        <span style="color:#f7971e;font-size:12px;">{badge}</span></div>
                    <p>🔋 {bat['type']} | {bat['rating']}</p>
                    <p>⚡ Capacity: {bat['capacity']}</p>
                    <p>📅 Life: {bat['life']}</p>
                    <p>🛡️ Warranty: {bat['warranty']}</p>
                    <p>💰 <b>PKR {bat['price']:,}</b></p>
                    <small style="color:#666;">{bat['desc']}</small>
                </div>""", unsafe_allow_html=True)

    with mp_tabs[3]:
        st.markdown("### 📦 Complete Solar Packages")
        packages_mp = [
            {"name":"🏠 Starter Package",  "size":"3 KW",  "for":"Small homes: 2–3 fans, lights, 1 fridge",        "panels":"6 × Longi 550W",  "inverter":"Solis 3KW",          "battery":"2 × Pylontech","price":750000,  "saving":"PKR 4,000–6,000/month",   "payback":"5–6 years"},
            {"name":"🏡 Family Package",   "size":"5 KW",  "for":"Medium homes: 4–5 fans, 1 AC, 1 fridge",         "panels":"9 × Longi 550W",  "inverter":"Growatt 5KW Hybrid", "battery":"4 × Pylontech","price":1250000, "saving":"PKR 8,000–12,000/month",  "payback":"5–6 years"},
            {"name":"🏘️ Premium Package",  "size":"10 KW", "for":"Large homes: 2 ACs, multiple appliances",        "panels":"18 × Longi 550W", "inverter":"Huawei 10KW",        "battery":"6 × Pylontech","price":2400000, "saving":"PKR 15,000–22,000/month", "payback":"5–6 years"},
            {"name":"🏰 Luxury Package",   "size":"20 KW", "for":"Large homes/offices: Full coverage",             "panels":"36 × Longi 550W", "inverter":"Sungrow 20KW",       "battery":"10 × BYD",     "price":4500000, "saving":"PKR 30,000–45,000/month", "payback":"5–6 years"},
        ]
        for pkg in packages_mp:
            st.markdown(f"""
            <div class="feature-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <h3>{pkg['name']} — {pkg['size']}</h3>
                    <h3 style="color:#f7971e;">PKR {pkg['price']/100_000:.0f}L</h3></div>
                <p>👨‍👩‍👧 <b>Best for:</b> {pkg['for']}</p>
                <p>🔆 Panels: {pkg['panels']}</p>
                <p>🔌 Inverter: {pkg['inverter']}</p>
                <p>🔋 Battery: {pkg['battery']}</p>
                <p>💰 Monthly Saving: <b>{pkg['saving']}</b></p>
                <p>📅 Payback Period: <b>{pkg['payback']}</b></p>
            </div>""", unsafe_allow_html=True)
            st.markdown("")
# ══════════════════════════════════════════
# TAB 11 — FIND INSTALLER
# ══════════════════════════════════════════
with tab11:
    st.markdown("## 🧑‍💼 Find Solar Installer")
    st.info(
        "💡 NEPRA approved installers — "
        "Pakistan ke best solar companies!")

    inst_city = st.selectbox(
        "📍 Apni city select karo",
        cities, key='inst_city')

    # Pakistan solar installers database
    installers = {
        "Lahore": [
            {"name": "Reon Energy",
             "rating": "⭐⭐⭐⭐⭐",
             "exp": "10+ years",
             "phone": "0321-1234567",
             "nepra": "NEPRA-001",
             "speciality": "Residential & Commercial",
             "min_system": "3 KW",
             "warranty": "10 years workmanship"},
            {"name": "Beacon Energy",
             "rating": "⭐⭐⭐⭐⭐",
             "exp": "8+ years",
             "phone": "0300-9876543",
             "nepra": "NEPRA-045",
             "speciality": "Large commercial",
             "min_system": "10 KW",
             "warranty": "5 years"},
            {"name": "Zorays Solar",
             "rating": "⭐⭐⭐⭐",
             "exp": "7+ years",
             "phone": "042-35761234",
             "nepra": "NEPRA-067",
             "speciality": "All types",
             "min_system": "1 KW",
             "warranty": "5 years"},
        ],
        "Karachi": [
            {"name": "Nizam Energy",
             "rating": "⭐⭐⭐⭐⭐",
             "exp": "12+ years",
             "phone": "021-35456789",
             "nepra": "NEPRA-012",
             "speciality": "Residential",
             "min_system": "3 KW",
             "warranty": "10 years"},
            {"name": "SunTech Pakistan",
             "rating": "⭐⭐⭐⭐",
             "exp": "6+ years",
             "phone": "0333-2345678",
             "nepra": "NEPRA-089",
             "speciality": "Industrial",
             "min_system": "20 KW",
             "warranty": "5 years"},
        ],
        "Islamabad": [
            {"name": "Zelin Pakistan",
             "rating": "⭐⭐⭐⭐⭐",
             "exp": "9+ years",
             "phone": "051-2876543",
             "nepra": "NEPRA-023",
             "speciality": "Residential & Commercial",
             "min_system": "5 KW",
             "warranty": "10 years"},
            {"name": "Premier Energy",
             "rating": "⭐⭐⭐⭐⭐",
             "exp": "15+ years",
             "phone": "0300-5678901",
             "nepra": "NEPRA-003",
             "speciality": "All sizes",
             "min_system": "1 KW",
             "warranty": "10 years"},
        ],
        "Multan": [
            {"name": "SolarMax Multan",
             "rating": "⭐⭐⭐⭐",
             "exp": "5+ years",
             "phone": "061-4567890",
             "nepra": "NEPRA-134",
             "speciality": "Residential",
             "min_system": "3 KW",
             "warranty": "5 years"},
        ],
        "Peshawar": [
            {"name": "KPK Solar Solutions",
             "rating": "⭐⭐⭐⭐",
             "exp": "6+ years",
             "phone": "091-2345678",
             "nepra": "NEPRA-156",
             "speciality": "Residential",
             "min_system": "3 KW",
             "warranty": "5 years"},
        ],
        "Quetta": [
            {"name": "Balochistan Solar",
             "rating": "⭐⭐⭐⭐",
             "exp": "4+ years",
             "phone": "081-2876543",
             "nepra": "NEPRA-178",
             "speciality": "All types",
             "min_system": "3 KW",
             "warranty": "5 years"},
        ],
    }

    # Get installers for selected city
    city_installers = installers.get(
        inst_city, None)

    if not city_installers:
        # Default installers for other cities
        city_installers = [
            {"name": "National Solar Pakistan",
             "rating": "⭐⭐⭐⭐",
             "exp": "5+ years",
             "phone": "0311-7777777",
             "nepra": "NEPRA-200",
             "speciality": "All Pakistan",
             "min_system": "3 KW",
             "warranty": "5 years"},
            {"name": "SolarPK",
             "rating": "⭐⭐⭐⭐⭐",
             "exp": "8+ years",
             "phone": "0321-8888888",
             "nepra": "NEPRA-201",
             "speciality": "Residential",
             "min_system": "1 KW",
             "warranty": "10 years"},
        ]

    st.markdown(
        f"### 🏙️ Installers in {inst_city}:")
    for inst in city_installers:
        st.markdown(f"""
        <div class="feature-card">
            <div style="display:flex;
                        justify-content:space-between;
                        align-items:center;">
                <h4>🏢 {inst['name']}</h4>
                <span>{inst['rating']}</span>
            </div>
            <p>✅ NEPRA License: <b>{inst['nepra']}</b></p>
            <p>📅 Experience: <b>{inst['exp']}</b></p>
            <p>📞 Phone: <b>{inst['phone']}</b></p>
            <p>⚡ Min System: <b>{inst['min_system']}</b></p>
            <p>🔧 Speciality: {inst['speciality']}</p>
            <p>🛡️ Warranty: {inst['warranty']}</p>
        </div>""", unsafe_allow_html=True)

    # Tips for choosing installer
    st.markdown(
        "### 💡 Installer Choose Karne ke Tips:")
    tips_i = [
        "✅ Sirf NEPRA approved installer use karo",
        "✅ Kam se kam 3 quotes lo comparison ke liye",
        "✅ References check karo — puraane customers se poochho",
        "✅ Contract mein warranty terms zaroor likhwao",
        "✅ Payment installments mein karo — pehle kuch nahi",
        "✅ Site survey pehle karwao — free hoti hai",
        "✅ Net metering experience check karo",
    ]
    for t in tips_i:
        st.markdown(f"""
        <div class="feature-card"
             style="padding:10px 15px;margin:3px 0;">
            {t}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 12 — SMART ALERTS
# ══════════════════════════════════════════
with tab12:
    st.markdown("## 🔔 Smart Alerts & Tips")
    st.info(
        "💡 AI automatically detect karta "
        "hai problems aur suggestions "
        "deta hai!")

    al1, al2 = st.columns(2)
    with al1:
        alert_city = st.selectbox(
            "📍 City", cities,
            key='alert_city')
        alert_kw = st.number_input(
            "⚡ System Size (KW)",
            0.5, 50.0, 5.0,
            key='alert_kw')
        alert_roof = st.number_input(
            "📐 Roof Area (sqft)",
            100, 5000, 500,
            key='alert_roof')
    with al2:
        alert_fans = st.number_input(
            "🌀 Fans", 0, 20, 4,
            key='alert_fans')
        alert_ac = st.number_input(
            "❄️ ACs", 0, 10, 2,
            key='alert_ac')
        alert_bill = st.number_input(
            "💰 Monthly Bill (PKR)",
            500, 200000, 15000,
            key='alert_bill')
        alert_budget = st.number_input(
            "💵 Budget (PKR Lakh)",
            1, 100, 10,
            key='alert_budget')

    if st.button(
        "🔔 Generate Smart Alerts",
        key='alert_btn'):

        sun_h = city_sun.get(alert_city, 5.0)

        # Calculate what they need
        total_w = (alert_fans*75 +
                   alert_ac*1500 + 500)
        daily_kwh = (total_w/1000) * 10
        kw_needed = round(
            daily_kwh/sun_h*1.25, 1)
        panels_needed = int(
            np.ceil(kw_needed*1000/PANEL_WATT))
        roof_needed = panels_needed * 20
        cost_needed = int(kw_needed*COST_PER_KW)
        saving = int(daily_kwh*30*UNIT_RATE)

        st.markdown("### 🚨 Alert Analysis:")

        alerts_list = []

        # Alert 1: Roof too small
        if alert_roof < roof_needed:
            shortage = roof_needed - alert_roof
            alerts_list.append({
                "type": "ERROR",
                "icon": "🚨",
                "title": "Roof Too Small!",
                "msg": (
                    f"Aapko {roof_needed} sqft "
                    f"chahiye lekin sirf "
                    f"{alert_roof} sqft hai. "
                    f"{shortage} sqft short!"),
                "solution": (
                    "✅ System size reduce karo "
                    f"ya {int(shortage/20)} "
                    "panels kam lagwao"),
                "color": "#ff6b6b"
            })

        # Alert 2: System size mismatch
        if abs(alert_kw - kw_needed) > 1.5:
            if alert_kw < kw_needed:
                alerts_list.append({
                    "type": "WARNING",
                    "icon": "⚠️",
                    "title": "System Too Small!",
                    "msg": (
                        f"{alert_kw} KW system "
                        f"aapki zaroorat ke liye "
                        f"kam hai. Minimum "
                        f"{kw_needed} KW chahiye!"),
                    "solution": (
                        f"✅ System {kw_needed} KW "
                        "tak upgrade karo"),
                    "color": "#f7971e"
                })
            else:
                saving_kw = alert_kw - kw_needed
                alerts_list.append({
                    "type": "INFO",
                    "icon": "💡",
                    "title": "System Oversized!",
                    "msg": (
                        f"{alert_kw} KW system "
                        f"tumhari zaroorat se "
                        f"{saving_kw:.1f} KW "
                        f"zyada hai!"),
                    "solution": (
                        f"✅ {kw_needed} KW se "
                        f"kaam chalega. PKR "
                        f"{int(saving_kw*COST_PER_KW/1000)}K "
                        "bacha sakte ho!"),
                    "color": "#2e86de"
                })

        # Alert 3: Budget vs cost
        budget_pkr = alert_budget * 100_000
        if budget_pkr < cost_needed:
            shortage_b = cost_needed - budget_pkr
            alerts_list.append({
                "type": "ERROR",
                "icon": "💰",
                "title": "Budget Kam Hai!",
                "msg": (
                    f"Aapka budget PKR "
                    f"{alert_budget}L lekin "
                    f"system cost PKR "
                    f"{cost_needed/100_000:.1f}L. "
                    f"PKR "
                    f"{shortage_b/100_000:.1f}L "
                    f"short!"),
                "solution": (
                    "✅ Installment plan lo "
                    "ya chota system lagwao"),
                "color": "#ff6b6b"
            })

        # Alert 4: AC saving opportunity
        if alert_ac >= 2:
            saving_kw = alert_ac * 0.6
            cost_s = int(
                saving_kw * COST_PER_KW)
            alerts_list.append({
                "type": "TIP",
                "icon": "💡",
                "title": "Inverter AC se Bachao!",
                "msg": (
                    f"{alert_ac} ACs ko inverter "
                    f"AC se replace karo — "
                    f"40% less bijli!"),
                "solution": (
                    f"✅ System size "
                    f"{saving_kw:.1f} KW "
                    f"reduce hoga = PKR "
                    f"{cost_s/1000:.0f}K bachega "
                    f"system cost mein!"),
                "color": "#26de81"
            })

        # Alert 5: City solar potential
        if sun_h < 4.8:
            alerts_list.append({
                "type": "WARNING",
                "icon": "☁️",
                "title": "Low Solar Potential",
                "msg": (
                    f"{alert_city} mein sirf "
                    f"{sun_h} sun hours/day — "
                    f"Pakistan average se kam!"),
                "solution": (
                    "✅ Bigger system lagao "
                    "ya hybrid system consider "
                    "karo"),
                "color": "#f7971e"
            })
        elif sun_h >= 6.0:
            alerts_list.append({
                "type": "SUCCESS",
                "icon": "🌟",
                "title": "Excellent Solar City!",
                "msg": (
                    f"{alert_city} mein "
                    f"{sun_h} sun hours/day — "
                    f"Pakistan ke top cities mein!"),
                "solution": (
                    "✅ Solar here is BEST "
                    "investment — "
                    "invest karo zaroor!"),
                "color": "#26de81"
            })

        # Alert 6: Bill vs saving
        if alert_bill > saving:
            extra = alert_bill - saving
            alerts_list.append({
                "type": "INFO",
                "icon": "📊",
                "title": "Partial Coverage",
                "msg": (
                    f"Solar se PKR "
                    f"{saving:,}/month bachega "
                    f"lekin bill PKR "
                    f"{alert_bill:,} hai. "
                    f"PKR {extra:,} abhi bhi "
                    f"dena hoga."),
                "solution": (
                    f"✅ System size barhao "
                    f"ya bijli ka use "
                    f"optimize karo"),
                "color": "#2e86de"
            })
        else:
            alerts_list.append({
                "type": "SUCCESS",
                "icon": "🎉",
                "title": "Full Bill Coverage!",
                "msg": (
                    f"Solar se pura bill "
                    f"cover ho jayega! "
                    f"PKR {saving-alert_bill:,} "
                    f"extra grid ko export!"),
                "solution": (
                    "✅ Net metering apply "
                    "karo — extra earn karo!"),
                "color": "#26de81"
            })

        # Display alerts
        if not alerts_list:
            st.success("✅ Sab theek lag "
                       "raha hai!")
        for alert in alerts_list:
            st.markdown(f"""
            <div style="background:{alert['color']};
                        padding:15px 20px;
                        border-radius:12px;
                        color:white;
                        margin:8px 0;">
                <h4>{alert['icon']} {alert['title']}</h4>
                <p>{alert['msg']}</p>
                <p><b>{alert['solution']}</b></p>
            </div>""",
                unsafe_allow_html=True)

        # Summary
        st.markdown("### 📊 Your Summary:")
        sm1,sm2,sm3,sm4 = st.columns(4)
        with sm1:
            st.metric(
                "KW Needed", f"{kw_needed} KW",
                f"{alert_kw-kw_needed:+.1f} KW")
        with sm2:
            st.metric(
                "Panels Needed",
                f"{panels_needed}",
                f"Roof: {roof_needed} sqft")
        with sm3:
            st.metric(
                "System Cost",
                f"PKR {cost_needed/100_000:.1f}L",
                f"Budget: {alert_budget}L")
        with sm4:
            st.metric(
                "Monthly Saving",
                f"PKR {saving:,}",
                f"Bill: {alert_bill:,}")

# ══════════════════════════════════════════
# TAB 13 — USER PROFILES
# ══════════════════════════════════════════
with tab13:
    st.markdown("## 👤 My Solar Profiles")
    st.info(
        "💡 Multiple ghar save karo aur "
        "compare karo!")

    # Session state for profiles
    if 'profiles' not in st.session_state:
        st.session_state.profiles = {}

    profile_tabs = st.tabs([
        "➕ Add Profile",
        "📋 My Profiles",
        "📊 Compare Profiles"
    ])

    with profile_tabs[0]:
        st.markdown("### ➕ New Home Profile")

        pf1, pf2 = st.columns(2)
        with pf1:
            p_name = st.text_input(
                "🏠 Profile Name",
                placeholder="e.g. Main House, Office",
                key='p_name')
            p_city = st.selectbox(
                "📍 City", cities,
                key='p_city')
            p_fans = st.number_input(
                "🌀 Fans", 0, 20, 4,
                key='p_fans')
            p_ac = st.number_input(
                "❄️ ACs (1 ton)", 0, 10, 1,
                key='p_ac')
            p_fridge = st.number_input(
                "🧊 Fridges", 0, 5, 1,
                key='p_fridge')
        with pf2:
            p_lights = st.number_input(
                "💡 Lights", 0, 30, 8,
                key='p_lights')
            p_tv = st.number_input(
                "📺 TVs", 0, 10, 2,
                key='p_tv')
            p_washing = st.number_input(
                "🫧 Washing Machine", 0, 3, 1,
                key='p_wash')
            p_hours = st.selectbox(
                "⏰ Usage Hours",
                list(range(4,20)), index=4,
                key='p_hours')
            p_bill = st.number_input(
                "💰 Monthly Bill (PKR)",
                500, 200000, 8000,
                key='p_bill')

        if st.button(
            "💾 Save Profile",
            key='save_profile'):
            if p_name:
                sun_h = city_sun.get(
                    p_city, 5.0)
                total_w = (
                    p_fans*75 + p_lights*15 +
                    p_fridge*150 + p_ac*1500 +
                    p_tv*100 + p_washing*500)
                daily_kwh = (
                    total_w/1000) * p_hours
                kw_need = round(
                    daily_kwh/sun_h*1.25, 1)
                cost = int(kw_need*COST_PER_KW)
                saving = int(
                    daily_kwh*30*UNIT_RATE)

                st.session_state.profiles[
                    p_name] = {
                    'city': p_city,
                    'fans': p_fans,
                    'ac': p_ac,
                    'fridge': p_fridge,
                    'lights': p_lights,
                    'tvs': p_tv,
                    'washing': p_washing,
                    'hours': p_hours,
                    'bill': p_bill,
                    'kw_needed': kw_need,
                    'cost': cost,
                    'saving': saving,
                    'daily_kwh': round(
                        daily_kwh, 1),
                    'panels': int(np.ceil(
                        kw_need*1000/PANEL_WATT)),
                    'payback': round(
                        cost/(saving*12), 1),
                    'saved_on': datetime.now(
                        ).strftime("%d/%m/%Y")
                }
                st.success(
                    f"✅ Profile '{p_name}' "
                    f"saved!")
            else:
                st.warning(
                    "Profile ka naam dena "
                    "zaroor hai!")

    with profile_tabs[1]:
        st.markdown("### 📋 My Saved Profiles")

        if not st.session_state.profiles:
            st.info(
                "Koi profile nahi hai! "
                "Add Profile tab se "
                "add karo!")
        else:
            for pname, pdata in \
                    st.session_state.profiles\
                    .items():
                with st.expander(
                    f"🏠 {pname} — "
                    f"{pdata['city']} — "
                    f"{pdata['kw_needed']} KW"
                ):
                    pp1,pp2 = st.columns(2)
                    with pp1:
                        st.markdown(f"""
                        <div class="feature-card">
                            <h4>🏠 {pname}</h4>
                            <p>📍 City: {pdata['city']}</p>
                            <p>🌀 Fans: {pdata['fans']}</p>
                            <p>❄️ ACs: {pdata['ac']}</p>
                            <p>🧊 Fridges: {pdata['fridge']}</p>
                            <p>💡 Lights: {pdata['lights']}</p>
                            <p>⏰ Hours: {pdata['hours']}hrs</p>
                        </div>""",
                            unsafe_allow_html=True)
                    with pp2:
                        st.markdown(f"""
                        <div class="result-box">
                            <p>⚡ {pdata['kw_needed']} KW</p>
                            <p>🔆 {pdata['panels']} Panels</p>
                            <p>💰 PKR {pdata['cost']/100_000:.1f}L</p>
                            <p>💵 PKR {pdata['saving']:,}/month</p>
                            <p>📅 {pdata['payback']} yr payback</p>
                        </div>""",
                            unsafe_allow_html=True)

                    if st.button(
                        f"🗑️ Delete {pname}",
                        key=f'del_{pname}'):
                        del st.session_state\
                            .profiles[pname]
                        st.success("Deleted!")
                        st.rerun()

    with profile_tabs[2]:
        st.markdown(
            "### 📊 Compare All Profiles")

        if len(st.session_state.profiles) < 2:
            st.info(
                "Compare ke liye kam se "
                "kam 2 profiles chahiye!")
        else:
            names = list(
                st.session_state.profiles.keys())
            costs = [
                st.session_state.profiles[n]['cost']
                for n in names]
            savings = [
                st.session_state.profiles[n]['saving']
                for n in names]
            kws = [
                st.session_state.profiles[n]['kw_needed']
                for n in names]

            fig, axes = plt.subplots(
                1, 3, figsize=(14, 5))

            axes[0].bar(names, kws,
                        color='#f7971e')
            axes[0].set_title('System Size (KW)')
            axes[0].tick_params(
                axis='x', rotation=45)

            axes[1].bar(
                names,
                [c/100_000 for c in costs],
                color='#2e86de')
            axes[1].set_title('Cost (Lakh PKR)')
            axes[1].tick_params(
                axis='x', rotation=45)

            axes[2].bar(
                names,
                [s/1000 for s in savings],
                color='#26de81')
            axes[2].set_title(
                'Monthly Saving (K PKR)')
            axes[2].tick_params(
                axis='x', rotation=45)

            plt.tight_layout()
            st.pyplot(fig)

            # Table comparison
            comp_data = [{
                'Profile': n,
                'City': st.session_state
                    .profiles[n]['city'],
                'KW': st.session_state
                    .profiles[n]['kw_needed'],
                'Cost (L)': f"{st.session_state.profiles[n]['cost']/100_000:.1f}",
                'Saving/Month': f"PKR {st.session_state.profiles[n]['saving']:,}",
                'Payback': f"{st.session_state.profiles[n]['payback']} yrs"
            } for n in names]

            st.dataframe(
                pd.DataFrame(comp_data),
                use_container_width=True)

            # Best profile
            best_idx = savings.index(
                max(savings))
            st.success(
                f"🏆 Best investment: "
                f"**{names[best_idx]}** — "
                f"highest monthly saving of "
                f"PKR {max(savings):,}!")

# ══════════════════════════════════════════
# TAB 14 — AI ASSISTANT
# ══════════════════════════════════════════
with tab14:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#f7971e);
        padding:20px;border-radius:15px;text-align:center;color:white;margin-bottom:20px;">
        <h2 style="margin:0;">🤖 Solar AI Assistant</h2>
        <p style="margin:5px 0 0 0;opacity:0.9;">Solar ke baare mein kuch bhi poochho!</p>
    </div>""", unsafe_allow_html=True)

    def solar_fallback(msg):
        m = msg.lower()
        if any(w in m for w in ['hello','hi','salam','hey']):
            return ("👋 Salam! Main Solar Finder AI hoon! ☀️\n\nPoochho:\n"
                    "• Kitna solar system chahiye?\n• Kaunsa city best hai?\n"
                    "• Batteries ke baare mein?\n• Bill saving kaise hogi?")
        if any(w in m for w in ['best city','konsa city','which city']):
            return ("☀️ Best cities in Pakistan:\n\n"
                    "🥇 Quetta/Kharan — 6.5 hrs/day\n🥈 Chaman — 6.4 hrs/day\n"
                    "🥉 Turbat/Sibi — 6.3 hrs/day\n4️⃣ Sukkur — 6.2 hrs/day")
        if any(w in m for w in ['kitna cost','price','how much']):
            return ("💰 Solar costs 2025:\n\n• 3 KW: PKR 6 Lakh\n"
                    "• 5 KW: PKR 10 Lakh\n• 10 KW: PKR 20 Lakh\n• 20 KW: PKR 40 Lakh")
        if any(w in m for w in ['battery','backup','load shedding']):
            return ("🔋 Battery guide:\n\n✅ Lithium — Best! 10–15 year life\n"
                    "⚠️ AGM/Gel — 3–5 years\n❌ Lead Acid — 2–3 years only")
        if any(w in m for w in ['ac','air condition']):
            return ("❄️ AC aur solar:\n\n• AC 1 ton = 1500W\n"
                    "• 2 ACs ke liye minimum 8 KW system\n"
                    "✅ Inverter AC use karo — 40% less bijli!")
        if any(w in m for w in ['panel','kitne panel']):
            return ("🔆 550W panels:\n\n• 3 KW = 6 panels\n• 5 KW = 10 panels\n"
                    "• 10 KW = 19 panels\n📐 Har panel = 20 sqft roof")
        if any(w in m for w in ['saving','bill','bachhat']):
            return ("💰 Monthly savings:\n\n• 3 KW = PKR 4,000–6,000\n"
                    "• 5 KW = PKR 7,000–10,000\n• 10 KW = PKR 14,000–18,000\n"
                    "📅 Payback: 4–6 years")
        if any(w in m for w in ['net meter','export','sell bijli']):
            return ("🌐 Net Metering:\n\n• NEPRA rate: PKR 19.32/unit\n"
                    "• Extra bijli grid ko sell karo\n• DISCO se apply karo\n"
                    "• 1–3 mahine mein approval")
        if any(w in m for w in ['summer','winter','season','sardi','garmi']):
            return ("🌡️ Solar & seasons:\n\n☀️ Summer: Best generation!\n"
                    "  (longer days, more sun)\n❄️ Winter: 25–30% less output\n"
                    "  (but panels run cooler = efficient)\n\nAnnual average still great!")
        if any(w in m for w in ['roof','chhat','space','area']):
            return ("🏠 Roof space needed:\n\n• 3 KW (6 panels) = 120 sqft\n"
                    "• 5 KW (10 panels) = 200 sqft\n• 10 KW (19 panels) = 380 sqft\n"
                    "💡 Roof Layout tab mein visual dekho!")
        if any(w in m for w in ['document','kagaz','apply']):
            return ("📋 Net Metering documents:\n\n1. CNIC copy\n"
                    "2. Latest electricity bill\n3. Solar system invoice\n"
                    "4. Single Line Diagram\n5. Ownership proof")
        if any(w in m for w in ['brand','best panel','which brand']):
            return ("🏆 Best brands in Pakistan:\n\n🥇 Longi Solar — Most popular\n"
                    "🥈 Jinko Solar — World's largest\n🥉 Trina Solar — Tier 1\n"
                    "4️⃣ Canadian Solar — Premium\n5️⃣ Risen Energy — Budget")
        return ("🤔 Thoda aur detail mein poochho!\n\nMain help kar sakta hoon:\n"
                "☀️ System size  💰 Cost & savings\n"
                "🔋 Battery backup  🏙️ City potential")

    def get_solar_response(user_msg, history):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY", ""))
            msgs = []
            for m in history[-8:]:
                role = 'assistant' if m['role'] in ('bot','assistant') else 'user'
                msgs.append({"role": role, "content": m['content']})
            msgs.append({"role": "user", "content": user_msg})
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                system="""You are Solar Finder AI — Pakistan's smartest solar energy assistant!

LANGUAGE RULE: Match user's language (English or Roman Urdu). Use emojis!

PAKISTAN SOLAR FACTS 2025:
- 1 KW solar = PKR 200,000 approx
- 550W panels standard, UNIT_RATE = PKR 62/unit
- Net metering NEPRA buyback = PKR 19.32/unit
- Best cities: Quetta/Kharan 6.5h, Chaman 6.4h, Turbat 6.3h, Multan 6.0h
- Best brands: Longi, Jinko, Trina Solar
- 5 KW system = PKR 10 Lakh approx, payback 4–6 years

RULES: Max 150 words. Always PKR amounts. Be specific with numbers.""",
                messages=msgs)
            return resp.content[0].text
        except Exception:
            return solar_fallback(user_msg)

    with st.container():
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="chat-msg-bot">
                🤖 <b>Solar Finder AI mein khush aamdeed!</b> ☀️<br><br>
                ☀️ Kitna solar system chahiye?<br>
                💰 Kitna cost aayega?<br>
                🔋 Kitni batteries chahiye?<br>
                🏙️ Konsa city best hai?<br><br>
                <i>English ya Roman Urdu mein poochho! 😊</i>
            </div><div style="clear:both"></div>""", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(
                    f'<div class="chat-msg-user">👤 {msg["content"]}</div>'
                    '<div style="clear:both"></div>', unsafe_allow_html=True)
            else:
                content = msg['content'].replace('\n', '<br>')
                st.markdown(
                    f'<div class="chat-msg-bot">🤖 {content}</div>'
                    '<div style="clear:both"></div>', unsafe_allow_html=True)

    st.markdown("#### ⚡ Quick Questions:")
    SOLAR_Q = [
        ("🏙️ Best City",     "Which city is best for solar in Pakistan and why?"),
        ("💰 5KW Cost",       "How much does a 5 KW solar system cost in Pakistan in 2025?"),
        ("🔋 Best Battery",   "Which battery is best for solar in Pakistan — lithium or lead acid?"),
        ("❄️ AC on Solar",    "How much solar do I need to run 2 ACs in Lahore?"),
        ("💵 Bill Saving",    "How much monthly bill will I save with 5 KW solar in Karachi?"),
        ("🔆 Panels Count",   "How many panels do I need for a 5 KW solar system in Pakistan?"),
        ("📅 Payback",        "In how many years will a 5 KW solar system pay back in Pakistan?"),
        ("🌐 Net Metering",   "How does net metering work in Pakistan and how much can I earn?"),
        ("🌡️ Summer vs Winter","Does solar work better in summer or winter in Pakistan?"),
        ("🏠 Roof Size",      "How much roof space do I need for a 5 KW solar system?"),
        ("📋 Documents",      "What documents are needed for solar net metering in Pakistan?"),
        ("🏆 Best Brand",     "Which is the best solar panel brand available in Pakistan?"),
    ]

    for row in range(0, len(SOLAR_Q), 4):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row + i
            if idx < len(SOLAR_Q):
                label, question = SOLAR_Q[idx]
                with col:
                    if st.button(label, key=f'sq_{idx}', use_container_width=True):
                        st.session_state.chat_history.append({'role':'user','content':question})
                        with st.spinner("🤖 Thinking..."):
                            resp = get_solar_response(question, st.session_state.chat_history[:-1])
                        st.session_state.chat_history.append({'role':'assistant','content':resp})
                        st.rerun()

    st.markdown("---")
    ci, cs, cc = st.columns([5, 1, 1])
    with ci:
        user_input = st.text_input("msg", key="solar_chat", label_visibility="collapsed")
    with cs:
        send = st.button("📤", key='solar_send', use_container_width=True)
    with cc:
        if st.button("🗑️", key='solar_clear', use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({'role':'user','content':user_input})
        with st.spinner("🤖 Thinking..."):
            response = get_solar_response(user_input, st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({'role':'assistant','content':response})
        st.rerun()

# ══════════════════════════════════════════
# TAB 15 — HISTORY
# ══════════════════════════════════════════
with tab15:
    st.markdown("## 📋 Calculation History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        csv = hist_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="solar_history.csv">'
                    f'📥 Download History as CSV</a>', unsafe_allow_html=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.success("Cleared!")
    else:
        st.info("Abhi koi calculation nahi ki! Solar Calculator tab use karo! ☀️")

# ── Footer ────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center;color:grey;font-size:14px;">
    ☀️ Solar Finder AI | Powered by Machine Learning  | Built with ❤️ | © 2026 
</p>""", unsafe_allow_html=True) 
