# CrowdSense: Smart Crowd Management System 🚨👥

Real-time Crowd Intelligence for Public Safety  
Built by **Team Horizon** – Hack-O-Clock 2025 Submission  
Theme: **Public Safety**

---

## 🚩 Problem Statement

> “No real-time crowd data. Delayed response. Stampedes happen. Manual surveillance isn't enough.”

Large public gatherings often lack effective crowd monitoring, leading to delayed communication, panic, stampedes, and safety hazards. Examples like the **Delhi Station incident (March 2024)** and **Mahakumbh stampede** highlight the critical need for real-time crowd data.

---

## ✅ Solution: CrowdSense

CrowdSense is a real-time AI-powered system for smart crowd detection, zone-based redirection, and safety alerts.

### 🔍 Key Features

- **🧠 Real-Time Crowd Detection & Counting**  
  AI-powered live people detection using surveillance feeds.

- **🌡️ Density-Based Heatmaps**  
  Color-coded visualization of crowd intensity per zone.

- **🧭 Zone-Based Redirection Logic**  
  Suggests safe navigation using directional arrows when zones are overcrowded.

- **📊 Graphs & Maps**  
  Dynamic visualization of crowd flow and density over time.

---

## 🛠️ Tech Stack

| Tech            | Role                                        |
|-----------------|---------------------------------------------|
| **Python**      | Core backend & logic                        |
| **YOLOv5**      | Real-time crowd detection                   |
| **OpenCV**      | Video feed processing & API handling        |
| **Matplotlib**  | Visualization of time-based density trends  |
| **HTML/CSS/JS** | Interactive frontend                        |

---

## 🎯 Use Cases

- 🛕 Religious Gatherings (e.g. Mahakumbh)
- 🛫 Airports and Railway Stations
- 🪧 Political Rallies & Protests
- 🎉 Festive Celebrations & Concerts

---

## 👨‍💻 Team Horizon

- **Saurya Tripathi** – ML Engineer  
- **Parth Gupta** – UI/UX Developer  
- **Saurabh Rajput** – App Developer  
- **Saubhagya Sharma** – Web Developer  

Built under the banner of **Google Developer Group on Campus, IILM**.

---

## 📸 Screenshots

*(Add your screenshots or demo gifs here)*

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/crowdsense.git
cd crowdsense

# Install backend requirements
pip install -r requirements.txt

# Run Flask backend
python app.py
