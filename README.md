🌍 AI-Powered Rural Air Quality Monitor

Satellite + AI | Real-Time AQI | No Physical Sensors Required

Team NEWBEES

📌 Project Overview

In most rural areas of India, there are no air quality monitoring stations, making it difficult to assess pollution levels in real time. This project proposes a cost-effective, scalable, AI-powered solution to estimate real-time AQI using satellite and meteorological data, without relying on ground sensors.

The system predicts PM₂.₅ concentrations using machine learning models and converts them into CPCB-standard AQI categories, enabling health-aware decision making for rural populations.

👥 Team Members
Name	Role
Vishal	Team Leader
Rishav	Backend Developer
Nikhil	UI/UX Designer
Faiz	Frontend Developer
Ashutosh	AI/ML Engineer
🚨 Why This Project Matters

🚫 Lack of Monitoring: Most rural areas have no real-time AQI stations

🌾 Hidden Health Risks: Crop residue burning, biomass fuel, dust pollution

👨‍👩‍👧‍👦 Coverage Gap: ~62% of India’s population lives beyond 50 km of AQI stations

🌱 SDG Alignment: Supports SDG 3 (Health), 11 (Sustainable Cities), 13 (Climate Action)

India faces one of the largest air-quality monitoring gaps globally.

💡 Key Idea

Machine-learning-based real-time AQI prediction using Satellite AOD and ERA5 meteorological data

🛰️ Data Sources
Source	Purpose
MODIS / VIIRS / OCM-3	Aerosol Optical Depth (AOD)
ERA5 (Copernicus)	Weather variables (temperature, humidity, wind, pressure)
CPCB Ground Data	PM₂.₅ & AQI labels (training reference)
🧠 Model Training Pipeline
Training Phase

Historical AOD + ERA5 + CPCB PM₂.₅ data used

ML model learns correlation between AOD, weather & PM₂.₅

Models tested:

Random Forest

Regression

Output:

Predicted PM₂.₅

Converted to AQI using CPCB formula

Data Pre-Processing

Spatial & temporal synchronization

Missing value handling

Feature normalization

Dataset merging:

AOD + ERA5 + CPCB PM₂.₅ → Unified Training Dataset

🏗️ System Architecture
User Flow

Capture latitude & longitude

Send coordinates to backend API

Receive AQI, category & health tips

Backend Flow
Receive Coordinates
→ Fetch Real-time AOD + ERA5
→ Feature Preprocessing
→ ML Model Inference
→ PM₂.₅ Prediction
→ AQI & Category Calculation
→ JSON Response

⚙️ Technologies Used
Frontend

HTML / React

Tailwind CSS

Backend

FastAPI / Flask

REST APIs

ML & Data

Python

Random Forest

Satellite + ERA5 data

Deployment

Cloud-based API deployment

📊 Final Output

Real-time AQI value

CPCB AQI category

Health advisories & precautions

🎯 Key Benefits

✅ Coverage Expansion: AQI access for underserved rural regions

🏥 Health Awareness: Converts PM₂.₅ to CPCB AQI & advisories

🏛️ Policy Support: NCAP evaluation & hotspot identification

💸 Cost-Effective: No dense sensor networks required

🌍 Equity & Inclusion: Rural-first AQI delivery

🔗 Standards-Compliant: CPCB AQI breakpoints applied

🔬 Research Validation

This approach is inspired and validated by recent peer-reviewed research:

Science Advances (2025)
ML-based PM₂.₅ mapping using satellite AOD & meteorology

Environmental Science & Technology (2023–2024)
ERA5 & MERRA-2 based long-term PM₂.₅ reconstruction for India

These studies demonstrate high accuracy and scalable rural coverage.

🚀 Future Scope

Interactive AQI maps

Mobile app integration

Early-warning alerts

Higher spatial resolution

Government & NGO deployment

📜 License

This project is developed for academic and research purposes.
Open-source usage encouraged with proper attribution.

