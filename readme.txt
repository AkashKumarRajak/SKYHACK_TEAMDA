# ORD Flight Difficulty Score Dashboard

This project is a Streamlit-based analytics dashboard that ranks ORD (O'Hare) departures by operational difficulty.  
It combines key operational data such as flight schedules, baggage information, passenger counts, and special service requests to identify which flights are most challenging to manage on any given day.

---
## Setup Instructions
1. Create and activate a virtual environment


2. Install the dependencies
pip install -U pip
pip install -r requirements.txt

3. Run the dashboard

streamlit run app.py
The following CSV files must be in the same folder as the app:

- Flight Level Data.csv
- Bag+Level+Data.csv
- PNR+Flight+Level+Data.csv
- PNR Remark Level Data.csv (optional but recommended)
- Airports Data.csv
- requirements.txt

---

# Project Overview

### Objective

- Airport operations need a simple way to see which flights are likely to be the most difficult to manage each day.
Flights with short turnaround times, heavy transfer loads, high passenger counts, or many special service requests (SSR) tend to create operational challenges.
This tool provides a clear, data-driven way to highlight those flights before problems occur.

### Core Idea

#### Each flight is scored and ranked based on four measurable features:
- Ground Risk – how close the scheduled ground time is to the minimum turnaround time.
- Transfer Ratio – proportion of transfer bags compared to all checked bags.
- Load Factor – ratio of passengers to total seats.
- SSR Total – count of special service requests such as wheelchair, medical, or minor assistance.

#### The application calculates a standardized score for every flight within each day, ranks them, and groups them into three categories:

---


### Features

1. Exploratory Data Analysis (EDA)

- Calculates average departure delay and the percentage of late flights.
- Identifies how many flights have scheduled ground time close to or below the minimum turn time.
- Summarizes transfer bag ratios, load factors, and SSR patterns.
- Visualizes delay distributions and load factor versus delay scatter plots.

2. Daily Difficulty Score

- Normalizes each selected feature by day using z-scores.
- Applies weighted combinations (adjustable in the sidebar).
- Generates a difficulty score and rank for every flight.
- Classifies flights into Difficult, Medium, or Easy categories.

3. Post-Analysis and Insights

- ighlights destinations that consistently appear in the Difficult category.
- Shows which features contribute most to difficulty.
- Suggests practical operational improvements.

---

# Project Structure

ORD-Difficulty-Score/
│
├── app.py                 # Streamlit main app
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
│
├── Flight Level Data.csv
├── Bag+Level+Data.csv
├── PNR+Flight+Level+Data.csv
├── PNR Remark Level Data.csv
└── Airports Data.csv

---

# Example Insights

- ORD to BOS flights often have very short turn times and show higher delay risk.
- ORD to LGA flights frequently carry high SSR loads and need better pre-boarding coordination.
- ORD to DEN flights have high load factors and require early boarding management.
- ORD to CDG flights involve many transfer bags and need stronger baggage team support.


---

# Recommended Actions

- Protect flights operating at or below minimum turn time by assigning additional ramp and gate resources.
- Schedule transfer-bag teams during heavy bank waves to reduce missed connections.
- Prepare SSR-related equipment and staff in advance for high-demand flights.

---

## Impact Summary

- **Turn Management:**  
  Previously, gate and turnaround planning was reactive and involved manual adjustments after issues arose.  
  With the new system, decisions are proactive and data-driven, allowing teams to anticipate tight turns and allocate resources efficiently.

- **Transfer Handling:**  
  Earlier, transfer baggage volumes were unpredictable, often leading to connection delays.  
  The dashboard now identifies high-transfer flights in advance, enabling smoother coordination and targeted staffing during peak periods.

- **SSR Coordination:**  
  Special Service Requests (such as wheelchair or medical assistance) were manually tracked and prone to oversight.  
  Automated detection through PNR remarks now ensures that SSR requirements are recognized early and prepared for before departure.

- **Resource Planning:**  
  Resource allocation used to rely largely on individual experience and intuition.  
  It is now guided by quantified difficulty scores, providing a consistent, objective basis for operational planning and shift assignments---
Author and Acknowledgment

Developed by Aviral and Team
Built with Streamlit, Pandas, NumPy, and Matplotlib
Designed to support operational efficiency and better decision-making through data analysis.---


## Future Enhancements

- **Gate-Level or Bank-Level Analysis:**  
  Incorporate gate and bank identifiers to enable more precise resource planning and identify bottlenecks within specific terminal areas.

- **Feature Importance Visualization:**  
  Add interpretability tools such as feature importance charts to show which variables (ground risk, load factor, transfer ratio, etc.) drive the difficulty score the most.

- **Integration with Live Flight or OPS Data Feeds:**  
  Connect to live operational systems to update scores and dashboards in real time for better situational awareness.

- **Daily Summary Storage:**  
  Implement automated saving of daily difficulty results to support historical tracking, performance benchmarking, and trend analysis over time.


---
### Author and Acknowledgment

- Developed by Akash Kumar Rajak & Mayank Yadav (TEAMDA)
- Built with Streamlit, Pandas, NumPy, and Matplotlib
- Designed to support operational efficiency and better decision-making through data analysis.
