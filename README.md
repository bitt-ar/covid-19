# Interactive COVID-19 Dashboard

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Data Source](https://img.shields.io/badge/Data-Our%20World%20in%20Data-1D4ED8)

An interactive **Streamlit** application for exploring global COVID-19 data with maps, key indicators, country comparisons, and time-series analysis.

## Preview Image

![SARS-CoV-2 Illustration](https://upload.wikimedia.org/wikipedia/commons/3/35/SARS-CoV-2_without_background.png)

## Features

- Key indicators: total cases, deaths, vaccinated people, and recovered estimates.
- Interactive global map based on the selected metric.
- Top countries ranking by any selected metric.
- Time-series analysis (Line / Area / Bar) with moving average.
- Start vs end period comparison.
- Correlation analysis between selected metrics.
- Country detail snapshot with core indicators.

## Project Structure

- `app.py`: main application.
- `data/owid-covid-data.csv`: data source.
- `requirements.txt`: project dependencies.
- `covid19-case-study.ipynb`: additional analysis notebook.
- `covid_case_study_presentation.pptx`: project presentation.

## Requirements

- Python 3.10+
- pip

## Installation and Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Data Source

- **Our World in Data (OWID)**
- Dataset file used: `owid-covid-data.csv`

## Notes

- The app automatically looks for the dataset in `data/`.
- The app currently caps the displayed date range at `2024-08-01`.

## Screenshots (Optional)

You can add screenshots from the app in `docs/images` and reference them here, for example:

```md
![Dashboard Overview](docs/images/dashboard-overview.png)
![Global Map](docs/images/global-map.png)
```
