# Nepal Climate Service

This project is based on the [Open Climate Service](https://dhis2.github.io/climate-api/) and provides climate data for Nepal, including ERA5-Land reanalysis, CHIRPS3 precipitation, and WorldPop population data.

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd nepal-climate-service
   ```

2. **Install dependencies:**
   ```sh
   make install
   ```

3. **Configure environment:**
   ```sh
   cp .env.example .env
   # Edit .env and set CLIMATE_API_CONFIG to the absolute path of climate-api.yaml
   # Set PROJ_DATA to the rasterio proj_data path in your venv
   ```

## Usage

```sh
make run
```

The API starts at http://localhost:8001. Visit `/manage` to ingest datasets.

## Available Datasets

| Dataset | Period | Source |
|---|---|---|
| 2m temperature (ERA5-Land) | Hourly | DestinE Earth Data Hub |
| Total precipitation (ERA5-Land) | Hourly | DestinE Earth Data Hub |
| Total precipitation (CHIRPS3) | Daily | CHC UCSB |
| Total population (WorldPop) | Yearly | WorldPop |
