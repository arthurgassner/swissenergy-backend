# Swiss-energy

Inspired by the [SFOE's energy consumption dashboard](https://www.energiedashboard.admin.ch/strom/stromverbrauch), I wondered how accurate these energy consumption predictions could get only using public data.

This repo gathers the early-stages of my efforts to answer that question.

## Data

The data is sourced from [ENTSO-E](https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show?name=&defaultValue=true&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime=08.12.2022+00:00|CET|DAY&biddingZone.values=CTY|10YCH-SWISSGRIDZ!BZN|10YCH-SWISSGRIDZ&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)).

> The ENTSO-E is TODO.

> Note that they say themselves that the data is not to be trusted blindly [here TODO].


## TODOs

### Data
- [x] Locally get all data up to some recent point in time 

### Modelling
- [x] Measure the performance of their current approach
- [x] Build a dummy baseline that predicts the consumption 24h-from-now as last hour's consumption
- [x] Build a back-testing approach, to accurately compare models' performances
- [x] Build a LGBM-based model leveraging the last hour's consumption
- [x] Perform feature engineering to enrich the data with the hourly-consumption 24h ago, and a week ago.
- [x] Perform feature engineering to enrich the data with statistics about the previous hourly-consumptions, over 24h and 7d.
- [x] Build a training notebook, where the best model can be trained from scratch.
- [ ] Perform hyperparameters search (`optuna`), potentially also through the features used.
- [ ] Use weather-related covariates -- both past and future -- as features.
- [ ] Use Swiss-stock-market covariates -- both past and future -- as features.
- [ ] Leverage the energy outage data as past covariate.

### MLOps

- [x] Move the modelling efforts into a `.py`
- [x] Build a FastAPI server
- [x] Dockerize the `.py`
- [x] Deploy the docker image on a VPS
- [ ] Setup a Cron job to retrain the model hourly
- [ ] Build a small webserver allowing access to the ENTSOE-DF data and the latest model's prediction
- [ ] Host that webserver on the VPS, allowing access through the internet
- [ ] Automatically update the software on the VPS through GitHub Actions

### Website

- [ ] Build a rudimentary website showcasing a plot of the current prediction.
- [ ] Use the website to explain your approach to EDA, modelling and MLOPs
- [ ] EDA: Talk about autocorrelation, window-average
- [ ] Modelling: Talk about back-testing, back-testing with sampling to avoid losing too much time on compute, how to represent a time series as a regression problem
- [ ] MLOps: Talk about how to deploy on a VPS