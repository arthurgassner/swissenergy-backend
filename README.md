> [!WARNING]
> This repo has been **archived**. <br>
> Its content was moved to this repo: [github.com/arthurgassner/swissenergyforecast](https://github.com/arthurgassner/swissenergyforecast).

---

# swissenergy-backend

<br>

<p align="center">🚀 <a href="https://swissenergyforecast.com"><strong>live dashboard & detailed write-up</strong></a> 🚀</p>

<br>

<p align="center"><img src="img/dashboard.gif" width="100%"><p>

This repository contains the ML backend powering an **energy consumption prediction dashboard**.

Inspired by the [SFOE's energy consumption dashboard](https://www.energiedashboard.admin.ch/strom/stromverbrauch), I figured it would be a great opportunity to talk about an end-to-end ML project, going over the challenges one encounters during

- Problem Understanding
- Data Ingestion
- Exploratory Data Analysis
- Machine Learning Modelling
- Industrialization
- Deployment

> [!IMPORTANT]
> I _heavily_ encourage you to check out the 🚀 [**write-up**](https://swissenergyforecast.com) 🚀 to make sense of this repo, as it goes through each stage methodically.

> [!NOTE]  
> The code for the frontend can be found [here](https://github.com/arthurgassner/swissenergy-frontend).

## Repo structure

The repo is structured as follows

```bash 
├── app/ # ML backend
├── img/ 
├── viz/ # Visualization built for the writeup
├── nb-dev/ # Notebooks created during the EDA/Modelling phase
├── tests/ # pytests
├── .gitignore 
├── .pre-commit-config.yaml 
├── .pyproject.toml
├── uv.lock
├── Dockerfile
├── compose.yaml 
├── sanity_checks.ipynb # Used to manually check our some inputs
├── data_checks.ipynb # Used to manually check our data
└── README.md
```

## Running the backend

The backend is meant to be run as a dockerized app, running off some machine. This project's [write-up](https://swissenergyforecast.com) goes in depth about how to run the backend.

![](img/backend.png)
