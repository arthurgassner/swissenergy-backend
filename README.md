# `swissenergy-backend`

> [!IMPORTANT] 
> The project is better understood through its live 🚀 [**dashboard and detailed write-up**](https://swissenergy.arthurgassner.ch) 🚀.

<br>

This repository contains the ML backend powering an energy consumption prediction dashboard.

Inspired by the [SFOE's energy consumption dashboard](https://www.energiedashboard.admin.ch/strom/stromverbrauch), I figured it would be a great opportunity to showcase an end-to-end ML project, going over the challenges one encounters during

- Problem Understanding
- Data Ingestion
- Exploratory Data Analysis
- Machine Learning Modelling
- Industrialization
- Deployment

![](img/dashboard.gif)

> [!IMPORTANT]
> I _heavily_ encourage you to check out the 🚀 [**write-up**](https://swissenergy.arthurgassner.ch) 🚀 to make sense of this repo, as it goes through each stage methodically.

## Repo structure

The repo is structured as follows

```bash 
├── img 
├── model_server # ML backend
├── nb-dev # Notebooks created during the EDA/Modelling phase
├── tests # pytests
├── viz # Visualization built for the writeup
├── .gitignore 
├── .pre-commit-config.yaml 
├── Dockerfile
├── README.md
├── data_checks.ipynb # Used to manually check our data
├── docker-compose.yml 
├── requirements.txt
└── sanity_checks.ipynb # Used to manually check our some inputs
```

## Running the backend

The backend is meant to be run as a dockerized app, running off some machine. This project's [write-up](https://swissenergy.arthurgassner.ch) goes in depth about how to run the backend.

![](img/backend.png)
