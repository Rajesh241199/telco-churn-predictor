# Telco Customer Churn Prediction  
An end-to-end machine learning project that predicts telecom customer churn using **XGBoost**, with **MLflow experiment tracking**, **FastAPI backend serving**, **Gradio frontend UI**, **Docker**, **GitHub Actions CI/CD**, and **AWS ECS Fargate deployment**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Objectives](#objectives)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Project Workflow](#project-workflow)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Modeling Approach](#modeling-approach)
- [Threshold Selection Strategy](#threshold-selection-strategy)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Backend API](#backend-api)
- [Frontend UI](#frontend-ui)
- [Dockerization](#dockerization)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [AWS Deployment](#aws-deployment)
- [Local Setup](#local-setup)
- [Run Locally](#run-locally)
- [Sample Request and Response](#sample-request-and-response)
- [Cost-Aware Deployment Decisions](#cost-aware-deployment-decisions)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)
- [Key Learnings](#key-learnings)
- [Resume / Portfolio Summary](#resume--portfolio-summary)

---

## Project Overview

Customer churn is one of the most important business challenges in subscription-based industries like telecom. Losing customers directly affects recurring revenue, customer lifetime value, and acquisition efficiency.

This project predicts whether a telecom customer is likely to churn based on:
- demographic details
- account information
- service usage
- contract type
- support features
- payment behavior
- billing-related signals

The project was built as a **production-style ML system**, not just a notebook experiment. It includes:

- exploratory data analysis
- modularized training pipeline
- feature engineering
- model tuning
- MLflow experiment tracking
- FastAPI inference API
- Gradio frontend UI
- Dockerized deployment
- GitHub Actions CI/CD
- AWS ECS Fargate deployment

---

## Business Problem

Telecom companies lose revenue when customers discontinue their subscriptions. The earlier churn risk is identified, the more time the business has to take proactive retention actions such as:
- discounts
- support outreach
- contract upgrades
- service improvement offers

The goal of this project is to build a system that identifies likely churners so retention teams can act before customers leave.

---

## Objectives

The key objectives of this project were:

1. Build a reliable churn prediction model
2. Create a modular training and inference pipeline
3. Track experiments and compare model runs
4. Serve the model through a production-friendly API
5. Expose predictions through a user-facing UI
6. Containerize both backend and frontend
7. Add CI/CD validation
8. Deploy the full solution on AWS

---

## Tech Stack

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Optuna

### Experiment Tracking
- MLflow

### Backend
- FastAPI
- Uvicorn

### Frontend
- Gradio

### DevOps / MLOps
- Docker
- GitHub Actions

### Cloud
- Amazon ECR
- Amazon ECS Fargate
- Amazon CloudWatch Logs
- Amazon VPC
- Security Groups

---

## System Architecture

The final deployed solution uses separate frontend and backend services.

```text
User Browser
    ↓
Gradio UI (ECS Fargate, port 7860)
    ↓
FastAPI Backend (ECS Fargate, port 8000)
    ↓
XGBoost Model + Preprocessing Artifacts
```
## High-Level Deployment Flow

Local Development
    ↓
Docker Build
    ↓
Amazon ECR
    ↓
Amazon ECS Fargate
    ↓
Public UI + Backend API
