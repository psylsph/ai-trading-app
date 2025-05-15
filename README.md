# Automated Crypto Trading App - Installation Guide

This guide will help you set up and run the Automated Crypto Trading Application.

## Prerequisites

- Python 3.8 or higher
- An Alpaca trading account with API access
- A Google Gemini API key

## Installation Steps

### 1. Clone or download the repository

Create a directory for your project and save the Python script there.

### 2. Install required packages

Run the following command to install all required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set up your environment variables

1. Create a file named `.env` in the same directory as the script
2. Add your API keys to the file using the provided template:

```
# Alpaca API credentials
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# GROQ API credentials
GROQ_API_KEY==your_qroq_api_key_here
```

3. Replace the placeholder values with your actual API keys

### 4. Run the application

Execute the script with Python:

```bash
python ai_trading_app.py
```

## Getting API Keys

### Alpaca API Keys

1. Create an account at [Alpaca](https://alpaca.markets/)
2. Navigate to your Paper Trading account (for testing) or Live Trading account
3. Go to "API Keys" section and generate your keys
4. Copy the API Key and Secret Key to your `.env` file

### GROQ API Key

1. TBC

## Important Notes

- The application uses Alpaca's paper trading by default (`paper=True`). Set to `False` in the script to use live trading.
- Be cautious when switching to live trading as real assets will be involved.
- The web scraping functionality may need periodic updates as websites change their structure.

## Troubleshooting

If you encounter errors related to missing environment variables:
- Make sure your `.env` file is in the same directory as the script
- Check that you've correctly named the environment variables (case-sensitive)
- Verify that python-dotenv is properly installed

For any other issues, check the error messages which usually contain helpful information about what went wrong.