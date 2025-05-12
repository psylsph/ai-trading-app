#!/usr/bin/env python3
"""
Automated Cryptocurrency Trading Application

This app:
1. Checks current cryptocurrency balance in an Alpaca account
2. Fetches Bitcoin price and performance data
3. Uses Google Gemini to analyze data and make trading decisions
4. Executes the determined trade on the Alpaca account
"""

import os
import json
import time
import json_repair
import requests
import pandas as pd
from datetime import datetime
import google.generativeai as genai
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from tradingview_scraper.symbols.technicals import Indicators
from tradingview_scraper.symbols.news import NewsScraper
from json_repair import repair_json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_account_info():
    """Get the current account information from Alpaca"""
    try:
        account = trading_client.get_account()
        print(f"Account ID: {account.id}")
        print(f"Cash: ${float(account.cash)}")
        print(f"Portfolio Value: ${float(account.portfolio_value)}")
        
        # Get crypto positions specifically
        positions = trading_client.get_all_positions()
        crypto_positions = [p for p in positions if p.symbol.endswith('USD')]
        
        crypto_data = {
            "account_cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "crypto_positions": []
        }
        
        print("\nCryptocurrency Positions:")
        for position in crypto_positions:
            pos_data = {
                "symbol": position.symbol,
                "qty": float(position.qty),
                "market_value": float(position.market_value),
                "cost_basis": float(position.cost_basis),
                "unrealized_pl": float(position.unrealized_pl),
                "current_price": float(position.current_price)
            }
            crypto_data["crypto_positions"].append(pos_data)
            
            print(f"  {position.symbol}: {position.qty} shares worth ${float(position.market_value)}")
            print(f"    Cost Basis: ${float(position.cost_basis)}")
            print(f"    Unrealized P/L: ${round(float(position.unrealized_pl),2)}")
            print(f"    Current Price: ${float(position.current_price)}")
        if len(crypto_data["crypto_positions"]) == 0:
            print("  No cryptocurrency positions found.")
        
        return crypto_data
    except Exception as e:
        print(f"Error getting account info: {e}")
        return None
    
def get_technical_indicators(bitcoin_data: dict):

    technical_indicators = {}
    try:
        # Scrape indicators for the BTCUSD symbol
        indicators_scraper = Indicators(export_result=False, export_type='json')
        technical_indicators = indicators_scraper.scrape(
            symbol="BTCUSD",
            timeframe="1h",
            indicators=["RSI", "Stoch.K", "CCI20", "AO", "Mom", "MACD.macd", "MACD.signal"]
        )
            
        if technical_indicators['status'] == 'success':
            bitcoin_data["technical_indicators"]["Oscillators"] = technical_indicators['data']
        else:
            bitcoin_data["technical_indicators"]["Oscillators"] = {
                "error": f"Failed to retrieve technical data."
            }

        # Scrape indicators for the BTCUSD symbol
        indicators_scraper = Indicators(export_result=False, export_type='json')
        technical_indicators = indicators_scraper.scrape(
            symbol="BTCUSD",
            timeframe="4h",
            indicators=["SMA10", "EMA10", "HullMA9", "SMA20", "EMA20", "SMA30", "EMA30"]
        )
            
        if technical_indicators['status'] == 'success':
            bitcoin_data["technical_indicators"]["Moving Averages"] = technical_indicators['data']
        else:
            bitcoin_data["technical_indicators"]["Moving Averages"] = {
                "error": f"Failed to retrieve technical data."
            }
    except Exception as e:
        print(f"Error getting Bitcoin technical indicators: {e}")
        # Fallback to placeholder if all attempts fail
        bitcoin_data["technical_indicators"] = []
    
    return bitcoin_data["technical_indicators"]

def summarize_news(news_as_list:list) :

    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    
    # Structure the prompt with all relevant information
    prompt = f"""
    As a cryptocurrency trading advisor, please summarize the following news in 200-300 words:
    {str(news_as_list)}"""
    
    print("Sending news to Google Gemini to summarize...")
    response = model.generate_content(prompt)
    return clean_string(response.text)

# Function to extract Bitcoin news from TradingView
def get_bitcoin_news():
    """
    Get Bitcoin news from TradingView
    """
    bitcoin_data = {}
    try:
        news_scraper = NewsScraper(export_result=False, export_type='json')
        news_headlines = news_scraper.scrape_headlines(
            symbol='BTCUSD',      # Uncomment and specify if needed
            exchange='BINANCE', # Uncomment and specify if needed
            sort='latest',
        )
        bitcoin_data["recent_news"] = []
        for article in news_headlines[:10]:
            news_content = news_scraper.scrape_news_content(story_path=article['storyPath'])
            story = {}
            story['title'] = news_content['title']
            story['date'] = news_content['published_datetime']
            story['content'] = ""
            for line in news_content['body']:
                try:
                    story['content'] = story['content'] + line['content']
                except KeyError as e:
                    pass
            bitcoin_data["recent_news"].append(story)
        
    except Exception as e:
        print(f"Error getting Bitcoin news: {e}")
        # Fallback to placeholder news if all attempts fail
        bitcoin_data["recent_news"] = []
        
    return bitcoin_data["recent_news"]

def clean_string(input_string):
    output_string  = input_string.replace('\u2013', '-').replace('\u2014', '-').replace('\u2018', "'").replace('\u2019', "'")
    output_string = output_string.replace('\u201c', "'").replace('\u201d', "'").replace('\u00a0', " ").replace('\u202f', " ")
    output_string = output_string.replace('"', "'").replace('\u2011', '-').replace('\u003c', '<').replace('\u003e', '>')
    return output_string

def get_bitcoin_data():
    
    bitcoin_data = {
        "price": None,
        "technical_indicators": {},
        "recent_news": []
    }
    
    # Get news
    try:
        bitcoin_data["recent_news"] = summarize_news(get_bitcoin_news())
        print(f"Recent News Summary:\n{bitcoin_data["recent_news"]}")
    except Exception as e:
        print(f"Error fetching news: {e}")

    # Get technical indicators
    try:
        bitcoin_data["technical_indicators"] = get_technical_indicators(bitcoin_data)

    except Exception as e:
        print(f"Error fetching technical indicators: {e}")
    
    # If scraping fails, use CoinGecko API as fallback for basic price data
    try:
        response = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin")
        data = response.json()
        bitcoin_data["price"] = data["market_data"]["current_price"]["usd"]
        bitcoin_data["24h_change"] = data["market_data"]["price_change_percentage_24h"]
        bitcoin_data["7d_change"] = data["market_data"]["price_change_percentage_7d"]
        bitcoin_data["market_cap"] = data["market_data"]["market_cap"]["usd"]
        bitcoin_data["24h_volume"] = data["market_data"]["total_volume"]["usd"]
    except Exception as e:
        print(f"Error fetching from fallback API: {e}")
    
    return bitcoin_data

def format_technical_indicators(indicators_dict):
    """Formats the technical indicators dictionary into a human-readable string."""
    if not indicators_dict or not isinstance(indicators_dict, dict):
        return "No technical indicators available or data is in an unexpected format."

    formatted_string = []
    for category, indicators in indicators_dict.items():
        if isinstance(indicators, dict):
            if "error" in indicators:
                formatted_string.append(f"\t  {category}: {indicators['error']}")
            else:
                formatted_string.append(f"\t  {category}:")
                for name, value in indicators.items():
                    formatted_string.append(f"\t    {name}: {round(float(value), 2)}")
        else:
            formatted_string.append(f"  {category}: Data for this category is not in the expected format.")
    
    if not formatted_string:
        return "No technical indicators processed."
        
    return "\n" + "\n".join(formatted_string)

def ask_gemini_for_decision(account_data, bitcoin_data):
    """Send data to Google Gemini and get a trading decision"""

    model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
    
    # Structure the prompt with all relevant information
    prompt = f"""
    As a cryptocurrency trading advisor, please analyze the following data and recommend whether to buy, sell or hold Bitcoin,
    along with a suggested quantity. Please consider transaction fees in your decision.
    
    Current date and time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    ACCOUNT INFORMATION:
    - Available Cash: ${account_data.get('account_cash', 'Unknown')}
    - Total Portfolio Value: ${account_data.get('portfolio_value', 'Unknown')}
    - Maker Fee: 0.15%, Taker Fee: 0.25%
    
    CURRENT BITCOIN HOLDINGS:
    {json.dumps(account_data.get('crypto_positions', []), indent=2)}
    
    BITCOIN MARKET DATA:
    - Current Price: ${bitcoin_data.get('price', 'Unknown')}
    - 24h Change: {bitcoin_data.get('24h_change', 'Unknown')}%
    - 7d Change: {bitcoin_data.get('7d_change', 'Unknown')}%
    - Technical Indicators: {format_technical_indicators(bitcoin_data.get('technical_indicators', {}))}
    
    RECENT NEWS:
    {json.dumps(bitcoin_data.get('recent_news', []), indent=2)}
    
    Based on this information, please provide:
    1. A clear BUY, SELL or HOLD recommendation for Bitcoin
    2. The quantity to buy or sell (in USD or BTC)
    3. Your reasoning for this decision
    4. A confidence level in your recommendation (low, medium, high)
    
    Format your response as JSON with the following structure:
    {{
        "decision": "BUY, SELL or HOLD",
        "quantity": 123.45,
        "quantity_unit": "USD or BTC",
        "reasoning": "Your detailed reasoning here",
        "confidence": "low/medium/high"
    }}
    """
    
    print("Sending data to Google Gemini for analysis...")
    open ('gemini-input.txt', 'w').write(prompt)
    response = model.generate_content(prompt)
    
    try:
        # Parse the response - assuming Gemini returns properly formatted JSON
        response_text = response.text
        
        # Extract JSON from the response (if it's wrapped in other text or code blocks)
        import re
        json_match = re.search(r'({[\s\S]*})', response_text)
        if json_match:
            json_str = json_match.group(1)
            decision_data = json_repair.loads(json_str)
        else:
            decision_data = json_repair.loads(response_text)
            
        return decision_data
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {response.text}")
        return {
            "decision": "HOLD",  # Default to HOLD if there's an error
            "quantity": 0,
            "quantity_unit": "USD",
            "reasoning": f"Error parsing Gemini response: {e}",
            "confidence": "low"
        }

def execute_trade(decision):
    """Execute the trade on Alpaca based on the Gemini decision"""
    try:
        # Validate decision
        if decision.get("decision").upper() not in ["BUY", "SELL"]:
            print(f"Invalid decision: {decision.get('decision')}. Must be BUY or SELL.")
            return False
            
        # Convert decision to Alpaca parameters
        side = OrderSide.BUY if decision.get("decision").upper() == "BUY" else OrderSide.SELL
        
        # Determine the quantity based on whether it's in USD or BTC
        quantity = float(decision.get("quantity", 0))
        quantity_unit = decision.get("quantity_unit", "").upper()
        
        if quantity <= 0:
            print("Cannot execute trade with zero or negative quantity")
            return False
            
        # For BTC quantity, we pass it directly
        # For USD quantity, we need to use notional value
        if quantity_unit == "USD":
            # Create a market order with notional value
            order_data = MarketOrderRequest(
                symbol="BTCUSD",
                side=side,
                notional=quantity,  # Amount in USD
                time_in_force=TimeInForce.GTC
            )
            print(f"Placing order to {side.name} Bitcoin worth ${quantity}")
        else:  # Assume BTC
            # Create a market order with qty
            order_data = MarketOrderRequest(
                symbol="BTCUSD",
                side=side,
                qty=quantity,  # Amount in BTC
                time_in_force=TimeInForce.GTC
            )
            print(f"Placing order to {side.name} {quantity} BTC")
        
        # Submit the order
        order = trading_client.submit_order(order_data)
        
        print(f"Order placed successfully!")
        print(f"Order ID: {order.id}")
        print(f"Symbol: {order.symbol}")
        print(f"Side: {order.side}")
        print(f"Qty: {order.qty if hasattr(order, 'qty') and order.qty else 'N/A'}")
        print(f"Notional value: ${order.notional if hasattr(order, 'notional') and order.notional else 'N/A'}")
        print(f"Type: {order.type}")
        print(f"Status: {order.status}")
        
        return True
    except Exception as e:
        print(f"Error executing trade: {clean_string(str(e))}")
        return False

def check_position():
    """check_position function to run the entire trading process"""
    print("=" * 50)
    print("AUTOMATED CRYPTO TRADING APPLICATION")
    print("=" * 50)
    
    # Check if environment variables are properly loaded
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY or not GOOGLE_API_KEY:
        print("ERROR: Required API keys not found in environment variables.")
        print("Please make sure your .env file is set up correctly with the following variables:")
        print("  - ALPACA_API_KEY")
        print("  - ALPACA_SECRET_KEY")
        print("  - GOOGLE_GEMINI_API_KEY")
        return
    
    # Step 1: Get account info
    print("\n1. Checking Alpaca account information...")
    account_data = get_account_info()
    if not account_data:
        print("Failed to retrieve account data. Exiting.")
        return
    
    # Step 2: Get Bitcoin data
    print("\n2. Fetching Bitcoin price, performance data and news...")
    bitcoin_data = get_bitcoin_data()
    if not bitcoin_data or not bitcoin_data.get("price"):
        print("Failed to retrieve adequate Bitcoin data. Exiting.")
        return
    print(f"Bitcoin current price: ${bitcoin_data.get('price')}")
    
    # Step 3: Get trading decision from Gemini
    print("\n3. Analyzing data with Google Gemini...")
    decision = ask_gemini_for_decision(account_data, bitcoin_data)
    print("\nGemini's Trading Decision:")
    print(f"Decision: {decision.get('decision')}")
    print(f"Quantity: {decision.get('quantity')} {decision.get('quantity_unit')}")
    print(f"Confidence: {decision.get('confidence')}")
    print(f"Reasoning: {decision.get('reasoning')}")
    
    # Step 4: Execute the trade
    print("\n4. Executing the recommended trade...")
    if decision.get("decision").upper() == "HOLD":
        print("Decision is to HOLD. No trade will be executed.")
    else:
        success = execute_trade(decision)
        if success:
            print("\nTrade executed successfully!")
        else:
            print("\nFailed to execute trade.")
    
    print("\n" + "=" * 50)
    print("Trading process completed.")
    print("=" * 50)

if __name__ == "__main__":
    # Set up API clients
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)  # Use paper=False for live trading
    genai.configure(api_key=GOOGLE_API_KEY)
    
    while True:
        try:
            check_position()
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for 5 minutes before the next run...")
            time.sleep(300)  # 5 minutes * 60 seconds
        except KeyboardInterrupt:
            print("\nApplication terminated by user.")
            break
        except Exception as e:
            print(f"An error occurred during execution: {e}")
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Retrying in 5 minutes...")
            time.sleep(300)
