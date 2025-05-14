#!/usr/bin/env python3
"""
Automated Cryptocurrency Trading Application

This app:
1. Checks current cryptocurrency balance in an Alpaca account
2. Fetches coin price and performance data
3. Uses local llm to analyze data and make trading decisions
4. Executes the determined trade on the Alpaca account
"""

import os
import json
import time
import requests
from datetime import datetime
from openai import OpenAI
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from tradingview_scraper.symbols.technicals import Indicators
from tradingview_scraper.symbols.news import NewsScraper
from dotenv import load_dotenv
import sys # Added for sys.exit()
import re # Added for regex in llm response parsing
from json_repair import loads as repair_json_loads # Use specific import
from alpaca.common.exceptions import APIError # Specific Alpaca exception
from requests.exceptions import RequestException # Specific requests exception

# --- Constants ---
SYMBOL = "XRPUSD" # Define symbol constant
PRICING_SYMBOL = "ripple"
SLEEP_INTERVAL_SECONDS = 600 # Define sleep interval constant
DEBUG = True # Control debug file writing
llm_client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- Global Clients/Scrapers (Instantiate once) ---
# Note: API key check happens before this in __main__
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True) # Use paper=False for live trading
indicators_scraper = Indicators(export_result=False, export_type='json')
news_scraper = NewsScraper(export_result=False, export_type='json')


def get_account_info(client):
    """Get the current account information from Alpaca"""
    try:
        account = client.get_account()
        print(f"Account ID: {account.id}")
        print(f"Cash: ${float(account.cash)}")
        print(f"Portfolio Value: ${float(account.portfolio_value)}")

        # Get crypto positions specifically
        positions = client.get_all_positions()
        # Filter specifically for the target symbol if needed, or all crypto
        # crypto_positions = [p for p in positions if p.symbol == SYMBOL]
        crypto_positions = [p for p in positions if p.asset_class == 'crypto'] # More general crypto check

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
    except (APIError, RequestException) as e: # Catch specific exceptions
        print(f"Error getting account info: {e}")
        return None
    except Exception as e: # Catch other unexpected errors
        print(f"Unexpected error getting account info: {e}")
        return None

# Removed duplicated code block that was inserted here

def get_technical_indicators(scraper, symbol: str):
    """Fetches technical indicators using the provided scraper instance."""
    indicators_data = {}
    try:
        # Scrape Oscillators
        oscillators = scraper.scrape(
            symbol=symbol,
            timeframe="1h",
            indicators=["RSI", "Stoch.K", "CCI20", "AO", "Mom", "MACD.macd", "MACD.signal"]
        )
        if oscillators.get('status') == 'success':
            indicators_data["Oscillators"] = oscillators.get('data', {})
        else:
            print(f"Warning: Failed to retrieve Oscillators for {symbol}. Status: {oscillators.get('status')}")
            indicators_data["Oscillators"] = {"error": "Failed to retrieve Oscillator data."}

        # Scrape Moving Averages
        moving_averages = scraper.scrape(
            symbol=symbol,
            timeframe="4h",
            indicators=["SMA10", "EMA10", "HullMA9", "SMA20", "EMA20", "SMA30", "EMA30"]
        )
        if moving_averages.get('status') == 'success':
            indicators_data["Moving Averages"] = moving_averages.get('data', {})
        else:
            print(f"Warning: Failed to retrieve Moving Averages for {symbol}. Status: {moving_averages.get('status')}")
            indicators_data["Moving Averages"] = {"error": "Failed to retrieve Moving Average data."}

    except Exception as e:
        print(f"Error getting technical indicators for {symbol}: {e}")
        # Fallback to error placeholder if scraping fails
        indicators_data = {"error": f"Failed to retrieve technical indicators due to exception: {e}"}

    return indicators_data


def get_llm_response(system_prompt: str, user_prompt: str, llm_model: str, temperature: float):

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        top_p=0.95,
        max_completion_tokens=8192,
        timeout=6000,
        stream=True
    )
    
    # create variables to collect the stream of chunks
    collected_messages = []
    # iterate through the stream of events
    for chunk in response:
        if chunk == None:
            continue
        if chunk.choices == None:
            continue
        if chunk.choices[0] == None:
            continue
        if chunk.choices[0].delta == None:
            continue
        if chunk.choices[0].delta.content == None:
            continue
        chunk_message = chunk.choices[0].delta.content  # extract the message
        collected_messages.append(chunk_message)  # save the message
        if DEBUG:
            print(chunk_message, end="")
    # clean None in collected_messages
    collected_messages = [m for m in collected_messages if m is not None]
    full_reply_content = ''.join(collected_messages)
    
    cleaned_response = full_reply_content.replace("*", "").replace("#", "").replace("\n\n", "\n")
    cleaned_response = cleaned_response.replace("&", "").replace("\\", "")
    cleaned_response = cleaned_response.replace("\\hline", "").replace("\\boxed", "").replace("\\textbf", "").replace("\\begin", "")
    cleaned_response = cleaned_response.replace("</s>", "").replace("[TOOL_CALLS]", "")
    cleaned_response= cleaned_response.replace("[APPROVED]", "").replace("[REVISE]", "")
    cleaned_response= cleaned_response.replace("<strong>", "").replace("</strong>", "")
    
    if "</think>" in cleaned_response:
            cleaned_response = cleaned_response.split("</think>")[1]

    return cleaned_response

def summarize_news(news_as_list: list, model):
    """Summarizes a list of news articles using the provided llm model."""
    if not news_as_list:
        return "No news articles provided for summarization."

    # Structure the prompt with all relevant information
    prompt = f"""
    As a cryptocurrency trading advisor, please summarize the following news in 200-300 words:
    {str(news_as_list)}"""
    
    print(f"Sending news to {summary_model} for summarization...")
    try:
        system_prompt = "You are crypto currency expert."
        response = get_llm_response(system_prompt, prompt, summary_model, temperature=0.2)
        return clean_string(response)
    except Exception as e:
        print(f"Error generating news summary: {e}")
        return "Error generating news summary."


def get_coin_news(scraper, symbol: str, exchange: str = 'BINANCE'):
    """Gets coin news using the provided scraper instance."""
    news_list = []
    try:
        news_headlines = scraper.scrape_headlines(
            symbol=symbol,
            exchange=exchange,
            sort='latest',
        )
        # Limit to top 10 articles
        for article in news_headlines[:10]:
            try:
                news_content = scraper.scrape_news_content(story_path=article['storyPath'])
                story = {
                    'title': news_content.get('title', 'N/A'),
                    'date': news_content.get('published_datetime', 'N/A'),
                    'content': ""
                }
                # Safely concatenate content lines
                for line in news_content.get('body', []):
                    try:
                        # Check if 'content' key exists and is not None
                        if 'content' in line and line['content'] is not None:
                            story['content'] += str(line['content']) # Ensure content is string
                    except KeyError:
                         # This handles if 'content' key is missing entirely
                         print(f"Warning: 'content' key missing in news body line for article: {story['title']}")
                    except Exception as inner_e: # Catch other potential errors during concatenation
                         print(f"Warning: Error processing content line for article {story['title']}: {inner_e}")

                news_list.append(story)
            except Exception as article_e:
                print(f"Error processing article {article.get('storyPath', 'N/A')}: {article_e}")

    except Exception as e:
        print(f"Error getting coin news headlines for {symbol}: {e}")
        # Return empty list on error
        return []

    return news_list


def clean_string(input_string):
    """Cleans special characters from a string."""
    if not isinstance(input_string, str):
        return str(input_string) # Return string representation if not a string
    output_string = input_string.replace('\u2013', '-').replace('\u2014', '-').replace('\u2018', "'").replace('\u2019', "'")
    output_string = output_string.replace('\u201c', "'").replace('\u201d', "'").replace('\u00a0', " ").replace('\u202f', " ")
    output_string = output_string.replace('"', "'").replace('\u2011', '-').replace('\u003c', '<').replace('\u003e', '>')
    return output_string

def get_coin_data(news_scraper_instance, indicators_scraper_instance, summary_model, symbol: str):
    """Aggregates coin data: news, technical, and price."""
    coin_data = {
        "price": None,
        "technical_indicators": {},
        "recent_news_summary": "Not available" # Changed key name
    }

    # Get news and summarize
    try:
        raw_news = get_coin_news(news_scraper_instance, symbol)
        if raw_news:
             # Pass the separate summary model here
             coin_data["recent_news_summary"] = summarize_news(raw_news, summary_model)
             print(f"Recent News Summary:\n{coin_data['recent_news_summary']}")
        else:
             print("No raw news fetched to summarize.")
             coin_data["recent_news_summary"] = "No recent news found."
    except Exception as e:
        print(f"Error fetching or summarizing news: {e}")
        coin_data["recent_news_summary"] = f"Error fetching/summarizing news: {e}"

    # Get technical indicators
    try:
        coin_data["technical_indicators"] = get_technical_indicators(indicators_scraper_instance, symbol)
    except Exception as e:
        print(f"Error fetching technical indicators in get_coin_data: {e}")
        coin_data["technical_indicators"] = {"error": f"Failed to get technical indicators: {e}"}

    # Use CoinGecko API for price data (more reliable than scraping for just price)
    try:
        # Assuming 'PRICING_SYMBOL' is the correct ID for CoinGecko
        response = requests.get(f"https://api.coingecko.com/api/v3/coins/{PRICING_SYMBOL}", timeout=10) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        market_data = data.get("market_data", {})
        current_price = market_data.get("current_price", {}).get("usd")

        if current_price is not None:
             coin_data["price"] = float(current_price) # Ensure float
             coin_data["24h_change"] = market_data.get("price_change_percentage_24h")
             coin_data["7d_change"] = market_data.get("price_change_percentage_7d")
             coin_data["market_cap"] = market_data.get("market_cap", {}).get("usd")
             coin_data["24h_volume"] = market_data.get("total_volume", {}).get("usd")
        else:
             print("Error: Could not find 'usd' price in CoinGecko response.")
             coin_data["price"] = None # Explicitly set to None if not found

    except RequestException as e:
        print(f"Error fetching from CoinGecko API: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding CoinGecko JSON response: {e}")
    except Exception as e:
        print(f"Unexpected error fetching price data: {e}")
        # Keep price as None if any error occurs

    return coin_data


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
                # Safely format values, handling potential non-numeric data
                for name, value in indicators.items():
                    try:
                        # Attempt to convert to float for rounding, handle errors
                        formatted_value = round(float(value), 2)
                    except (ValueError, TypeError):
                        formatted_value = value # Keep original value if conversion fails
                    formatted_string.append(f"\t    {name}: {formatted_value}")
        elif isinstance(indicators, str): # Handle case where value is just an error string
             formatted_string.append(f"\t  {category}: {indicators}")
        else:
            # Log unexpected format instead of just printing to output string
            print(f"Warning: Unexpected format for technical indicator category '{category}': {type(indicators)}")
            formatted_string.append(f"  {category}: Data for this category is in an unexpected format.")

    if not formatted_string:
        return "No technical indicators processed."
        
    return "\n" + "\n".join(formatted_string)

def ask_llm_for_decision(account_data, coin_data, gen_model, sum_model):
    """Send data to llm and get a trading decision"""
    # Model is now passed in

    # Structure the prompt with all relevant information
    # Use .get with defaults for safer access
    account_cash = account_data.get('account_cash', 'Unknown')
    portfolio_value = account_data.get('portfolio_value', 'Unknown')
    crypto_positions = account_data.get('crypto_positions', [])
    current_price = coin_data.get('price', 'Unknown')
    change_24h = coin_data.get('24h_change', 'Unknown')
    change_7d = coin_data.get('7d_change', 'Unknown')
    tech_indicators = coin_data.get('technical_indicators', {})
    news_summary = coin_data.get('recent_news_summary', 'Not available') # Use updated key
    clean_symbol = SYMBOL.replace("USD", "")

    prompt = f"""
    As a cryptocurrency trading advisor, please analyze the following data and recommend whether to buy, sell or hold coin,
    along with a suggested quantity. Please consider transaction fees in your decision.
    
    Current date and time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    ACCOUNT INFORMATION:
    - Available Cash: ${account_cash}
    - Total Portfolio Value: ${portfolio_value}
    - Maker Fee: 0.15%, Taker Fee: 0.25%

    CURRENT {clean_symbol} HOLDINGS:
    {json.dumps(crypto_positions, indent=2)}

    {clean_symbol} MARKET DATA:
    - Current Price: ${current_price}
    - 24h Change: {f'{change_24h}%' if change_24h is not None else 'Unknown'}
    - 7d Change: {f'{change_7d}%' if change_7d is not None else 'Unknown'}
    - Technical Indicators: {format_technical_indicators(tech_indicators)}

    RECENT NEWS SUMMARY:
    {news_summary}

    Based on this information, please provide:
    1. A clear BUY, SELL or HOLD recommendation for {clean_symbol}
    2. The quantity to buy or sell (in USD or {clean_symbol})
    3. Your reasoning for this decision
    4. A confidence level in your recommendation (low, medium, high)
    5. Always HOLD if the proposed trade would worth less than $5
    
    Format your response as JSON with the following structure:
    {{
        "decision": "BUY, SELL or HOLD",
        "quantity": 123.45,
        "quantity_unit": "USD or SYMBOL",
        "reasoning": "Your detailed reasoning here",
        "confidence": "LOW/MEDIUM/HIGH"
    }}
    """

    prompt = prompt.replace("SYMBOL", clean_symbol)
    
    print(f"Sending data to {decision_model} for analysis...")
    if DEBUG:
        try:
            # Use with statement and specify encoding
            with open('llm-input.txt', 'w', encoding='utf-8') as f:
                f.write(prompt)
        except IOError as e:
            print(f"Warning: Could not write debug prompt file: {e}")

    try:
        system_prompt = "You are crypto currency expert."
        try:
            response_text = get_llm_response(system_prompt, prompt, decision_model, temperature=0.2)
        except Exception as gen_ex:
            print(f"Warning: could not use {gen_model}: {gen_ex}")
        
        # Attempt to extract JSON using regex
        # Improved regex for markdown code blocks and plain objects
        json_match = re.search(r'```json\s*({[\s\S]*?})\s*```|({[\s\S]*})', response_text)
        json_str = None
        if json_match:
            # Prioritize the markdown block match, then the plain object match
            json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)

        if json_str:
            try:
                # Use the imported repair function
                decision_data = repair_json_loads(json_str)
                return decision_data
            except json.JSONDecodeError as json_e:
                print(f"Error parsing extracted JSON from llm response: {json_e}")
                print(f"Extracted JSON string: {json_str}")
                # Fall through to try parsing the whole text if regex extraction failed to parse
            except Exception as repair_e: # Catch potential errors from repair_json_loads itself
                 print(f"Error during JSON repair: {repair_e}")
                 print(f"Extracted JSON string: {json_str}")
                 # Fall through

        # If regex fails or extracted JSON fails to parse, try repairing the whole response
        print("Warning: Could not find or parse JSON block via regex, attempting to repair entire response.")
        try:
            decision_data = repair_json_loads(response_text)
            return decision_data
        except Exception as full_repair_e:
            print(f"Error parsing or repairing full llm response: {full_repair_e}")
            print(f"Raw response: {response_text}")
            # Fall through to default HOLD response

    except Exception as gen_e: # Catch errors during model.generate_content
        print(f"Error communicating with llm: {gen_e}")

    # Default response if anything fails
        return {
            "decision": "HOLD",  # Default to HOLD if there's an error
            "quantity": 0,
            "quantity_unit": "USD",
            "reasoning": f"Error parsing llm response: {gen_e}",
            "confidence": "LOW"
        }

def execute_trade(client, decision, symbol: str):
    """Execute the trade on Alpaca based on the llm decision"""
    decision_action = decision.get("decision")
    confidence = decision.get("confidence")
    quantity = decision.get("quantity")
    # Default to empty string, then upper for quantity_unit
    quantity_unit = decision.get("quantity_unit", "").upper()

    # --- Robust Validation ---
    if not decision_action:
        print("Error: Missing 'decision' in llm response.")
        return False
    
    if not confidence:
        print("Error: Missing 'confidence' in llm response.")
        return False


    decision_action_upper = decision_action.upper()
    confidence_upper = confidence.upper()

    # Allow HOLD as a valid non-trading action
    if decision_action_upper == "HOLD":
        print("Decision is HOLD. No trade will be executed.")
        return True # Indicate successful handling of HOLD
    
    if confidence_upper == "LOW":
        print("Confidence is LOW. No trade will be executed.")
        return True # Indicate successful handling of LOW

    if decision_action_upper not in ["BUY", "SELL"]:
        print(f"Error: Invalid decision action '{decision_action}'. Must be BUY, SELL, or HOLD.")
        return False
    
    if confidence_upper not in ["MEDIUM", "HIGH"]:
        print(f"Error: Invalid confidence action '{confidence_upper}'. Must be LOW, MEDIUM, HIGH.")
        return False

    # Validate quantity only if BUY or SELL
    try:
        quantity_float = float(quantity)
        if quantity_float <= 0:
            print(f"Error: Invalid quantity ({quantity}). Must be a positive number for BUY/SELL.")
            return False
    except (ValueError, TypeError):
        print(f"Error: Invalid quantity format ({quantity}). Must be a number for BUY/SELL.")
        return False

    # Validate quantity unit only if BUY or SELL
    # Adjust 'BTC' based on the actual base currency of the symbol if needed
    base_currency = symbol[:3] # Simple assumption, might need refinement
    if quantity_unit not in ["USD", base_currency]:
         print(f"Error: Invalid quantity_unit '{decision.get('quantity_unit', '')}'. Must be 'USD' or '{base_currency}'.")
         return False
    # --- End Validation ---

    side = OrderSide.BUY if decision_action_upper == "BUY" else OrderSide.SELL

    try:
        order_data = None
        # For base currency quantity (e.g., BTC), use 'qty'
        # For USD quantity, use 'notional' value
        if quantity_unit == "USD":
            order_data = MarketOrderRequest(
                symbol=symbol,
                side=side,
                notional=quantity_float,  # Amount in USD
                time_in_force=TimeInForce.GTC # Good Till Cancelled
            )
            print(f"Placing market order to {side.name} {symbol} worth ${quantity_float}")
        else: # Assume base currency (e.g., BTC)
            order_data = MarketOrderRequest(
                symbol=symbol,
                side=side,
                qty=quantity_float,  # Amount in base currency (e.g., BTC)
                time_in_force=TimeInForce.GTC
            )
            print(f"Placing market order to {side.name} {quantity_float} {symbol}") # Log includes symbol

        # Submit the order
        order = client.submit_order(order_data=order_data)

        print("Order submitted successfully!")
        print(f"Order ID: {order.id}")
        print(f"Symbol: {order.symbol}")
        print(f"Side: {order.side}")
        print(f"Qty: {order.qty if hasattr(order, 'qty') and order.qty else 'N/A'}")
        print(f"Notional value: ${order.notional if hasattr(order, 'notional') and order.notional else 'N/A'}")
        print(f"Type: {order.type}")
        print(f"Status: {order.status}")
        
        return True
    except APIError as e:
        # Handle specific Alpaca errors (e.g., insufficient funds, invalid order)
        print(f"Alpaca API Error executing trade: {clean_string(str(e))}")
        # You might want to inspect e.response or e.status_code for more details
        return False
    except Exception as e:
        print(f"Unexpected Error executing trade: {clean_string(str(e))}")
        return False


def run_trading_cycle(alpaca_client, decision_model, summary_model, news_scraper_instance, indicators_scraper_instance, symbol: str):
    """Runs one cycle of the trading logic."""
    print("=" * 50)
    print(f"Starting Trading Cycle for {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Step 1: Get account info
    print("\n1. Checking Alpaca account information...")
    account_data = get_account_info(alpaca_client)
    if not account_data:
        print("Failed to retrieve account data. Skipping cycle.")
        return # Skip this cycle if account data fails

    # Step 2: Get coin data (Price, News Summary, Technical Indicators)
    print(f"\n2. Fetching {symbol} price, performance data and news...")
    # Pass the instantiated summary model here
    coin_data = get_coin_data(
        news_scraper_instance,
        indicators_scraper_instance,
        summary_model, # Pass summary model
        symbol
    )

    # Check if essential price data was retrieved
    if coin_data.get("price") is None:
        print(f"Failed to retrieve {symbol} price data. Skipping cycle.")
        return # Skip cycle if price is missing

    print(f"{symbol} current price: ${coin_data.get('price')}")

    # Step 3: Get trading decision from llm
    print("\n3. Analyzing data with Google llm...")
    # Pass the instantiated decision model here
    decision = ask_llm_for_decision(account_data, coin_data, decision_model, summary_model)
    print(f"\n{decision_model} Trading Decision:")
    # Use .get with defaults for safer printing
    print(f"  Decision: {decision.get('decision', 'N/A')}")
    print(f"  Quantity: {decision.get('quantity', 'N/A')} {decision.get('quantity_unit', 'N/A')}")
    print(f"  Confidence: {decision.get('confidence', 'N/A')}")
    print(f"  Reasoning: {decision.get('reasoning', 'N/A')}")

    # Step 4: Execute the trade (or handle HOLD)
    print("\n4. Processing the recommended action...")
    # Pass the client and symbol to execute_trade
    action_success = execute_trade(alpaca_client, decision, symbol)

    if action_success:
        # execute_trade handles logging for BUY/SELL/HOLD success
        print("\nAction processed successfully.")
    else:
        # execute_trade already printed the error
        print("\nAction processing failed.")

    print("\n" + "=" * 50)
    print("Trading cycle completed.")
    print("=" * 50)


if __name__ == "__main__":
    print("Initializing Trading Application...")

    # --- Initial Setup & Checks ---
    # Check for essential API keys first
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("\nERROR: Required API keys not found in environment variables.")
        print("Please make sure your .env file is set up correctly with:")
        print("  - ALPACA_API_KEY")
        print("  - ALPACA_SECRET_KEY")
        sys.exit(1) # Exit if keys are missing

    # Select llm Model (can be configured)
    # Using Pro for decisions, Flash for summaries (as example)
    try:
        # Initialize models needed
        decision_model = "deepseek-r1:14b"
        summary_model = "deepseek-r1:14b"
    except Exception as e:
        print(f"FATAL: Failed to initialize llm models: {e}")
        sys.exit(1)

    # --- Main Loop ---
    print("Starting main trading loop...")
    while True:
        try:
            # Pass dependencies to the cycle function
            run_trading_cycle(
                alpaca_client=trading_client,
                decision_model=decision_model,
                summary_model=summary_model,
                news_scraper_instance=news_scraper,
                indicators_scraper_instance=indicators_scraper,
                symbol=SYMBOL
            )
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for {SLEEP_INTERVAL_SECONDS} seconds before the next run...")
            time.sleep(SLEEP_INTERVAL_SECONDS) # Use constant
        except KeyboardInterrupt:
            print("\nApplication termination requested by user.")
            break
        except Exception as e:
            # Catch unexpected errors in the main loop
            print(f"\n--- An unexpected error occurred in the main loop: {e} ---")
            # Add more detailed error logging here if needed (e.g., traceback)
            import traceback
            traceback.print_exc()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Retrying in {SLEEP_INTERVAL_SECONDS} seconds...")
            time.sleep(SLEEP_INTERVAL_SECONDS) # Use constant

    print("\nTrading application stopped.")
