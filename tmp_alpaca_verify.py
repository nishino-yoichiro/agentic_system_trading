import os, requests, datetime as dt
api=os.getenv("ALPACA_API_KEY"); sec=os.getenv("ALPACA_SECRET_KEY")
print("API key present:", bool(api), " Secret present:", bool(sec))
masked = (api[:4] + "..." + api[-3:]) if api else "None"
print("Using APCA-API-KEY-ID:", masked)

headers={"APCA-API-KEY-ID": api, "APCA-API-SECRET-KEY": sec}

# Trading API (account)
acct = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers)
print("Account endpoint:", acct.status_code)

# Market Data v2 (stocks bars)
end=(dt.datetime.utcnow()).replace(microsecond=0).isoformat()+"Z"
start=(dt.datetime.utcnow()-dt.timedelta(days=1)).replace(microsecond=0).isoformat()+"Z"
params={"symbols":"SPY","timeframe":"1Min","start":start,"end":end,"limit":10000}
bars = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=headers, params=params)
print("Bars endpoint:", bars.status_code)
print("Bars response (first 200):", bars.text[:200])
