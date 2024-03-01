import yahooquery as yq
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import operator
import kaleido
from datetime import timedelta, datetime

import warnings
warnings.filterwarnings("ignore")

# put images into 'images' folder in the same directory
def ants_indicator(tickers:list) -> None:
  main_df = yq.Ticker(tickers, formatted=True, validate=True).history(period="1y", interval="1d")
  main_df = main_df.reset_index()
  main_df['date'] = [datetime.strptime(x.strftime('%Y-%m-%d'), ('%Y-%m-%d')) for x in main_df['date']]

  for n, ticker in enumerate(tickers):
    df = main_df[main_df['symbol'] == ticker]
    df.reset_index(drop=True, inplace=True)

    df['momentum'] = df['adjclose'] > df['adjclose'].shift(1)
    df['momentum'] = [df['momentum'][x:15+x].value_counts()[True] if True in df['momentum'][x:15+x].value_counts().keys() else 0 for x in range(len(df))]

    df['avg_volume'] = df['volume'].rolling(30).mean()

    df['last_15_price'] = df['adjclose'].shift(15)

    # momentum only
    grey_index = set()
    for x in range(len(df) - 15):
      if df['momentum'][x] >= 12:
        grey_index.add(x)

    # momentum and volume
    orange_index = set()
    for x in grey_index:
      if df['volume'][x:15+x].mean() > df['avg_volume'][x] * 1.25:
        orange_index.update(range(x, x+15))

    # momentum and price
    blue_index = set()
    for x in grey_index:
      if df['adjclose'][x:15+x].mean() > df['last_15_price'][x] * 1.2:
        blue_index.update(range(x, x+15))

    green_index = orange_index.intersection(blue_index)

    # add on for momentum only
    temp = grey_index.copy()
    for x in temp:
      grey_index.update(range(x, x+15))

    # make sure indexes are not duplicated
    blue_index -= green_index
    orange_index -= green_index
    grey_index = grey_index - green_index - blue_index - orange_index

    indexes = [grey_index, blue_index, orange_index, green_index]
    new_indexes = []

    for index in indexes:
      consec_count = [0] + list(itertools.accumulate([sum(1 for _ in g) for _, g in itertools.groupby([e-i for i, e in enumerate(sorted(list(index)))])], operator.add))
      temp = sorted(list(index.copy()))
      cnt = 0
      for i in range(1, len(consec_count)):
        if consec_count[i] - consec_count[i-1] < 15:
          del temp[consec_count[i-1] - cnt : consec_count[i] - cnt]
          cnt += consec_count[i] - consec_count[i-1]
      new_indexes.append(temp)

    buy_signal = ""
    # generate buy signal through finding recent green ant
    if len(new_indexes[3]) > 0:
      if max(new_indexes[3]) > len(df) - 45:  # see if last 60 days has at least one green ant
        buy_signal = "[BUY]"

    ant_dates, ant_price = [[] for _ in range(4)], [[] for _ in range(4)]

    for i in range(4):
      for x in new_indexes[i]:
        ant_dates[i].append(df['date'][x])
        ant_price[i].append(df['high'][x] * 1.03)

    missing_dates = sorted(set(df.date[0] + timedelta(x) for x in range((df.date[len(df.date)-1] - df.date[0]).days)) - set(df.date))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
              vertical_spacing=0.03, row_width=[0.2, 0.7])
    
    # price candlestick
    fig.add_trace(go.Candlestick(x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']),
                    row=1, col=1)
    
    # volume
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], 
                          showlegend=False,
                          marker_color='skyblue',
                          marker_line=dict(width=1,
                                    color='darkslategrey')), 
                          row=2, col=1)

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="1m",
                        step="month",
                        stepmode="backward"),
                    dict(count=6,
                        label="6m",
                        step="month",
                        stepmode="backward"),
                    dict(count=1,
                        label="YTD",
                        step="year",
                        stepmode="todate"),
                    dict(count=1,
                        label="1y",
                        step="year",
                        stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),
        autosize=False,
        width=1500,
        height=750,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_rangeslider_visible=False)

    fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(values=missing_dates)
                # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
            ]
        )

    color = ['grey', 'navy', 'goldenrod', 'lightgreen']

    for i in range(4):
      fig.add_trace(
          go.Scatter(
              x=ant_dates[i],
              y=ant_price[i],
              mode="markers",
              marker=dict(symbol='circle', 
                          size=6, 
                          color=color[i],
                          line=dict(width=1,
                                    color='darkslategrey'))))

    # fig.show()
    print(f'{n+1}: {ticker}')
    fig.write_image(f"images/{buy_signal} {ticker}.png", format='png', engine='kaleido')
  
  return


tickers = pd.read_csv('large_cap.csv').Symbol.to_list()
ants_indicator(tickers)