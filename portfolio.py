import pandas as pd
from openbb import obb
import riskfolio as rp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

assets = [
    'NVDA', 
    'OXY', 
    'NIO', 
    'QQQ', 
    'PBR', 
    'SQ', 
    'AAPL', 
    'GOOG', 
    'DBX', 
    'AZN', 
    'EXPE', 
    'WIMI',
    'FSLR'
]

data = (
    obb
    .equity
    .price
    .historical(assets, provider="yfinance")
    .to_df()
    .pivot(columns="symbol", values="close")
)

initial_returns = data.pct_change().dropna()

rf = round((
    obb
    .equity
    .price
    .historical('^TNX', provider='yfinance')
    .to_df()
    .close
    .tail(1)[0]
) / 100 / 252, 5)

port = rp.Portfolio(returns=initial_returns)
rm = 'MV'
method_mu='ewma1' # Method to estimate expected returns based on historical data.
method_cov='ewma1' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Risk parity portfolio with same risk contribution of each asset
w1 = port.rp_optimization(model='Classic', hist=True)

# # Classic portfolio that only minimizes the risk aka standard deviation
# port.wc_stats()
# w2 = port.wc_optimization(obj='Sharpe', rf=rf, l=2)

w3 = port.optimization()

# Estimate points in the efficient frontier mean - standard deviation
ws = port.efficient_frontier(model='Classic', rm=rm, kelly='approx', points=20, rf=rf, hist=True)

label = 'Optimal Portfolio'
# optimized returns and cov
mu = port.mu
cov = port.cov
returns = port.returns
model = w3

ax = rp.plot_frontier(w_frontier=ws,
                      mu=mu,
                      cov=cov,
                      returns=returns,
                      rm=rm,
                      rf=rf,
                      alpha=0.05,
                      cmap='viridis',
                      w=model,
                      label=label,
                      marker='*',
                      s=16,
                      c='r',
                      height=6,
                      width=10,
                      t_factor=252,
                      ax=None)
plt.savefig('riskfolio/frontier.png', bbox_inches='tight')
plt.close()


ax = rp.plot_pie(w=model,
                 title='Portfolio',
                 height=6,
                 width=10,
                 cmap="tab20",
                 ax=None)
plt.savefig('riskfolio/pie.png', bbox_inches='tight')
plt.close()


ax = rp.plot_risk_con(w=model,
                      cov=cov,
                      returns=returns,
                      rm=rm,
                      rf=rf,
                      alpha=0.05,
                      color="tab:blue",
                      height=6,
                      width=10,
                      t_factor=252,
                      ax=None)
plt.savefig('riskfolio/risk_con.png', bbox_inches='tight')
plt.close()


ax = rp.plot_table(returns=returns,
                   w=model,
                   MAR=rf,
                   alpha=0.05,
                   ax=None)
plt.savefig('riskfolio/table.png', bbox_inches='tight')
plt.close()

sharpe = rp.RiskFunctions.Sharpe(w=model,
                                 mu=mu,
                                 cov=cov,
                                 returns=returns)
print(f"Sharpe Ratio: {sharpe}")