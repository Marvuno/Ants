import pandas as pd
from openbb import obb
import riskfolio as rp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def dendrogram(returns):
    ax = rp.plot_dendrogram(
        returns=returns,
        codependence="pearson",
        linkage="single",
        k=None,
        max_k=10,
        leaf_order=True,
        ax=None,
    )
    return ax

def hcportfolio(returns, rf, method):
    port = rp.HCPortfolio(returns=returns)
    w = port.optimization(
        model="HRP",
        codependence="pearson",
        rm=method, # MV, MAD, CVaR, VaR
        rf=rf,
        linkage="single",
        max_k=10,
        leaf_order=True,
    )
    return w

def pie_chart(returns, rf, method):
    ax = rp.plot_pie(
        w=hcportfolio(returns, rf, method),
        title="HRP Naive Risk Parity",
        others=0.05,
        nrow=25,
        cmap="tab20",
        height=8,
        width=10,
        ax=None,
    )
    return ax

def risk_contribution(returns, rf, method):
    ax = rp.plot_risk_con(
        w=hcportfolio(returns, rf, method),
        cov=returns.cov(),
        returns=returns,
        rm=method,
        rf=rf,
        alpha=0.05,
        color="tab:blue",
        height=6,
        width=10,
        t_factor=252,
        ax=None,
    )
    return ax
    
# assets = [
#     "XLE", "XLF", "XLU", "XLI", "GDX",
#     "XLK", "XLV", "XLY", "XLP", "XLB",
#     "XOP", "IYR", "XHB", "ITB", "VNQ",
#     "GDXJ", "IYE", "OIH", "XME", "XRT",
#     "SMH", "IBB", "KBE", "KRE", "XTL",
# ]

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

returns = data.pct_change().dropna()

rf = (
    obb
    .equity
    .price
    .historical('^TNX', provider='yfinance')
    .to_df()
    .close
    .tail(1)[0]
) / 100

ax = dendrogram(returns)
plt.savefig('hrp/dendrogram.png', bbox_inches='tight')
plt.close()

ax = pie_chart(returns, rf, 'MV')
plt.savefig('hrp/pie.png', bbox_inches='tight')
plt.close()

ax = risk_contribution(returns, rf, 'MV')
plt.savefig('hrp/RC.png', bbox_inches='tight')
plt.close()