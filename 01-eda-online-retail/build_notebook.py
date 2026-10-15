"""
Builds eda_online_retail.ipynb from the cells below, then execute it with:
    jupyter nbconvert --to notebook --execute --inplace eda_online_retail.ipynb
Keeping the notebook source in a .py file makes it easy to review and diff.
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Exploratory Data Analysis — Online Retail (UK e-commerce, 2010–2011)

**Dataset:** [E-Commerce Data on Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data), originally from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail) (Chen, Sain & Guo, 2012).
Every transaction of a UK-based online gift retailer between **01-Dec-2010 and 09-Dec-2011**. Many of its customers are wholesalers.

| Column | Meaning |
|---|---|
| InvoiceNo | invoice number. A leading **C** means a cancellation |
| StockCode | product code |
| Description | product name |
| Quantity | units per line |
| InvoiceDate | date and time |
| UnitPrice | price per unit (GBP) |
| CustomerID | customer number |
| Country | customer's country |

**Questions**
1. How clean is the data, and what has to be fixed before analysis?
2. How do sales move over time (month, weekday, hour)?
3. Which countries, products and customers drive revenue? How concentrated is it?
4. How do customer-level variables relate to each other (recency, frequency, spend, basket size)?
5. How well are customers retained, and what gets returned?
""")

code(r"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Colour roles (validated categorical palette + one-hue sequential ramp for magnitude)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, MUTED, INK = SERIES[0], SERIES[1], "#b9b8b2", "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "grid.color": "#ebeae6", "grid.linewidth": 0.8, "lines.linewidth": 2,
})
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
pd.set_option("display.max_columns", 20)

FIG = Path("figures")
FIG.mkdir(exist_ok=True)

def save(fig, name):
    fig.savefig(FIG / f"{name}.png")

def _gbp(v, _):
    if abs(v) >= 1e6: return f"£{v/1e6:.1f}M"
    if abs(v) >= 1e4: return f"£{v/1e3:.0f}K"
    if abs(v) >= 1e3: return f"£{v/1e3:.1f}K"
    return f"£{v:.0f}"
gbp = mtick.FuncFormatter(_gbp)
""")

md("## 1. Load and first look")
code(r"""
raw = pd.read_excel("data/Online Retail.xlsx", dtype={"InvoiceNo": str, "StockCode": str})
print(raw.shape)
raw.head()
""")
code(r"""
raw.info()
""")
code(r"""
raw.describe()
""")
md(r"""
Already visible: `Quantity` and `UnitPrice` have **negative minimums** and huge maximums (±80,995 units), and `CustomerID`
was read as a float because it has missing values.
""")

md("## 2. Data-quality audit")
code(r"""
is_cancel = raw["InvoiceNo"].str.startswith("C")
is_product = raw["StockCode"].str.match(r"^\d{5}[A-Za-z]{0,2}$")
desc = raw["Description"].astype("string")

audit = pd.DataFrame([
    ("Missing CustomerID",                          raw["CustomerID"].isna().sum()),
    ("Missing Description",                         raw["Description"].isna().sum()),
    ("Exact duplicate rows",                        raw.duplicated().sum()),
    ("Cancellation invoices (InvoiceNo starts 'C')", is_cancel.sum()),
    ("Quantity <= 0 on a normal invoice",           ((raw["Quantity"] <= 0) & ~is_cancel).sum()),
    ("UnitPrice <= 0",                              (raw["UnitPrice"] <= 0).sum()),
    ("Non-product StockCode (postage, fees, ...)",  (~is_product).sum()),
    ("Description with extra spaces",               (desc != desc.str.strip()).sum()),
    ("|Quantity| >= 10,000",                        (raw["Quantity"].abs() >= 10_000).sum()),
], columns=["Issue", "Rows"])
audit["% of rows"] = 100 * audit["Rows"] / len(raw)
audit
""")
code(r"""
# Non-product codes: what are they?
(raw.loc[~is_product].groupby("StockCode")
    .agg(Description=("Description", "first"), Rows=("InvoiceNo", "size"))
    .sort_values("Rows", ascending=False).head(12))
""")
code(r"""
# The same StockCode is written with several descriptions (typos, renamed products)
n_desc = raw.dropna(subset=["Description"]).groupby("StockCode")["Description"].nunique()
print(f"{(n_desc > 1).sum()} stock codes have more than one description, e.g. 23196:")
raw.loc[raw["StockCode"] == "23196", "Description"].value_counts()
""")
code(r"""
# The two extreme orders were cancelled minutes later: data-entry errors, not real demand
cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "CustomerID", "InvoiceDate"]
raw.loc[raw["Quantity"].abs() >= 70_000, cols]
""")
code(r"""
# Who are the customers without an ID?
no_id = raw[raw["CustomerID"].isna() & ~is_cancel & (raw["Quantity"] > 0) & (raw["UnitPrice"] > 0)]
share_rev = (no_id["Quantity"] * no_id["UnitPrice"]).sum() / (raw.loc[~is_cancel & (raw["UnitPrice"] > 0) & (raw["Quantity"] > 0), "Quantity"]
                                                              * raw.loc[~is_cancel & (raw["UnitPrice"] > 0) & (raw["Quantity"] > 0), "UnitPrice"]).sum()
print(f"Rows without CustomerID: {raw['CustomerID'].isna().mean():.1%} of rows, but only {share_rev:.1%} of sales value")
print("Top countries of rows without CustomerID:")
no_id["Country"].value_counts(normalize=True).head(3)
""")

md(r"""
### Cleaning decisions

| Issue | Decision | Why |
|---|---|---|
| Exact duplicates | **Remove**, before and again after name cleaning | The same line entered twice would double-count sales |
| Cancellations (`C…`) | **Move to a separate `returns` table** | They are returns, not sales. Analysed separately in section 3.9 |
| Non-product codes (POST, DOT, M, AMAZONFEE, BANK CHARGES…) | **Remove** | Postage, fees and manual adjustments are not product demand |
| Quantity ≤ 0 on normal invoices | **Remove** | Stock write-offs ("damages", "check", "?"), none have a customer |
| UnitPrice ≤ 0 | **Remove** | Free samples or bad-debt adjustments with no revenue |
| Orders fully reversed within a day (same customer, product, quantity) | **Remove both the sale and its cancellation** | Includes the 80,995- and 74,215-unit entry errors |
| Messy descriptions | **Trim, upper-case, one canonical name per StockCode** (the most frequent) | So each product is grouped under one name |
| Missing CustomerID (25% of rows) | **Keep**, flag `HasCustomer` | They carry 16.5% of sales, almost all from the UK (most likely guest checkouts). Dropping them would under-state revenue. They are left out **only** in customer-level analysis |
""")

code(r"""
log = []
def step(name, before, after):
    log.append({"Step": name, "Rows before": len(before), "Removed": len(before) - len(after), "Rows after": len(after)})
    return after

df = step("Remove exact duplicates", raw, raw.drop_duplicates())

cancel_mask = df["InvoiceNo"].str.startswith("C")
returns = df[cancel_mask].copy()
df = step("Move cancellations to `returns`", df, df[~cancel_mask])

df = step("Remove non-product stock codes", df, df[df["StockCode"].str.match(r"^\d{5}[A-Za-z]{0,2}$")])
df = step("Remove quantity <= 0 (stock adjustments)", df, df[df["Quantity"] > 0])
df = step("Remove unit price <= 0", df, df[df["UnitPrice"] > 0])

# Sales fully reversed by a cancellation of the same customer/product/quantity within 1 day
c = (returns.dropna(subset=["CustomerID"])
            .assign(Quantity=lambda x: -x["Quantity"])
            [["CustomerID", "StockCode", "Quantity", "InvoiceDate"]]
            .rename(columns={"InvoiceDate": "CancelDate"})
            .rename_axis("cancel_idx").reset_index())
pairs = df.reset_index().merge(c, on=["CustomerID", "StockCode", "Quantity"])
lag = pairs["CancelDate"] - pairs["InvoiceDate"]
pairs = pairs[(lag >= pd.Timedelta(0)) & (lag <= pd.Timedelta(days=1))]
df = step("Remove sales reversed within 1 day", df, df.drop(index=pairs["index"].unique()))
# ...and their cancellations, so the returns analysis does not count entry errors as returns
returns = returns.drop(index=pairs["cancel_idx"].unique())
print(f"Reversed pairs removed: {pairs['index'].nunique()} sales lines and {pairs['cancel_idx'].nunique()} cancellation lines")

# Standardise product names: trim, collapse spaces, upper-case, then one canonical name per code
clean_desc = df["Description"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
canonical = clean_desc.groupby(df["StockCode"]).agg(lambda s: s.value_counts().idxmax())
df = df.assign(Description=df["StockCode"].map(canonical))
# rows that differed only in how the description was typed are now exact duplicates
df = step("Remove duplicates revealed by name cleaning", df, df.drop_duplicates())

# Types and derived columns
df = df.assign(
    CustomerID=df["CustomerID"].astype("Int64").astype("string"),
    HasCustomer=df["CustomerID"].notna(),
    Revenue=df["Quantity"] * df["UnitPrice"],
    Month=df["InvoiceDate"].dt.to_period("M"),
    Weekday=df["InvoiceDate"].dt.day_name(),
    Hour=df["InvoiceDate"].dt.hour,
)
returns = returns[returns["StockCode"].str.match(r"^\d{5}[A-Za-z]{0,2}$")].assign(
    Revenue=lambda x: x["Quantity"] * x["UnitPrice"],
    CustomerID=lambda x: x["CustomerID"].astype("Int64").astype("string"))

cleaning_log = pd.DataFrame(log)
cleaning_log
""")
code(r"""
print(f"Clean sales lines : {len(df):,}  ({len(df) / len(raw):.1%} of raw rows kept)")
print(f"Invoices          : {df['InvoiceNo'].nunique():,}")
print(f"Products          : {df['StockCode'].nunique():,}")
print(f"Customers (with ID): {df['CustomerID'].nunique():,}")
print(f"Countries         : {df['Country'].nunique()}")
print(f"Revenue           : £{df['Revenue'].sum():,.0f}")
print(f"Period            : {df['InvoiceDate'].min():%d-%b-%Y} to {df['InvoiceDate'].max():%d-%b-%Y}")
assert df.duplicated().sum() == 0 and (df["Quantity"] > 0).all() and (df["UnitPrice"] > 0).all()
""")

md("## 3. Exploratory analysis")
md("### 3.1 Distributions: what does a typical order look like?")
code(r"""
orders = df.groupby("InvoiceNo").agg(
    OrderValue=("Revenue", "sum"), Lines=("StockCode", "size"), Units=("Quantity", "sum"),
    HasCustomer=("HasCustomer", "first"), Country=("Country", "first"), InvoiceDate=("InvoiceDate", "first"))
orders[["OrderValue", "Lines", "Units"]].describe(percentiles=[.25, .5, .75, .95, .99])
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(orders["OrderValue"], bins=60, log_scale=True, color=BLUE, edgecolor="white", linewidth=0.5, ax=axes[0])
med = orders["OrderValue"].median()
axes[0].axvline(med, color=INK, ls="--", lw=1)
axes[0].text(med * 1.15, axes[0].get_ylim()[1] * 0.9, f"median £{med:,.0f}", color=INK)
axes[0].set(title="Order value is right-skewed (log scale)", xlabel="Order value (£, log scale)", ylabel="Orders")

sns.histplot(orders["Lines"].clip(upper=100), bins=50, color=BLUE, edgecolor="white", linewidth=0.5, ax=axes[1])
axes[1].set(title="Products per order (capped at 100)", xlabel="Distinct lines per order", ylabel="Orders")
save(fig, "01_distributions")
plt.show()
""")
md(r"""
Order values are strongly right-skewed. The **median (not the mean)** is the right "typical order" here. A long tail of large
orders, a few thousand pounds each, comes from wholesale buyers. Because of this skew the analysis below uses log scales,
medians and rank-based (Spearman) correlation.
""")

md("### 3.2 Trend over time")
code(r"""
monthly = df.groupby("Month").agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"),
                                  Customers=("CustomerID", "nunique"))
monthly["AOV"] = monthly["Revenue"] / monthly["Orders"]
monthly["Days"] = df.groupby("Month")["InvoiceDate"].apply(lambda s: s.dt.date.nunique())
monthly
""")
code(r"""
fig, ax = plt.subplots(figsize=(12, 4.2))
labels = monthly.index.strftime("%b\n%Y")
colors = [MUTED if m == pd.Period("2011-12") else BLUE for m in monthly.index]
ax.bar(labels, monthly["Revenue"], color=colors, width=0.7)
ax.yaxis.set_major_formatter(gbp)
peak = monthly["Revenue"].idxmax()
ax.text(list(monthly.index).index(peak) - 0.45, monthly.loc[peak, "Revenue"],
        f"Peak: {peak.strftime('%b %Y')}  \n£{monthly.loc[peak, 'Revenue']/1e6:.2f}M  ", ha="right", va="top", color=INK)
ax.text(len(monthly) - 1, monthly["Revenue"].iloc[-1], "partial\n(1–9 Dec)", ha="center", va="bottom", color=INK, fontsize=9)
ax.set(title="Monthly revenue: steady growth into a Sep–Nov peak", ylabel="Revenue")
save(fig, "02_monthly_revenue")
plt.show()
""")
code(r"""
# Orders and customers on separate panels (different scales -> no dual axis)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharex=True)
full = monthly.loc[:"2011-11"]
for ax, col, title in [(axes[0], "Orders", "Orders per month"), (axes[1], "Customers", "Active customers per month")]:
    ax.plot(full.index.strftime("%b %y"), full[col], color=BLUE, marker="o", markersize=5)
    ax.set(title=title, ylabel=col)
    ax.tick_params(axis="x", rotation=45)
save(fig, "03_orders_customers")
plt.show()

nov, dec10 = monthly.loc["2011-11"], monthly.loc["2010-12"]
print(f"Nov-2011 vs Dec-2010: revenue {nov.Revenue / dec10.Revenue - 1:+.0%}, orders {nov.Orders / dec10.Orders - 1:+.0%}, "
      f"customers {nov.Customers / dec10.Customers - 1:+.0%}, AOV {nov.AOV / dec10.AOV - 1:+.0%}")
q4 = monthly.loc["2011-09":"2011-11", "Revenue"].sum() / monthly["Revenue"].sum()
print(f"Sep-Nov 2011 = {q4:.1%} of all revenue in the dataset")
""")
md("### 3.3 When do customers buy? (weekday × hour)")
code(r"""
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heat = (df.pivot_table(index="Weekday", columns="Hour", values="Revenue", aggfunc="sum")
          .reindex(days).fillna(0))
fig, ax = plt.subplots(figsize=(12, 3.8))
sns.heatmap(heat / 1e3, cmap=SEQ, linewidths=1, linecolor="white", cbar_kws={"label": "Revenue (£K)"}, ax=ax)
ax.set(title="Revenue by weekday and hour: no sales on Saturday, peak at 10:00–15:00", xlabel="Hour of day", ylabel="")
save(fig, "04_weekday_hour")
plt.show()

by_day = df.groupby("Weekday")["Revenue"].sum().reindex(days).fillna(0)
(by_day / by_day.sum()).to_frame("Share of revenue")
""")

md("### 3.4 Where: countries")
code(r"""
country = df.groupby("Country").agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"),
                                    Customers=("CustomerID", "nunique"))
country["AOV"] = country["Revenue"] / country["Orders"]
country["Share"] = country["Revenue"] / country["Revenue"].sum()
print(f"United Kingdom: {country.loc['United Kingdom', 'Share']:.1%} of revenue, "
      f"{country.loc['United Kingdom', 'Customers'] / country['Customers'].sum():.1%} of customers")
country.sort_values("Revenue", ascending=False).head(11)
""")
code(r"""
intl = country.drop("United Kingdom").sort_values("Revenue", ascending=False).head(10)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].barh(intl.index[::-1], intl["Revenue"][::-1], color=BLUE, height=0.65)
axes[0].xaxis.set_major_formatter(gbp)
axes[0].set(title="Top 10 international markets", xlabel="Revenue")

aov = country[country["Orders"] >= 50].sort_values("AOV", ascending=False).head(10)
bar_colors = [ORANGE if c == "United Kingdom" else BLUE for c in aov.index[::-1]]
axes[1].barh(aov.index[::-1], aov["AOV"][::-1], color=bar_colors, height=0.65)
axes[1].axvline(country.loc["United Kingdom", "AOV"], color=ORANGE, ls="--", lw=1)
axes[1].text(country.loc["United Kingdom", "AOV"], len(aov) - 0.45, " UK average", color=INK, fontsize=9)
axes[1].xaxis.set_major_formatter(gbp)
axes[1].set(title="Highest average order value (countries with 50+ orders)", xlabel="Average order value")
plt.tight_layout()
save(fig, "05_countries")
plt.show()
""")

md("### 3.5 What: products")
code(r"""
products = (df.groupby(["StockCode", "Description"])
              .agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"), Orders=("InvoiceNo", "nunique"),
                   AvgPrice=("UnitPrice", "median"))
              .sort_values("Revenue", ascending=False).reset_index())
products["CumShare"] = products["Revenue"].cumsum() / products["Revenue"].sum()
n80 = (products["CumShare"] <= 0.80).sum() + 1
print(f"{n80:,} of {len(products):,} products ({n80 / len(products):.1%}) generate 80% of revenue")
products.head(10)
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
top = products.head(10)
axes[0].barh(top["Description"].str.title()[::-1], top["Revenue"][::-1], color=BLUE, height=0.65)
axes[0].xaxis.set_major_formatter(gbp)
axes[0].set(title="Top 10 products by revenue", xlabel="Revenue")

x = np.arange(1, len(products) + 1) / len(products)
axes[1].plot(x, products["CumShare"], color=BLUE)
axes[1].axhline(0.8, color=MUTED, lw=1, ls="--")
axes[1].axvline(n80 / len(products), color=MUTED, lw=1, ls="--")
axes[1].annotate(f"{n80 / len(products):.0%} of products\n= 80% of revenue", xy=(n80 / len(products), 0.8),
                 xytext=(25, -45), textcoords="offset points", color=INK, arrowprops=dict(arrowstyle="-", color=INK))
axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(1))
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
axes[1].set(title="Product Pareto curve", xlabel="Share of products (ranked)", ylabel="Cumulative share of revenue")
plt.tight_layout()
save(fig, "06_products")
plt.show()
""")

md("### 3.6 Who: customers and RFM segments")
md("Customer-level analysis uses only rows **with** a CustomerID.")
code(r"""
snapshot = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
known = df[df["HasCustomer"]]
cust = known.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum"),
    UniqueProducts=("StockCode", "nunique"),
    Units=("Quantity", "sum"),
    Tenure=("InvoiceDate", lambda s: (snapshot - s.min()).days),
    Country=("Country", "first"),
)
cust["AvgOrderValue"] = cust["Monetary"] / cust["Frequency"]
cust["UnitsPerOrder"] = cust["Units"] / cust["Frequency"]

cust = cust.sort_values("Monetary", ascending=False)
cum = cust["Monetary"].cumsum() / cust["Monetary"].sum()
top20 = cust["Monetary"].head(int(len(cust) * 0.2)).sum() / cust["Monetary"].sum()
print(f"{len(cust):,} customers | top 20% generate {top20:.1%} of their revenue | "
      f"top 10 customers: {cust['Monetary'].head(10).sum() / cust['Monetary'].sum():.1%}")
print(f"One-time buyers: {(cust['Frequency'] == 1).mean():.1%} of customers")
cust.describe().T[["mean", "50%", "max"]]
""")
code(r"""
# RFM scores 1-5 by quintile (rank first so ties don't break qcut)
r = pd.qcut(cust["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
f = pd.qcut(cust["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
m = pd.qcut(cust["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
cust = cust.assign(R=r, F=f, M=m)

def segment(row):
    if row.R >= 4 and row.F >= 4 and row.M >= 4: return "Champions"
    if row.R >= 3 and row.F >= 3:                return "Loyal"
    if row.R >= 4 and row.F <= 2:                return "New / Promising"
    if row.R <= 2 and row.F >= 3:                return "At Risk"
    if row.R <= 2 and row.F <= 2:                return "Lost / Hibernating"
    return "Needs Attention"

cust["Segment"] = cust.apply(segment, axis=1)
order = ["Champions", "Loyal", "New / Promising", "Needs Attention", "At Risk", "Lost / Hibernating"]
seg = (cust.groupby("Segment").agg(Customers=("Monetary", "size"), Revenue=("Monetary", "sum"),
                                   AvgRecency=("Recency", "median"), AvgOrders=("Frequency", "median"))
           .reindex(order))
seg["% Customers"] = 100 * seg["Customers"] / seg["Customers"].sum()
seg["% Revenue"] = 100 * seg["Revenue"] / seg["Revenue"].sum()
seg
""")
code(r"""
fig, ax = plt.subplots(figsize=(11, 3.8))
y = np.arange(len(seg))
ax.barh(y + 0.2, seg["% Customers"], height=0.38, color=MUTED, label="% of customers")
ax.barh(y - 0.2, seg["% Revenue"], height=0.38, color=BLUE, label="% of revenue")
for yi, v in zip(y - 0.2, seg["% Revenue"]):
    ax.text(v + 0.8, yi, f"{v:.0f}%", va="center", color=INK, fontsize=9)
ax.set_yticks(y, seg.index)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc="lower right", frameon=False)
ax.set(title="RFM segments: share of customers vs share of revenue", xlabel="")
save(fig, "07_rfm_segments")
plt.show()
""")

md("### 3.7 Relationships between variables")
code(r"""
features = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UnitsPerOrder", "UniqueProducts", "Tenure"]
corr = cust[features].corr(method="spearman")      # rank-based: robust to the heavy skew seen in 3.1
fig, ax = plt.subplots(figsize=(7.5, 6))
div = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
sns.heatmap(corr, annot=True, fmt=".2f", cmap=div, vmin=-1, vmax=1, center=0, square=True,
            linewidths=1, linecolor="white", cbar_kws={"shrink": 0.75}, ax=ax)
ax.set(title="Customer features: Spearman correlation")
save(fig, "08_correlation")
plt.show()
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
axes[0].scatter(cust["Frequency"], cust["Monetary"], s=10, alpha=0.35, color=BLUE, edgecolors="none")
axes[0].set(xscale="log", yscale="log", title="More orders → more spend (log–log)",
            xlabel="Orders per customer", ylabel="Total spend (£)")
plain = mtick.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")

lines = df.sample(60_000, random_state=1)
hb = axes[1].hexbin(lines["UnitPrice"], lines["Quantity"], xscale="log", yscale="log", gridsize=35, cmap=SEQ, mincnt=1)
axes[1].set(title="Cheaper items are bought in larger quantities", xlabel="Unit price (£, log)", ylabel="Quantity per line (log)")
fig.colorbar(hb, ax=axes[1], label="Lines")
for ax in axes:
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
plt.tight_layout()
save(fig, "09_relationships")
plt.show()

rho = df[["UnitPrice", "Quantity"]].corr(method="spearman").iloc[0, 1]
print(f"Spearman correlation, unit price vs quantity per line: {rho:.2f}")
""")

md("### 3.8 Retention: monthly cohorts")
code(r"""
first = known.groupby("CustomerID")["Month"].min().rename("Cohort")
# December 2011 has only 9 days of data, so it is left out to avoid a false drop in the last column
act = (known.loc[known["Month"] <= pd.Period("2011-11"), ["CustomerID", "Month"]]
            .drop_duplicates().join(first, on="CustomerID"))
act["Age"] = (act["Month"] - act["Cohort"]).apply(lambda d: d.n)
cohort = act.pivot_table(index="Cohort", columns="Age", values="CustomerID", aggfunc="nunique")
retention = cohort.div(cohort[0], axis=0)

fig, ax = plt.subplots(figsize=(12, 5.2))
sns.heatmap(retention.iloc[:, 1:], annot=True, fmt=".0%", cmap=SEQ, vmin=0, vmax=0.5,
            linewidths=1, linecolor="white", cbar=False, annot_kws={"fontsize": 8}, ax=ax)
ax.set_yticklabels([p.strftime("%b %Y") for p in retention.index], rotation=0)
ax.set(title="Share of each cohort that bought again N months after its first purchase",
       xlabel="Months since first purchase", ylabel="First-purchase cohort")
save(fig, "10_cohort_retention")
plt.show()

m1 = retention[1].drop(pd.Period("2010-12")).mean()
m3 = retention[3].drop(pd.Period("2010-12")).mean()
print(f"Dec-2010 cohort: {retention.loc['2010-12', 1]:.0%} came back the next month; "
      f"later cohorts average {m1:.0%} in month 1 and {m3:.0%} in month 3")
""")

md("### 3.9 Returns")
code(r"""
ret_value = -returns["Revenue"].sum()
print(f"Returned value: £{ret_value:,.0f} = {ret_value / df['Revenue'].sum():.1%} of sales value; "
      f"{returns['InvoiceNo'].nunique():,} cancellation invoices")

ret_prod = (returns.groupby("StockCode").agg(ReturnedValue=("Revenue", lambda s: -s.sum()),
                                              ReturnedUnits=("Quantity", lambda s: -s.sum()))
            .join(products.set_index("StockCode")[["Description", "Revenue", "Units"]], how="inner"))
ret_prod["ReturnRateUnits"] = ret_prod["ReturnedUnits"] / ret_prod["Units"]
ret_prod.query("Units >= 500").sort_values("ReturnedValue", ascending=False).head(8)
""")
code(r"""
ret_country = (returns.groupby("Country")["Revenue"].sum().mul(-1).rename("Returned")
               .to_frame().join(country["Revenue"]))
ret_country["ReturnRate"] = ret_country["Returned"] / ret_country["Revenue"]
ret_country.query("Revenue >= 50_000").sort_values("ReturnRate", ascending=False).head(8)
""")

md("## 4. Key findings and recommendations")
md(r"""
### Data quality
- **96.3% of rows kept** (541,909 → 521,819) after 7 documented cleaning steps. The largest issues were cancellations (9,251 rows),
  duplicates (5,268) and non-product codes such as postage and fees (2,406).
- Two orders of **80,995 and 74,215 units** were cancelled 12–16 minutes later. Left in, they would add about £246K of fake sales
  and make two minor products the best sellers by units.
- **25% of rows have no CustomerID** but only 16.5% of sales value. Dropping them, as many tutorials do, would under-state revenue.
  They are kept for revenue analysis and excluded only from customer-level work.

### Findings
| # | Finding | Evidence |
|---|---|---|
| 1 | **Strong Q4 seasonality**: Sep–Nov 2011 brought in 35.8% of the year's revenue. November alone was £1.44M | 3.2 |
| 2 | **Growth comes from more customers, not bigger baskets**: Nov-2011 vs Dec-2010 shows customers +88% and revenue +87%, but average order value only +5% | 3.2 |
| 3 | **A B2B weekday pattern**: no orders at all on Saturday, Sunday is only 8% of revenue, and most revenue lands on weekdays between 10:00 and 15:00 | 3.3 |
| 4 | **UK = 85% of revenue.** Abroad, a handful of wholesale accounts dominate: the Netherlands brings £284K from just 9 customers (average order £3,053, 6.5× the UK) | 3.4 |
| 5 | **The long tail is real**: 22% of the 3,899 products generate 80% of revenue | 3.5 |
| 6 | **Customers are concentrated**: the top 20% generate 74% of revenue. "Champions" (21% of customers) bring 66%. 35% of customers bought only once | 3.6 |
| 7 | **£705K sits with "At Risk" customers** (15% of customers): they used to buy regularly but have been inactive for a median of 134 days | 3.6 |
| 8 | **Spend has two independent drivers**: *how often* (frequency ↔ spend ρ = 0.81) and *how big* (order value ↔ units per order ρ = 0.81). Frequency and order value are almost unrelated (ρ = 0.16) | 3.7 |
| 9 | **Price vs quantity**: cheaper items are bought in bulk (ρ = −0.41 between unit price and quantity per line), a typical wholesale pattern | 3.7 |
| 10 | **Retention is the main weakness**: only about 20% of a new cohort buys again the next month (23% by month 3). The Dec-2010 cohort looks better (37%) because it contains existing customers whose history starts before the dataset (left-censoring) | 3.8 |
| 11 | **Returns are low overall (1.8% of sales value) but concentrated**: Tea Time Party Bunting had 59% of its units returned, and Feltcraft Doll Molly and Fairy Cake Flannel about 31–32% | 3.9 |

### Recommendations
1. **Plan stock, staff and cash flow for Sep–Nov.** Place orders for the top 22% of products by August at the latest.
2. **Build a retention programme for first-time buyers**, for example a follow-up offer 30 days after the first order. Raising month-1 retention
   from 20% to 25% is worth more than any single campaign, because 35% of customers never come back.
3. **Win back the At-Risk segment** (656 customers, £705K in past spend) with personal outreach, since many are wholesalers.
4. **Grow international wholesale deliberately.** A single Dutch or Australian account is worth dozens of UK retail customers.
   Give key accounts dedicated account management.
5. **Investigate the high-return products** (quality, description, photos) before the next season.
6. **Fix data capture at the source**: make CustomerID mandatory for online orders, and block orders above a quantity limit without a confirmation step.

### Limitations
- Only 12 months of data, so year-over-year seasonality cannot be confirmed.
- December 2011 is partial (1–9 Dec) and is flagged in all monthly views.
- Only revenue is available (no cost), so profitability cannot be analysed.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "eda_online_retail.ipynb")
print("notebook written:", len(cells), "cells")
