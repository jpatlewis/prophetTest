import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

dataFrame = pd.read_csv('./example_wp_log_peyton_manning.csv')
p = Prophet()
p.fit(dataFrame)
future = p.make_future_dataframe(periods=365)
forecast = p.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig2 = p.plot_components(forecast)
plt.show()