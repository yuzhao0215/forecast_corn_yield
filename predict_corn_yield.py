import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.io as pio
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn


data_file = "./data/corn_yield/total_data.csv"      # data file path

df = pd.read_csv(data_file, index_col="Year")       # read as dataframe

years = df.index                                    # set "years" column as index column

df = df.loc[:"2010", :]                             # choose data between 1950-2010, comment this line if need all data

pio.templates.default = "plotly_white"              # set plotly style
plot_template = dict(
    layout=go.Layout({
        "font_size": 18,
        "xaxis_title_font_size": 24,
        "yaxis_title_font_size": 24})
)

# uncomment following lines to plot corn_yield versus year

# fig = px.line(df, x=years, y="Corn_Yield(BU/ACRE)", labels=dict(
#     created_at="Year", value="BU / ACRE", variable="Corn_Yield"
# ))
# fig.update_layout(
#   template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
# )
# fig.show()
#

target_column = "Corn_Yield(BU/ACRE)"
# features = list(df.columns.difference([target_column]))       # this line of code use all features except Corn_Yield
features = df.columns                                           # this line uses all features including Corn_Yield for prediction

forecast_lead = 1                                               # "1" means using sequence data to predict the following one year's corn yield

# following lines shift target column by "forecast_lead" rows
# This means that when selecting data on i'th row, the target will be df['Target_forecast'][i]
#   example:    Year    Target    Target_forecast
#               1997      0.5           0.2
#               1998      0.2           0.3
#               1999      0.3           NaN
target = f"{target_column}_lead{forecast_lead}"
df[target] = df[target_column].shift(-forecast_lead)
df = df.iloc[:-forecast_lead]

torch.manual_seed(101)

batch_size = 1
sequence_length = 8

test_start = 1998                                       # start year of test data set

df_train = df.loc[:test_start].copy()
df_test = object()

if test_start-sequence_length+1 >= 0:
    df_test = df.loc[test_start-sequence_length+1:].copy()
else:
    print("Not enough sequence for test dataset, exiting.")
    exit()
# exp. test_start = 10; when seq = 1, then no need to go backward. when seq = 2, then need to go back additional one.

print("Test set fraction:", (len(df_test) - sequence_length + 1) / len(df))

# normalize train and test data sets using mean and std of train data set
target_mean = df_train[target].mean()
target_stdev = df_train[target].std()

for c in df_train.columns:
    mean = df_train[c].mean()
    stdev = df_train[c].std()

    df_train[c] = (df_train[c] - mean) / stdev
    df_test[c] = (df_test[c] - mean) / stdev

    df_train[c].fillna(value=0, inplace=True)
    df_test[c].fillna(value=0, inplace=True)

print(df_train)


# Customized DataLoader class
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, is_test, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length

        self.is_test = is_test
        self.previous_data = []  # this is used for test data set. when the index is smaller than sequence_length - 1,
                                 # previous_data will be used on top of test data for prediction
        if self.is_test:
            self.previous_data = torch.tensor(dataframe[features].values[0:sequence_length-1]).float()
            self.X = torch.tensor(dataframe[features].values[sequence_length-1:]).float()
            self.y = torch.tensor(dataframe[target].values[sequence_length-1:]).float()
        else:
            self.y = torch.tensor(dataframe[target].values).float()
            self.X = torch.tensor(dataframe[features].values).float()
        self.num_features = len(features)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if not self.is_test:
            if i >= self.sequence_length - 1:
                i_start = i - self.sequence_length + 1
                if self.num_features > 1:
                    x = self.X[i_start:(i + 1), :]
                else:
                    x = self.X[i_start:(i + 1)]
            else:
                padding = self.X[0].repeat(self.sequence_length - i - 1, 1)

                if self.num_features > 1:
                    x = self.X[0:(i + 1), :]
                else:
                    x = self.X[0:(i + 1)]

                x = torch.cat((padding, x), 0)
        else:
            if i >= self.sequence_length - 1:
                i_start = i - self.sequence_length + 1
                if self.num_features > 1:
                    x = self.X[i_start:(i + 1), :]
                else:
                    x = self.X[i_start:(i + 1)]
            else:
                if self.num_features > 1:
                    padding = self.previous_data[i:, :]
                    x = self.X[0:(i + 1), :]
                else:
                    padding = self.previous_data[i:]
                    x = self.X[0:(i + 1)]

                x = torch.cat((padding, x), 0)

        return x, self.y[i]


# training data set
train_dataset = SequenceDataset(
    df_train,
    target=target,
    features=features,
    is_test=False,
    sequence_length=sequence_length
)

# testing data set
test_dataset = SequenceDataset(
    df_test,
    target=target,
    features=features,
    is_test=True,
    sequence_length=sequence_length
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X, y = next(iter(train_loader))

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# class of LSTM
class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_features, hidden_units):
        super().__init__()
        self.num_sensors = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 3

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


learning_rate = 5e-5
num_hidden_units = 200

model = ShallowRegressionLSTM(num_features=len(features), hidden_units=num_hidden_units)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")


print("Untrained test\n--------")
test_model(test_loader, model, loss_function)
print()

epochs = 300

for ix_epoch in range(epochs):
    print(f"Epoch {ix_epoch}\n---------")
    train_model(train_loader, model, loss_function, optimizer=optimizer)
    test_model(test_loader, model, loss_function)
    print()


def predict(data_loader, model):
    """Just like `test_loop` function but keep track of the outputs instead of the loss
    function.
    """
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output


train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
ystar_col = "Model forecast"
df_train[ystar_col] = predict(train_eval_loader, model).numpy()

# drop the first sequence_len - 1 rows of df_test
df_test = df_test.iloc[sequence_length-1:, :]
df_test[ystar_col] = predict(test_loader, model).numpy()

df_out = pd.concat((df_train, df_test))[[target, ystar_col]]

for c in df_out.columns:
    df_out[c] = df_out[c] * target_stdev + target_mean

original_target = df_test[target] * target_stdev + target_mean
predict_target = df_test[ystar_col] * target_stdev + target_mean

mean_error_rate = np.mean(np.absolute((original_target - predict_target) / original_target))
print("The final predication error rate is: {}%".format(mean_error_rate * 100))

mean_mse_error = np.mean((original_target - predict_target)**2)
print("The final predication mse error is: {}\n".format(mean_mse_error))

mean_error_rate_using_target_mean = np.mean(np.absolute((original_target - target_mean) / original_target))
print("The prediction error rate using target mean is: {}%".format(mean_error_rate_using_target_mean * 100))

mean_mse_error_using_target_mean = np.mean((original_target - target_mean)**2)
print("The mse error using target mean is: {}\n".format(mean_mse_error_using_target_mean))

# print(df_out)

fig = px.line(df_out, labels={'value': "Corn Yield (BU/ACRE)", 'created_at': 'Year'})
fig.add_vline(x=test_start, line_width=4, line_dash="dash")
fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text="Test set start", showarrow=False)
fig.update_layout(
  template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
)
fig.show()

# data Source
'''
Weather:    https://mrcc.purdue.edu/CLIMATE/
Soil:       https://mrcc.purdue.edu/CLIMATE/
Corn_Yield: https://quickstats.nass.usda.gov/
Price:      https://quickstats.nass.usda.gov/
Wind:       https://www.wunderground.com
'''
