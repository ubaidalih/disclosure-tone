import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

COMPANY_CHOICES = ['ACES', 'ADRO', 'AKRA', 'AMRT', 'ANTM', 'ARTO', 'ASII', 'BBCA', 'BBNI', 'BBRI', 
                   'BMRI', 'BRPT', 'CPIN', 'GOTO', 'ICBP', 'INCO', 'INDF', 'INKP', 'KLBF', 'MAPI',
                   'MBMA', 'MDKA', 'MEDC', 'PGAS', 'PGEO', 'PTBA', 'SMGR', 'TLKM', 'UNTR', 'UNVR']
PRICE_CHOICES    = ["12-Weeks Input", "24-Weeks Input"]
TONE_CHOICES  = ["OHLC Only", "ZS ML Fin-R1", "ZS ML Qwen", "ICL ML Fin-R1", "ICL ML Qwen"
                  , "ZS MC2 Fin-R1", "ZS MC2 Qwen", "ICL MC2 Fin-R1", "ICL MC2 Qwen"
                  , "ZS MC1 Fin-R1", "ZS MC1 Qwen", "ICL MC1 Fin-R1", "ICL MC1 Qwen"]
DEFAULT_START  = datetime.strptime('2024-04-05', '%Y-%m-%d')
DEFAULT_END    = datetime.strptime('2025-03-28', '%Y-%m-%d')

DEVICE = 'cuda:0'
HIDDEN_DIM = 64
LAYER_DIM  = 2
DROPOUT    = 0.2

def set_seed(SEED=42):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataset(price_filename, tone_filename):
   weekly_ff = pd.read_csv(price_filename)
   weekly_ff['Date'] = pd.to_datetime(weekly_ff['Date'])
   tone_df = pd.read_csv(tone_filename)
   tone_df['Ticker'] = tone_df['Ticker'] + '.JK'

   d = weekly_ff['Date']
   weekly_ff['tone_year'] = d.dt.year - 1 - (d.dt.month < 4).astype(int)
   tone_df = tone_df.rename(columns={'Year':'tone_year'})

   merged = (
      weekly_ff
         .merge(
            tone_df,
            on=['Ticker','tone_year'],
            how='left'
         )
         .drop(columns=['tone_year'])
   )
   merged = merged.fillna(0)
   return merged

def create_dataloader(df, ticker, start_date, end_date, n_lags=12, tone=False, batch_size=64):
    if tone:
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 
            'Positive_Pred', 'Negative_Pred', 'Uncertainty_Pred', 'Litigious_Pred', 
            'Strong_Modal_Pred', 'Weak_Modal_Pred', 'Constraining_Pred'
        ]
    else:
        feature_cols = [
            'Open', 'High', 'Low', 'Close'
        ]
    X_test, y_test = [], []
    df = df[(df['Ticker'] == ticker)]
    df = df.sort_values('Date')
    feats  = df[feature_cols].values
    dates = df['Date'].values

    for i in range(n_lags, len(df)):
        Xw = feats[i-n_lags : i]
        yw = feats[i][3]
        label_date = dates[i]
        if label_date >= pd.to_datetime(start_date) and label_date <= pd.to_datetime(end_date):
            X_test.append(Xw)
            y_test.append(yw)
    
    X_test  = np.stack(X_test,  axis=0)
    y_test  = np.array(y_test)
    test_ds  = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    return X_test.shape[2], test_loader

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(BiLSTMModel, self).__init__()
        self.relu1 = nn.ReLU()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu2 = nn.ReLU()
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 16)
        self.fc2 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self.relu1(x)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x = self.relu2(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc1(x[:, -1, :])
        x = self.fc2(x)
        return x.squeeze(-1)

def evaluate_model(model_name, input_dim, test_loader):
    set_seed()
    model = BiLSTMModel(input_dim, HIDDEN_DIM, LAYER_DIM, DROPOUT)
    model.to(DEVICE)
    checkpoint_path = f"Model/{model_name}"
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)  
    model.load_state_dict(state_dict)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)

    return rmse, mae, all_labels, all_preds

def run_inference(price_choice, tone_choice, ticker_choice, start_date, end_date):
    start_date = datetime.fromtimestamp(start_date)
    end_date = datetime.fromtimestamp(end_date)
    if start_date is None or end_date is None:
        raise gr.Error("Please choose both start and end dates.")
    if start_date > end_date:
        raise gr.Error("End date must be after start date.")
    if start_date < DEFAULT_START:
        raise gr.Error("Start date must be after 2024-04-05.")
    if end_date > DEFAULT_END:
        raise gr.Error("End date must be before 2025-03-28.")
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    window = price_choice.split('-')[0]
    prompt = tone_choice.split(' ')[0].lower()
    if prompt == 'ohlc':
        model_name = f"ohlc_{window}.pt"
        csv_name = "tone_icl_multilabel_fin-r1.csv"
        tone = False
    else:
        classification = tone_choice.split(' ')[1]
        model = tone_choice.split(' ')[2].lower()
        if classification == 'ML':
            classification = 'multilabel'
        if classification == 'MC2':
            classification = 'multiclass-2'
        if classification == 'MC1':
            classification = 'multiclass-1'
        model_name = f"{prompt}_{classification}_{model}_{window}.pt"
        csv_name = f"tone_{prompt}_{classification}_{model}.csv"
        tone =True
    timeseries_df = create_dataset('Weekly_Stock_Price.csv', f"DisclosureTone/{csv_name}")
    input_dim, test_loader = create_dataloader(timeseries_df, f"{ticker_choice}.JK", start_date, end_date, n_lags=int(window), tone=tone)
    rmse, mae, labels, preds = evaluate_model(model_name, input_dim, test_loader)

    price_df = pd.read_csv('Weekly_Stock_Price.csv')
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    dates = price_df[(price_df['Ticker'] == f"{ticker_choice}.JK") & (price_df['Date'] >= start_date) & (price_df['Date'] <= end_date)]['Date'].values
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dates, labels, label="Actual")
    ax.plot(dates, preds, label="Predicted")
    ax.set_title(f"{ticker_choice} | {price_choice} | {tone_choice}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    return fig, rmse, mae

with gr.Blocks(title="Stock Price Forecasting With Disclosure Tone") as demo:
    gr.Markdown("## Stock Price Forecasting With Disclosure Tone")

    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            ticker_in = gr.Dropdown(choices=COMPANY_CHOICES, label="Ticker", value=COMPANY_CHOICES[0], interactive=True)
            price_in = gr.Radio(choices=PRICE_CHOICES, label="Price Window", value=PRICE_CHOICES[0])
            tone_in = gr.Dropdown(choices=TONE_CHOICES, label="Tone Type", value=TONE_CHOICES[0], interactive=True)
            start_in = gr.DateTime(label="Start Date", include_time=False, value=DEFAULT_START)
            end_in = gr.DateTime(label="End Date", include_time=False, value=DEFAULT_END)
            run_btn = gr.Button("Run", variant="primary")

        with gr.Column(scale=2, min_width=400):
            plot_out = gr.Plot(label="Forecast vs. Actual")
            with gr.Row():
                rmse_out = gr.Number(label="RMSE", precision=4)
                mae_out  = gr.Number(label="MAE",  precision=4)

    run_btn.click(
        fn=run_inference,
        inputs=[price_in, tone_in, ticker_in, start_in, end_in],
        outputs=[plot_out, rmse_out, mae_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
