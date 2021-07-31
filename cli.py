import typer
import pickle

import tqdm

import os
from rich.console import Console

from pathlib import Path

from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

import warnings
import yaml

warnings.filterwarnings("ignore")

app = typer.Typer(
    name="Cyber Anamoly",
    add_completion=False,
    help="This is an app to check PCAP CSV files for attack.",
)
console = Console()


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(os.path.curdir, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("config.yaml")

attack_model = pickle.load(open(config["ATTACK_MODEL_PATH"], "rb"))
dos_attack_model = pickle.load(open(config["DOS_ATTACK_MODEL_PATH"], "rb"))

input_classes = config["INPUT_CLASSES"]

dos_classes = config["DOS_CLASSES"]


maps = {'ACK Flag Cnt': 'ACK Flag Count',
        'Active Max': 'Active Max',
        'Active Mean': 'Active Mean',
        'Active Min': 'Active Min',
        'Active Std': 'Active Std',
        'Pkt Size Avg': 'Average Packet Size',
        'Bwd Blk Rate Avg': 'Bwd Bulk Rate Avg',
        'Bwd Byts/b Avg': 'Bwd Bytes/Bulk Avg',
        'Bwd Header Len': 'Bwd Header Length',
        'Bwd IAT Max': 'Bwd IAT Max',
        'Bwd IAT Mean': 'Bwd IAT Mean',
        'Bwd IAT Min': 'Bwd IAT Min',
        'Bwd IAT Std': 'Bwd IAT Std',
        'Bwd IAT Tot': 'Bwd IAT Total',
        'Init Bwd Win Byts': 'Bwd Init Win Bytes',
        'Bwd PSH Flags': 'Bwd PSH Flags',
        'Bwd Pkt Len Max':'Bwd Packet Length Max',
        'Bwd Pkt Len Mean':'Bwd Packet Length Mean',
        'Bwd Pkt Len Min':'Bwd Packet Length Min',
        'Bwd Pkt Len Std':'Bwd Packet Length Std',
        'Bwd Pkts/b Avg': 'Bwd Packet/Bulk Avg',
        'Bwd Pkts/s': 'Bwd Packets/s',
        'Bwd Seg Size Avg': 'Bwd Segment Size Avg',
        'Bwd URG Flags': 'Bwd URG Flags',
        'CWR Flag Count': 'CWR Flag Count',
        'Down/Up Ratio': 'Down/Up Ratio',
        'Dst Port': 'Dst Port',
        'ECE Flag Cnt': 'ECE Flag Count',
        'FIN Flag Cnt': 'FIN Flag Count',
        'Init Fwd Win Byts': 'FWD Init Win Bytes',
        'Flow Byts/s': 'Flow Bytes/s',
        'Flow Duration': 'Flow Duration',
        'Flow IAT Max': 'Flow IAT Max',
        'Flow IAT Mean': 'Flow IAT Mean',
        'Flow IAT Min': 'Flow IAT Min',
        'Flow IAT Std': 'Flow IAT Std',
        'Flow Pkts/s': 'Flow Packets/s',
        'Fwd Act Data Pkts':'Fwd Act Data Pkts',
        'Fwd Blk Rate Avg': 'Fwd Bulk Rate Avg',
        'Fwd Byts/b Avg': 'Fwd Bytes/Bulk Avg',
        'Fwd Header Len': 'Fwd Header Length',
        'Fwd IAT Max': 'Fwd IAT Max',
        'Fwd IAT Mean': 'Fwd IAT Mean',
        'Fwd IAT Min': 'Fwd IAT Min',
        'Fwd IAT Std': 'Fwd IAT Std',
        'Fwd IAT Tot': 'Fwd IAT Total',
        'Fwd PSH Flags': 'Fwd PSH Flags',
        'Fwd Pkt Len Max':'Fwd Packet Length Max',
        'Fwd Pkt Len Mean':'Fwd Packet Length Mean',
        'Fwd Pkt Len Min':'Fwd Packet Length Min',
        'Fwd Pkt Len Std':'Fwd Packet Length Std',
        'Fwd Pkts/b Avg': 'Fwd Packet/Bulk Avg',
        'Fwd Pkts/s': 'Fwd Packets/s',
        'Fwd Seg Size Avg': 'Fwd Segment Size Avg',
        'Fwd Seg Size Min': 'Fwd Seg Size Min',
        'Fwd URG Flags': 'Fwd URG Flags',
        'Idle Max': 'Idle Max',
        'Idle Min': 'Idle Min',
        'Idle Mean': 'Idle Mean',
        'Idle Std': 'Idle Std',
        'Label' : 'Label',
        'PSH Flag Cnt': 'PSH Flag Count',
        'Pkt Len Max': 'Packet Length Max',
        'Pkt Len Mean': 'Packet Length Mean',
        'Pkt Len Min': 'Packet Length Min',
        'Pkt Len Std': 'Packet Length Std',
        'Pkt Len Var': 'Packet Length Variance',
        'Protocol': 'Protocol',
        'RST Flag Cnt': "RST Flag Count",
        'SYN Flag Cnt': 'SYN Flag Count',
        'Src Port': 'Src Port',
        'Subflow Bwd Byts': 'Subflow Bwd Bytes',
        'Subflow Bwd Pkts': 'Subflow Bwd Packets',
        'Subflow Fwd Byts': 'Subflow Fwd Bytes',
        'Subflow Fwd Pkts': 'Subflow Fwd Packets',
        'Tot Fwd Pkts': 'Total Fwd Packet',
        'Tot Bwd Pkts': 'Total Bwd packets',
        'TotLen Fwd Pkts': 'Total Length of Fwd Packet',
        'TotLen Bwd Pkts': 'Total Length of Bwd Packet',
        'URG Flag Cnt': 'URG Flag Count'
        }

  
le1 = LabelEncoder()
le1.fit(input_classes)

le2 = LabelEncoder()
le2.fit(dos_classes)


def check_file_exists(path):
    if not path.exists():
        print(f"The path you've supplied {path} does not exist!")
        raise typer.Exit(code=1)
    return path


@app.command()
def predict_attack(
    path: Path = typer.Argument(
        config["TEST_FILE_PATH"],
        help="The file to check Attack.",
        callback=check_file_exists,
    )
):
    """Predicts whether an input CSV file is Attack File or not.

    Warning:
        Make sure that you've trained model first.

    Args:
        file (Path): location of file. Defaults to a sample DOS File.

    Returns:
        Prediction for the input CSV.
    """
    if os.path.isdir(path):
        docs = []
        results = []
        preds_ratio = []
        at_type = []
        pbar = tqdm.tqdm(os.listdir(path))
        for file in pbar:
            pbar.set_description(file)
            dataf = pd.read_csv(f"{path}" + "/" + f"{file}")

            # Dropping irrelvant columns
            dataf = dataf[~dataf.isin([np.nan, np.inf, -np.inf]).any(1)]
            console.print(f"After filtering:{dataf.shape}", style= "bold orange_red1")

            # Dropping no info rows
            dataf = dataf[dataf['Src Port'] != 0]
            console.print(f"After filtering:{dataf.shape}", style= "bold orange_red1")

            if dataf.shape[0] == 0:
                console.print(f"{file} doesn't have valid flows")
            else:
                dataf.drop(
                    config["COLS_TO_DROP"],
                    axis=1,
                    inplace=True,
                )

                dataf.columns = dataf.columns.to_series().map(maps)
                preds = attack_model.predict(dataf)
                final = dataf.copy()
                final["Prediction"] = preds
                counts = Counter(preds)

                if any(x for x in list(counts.keys()) if x != 0):
                    attack_counts = Counter(le1.inverse_transform(preds))
                    out = attack_counts
                    ratio = attack_counts['BENIGN']/sum(attack_counts.values())
                    final.to_csv(f"output/{str(file).rsplit('/')[-1]}", index=False)
                else:
                    out = "Normal"
                    ratio = 1.0

                if any(x for x in list(counts.keys()) if x == 4):
                    out_type = "DOS"
                    req_df = final[final["Prediction"] == 4]
                    dos_pred = dos_attack_model.predict(
                        req_df.drop(["Prediction"], axis="columns")
                    )
                    dos_attack_counts = Counter(le2.inverse_transform(dos_pred))
                else:
                    out_type = "Other"

                docs.append(file)
                results.append(out)
                preds_ratio.append(ratio)
                at_type.append(out_type)
            save_results = pd.DataFrame(
                zip(docs, results, preds_ratio, at_type),
                columns=config["COLS_TO_RESULT"],
            )
            save_results.to_csv("save_results.csv", index=False)

    else:
        dataf = pd.read_csv(path)
        console.print(
            f"Shape of File: {dataf.shape[0]} Rows and, {dataf.shape[1]} Columns",
            style="bold gold1",
        )

        # Dropping irrelvant columns
        dataf = dataf[~dataf.isin([np.nan, np.inf, -np.inf]).any(1)]
        console.print(f"After filtering:{dataf.shape}", style="bold orange_red1")

        # Dropping no info rows
        dataf = dataf[dataf['Src Port'] != 0]
        console.print(f"After filtering:{dataf.shape}", style= "bold orange_red1")

        dataf.drop(
            config["COLS_TO_DROP"],
            axis=1,
            inplace=True,
        )


        preds = attack_model.predict(dataf)
        final = dataf.copy()
        final["Prediction"] = preds
        counts = Counter(preds)
        console.print(f"Counts: {counts}", style="bold yellow")

        if any(x for x in list(counts.keys()) if x != 0):
            attack_counts = Counter(le1.inverse_transform(preds))
            console.print(attack_counts)
            final.to_csv(f"output/{str(path).rsplit('/')[-1]}", index=False)
        else:
            console.print("Safe File", style="bold green")

        if any(x for x in list(counts.keys()) if x == 4):
            req_df = final[final["Prediction"] == 4]
            dos_pred = dos_attack_model.predict(
                req_df.drop(["Prediction"], axis="columns")
            )
            dos_attack_counts = Counter(le2.inverse_transform(dos_pred))
            console.print(dos_attack_counts)


if __name__ == "__main__":
    app()
