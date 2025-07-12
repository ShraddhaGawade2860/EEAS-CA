from preprocess import get_input_data, de_duplication, noise_remover
from modelling.modelling import train_type2, train_type3, train_type4
from embeddings import *
from modelling.data_model import *
from Config import Config
import numpy as np
import pandas as pd
import random
import os

random.seed(0)
np.random.seed(0)

OUTPUT_DIR = "final_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("🔄 Loading and preprocessing data...")
    df = get_input_data()
    df = de_duplication(df)
    df = noise_remover(df)

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype("U")
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype("U")

    print("Data loaded and cleaned.\n")

 # Level 1: Type 2 
    print("Training Level 1 - Type 2...")
    df = train_type2(df)
    if df is None:
        print("No valid Type 2 predictions. Exiting pipeline.")
        return
    df.to_csv(os.path.join(OUTPUT_DIR, "type2_predictions.csv"), index=False)

    #  Level 2 & 3: Type 3 & Type 4 
    all_type4 = []

    print("\nTraining Level 2 and 3 - Type 3 and Type 4...")
    for t2_class, df_t3 in train_type3(df):
        df_t3 = df_t3.copy()
        train_type4(df_t3, t2_class)
        all_type4.append(df_t3)

    # Combine 
    if all_type4:
        final_df = pd.concat(all_type4, ignore_index=True)
        final_output_path = os.path.join(OUTPUT_DIR, "hierarchical_output.csv")
        final_df.to_csv(final_output_path, index=False)
        print(f"\nFinal hierarchical predictions saved to: {final_output_path}")
    else:
        print("No valid Type 4 predictions generated.")

    print("\n Pipeline completed successfully.")
    


if __name__ == "__main__":
    main()
