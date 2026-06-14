import pandas as pd


def create_clv_segments(df: pd.DataFrame):

    print("\nCreating CLV segments...")

    q1 = df["CLV"].quantile(0.25)
    q2 = df["CLV"].quantile(0.50)
    q3 = df["CLV"].quantile(0.75)

    def segment(value):

        if value <= q1:
            return "Low Value"

        elif value <= q2:
            return "Medium Value"

        elif value <= q3:
            return "High Value"

        else:
            return "VIP"

    df["CLV_Segment"] = df["CLV"].apply(segment)

    print("CLV segmentation completed")

    return df