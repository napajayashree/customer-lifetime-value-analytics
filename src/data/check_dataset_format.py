import csv
import os
import pandas as pd

file_path = "data/raw/online_retail.xlsx"  # works for csv/xlsx/xls

extension = os.path.splitext(file_path)[1].lower()

print("\n===== FILE FORMAT CHECK =====")

if extension in [".xlsx", ".xls"]:

    print("\nExcel file detected\n")

    try:
        df = pd.read_excel(file_path)

        print("===== DATASET PREVIEW =====")
        print(df.head(10))

        print("\n===== DATASET INFO =====")
        print(df.info())

        print("\n===== COLUMN NAMES =====")
        print(df.columns.tolist())

        print("\n===== MISSING VALUES =====")
        print(df.isnull().sum())

        print("\n===== SHAPE =====")
        print(df.shape)

    except Exception as e:
        print("Error reading Excel file:", e)


elif extension == ".csv":

    print("\nCSV file detected\n")

    print("===== RAW FILE PREVIEW =====")

    with open(file_path, "r", encoding="ISO-8859-1") as f:
        for i in range(10):
            line = f.readline()
            print(f"Line {i+1}: {line.strip()}")

    print("\n===== DELIMITER DETECTION =====")

    with open(file_path, "r", encoding="ISO-8859-1") as f:
        sample = f.read(5000)
        sniffer = csv.Sniffer()

        try:
            dialect = sniffer.sniff(sample)
            delimiter = dialect.delimiter
            print("Detected delimiter:", repr(delimiter))
        except:
            delimiter = ","
            print("Could not detect delimiter automatically, assuming ','")

    print("\n===== COLUMN CONSISTENCY CHECK =====")

    expected_columns = None
    bad_rows = []

    with open(file_path, "r", encoding="ISO-8859-1") as f:
        reader = csv.reader(f, delimiter=delimiter)

        for i, row in enumerate(reader):

            column_count = len(row)

            if expected_columns is None:
                expected_columns = column_count

            if column_count != expected_columns:
                bad_rows.append((i + 1, column_count))

            if i > 1000:
                break

    print(f"Expected number of columns: {expected_columns}")

    if bad_rows:
        print("\nRows with inconsistent column counts:")
        for r in bad_rows[:10]:
            print(f"Row {r[0]} → {r[1]} columns")
    else:
        print("All rows appear consistent.")


else:
    print("Unsupported file format. Please use CSV, XLSX, or XLS.")