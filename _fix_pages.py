from pathlib import Path

pages = [
    "app/1_Data_Upload.py",
    "app/2_Preprocessing.py",
    "app/3_Model_Training.py",
    "app/4_Explainability.py",
    "app/5_Score_New_Data.py",
    "app/6_Fairness_Audit.py",
    "app/7_Credit_Analyst_Agent.py",
    "app/8_Business_Summary.py",
]

guard = 'if __name__ == "__main__":\n    main()'
replacement = "main()"

for path in pages:
    with open(path, encoding="utf-8") as f:
        src = f.read()

    if guard in src:
        src = src.replace(guard, replacement)
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"Fixed: {path}")
    else:
        print(f"WARNING — guard not found in: {path}")
