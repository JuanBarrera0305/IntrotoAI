import pandas as pd

# Load CSV
df = pd.read_csv("Heats_alpha_beta.csv")
df.columns = df.columns.str.strip()  # Clean whitespace in headers

# ---- Automatic Qualifiers: Top 2 per heat ----
auto_qualifiers = (
    df.sort_values(by=["Heat", "Seed Time"])
      .groupby("Heat")
      .head(2)
)

# ---- Remaining athletes ----
remaining = df[~df.index.isin(auto_qualifiers.index)]

# ---- Time Qualifiers: Next 6 fastest remaining ----
time_qualifiers = (
    remaining.sort_values(by="Seed Time")
             .head(6)
)

# ---- Combine ----
qualified_df = pd.concat([auto_qualifiers, time_qualifiers], ignore_index=False)

# Add Qualification label
qualified_df["Qualification"] = ["Auto" if i in auto_qualifiers.index else "Time"
                                 for i in qualified_df.index]

# Sort final list for nice output
qualified_df = qualified_df.sort_values(by=["Qualification", "Heat", "Seed Time"])

# ---- SAVE CSV FILE ----
output_file = "Semifinals_Qualifiers.csv"
qualified_df.to_csv(output_file, index=False)

print("=== Qualified Semifinalists (32) ===")
print(qualified_df[["Name","School","Heat","Lane","Seed Time","Qualification"]])
print(f"\nTotal Qualified: {qualified_df.shape[0]}")
print(f"\n✔ CSV has been created: {output_file}")
