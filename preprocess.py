import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\miniproject\processed_wind_data.csv')

# Print column names to verify
print("Columns in dataset:", df.columns)

# Check if 'Datetime' exists
if 'Datetime' in df.columns:
    # Try parsing different date formats
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d-%m-%Y %H:%M", errors='coerce')
    except:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')  # Auto-detect format
    
    # Check for any NaT (failed conversions)
    if df['Datetime'].isna().sum() > 0:
        print("⚠️ Warning: Some dates could not be converted. Check your dataset!")

    # Convert to Unix timestamp (numerical format)
    df['Timestamp'] = df['Datetime'].astype('int64') // 10**9  # Convert to seconds

    # Drop the original 'Datetime' column
    df.drop(columns=['Datetime'], inplace=True)

    # Save the preprocessed dataset
    df.to_csv(r'C:\miniproject\preprocessed_wind_data.csv', index=False)

    print("✅ Dataset preprocessed and saved as 'preprocessed_wind_data.csv'!")
else:
    print("❌ Error: 'Datetime' column is missing from the dataset!")
