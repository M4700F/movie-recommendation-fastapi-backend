import pandas as pd
import re
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
# Use transaction mode (port 6543) or direct connection
DB_HOST = "aws-1-us-east-1.pooler.supabase.com"
DB_PORT = "6543"  # Changed to transaction mode
DB_NAME = "postgres"

# Create engine with connection pooling disabled
engine = create_engine(
    f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}',
    poolclass=None  # Disable connection pooling
)

csv_path = "content_movie_list.csv"
df = pd.read_csv(csv_path)

def extract_title_and_year(title):
    match = re.search(r'^(.*)\s+\((\d{4})\)$', title)
    if match:
        clean_title = match.group(1).strip()
        year = int(match.group(2))
    else:
        clean_title = title.strip()
        year = None
    return pd.Series([clean_title, year])

df[['title', 'release_year']] = df['title'].apply(extract_title_and_year)
df = df[['title', 'release_year', 'genres']]

# Use context manager to ensure connection is closed
with engine.connect() as conn:
    df.to_sql('movies', conn, if_exists='append', index=False)

engine.dispose()  # Clean up all connections

print(f"✅ Successfully imported {len(df)} movies to Supabase!")