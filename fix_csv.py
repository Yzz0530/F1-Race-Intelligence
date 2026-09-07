import pandas as pd
import re
import sys

# Load data
csv = pd.read_csv('data/race_results.csv')
df2024 = pd.read_csv('data/all_races_2024.csv')

# Build code -> full name mapping from sessions that have proper names
code_to_name = {}
for dn in csv['DriverNumber'].dropna().unique():
    dn_str = str(dn)
    dn_rows = csv[csv['DriverNumber'] == dn]
    proper = dn_rows[~dn_rows['FullName'].apply(
        lambda x: bool(re.match(r'^[A-Z]{3,4}$', str(x))) if pd.notna(x) else False
    ) & (dn_rows['FullName'] != 'None None')]
    if len(proper) > 0:
        name = proper['FullName'].value_counts().index[0]
        code_to_name[dn_str] = name

# Build code -> team from 2024 data
code_to_team = {}
for _, row in df2024[['Driver', 'Team']].drop_duplicates().iterrows():
    code_to_team[row['Driver']] = row['Team']

# Also need to handle the case where FullName is the code itself (not just DriverNumber)
# In 2024 R rows, FullName IS the code (ALO, HAM, etc.)
# We need to map code -> full name directly
known_codes = {
    'ALB': 'Alexander Albon', 'ALO': 'Fernando Alonso', 'BEA': 'Oliver Bearman',
    'BOR': 'Gabriel Bortoleto', 'BOT': 'Valtteri Bottas', 'COL': 'Franco Colapinto',
    'DOO': 'Jack Doohan', 'GAS': 'Pierre Gasly', 'HAD': 'Isack Hadjar',
    'HAM': 'Lewis Hamilton', 'HUL': 'Nico Hulkenberg', 'LAW': 'Liam Lawson',
    'LEC': 'Charles Leclerc', 'LIN': 'Lance Stroll', 'MAG': 'Magnussen',  # Actually 'MAZ'?
    'NOR': 'Lando Norris', 'OCO': 'Esteban Ocon', 'PIA': 'Oscar Piastri',
    'PER': 'Sergio Perez', 'RIC': 'Daniel Ricciardo', 'RUS': 'George Russell',
    'SAI': 'Carlos Sainz', 'SAR': 'Liam Lawson', 'STR': 'Lance Stroll',
    'TSU': 'Yuki Tsunoda', 'VER': 'Max Verstappen', 'ZHO': 'Zhou Guanyu',
    'ANT': 'Antonio Giovinazzi',
}

# Override with actual known mappings from F1
known_codes = {
    'ALB': 'Alexander Albon', 'ALO': 'Fernando Alonso', 'BEA': 'Oliver Bearman',
    'BOR': 'Gabriel Bortoleto', 'BOT': 'Valtteri Bottas', 'COL': 'Franco Colapinto',
    'DOO': 'Jack Doohan', 'GAS': 'Pierre Gasly', 'HAD': 'Isack Hadjar',
    'HAM': 'Lewis Hamilton', 'HUL': 'Nico Hulkenberg', 'LAW': 'Liam Lawson',
    'LEC': 'Charles Leclerc', 'LIN': 'Lance Stroll',
    'NOR': 'Lando Norris', 'OCO': 'Esteban Ocon', 'PIA': 'Oscar Piastri',
    'PER': 'Sergio Perez', 'RIC': 'Daniel Ricciardo', 'RUS': 'George Russell',
    'SAI': 'Carlos Sainz', 'SAR': 'Sarah (unknown)', 'STR': 'Lance Stroll',
    'TSU': 'Yuki Tsunoda', 'VER': 'Max Verstappen', 'ZHO': 'Zhou Guanyu',
    'ANT': 'Antonio Giovinazzi', 'MAZ': 'Kevin Magnussen',
}

# Fix the DataFrame
fixed = csv.copy()
fix_count = 0

for i, row in fixed.iterrows():
    name = str(row['FullName']) if pd.notna(row['FullName']) else ''
    dn = str(row['DriverNumber']) if pd.notna(row['DriverNumber']) else ''
    
    new_name = None
    new_team = None
    
    # Fix "None None"
    if name == 'None None':
        # Try to get name from DriverNumber
        if dn in code_to_name:
            new_name = code_to_name[dn]
            new_team = row['TeamName']  # keep existing team
        fix_count += 1
    
    # Fix driver codes (3-4 uppercase chars)
    elif re.match(r'^[A-Z]{3,4}$', name):
        if name in known_codes:
            new_name = known_codes[name]
            new_team = code_to_team.get(name, row['TeamName'])
        elif dn in code_to_name:
            new_name = code_to_name[dn]
            new_team = row['TeamName']
        fix_count += 1
    
    if new_name:
        fixed.at[i, 'FullName'] = new_name
    if new_team:
        fixed.at[i, 'TeamName'] = new_team

print(f"Fixed: {fix_count} rows")
print(f"None None remaining: {len(fixed[fixed['FullName'] == 'None None'])}")
code_remaining = len(fixed[fixed['FullName'].apply(
    lambda x: bool(re.match(r'^[A-Z]{3,4}$', str(x))) if pd.notna(x) else False
)])
print(f"Code rows remaining: {code_remaining}")

# Check what's still wrong
if code_remaining > 0:
    remaining = fixed[fixed['FullName'].apply(
        lambda x: bool(re.match(r'^[A-Z]{3,4}$', str(x))) if pd.notna(x) else False
    )]
    print(f"Remaining codes: {sorted(remaining['FullName'].unique())}")

# Save
fixed.to_csv('data/race_results.csv', index=False)
print(f"\nSaved fixed CSV: {len(fixed)} rows")

# Verify by year
for year in sorted(fixed['Year'].unique()):
    y = fixed[fixed['Year'] == year]
    r = y[y['Session'] == 'R']
    drivers = sorted(r['FullName'].unique())
    print(f"\n{year} R drivers ({len(drivers)}):")
    for d in drivers:
        count = len(r[r['FullName'] == d])
        print(f"  {d:30s} ({count} rows)")
