"""Generate race_results.csv from all_races_master.csv for Results/Standings tabs."""
import pandas as pd
import os

def main():
    print("Generating race_results.csv from all_races_master.csv...")
    
    # Load existing race_results.csv if exists
    existing_path = "data/race_results.csv"
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        print(f"Existing: {len(existing)} rows")
        years_present = sorted(existing["Year"].unique())
        print(f"Years present: {years_present}")
    else:
        existing = pd.DataFrame()
        print("No existing race_results.csv")
    
    # Load master data
    master_path = "data/all_races_master.csv"
    master = pd.read_csv(master_path)
    print(f"\nMaster file: {len(master)} rows")
    print(f"Years in master: {sorted(master['Year'].unique())}")
    
    # For each year not in existing, extract race results
    new_rows = []
    
    for year in sorted(master["Year"].unique()):
        if year in years_present:
            print(f"\nYear {year} already exists, skipping...")
            continue
        
        print(f"\nProcessing year {year}...")
        year_laps = master[master["Year"] == year]
        
        # Get unique races
        races = year_laps["Race"].unique()
        print(f"  Found {len(races)} races")
        
        for race in races:
            race_laps = year_laps[year_laps["Race"] == race]
            
            # Get race session (TrackStatus=1 for running)
            race_laps_filtered = race_laps[race_laps["TrackStatus"] == 1]
            
            if race_laps_filtered.empty:
                print(f"  {race}: No race laps found")
                continue
            
            # Get final position for each driver by finding max LapInRace
            last_laps = race_laps_filtered.sort_values("LapInRace").groupby("Driver", as_index=False).last()
            
            # Extract race results
            for _, row in last_laps.iterrows():
                driver = row["Driver"]
                position = row["Position_normalized"]  # Normalized position (0-1)
                team = row.get("Team", "Unknown") if "Team" in race_laps.columns else "Unknown"
                
                # Get best lap time
                driver_laps = race_laps[race_laps["Driver"] == driver]
                best_lap = driver_laps["LapTime"].min()
                
                # Get total laps completed
                total_laps = len(driver_laps)
                
                # Convert normalized position to actual position
                # Position_normalized is 0-1, where 0 is first place
                # We need to reverse it
                actual_position = int((1 - position) * 20) + 1  # Approximate
                if actual_position < 1:
                    actual_position = 1
                elif actual_position > 20:
                    actual_position = 20
                
                # Status: DNF if very few laps or position is extreme
                if total_laps < 5 or actual_position > 20:
                    status = "DNF"
                else:
                    status = "Finished"
                
                new_rows.append({
                    "Year": year,
                    "Race": race,
                    "Session": "R",
                    "DriverNumber": "",  # Not in master data
                    "FullName": driver,
                    "TeamName": team,
                    "Position": actual_position,
                    "GridPosition": "",  # Not available
                    "Status": status,
                    "Time": "",  # Not available
                    "Points": "",  # Will calculate later
                    "Q1": "",
                    "Q2": "",
                    "Q3": "",
                    "BestLapTime": best_lap,
                    "Laps": total_laps,
                })
            
            print(f"  {race}: {len(last_laps)} drivers")
    
    # Create DataFrame
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        
        # Add to existing
        if not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        
        # Save
        combined.to_csv(existing_path, index=False)
        print(f"\nSaved {len(combined)} rows to {existing_path}")
        print(f"Years: {sorted(combined['Year'].unique())}")
        
        # Show sample
        print(f"\nSample rows:")
        print(combined.head(10)[['Year', 'Race', 'FullName', 'TeamName', 'Position', 'Status']].to_string())
    else:
        print("\nNo new data to add")

if __name__ == "__main__":
    main()
