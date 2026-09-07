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
    
    # Load 2024 data for team info (has Team column)
    team_info = {}
    try:
        df_2024 = pd.read_csv('data/all_races_2024.csv')
        if 'Team' in df_2024.columns and 'Driver' in df_2024.columns:
            team_info = df_2024[['Driver', 'Team']].drop_duplicates().set_index('Driver')['Team'].to_dict()
            print(f"Loaded team info for {len(team_info)} drivers from 2024 data")
    except Exception as e:
        print(f"Warning: Could not load 2024 team info: {e}")
    
    # For each year not in existing, extract race results
    new_rows = []
    
    for year in sorted(master["Year"].unique()):
        # Skip if year exists AND has valid data (team info AND points)
        if year in years_present:
            # Check if this year has valid data
            year_data = existing[existing["Year"] == year]
            unknown_teams = year_data["TeamName"].value_counts().get("Unknown", 0)
            
            # Check if points are present
            missing_points = year_data["Points"].isna().sum()
            total_rows = len(year_data)
            
            if unknown_teams == 0 and missing_points < total_rows * 0.1:
                print(f"\nYear {year} already exists with valid data, skipping...")
                continue
            else:
                print(f"\nYear {year} needs regeneration (Unknown: {unknown_teams}, Missing points: {missing_points}/{total_rows})")
        else:
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
                
                # Get team from 2024 data or use default
                team = team_info.get(driver, "Unknown")
                
                # Get best lap time
                driver_laps = race_laps[race_laps["Driver"] == driver]
                best_lap = driver_laps["LapTime"].min()
                
                # Get total laps completed
                total_laps = len(driver_laps)
                
                # Convert normalized position to actual position
                # Position_normalized is 0-1, where 0 is first place
                # Get actual position by finding rank among all drivers for this race
                race_positions = race_laps_filtered.groupby("Driver")["Position_normalized"].last().reset_index()
                race_positions = race_positions.sort_values("Position_normalized")
                race_positions["ActualPosition"] = range(1, len(race_positions) + 1)
                pos_map = dict(zip(race_positions["Driver"], race_positions["ActualPosition"]))
                actual_position = pos_map.get(driver, 21)
                
                # Ensure position is reasonable
                if actual_position < 1:
                    actual_position = 1
                elif actual_position > 25:
                    actual_position = 22  # Beyond points
                
                # Calculate F1 points based on position
                points_table = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
                points = points_table.get(actual_position, 0)
                
                # Status: DNF if very few laps
                if total_laps < 5:
                    status = "DNF"
                elif actual_position > 20:
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
                    "Points": points,  # F1 points based on position
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
