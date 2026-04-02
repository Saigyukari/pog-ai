# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import os
import time
import urllib.parse

# ===== Configuration =====
BASE_URL = "http://123.207.48.44:8080"
# List of expert players to scan and match against
EXPERTS = [
    "爱城华恋", 
    "bolanqiu", 
    "scpcpc", 
    "RRRRex", 
    "Sai", 
    "Rex", 
    "星絮", 
    "RRRRRRRRRRex", 
    "RRRex",
    "AnonTokyo",
    "Kaname Madoka",
    "Noooooob"
]
OUTPUT_DIR = "data/rtt_games"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_expert_games():
    total_downloaded = 0
    
    for target_player in EXPERTS:
        print(f"\n--- Scanning finished games for player: [{target_player}] ---")
        
        # 1. Construct the profile URL
        encoded_player = urllib.parse.quote(target_player)
        profile_url = f"{BASE_URL}/games/finished/{encoded_player}"
        
        try:
            response = requests.get(profile_url)
            if response.status_code != 200:
                print(f"Error: Cannot access profile page for {target_player}. Status: {response.status_code}")
                continue
        except Exception as e:
            print(f"Network error while accessing profile for {target_player}: {e}")
            continue

        # 2. Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all anchor tags pointing to Paths of Glory replays
        game_links = soup.find_all('a', href=lambda href: href and "paths-of-glory/play.html?game=" in href)
        
        player_download_count = 0

        for link in game_links:
            href = link['href']
            # Extract game ID
            game_id = href.split("game=")[-1].split("&")[0]
            
            # Extract text to check for specific opponents from the whole game container
            container = link.find_parent('div', class_='game_item')
            game_text = container.get_text() if container else link.get_text() + str(link.parent.get_text())
            
            # 3. Filter logic: check if opponent is also an expert
            is_target_match = False
            matched_opp = ""
            for opp in EXPERTS:
                # Don't match the player against themselves in the text search 
                # (though the server logic prevents self-play, "bolanqiu" is in "bolanqiu vs Rex")
                if opp != target_player and opp in game_text:
                    is_target_match = True
                    matched_opp = opp
                    break
                    
            if is_target_match:
                # 4. Construct the actual JSON API URL
                api_url = f"{BASE_URL}/api/replay/{game_id}"
                save_path = os.path.join(OUTPUT_DIR, f"game_{game_id}.json")
                
                # Skip if already downloaded
                if os.path.exists(save_path):
                    # Silently skip to avoid noise, or print if debugging
                    continue
                    
                print(f"Found match! Game ID: {game_id} ({target_player} vs {matched_opp})")
                
                # 5. Download the JSON data
                try:
                    api_response = requests.get(api_url)
                    if api_response.status_code == 200:
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(api_response.text)
                        print(f" -> Successfully saved to: {save_path}")
                        player_download_count += 1
                        total_downloaded += 1
                        
                        # Polite delay
                        time.sleep(1.0)
                    else:
                        print(f" -> Download failed. API Status Code: {api_response.status_code}")
                except Exception as e:
                    print(f" -> Network error during JSON download: {e}")

        print(f"Completed scan for {target_player}. Downloaded {player_download_count} new game(s).")

    print(f"\nAll scans complete! Successfully downloaded {total_downloaded} new expert game(s) in this session.")

if __name__ == "__main__":
    download_expert_games()
