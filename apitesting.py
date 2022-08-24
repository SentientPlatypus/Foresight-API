import requests
import app.constants as constants

data = requests.get("https://ForesightAPI.sentientplatypu.repl.co").json()
print(data)
print(data["maidens"])