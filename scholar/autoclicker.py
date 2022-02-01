
# xdotool search --name "Google Chrome" windowfocus --sync key ctrl+Return key ctrl+Return

import subprocess
import time

while True:
    subprocess.run(['xdotool', 'getactivewindow', 'search' , '--name', "Google Chrome", "windowactivate", "--sync", "key", "ctrl+Return", "key", "ctrl+Return", 'windowactivate', '%1'])
    time.sleep(600)
