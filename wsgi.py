import os
from dotenv import load_dotenv

# Load env variables from .env if present
load_dotenv()

from home import app

if __name__ == "__main__":
    app.run()
