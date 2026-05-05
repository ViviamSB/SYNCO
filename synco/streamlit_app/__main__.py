"""SYNCO Streamlit App Entry Point

This module allows running the SYNCO dashboard with:
    python -m synco.streamlit_app

Or from the synco package root with:
    streamlit run synco/streamlit_app/app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit app."""
    # Get the path to the app
    app_path = Path(__file__).parent / "app.py"

    # Run streamlit
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=True,
        )
    except FileNotFoundError:
        print("Error: Streamlit is not installed. Please install it with:")
        print("  pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
