"""
The `dies-app` command (installed by pyproject.toml): starts the DiES app in the browser,
as `streamlit run dies/app/streamlit_app.py` would. Extra arguments go to Streamlit,
e.g. `dies-app --server.port 8502`.
"""

import os
import sys


def main():
    """Entry point of `dies-app`."""
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        sys.stderr.write('The DiES app needs Streamlit: pip install -e .[app] (or pip install streamlit)\n')
        return 1
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'streamlit_app.py')
    # Streamlit reads its command line from sys.argv
    sys.argv = ['streamlit', 'run', app] + sys.argv[1:]
    return streamlit_cli.main()


if __name__ == '__main__':
    sys.exit(main())
