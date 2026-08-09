"""Entry point for the Grifols operations app.

Streamlit runs this file; all application code lives in the :mod:`app`
package, one sub-package per section.  Run locally with::

    streamlit run streamlit_app.py
"""

from app.main import main

if __name__ == "__main__":
    main()
