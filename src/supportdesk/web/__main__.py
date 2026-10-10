"""Start the web page: ``python -m supportdesk.web`` (same as ``supportdesk web``)."""
import sys

from ..cli import main

sys.exit(main(["web", *sys.argv[1:]]))
