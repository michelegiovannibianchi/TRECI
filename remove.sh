
#!/bin/bash

# Set the root directory (default to current directory if not provided)
ROOT_DIR="."

# Find and delete all .IDENTIFIER files
find "$ROOT_DIR" -type f -name ".IDENTIFIER" -exec rm -v {} \;
