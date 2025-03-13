#!/bin/bash

# Function to log messages
log() {
    # Ensure MPDG_LOG_FILE is defined only once and persists across script calls
    if [ -z "$MPDG_LOG_FILE" ]; then
        export MPDG_LOG_FILE="${MPDG_LOG_FILE:-install.log}"
        export MPDG_LOG_FILE=$(realpath "$MPDG_LOG_FILE")
        : > "$MPDG_LOG_FILE"  # Create and clear the log file
        echo "MELTPOOLDG:: log of installation will be written to $MPDG_LOG_FILE" | tee -a "$MPDG_LOG_FILE"
    fi

    # Log the message without clearing the file
    echo "MELTPOOLDG:: $1" | tee -a "$MPDG_LOG_FILE"
}
