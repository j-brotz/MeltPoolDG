LOG_FILE="install.log"

# Function to log messages
log() {
    echo "MELTPOOLDG:: $1" | tee -a "$LOG_FILE"
}
