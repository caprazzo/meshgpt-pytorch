#!/bin/bash

# Path to the cache file
CACHE_FILE="training_host_cache.txt"

# Function to determine TRAINING_HOST
get_training_host() {
    # Check if cache file exists
    if [ -f "$CACHE_FILE" ]; then
        # Read the TRAINING_HOST from the cache file
        TRAINING_HOST=$(cat "$CACHE_FILE")
        echo "Loaded TRAINING_HOST from cache: $TRAINING_HOST"
    else
        # Determine the TRAINING_HOST value
        TRAINING_HOST=$(gcloud compute instances list --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
        
        # Save the TRAINING_HOST to the cache file
        echo "$TRAINING_HOST" > "$CACHE_FILE"
        echo "Saved TRAINING_HOST to cache: $TRAINING_HOST"
    fi
}

# Get TRAINING_HOST
get_training_host

# Set JUPYTER_URL and export TRAINING_HOST
JUPYTER_URL=http://$TRAINING_HOST:8081
export TRAINING_HOST

# Example usage
echo "TRAINING_HOST: $TRAINING_HOST"
echo "JUPYTER_URL: $JUPYTER_URL"
