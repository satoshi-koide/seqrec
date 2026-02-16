#!/bin/bash

# Directory name to save
DOWNLOAD_DIR="dataset_archive"

# Main process as function
download_and_extract() {
    local GDRIVE_FOLDER_ID="$1"
    local DOWNLOAD_DIR="$2"
    
    if [ -z "$GDRIVE_FOLDER_ID" ]; then
        echo "Error: Please specify Google Drive folder ID."
        echo "Usage: $0 <GDRIVE_FOLDER_ID>"
        exit 1
    fi
    
    # 1. Check and install gdown
    if ! command -v gdown &> /dev/null; then
        echo "gdown not found. Installing..."
        pip install gdown
        export PATH=$PATH:$HOME/.local/bin
    else
        echo "gdown is already installed."
    fi

    # 2. Download entire folder
    echo "Downloading folder from Google Drive..."
    gdown --folder https://drive.google.com/drive/folders/$GDRIVE_FOLDER_ID -O $DOWNLOAD_DIR --remaining-ok

    if [ ! -d "$DOWNLOAD_DIR" ]; then
        echo "Error: Download directory not found. Please check the ID."
        exit 1
    fi

    # 3. Extract archives
    echo "Starting extraction process..."
    cwd=$(pwd)
    cd $DOWNLOAD_DIR

    find . -maxdepth 1 -name "*.part00" | while read filename; do
        basename=$(echo $filename | sed 's/\.part00$//')
        category=$(echo $basename | sed 's/\.\///' | sed 's/\.tar//')
        
        echo "Processing category: $category"
        
        cat ${basename}.part* | tar xvf -
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully extracted $category."
        else
            echo "❌ Failed to extract $category."
        fi
    done

    mv $category images
    cd $cwd
    echo "All processes completed."
}

# Call function (pass command line argument)

download_and_extract 1xdm8cmN9A_6dNxY5d2HRPQnykLcSBZT6 dataset/toys
download_and_extract 1hBCx7zpQXo4MG_uDjOByTWs8VokH0rHD dataset/sports
download_and_extract 1o8CTIuAqEtn3UtpTNLKrRGJtQ-PIqp1G dataset/beauty
