#! /bin/bash

# stop the script if not executed from `attendance-system` directory
if [ "$(basename "$(pwd)")" != "attendance-system" ]; then
    echo "Error: This script must be executed from the 'attendance-system' directory"
    exit 1
fi

# delete all the files in the enrollment_images directory
rm -rf enrollment_images/*

# delete the faiss_index.bin file
rm -f faiss_index.bin

# delete the faiss_user_id_map.npy file
rm -f faiss_user_id_map.npy

# delete the attendance_system.db file
rm -f attendance_system.db

uv run src/db/index.py

# this starts a flask server, I want to end this after the curl command
gunicorn --workers 1 --bind 0.0.0.0:1337 src.api.index:app &

# wait for 5 seconds
sleep 4

curl -X POST http://localhost:1337/setup-rooms

# kill the flask server
pkill -f "gunicorn --workers 1 --bind 0.0.0.0:1337 src.api.index:app"