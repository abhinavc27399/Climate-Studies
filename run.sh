#!/bin/bash

# Find all main.py files in immediate subdirectories
echo "🔍 Searching for main.py files in subdirectories..."
options=()
for dir in */; do
    if [[ -f "$dir/main.py" ]]; then
        options+=("${dir%/}")
    fi
done

# If no main.py found
if [ ${#options[@]} -eq 0 ]; then
    echo "❌ No subdirectories with main.py found."
    exit 1
fi

# Present menu
echo "Select which main.py you want to run:"
select opt in "${options[@]}"; do
    if [[ -n "$opt" ]]; then
        echo "▶️ Running: $opt/main.py"
        python "$opt/main.py"
        break
    else
        echo "❌ Invalid option. Please try again."
    fi
done
