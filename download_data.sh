# #!/usr/bin/env bash

# Function to lowercase all files in the specified directory
lowercase_files_in_dir() {
    local target_dir="$1"
    
    # Check if the target directory is provided
    if [ -z "$target_dir" ]; then
        echo "Usage: lowercase_files_in_dir <target_directory>"
        return 1
    fi

    # Check if the target directory exists
    if [ ! -d "$target_dir" ]; then
        echo "Directory not found: $target_dir"
        return 1
    fi

    # Loop through all files in the target directory
    for file in "$target_dir"/*; do
        # Check if it's a file (and not a directory)
        if [ -f "$file" ]; then
            # Extract the directory part and the filename part
            dir_part=$(dirname "$file")
            base_name=$(basename "$file")
            
            # Convert the filename to lowercase
            lowercase_file=$(echo "$base_name" | tr '[:upper:]' '[:lower:]')
            
            # Rename the file if the lowercase version is different
            if [ "$base_name" != "$lowercase_file" ]; then
                mv "$file" "$dir_part/$lowercase_file"
                echo "Renamed $file to $dir_part/$lowercase_file"
            fi
        fi
    done

    echo "All files in $target_dir have been lowercased."
}

datasets_dir="./data/datasets"
mkdir -p "$datasets_dir"
pushd "$datasets_dir"

years=(2018 2019 2021)

# Loop through the list and create a data dir per year
for year in "${years[@]}"
do
    internal_dir="FARS${year}NationalCSV"
    ZIP_FILE="${internal_dir}.zip"
    output_file="${year}.zip"
    
    wget "https://static.nhtsa.gov/nhtsa/downloads/FARS/${year}/National/${ZIP_FILE}" -O "${output_file}"
    
    if [ -f "$output_file" ]; then
        if [ "$year" -le 2019 ]; then
            # Create a directory named after the year variable
            mkdir -p "$year"
            
            # Unzip the file into the created directory
            unzip "$output_file" -d "$year"
            rm "$output_file"
        else    
            unzip ${output_file}
            mv "${internal_dir}" "${year}"
            rm ${output_file}
        fi
    else
        echo "ERROR: File $output_file not found in web URL"
    fi

    lowercase_files_in_dir "${year}"
done

popd
