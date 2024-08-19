# Guide to Setting Up a Google Cloud Free Tier Account, VM, Storage Bucket, and Service Account

This guide will walk you through the process of opening a free-tier Google Cloud account, creating an `n2-standard-2` Ubuntu 22.04 virtual machine (VM) with 50 GB of storage, setting up a storage bucket, creating a service account with a service account key, and assigning read permissions for all objects in the bucket.

## Step 1: Open a Free Tier Google Cloud Account

1. **Visit Google Cloud Platform (GCP)**: Go to the [Google Cloud homepage](https://cloud.google.com/).
2. **Sign Up**: Click on the "Get started for free" button. Sign in with your Google account, or create a new one if you don't have one.
3. **Set Up Billing**: Enter your billing details and agree to the terms of service. Google requires billing information, but you'll have $300 in free credits and won’t be charged as long as you stay within the free tier limits.
4. **Activate Free Tier**: Once your account is set up, you'll have access to various free tier products.

## Step 2: Create an SSH Key Pair for Secure VM Access

1. **Generate an SSH Key Pair on Your Local Machine**:
   - On Linux or macOS:
     ```bash
     ssh-keygen -t rsa -b 4096 -f ~/.ssh/gcp_ssh_key -C "your_email@example.com"
     ```
   - On Windows (using Git Bash or PowerShell):
     ```bash
     ssh-keygen -t rsa -b 4096 -f C:\Users\YourUsername\.ssh\gcp_ssh_key -C "your_email@example.com"
     ```
   - Press `Enter` to skip setting a passphrase.
   - The public key will be saved as `gcp_ssh_key.pub` and the private key as `gcp_ssh_key`.

2. **Copy the Public Key**:
   - Open the `gcp_ssh_key.pub` file and copy its content.

## Step 3: Create a Virtual Machine (VM)

1. **Navigate to the Compute Engine**: From the Google Cloud Console dashboard, click on the hamburger menu (≡) in the top-left corner. Select "Compute Engine" from the menu, then "VM instances."
2. **Enable Compute Engine**: If prompted, click "Enable" to activate the Compute Engine API.
3. **Create a New VM Instance**:
   - Click the "Create Instance" button.
   - **Name your VM**: Enter a name for your VM instance.
   - **Choose a Region and Zone**: Select a region and zone close to your location.
   - **Machine Configuration**:
     - Under "Machine configuration," select "General-purpose" and choose the `n2-standard-2` machine type (2 vCPUs, 8 GB memory).
   - **Boot Disk**:
     - Click "Change" next to "Boot disk."
     - Choose "Ubuntu" from the Operating System dropdown.
     - Select "Ubuntu 22.04 LTS" for the version.
     - Set the disk size to 50 GB.
   - **Identity and API access**:
        - Under "Access Scope", Choose "Set access for each API":
          - Set "Storage" as **"Read Write"**
          - Keep all the rest of the settings as is 
   - **Firewall Settings**:
     - Optionally, check "Allow HTTP traffic" and "Allow HTTPS traffic" if you need to serve web traffic.
   - **SSH Keys**:
     - Scroll down to the "Management, security, disks, networking, sole tenancy" section.
     - Click on "Security."
     - Under "SSH Keys," click "Show advanced options."
     - Paste the contents of your `gcp_ssh_key.pub` file into the "Enter SSH Key" field.
   - **Create the Instance**: Click "Create" to start your VM.

## Step 4: Set Up a Storage Bucket

1. **Navigate to Cloud Storage**: From the hamburger menu (≡), select "Storage" and then "Browser."
2. **Create a Bucket**:
   - Click "Create bucket."
   - **Name your bucket**: Enter a globally unique name.
   - **Location Type**: Choose a location type (e.g., Multi-region, Region).
   - **Storage Class**: Choose the appropriate storage class based on your needs (Standard is typically fine).
   - **Access Control**: Choose how you want to control access to objects in the bucket (e.g., Uniform for the same permissions across all objects).
   - **Finalize**: Click "Create" to create your bucket.

## Step 5: Create a Service Account

1. **Navigate to IAM & Admin**: From the hamburger menu (≡), go to "IAM & Admin" and select "Service accounts."
2. **Create a Service Account**:
   - Click "Create Service Account."
   - **Service Account Details**:
     - Enter a name for your service account.
     - Optionally, add a description.
   - **Service Account Permissions**:
     - Under "Grant this service account access to project," choose "Storage Object Viewer" to allow the service account to read objects in the bucket.
   - **Create Key**:
     - Before finalizing, click "Create Key."
     - Choose the key type as JSON and download the key file. This file contains the credentials needed for your applications to authenticate with Google Cloud services using the service account.
   - **Finalize**: Click "Done."

## Step 6: Grant Read Permissions to the Service Account for the Bucket

1. **Go to the Bucket Permissions**: Navigate back to the Cloud Storage bucket you created earlier.
2. **Manage Permissions**:
   - Click on the "Permissions" tab.
   - Click "Add" to add a new member.
   - **Enter the Service Account**: In the "New members" field, enter the email address of the service account you created.
   - **Assign a Role**: From the "Select a role" dropdown, choose "Storage Object Viewer."
   - **Save**: Click "Save" to apply the permissions.

## Step 7: Test the Setup

1. **Access the VM**: Go back to Compute Engine, select your VM instance, and click "SSH" to connect to it.
   - Alternatively, you can SSH into the VM from your local machine using the private key:
     ```bash
     ssh -i ~/.ssh/gcp_ssh_key username@external-ip-address
     ```
2. **Use the Service Account Key**:
   - Upload the service account key JSON file to your VM.
   - Install the Google Cloud SDK on your VM and authenticate using the service account key if needed.
3. **Verify Bucket Access**:
   - Use the `gsutil` command to list objects in the bucket:
     ```bash
     gsutil ls gs://your-bucket-name/
     ```

## Conclusion

You’ve successfully created a free-tier Google Cloud account, launched an Ubuntu 22.04 VM with 50 GB of storage, set up a storage bucket, and configured a service account with appropriate permissions. Your setup is now ready for various cloud-based tasks and services.
