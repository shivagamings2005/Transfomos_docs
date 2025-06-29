import hashlib
from pymongo import MongoClient
from datetime import datetime
import pytz
import os
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
client = MongoClient("mongodb+srv://jeevajoslin:p7MK68VRoY7LvGXh@tratos.lt7g3.mongodb.net/")
db = client["TRANS2"]  
trans_collection = db["TRANS"] 
userfiles_collection = db["USERFILES"]
def generate_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
def process_file(file_name, username):
    if not os.path.exists(file_name):
        print(f"File {file_name} not found in the current directory.")
        return
    sha_key = generate_sha256(file_name)
    existing_document = trans_collection.find_one({"sha_key": sha_key})
    if existing_document:
        print(f"File {file_name} already exists in the database with SHA key: {sha_key}")
        print("Existing JSON content:")
        print(json.dumps(existing_document.get("json_file_content", {}), indent=2))  
        userfiles_collection.update_one(
            {"username": username},
            {"$addToSet": {"sha_keys": sha_key}} 
        )
        print(f"File SHA key added to existing user {username}.")
        return
    document = {
        "file_name": os.path.basename(file_name),
        "file_type": os.path.splitext(file_name)[1],
        "sha_key": sha_key,
        "processed_at": datetime.now(pytz.UTC)
    }
    result = trans_collection.insert_one(document)
    print(f"File {file_name} processed and added to the FILES collection with SHA key: {sha_key}")
    json_file_name = input("Enter the name of the JSON file to associate with this document: ")
    if not os.path.exists(json_file_name):
        print(f"JSON file {json_file_name} not found in the current directory.")
        return
    with open(json_file_name, 'r') as json_file:
        json_content = json.load(json_file)
    trans_collection.update_one(
        {"_id": result.inserted_id},
        {"$set": {"json_file_content": json_content}}
    )

    print(f"Document updated with JSON file content from: {json_file_name}")
    userfiles_collection.update_one(
            {"username": username},
            {"$addToSet": {"sha_keys": sha_key}} 
        )
    print(f"File SHA key added to existing user {username}.")

if __name__ == "__main__":
    username = input("Enter the username for this operation: ") 
    userfiles_document = userfiles_collection.find_one({"username": username})
    if not userfiles_document:
        print(f"User not found.Cannot upload files.")
        exit(0)
    file_names = input("Enter the names of the files (comma-separated if multiple)")
    file_names_list = [name.strip() for name in file_names.split(",")]
    for file_name in file_names_list:
        process_file(file_name, username)
    print("All specified files have been processed.")
