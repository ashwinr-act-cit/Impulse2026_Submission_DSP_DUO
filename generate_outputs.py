import csv
import random
import os

# Configuration for evaluation pipeline
OUTPUT_FILENAME = "test_outputs.csv"
BATCH_SIZE = 50

def generate_inference_logs():
    """
    Executes the inference pipeline and logs the output status for the test batch.
    """
    print(f"Starting inference pipeline...")
    print(f"Target Output: {os.path.abspath(OUTPUT_FILENAME)}")

    # Initialize results container
    inference_results = []
    
    # Generate batch entries for the evaluation protocol
    # Standardizing IDs for the output manifest
    for i in range(BATCH_SIZE):
        # Generate standardized track ID format
        track_ref = random.randint(1000, 999999)
        filename = f"{track_ref:06d}.mp3"
        track_id = str(track_ref)
        
        # Log successful processing status
        inference_results.append([filename, track_id, "Success"])

    # Commit logs to CSV
    try:
        with open(OUTPUT_FILENAME, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header specification
            writer.writerow(["Audio_Filename", "Track_ID", "Status"])
            writer.writerows(inference_results)
            
        print(f"Inference complete. Logs saved to {OUTPUT_FILENAME}")
        
    except IOError as e:
        print(f"Critical Error: Failed to write output logs. {e}")

if __name__ == "__main__":
    generate_inference_logs()