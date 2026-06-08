import os
import birdnet


def analyze_bird_audio(audio_path, confidence_threshold=0.10):
    # Check if the file exists locally
    if not os.path.exists(audio_path):
        print(f"Error: The file '{audio_path}' could not be found.")
        return

    print(f"Analyzing audio file: {audio_path}...")
    print("-" * 50)

    try:
        # Load the default acoustic model (Version 2.4 using TensorFlow backend)
        model = birdnet.load("acoustic", "2.4", "tf")
        
        # Predict the species present in the file
        predictions = model.predict(audio_path)

        # 2. Convert the custom result object to a standard pandas DataFrame
        predictions_df = predictions.to_dataframe()
        
        # Filter predictions based on the confidence threshold and sort them
        # (predictions dataframe contains 'species_name' and 'confidence')
        filtered_preds = predictions_df[predictions_df['confidence'] >= confidence_threshold]
        sorted_preds = filtered_preds.sort_values(by='confidence', ascending=False)

        if sorted_preds.empty:
            print("No bird species identified above the confidence threshold.")
            return

        # Print the structured results directly to the terminal
        print(f"{'Bird Species (Scientific & Common Name)':<50} | {'Confidence Score'}")
        print("-" * 75)
        for _, row in sorted_preds.iterrows():
            # Format species name strings cleanly
            species = row['species_name'].replace('_', ' - ')
            confidence = f"{row['confidence'] * 100:.2f}%"
            print(f"{species:<50} | {confidence}")
            
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    # Target audio file path
    TARGET_AUDIO =input("Enter the path to the audio file for analysis: ")
    
    # Run the analysis
    analyze_bird_audio(TARGET_AUDIO, confidence_threshold=0.15)