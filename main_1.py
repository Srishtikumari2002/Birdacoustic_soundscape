import birdnet

# model = birdnet.load("acoustic", "2.4", "tf")

audio_path = input("Enter the path to the audio file for analysis: ")
confidence_threshold = 0.10
# # predictions = model.predict(
# #   path
# # )

# # print(type(predictions))

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