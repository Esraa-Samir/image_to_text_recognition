# image_to_text_recognition
 
# Steps to create the model
- Activate virtaul env
- Run: pip install -r requirements.txt
- Run: python Data_spliter.py once to split data between Data folder and Testing folder (Make sure you create the folders before running the script and place the data into the Data folder)
- Model hyperparameters were chosen using GridsearchCV to decide best combinations 
- Run: python image_to_text_recognition.py to train and evaluate the model
