![alt text](https://github.com/yusup-yusupov/daemon-dialoguers/blob/master/pictures/Logo.png?raw=true)
## Daemon Dialoguers: Queying log files with LLMs

Provides suits to converse with the log file from any system. The system can provide log embedding, editable summary and QnA chatbot for any .txt logfiles. <br>

Features to be added:
- [x] Create the 2 main backend functionality
- [x] Create a Streamlit UI
- [ ] Create a multi-platform UI with Flutter and Dart (in progress)
- [ ] Migrate files to Firebase
- [ ] Migrate hosting to Heroku
      
#### How to run:
- Install the requirements with 
```
pip install -r requirements.txt
```
- Move to the interface folder and enter
```
streamlit run interface.py
```

### Features:

#### Summarization:
Converts a log file into a ~500 word summary including general information along with potential warnings, errors and unusual behaviour.  The summary can be eddited using natural language commands.
![alt text](https://github.com/yusup-yusupov/daemon-dialoguers/blob/master/pictures/Summ_flow.png?raw=true)

#### Log File QnA:
Creates a Chat bot for you that can answer questions and provide suggestions based on input logfile. The chatbot can retrain memory to the continue the conversation in the future.
![alt text](https://github.com/yusup-yusupov/daemon-dialoguers/blob/master/pictures/qna_flow.png?raw=true)


