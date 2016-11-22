## Face Recognition

### Python Installation
1. Download Python 3.5.2 https://www.python.org/downloads
2. Put to your bash profile `export PATH="/Library/Frameworks/Python.framework/Versions/3.5/bin:${PATH}"`
 so that virtualenv is connected to Python 3.5.2
3. Install pip3.5 if is not installed already
4. Then, install virtualenv for Python 3.5: `pip3.5 install virtualenv`
5. cd into face_recognition
6. run command: `virtualenv venv` (this should be done only once)
7. run command: `source venv/bin/activate`
8. run command: `pip3.5 install -r requirements.txt` (in order to install dependencies in the current virtualenv)
