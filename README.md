
# About project

This is my first attempt at API implementation. The initial learning objective is the Titanic competition. Its goal is to predict whether a person survived, based on the input data.

So key of API implementation is to predict survivability depending on single-row input data. This is the simplest application of the API, created for familiarization with API basics.

# Requirements

See "requirements.txt"

# Run application 

The following code is automatically responsible for opening and executing the program:
```python
if __name__ == "__main__":
    uvicorn.run("api:app", reload=True)
```
In other cases, after running the main code, you should go to this ref: 
* http://127.0.0.1:8000;  
* http://127.0.0.1:800/docs (for integrated documentation of FastAPI).

