FROM python:3.9-slim

WORKDIR /webapp

COPY . . 
  
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

CMD streamlit run ./src/webapp.py