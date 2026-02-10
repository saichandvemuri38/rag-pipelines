python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn
python -m uvicorn app.main:app --reload
conda deactivate # to run packages in venv
deactivate
# Create new venv
python3.13 -m venv venv_3.13
source venv_3.13/bin/activate

# Reinstall dependencies
pip install -r requirements.txt


pyenv install 3.13.7
pyenv virtualenv 3.13.7 rag-pipelines
pyenv local rag-pipelines

rm -rf venv
