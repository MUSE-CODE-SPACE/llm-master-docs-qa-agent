.PHONY: install index serve eval test
install:; pip install -r requirements.txt
index:; python -m qa.indexer
serve:; uvicorn qa.api:app --reload
eval:; python -m qa.eval
test:; pytest -q
