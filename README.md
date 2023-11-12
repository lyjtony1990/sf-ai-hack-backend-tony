## MagicReply
Free yourself from replying repetitive emails. MagicReply learns from your past emails and your documents to help you automate email responses.

## Inspiration
We are tired of replying emails. A lot of my knowledge exists somewhere, I shouldn't have to write emails from scratch every time.

## What it does
It ingests knowledge from my past emails and documents, and synthesize a email response to answer incoming emails.

## How we built it
RAG + Chrome Extension built on Javascript

## Challenges we ran into
RAG latency, injecting responses to front-end

## Accomplishments
Built something end to end

## What we learned
Building a gen-AI App


### This repo is the back end code for our bot-extension project, which you can find [here](https://github.com/tarunmugunthan/bot-extension)

## Running the app locally
1. Install virtualenv and activate it
    > `apt install python3-virtualenv -y`
    > 
    > `virtualenv venv`
    >
    > `source venv/bin/activate`

2. Install the requirements
    > `pip install -r requirements.txt`

3. Create a file `config.yaml` with the OpenAPI keys by copying `config.yaml.sample`.

4. Run the service
    > `uvicorn app:app --host 0.0.0.0 --port 80`