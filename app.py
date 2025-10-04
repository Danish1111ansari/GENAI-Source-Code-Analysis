from langchain.vectorstores import Chroma
from src.helper import create_vector_store, get_embeddings,get_better_llm,create_conversation_chain,clone_repo, load_documents, split_documents
import os
from flask import Flask, render_template, jsonify, request



chainMemoryQa = None

app = Flask(__name__)





embeddings = get_embeddings()
persist_directory = "./chroma_db"
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


llm = get_better_llm()



@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():
    global chainMemoryQa  # ADD THIS LINE

    if request.method == 'POST':
        user_input = request.form['question']
        repo_path = clone_repo(user_input, persist_directory)
        documents = load_documents(repo_path)
        text_chunks = split_documents(documents)
        chainMemoryQa = create_conversation_chain(create_vector_store(text_chunks, embeddings), llm)

    return jsonify({"response": str(user_input) })






@app.route("/get", methods=["GET", "POST"])
def chat():
    if chainMemoryQa is None:
        return "please process a repository first"
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf ./chroma_db")

    result = chainMemoryQa({"question":input})
    print(result['answer'])
    return str(result["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)


