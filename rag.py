from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from typing import List

PROMPT = PromptTemplate.from_template(
    """You are an assistant. Based on the provided conversation content, answer the question. Keep your answer concise and cite the message ID.

    context:
    {context}

    Question: {question}
    Answer:"""
)

_llm = ChatOpenAI(model_name="gpt-4o-mini")


def _format_docs(docs: List):
    return "\n\n".join(f"<id={d.metadata['orig_id']}> {d.page_content}" for d in docs)

def get_chain(db: Chroma):
    retriever = db.as_retriever(search_kwargs={"k": 6})
    chain = (
        {
            "context": RunnableLambda(lambda q: retriever.get_relevant_documents(q)) | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | _llm
    )
    return chain

# Convenience wrapper

def answer(db: Chroma, question: str):
    chain = get_chain(db)
    return chain.invoke(question)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--db", default=".chroma")
    args = parser.parse_args()
    db = Chroma(persist_directory=args.db, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    print(answer(db, args.question))
