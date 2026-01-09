# Imports
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def main():
    """RAG Query Agent for Health Documents"""

    # Embeddings for the query and documents
    embeddings = HuggingFaceEmbeddings(
        #model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual model
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load the persisted Chroma vector store
    persist_directory = "chroma_db"
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    # LLM for generating answers
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2,  # Slightly creative responses
    )

    # Prompt template for RAG: context + question
    prompt = ChatPromptTemplate.from_template(
        """Tu es un assistant santé.
Réponds uniquement à partir du contexte ci-dessous.
Si l'information n'est pas présente, dis-le clairement.

Contexte :
{context}

Question :
{question}
"""
    )

    # RAG chain
    rag_chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Chat santé (prototype). Tape 'exit' pour quitter.\n")

    # Question and answer loop
    while True:
        question = input("➡️ Ta question : ")
        if question.lower() in ["exit", "quit"]:
            break

        # Retrieve relevant document chunks
        chunks_and_scores = vectorstore.similarity_search_with_score(
            question,
            k=10
        )

        # Filter chunks with score threshold
        filtered_chunks_and_scores = [
            (chunk, score) for chunk, score in chunks_and_scores if score >= 0.4
        ]
        print(f"\nNombre de chunks utilisés : {len(filtered_chunks_and_scores)}")

        if not filtered_chunks_and_scores:
            print("❌ Désolé, aucune information pertinente trouvée dans les documents.")
            print("\n" + "-" * 50 + "\n")
            continue

        # Sort by score descending
        filtered_chunks_and_scores.sort(key=lambda x: x[1], reverse=True)
        chunks = [chunk for chunk, score in filtered_chunks_and_scores]

        # Build context text with chunks
        context_text = "Context from documents:\n\n"
        for i, chunk in enumerate(chunks, 1):
            context_text += f"[Document {i}]\n{chunk}\n\n"  # Replace ___ with chunk

        # Invoke RAG chain
        answer = rag_chain.invoke({
            "context": context_text,
            "question": question
        })

        print("\n🧠 Réponse :")
        print(answer)

        # Display sources used
        print("\n📚 Sources utilisées :")
        for i, doc in enumerate(filtered_chunks_and_scores):
            print(f"  - Source {i+1}: {doc[0].metadata.get('source', 'inconnu')}")
            print(f"    Score de similarité : {doc[1]:.3f}")
            print(f"    Contenu : {doc[0].page_content[:20]}...")  # Affiche un extrait

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
