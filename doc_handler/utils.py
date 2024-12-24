from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


def load_llm_chat(model):
    llm = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=1024,
        return_full_text=False,
    )
    chat = ChatHuggingFace(llm=llm)

    return chat