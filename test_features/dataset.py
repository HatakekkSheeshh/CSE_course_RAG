from huggingface_hub import login, upload_folder, HfApi

# (optional) Login with your Hugging Face credentials
login()

# Push your dataset files
# upload_folder(folder_path="C:/project/CSE_course_RAG/data", repo_id="hatakekksheeshh/CSE_course_RAG", repo_type="dataset")
api = HfApi()
api.upload_large_folder(
    folder_path="C:/project/CSE_course_RAG/data",
    repo_id="hatakekksheeshh/CSE_course_RAG",
    repo_type="dataset",
)