# 4 - Streamlit Application

1. This repository contains the code for a streamlit applicaion, communicating with LangGraph RAG Backend, and containerize the frontend application.
2. Before using it, update your LANGGRAPH_API_BASE environment variable in the format: `http://35.224.251.70:80`.
3. Make sure your Google Cloud account has the Artifact Registry Admin and Storage Admin roles. Also, enable the `Artifact Registry API`.
4. To run this code on GKE, upload the entire folder (streamlit) to your Google Cloud CLI environment.
5. In your Google Cloud CLI, navigate to the streamlit/ directory.
6. Authenticate your Google Cloud CLI account by running `gcloud auth login` and follow the prompts.
7. Authenticate Docker with Artifact registry by running `gcloud auth configure-docker` and follow the prompts.
8. In the Google Cloud CLI, build and push your Docker image using the following command (and note the ending period) `gcloud builds submit --tag gcr.io/yourGoogleProjectID/streamlit-app:1.0 .`
9. From the Google Cloud UI, navigate to the Artifact registry and locate your image (with the exact tag/version).
10. Click `Deploy to GKE` to deploy your image to GKE. 
11. Use a proper deployment name that doesn't include spaces or special characters.
12. Use the default namespace for the deployment.
13. Select User-managed nodes for the deployment.
14. In the "Container Details" section, add an environment variable named `LANGGRAPH_API_BASE` with your LangGraph Service IP and port (e.g., `http://38.222.251.80:80`)
15. In the Expose section, configure the port mapping to map port `80` to target port `8080`.
16. If deployed successfully, the deployment should display a green checkmark. 
17. In your web browser, navigate to your streamlit app service IP and Port. 
18. If your LangGraph Service IP address changes, delete the Streamlit app service first, then the Streamlit app deployment, and redeploy the application starting from Step 9.