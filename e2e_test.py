import subprocess
import sys
import os

# --- Configuration ---
# On Windows using Git Bash, environment variable modification is needed.
# This script handles it automatically.
IS_WINDOWS_GITBASH = sys.platform == "win32" and "MSYSTEM" in os.environ

# Paths inside the Docker container
# Using the sample files mentioned in the README
INGEST_FILE_PATH = "/app/data/sample.txt"
EVAL_CSV_PATH = "/app/examples/eval_dataset_example.csv"
EVAL_OUTPUT_PATH = "/app/output/e2e_test_report.txt"

# --- Helper Function ---
def run_command(command):
    """Runs a command within the docker container and handles potential errors."""
    print(f"🚀 Executing: '''{' '.join(command)}'''")
    
    # Create a copy of the current environment variables
    env = os.environ.copy()
    
    # As per README, this is required for Git Bash on Windows to correctly interpret paths
    if IS_WINDOWS_GITBASH:
        env["MSYS_NO_PATHCONV"] = "1"
        print("   (Applying MSYS_NO_PATHCONV=1 for Windows/Git Bash)")

    try:
        # We build the full command to be executed by the shell
        full_command = command
        
        # For Windows Git Bash, we need to prepend the env var setting
        if IS_WINDOWS_GITBASH:
             # This is a bit of a workaround to make sure the env var is set for the docker command
             # In a real shell, you'd do `export MSYS_NO_PATHCONV=1 && docker-compose ...`
             # Here, we pass the modified environment directly to subprocess.run
             pass

        result = subprocess.run(
            full_command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',  # Force UTF-8 encoding for Windows
            errors='replace',   # Replace invalid characters instead of crashing
            env=env, # Pass the modified environment
            shell=False # It's safer to run without shell=True
        )
        print("✅ Success!")
        print("--- STDOUT ---")
        print(result.stdout)
        if result.stderr:
            print("--- STDERR ---")
            print(result.stderr)
    except FileNotFoundError:
        print("❌ Error: 'docker-compose' command not found.")
        print("   Please ensure Docker Desktop is running and 'docker-compose' is in your system's PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing command: '''{' '.join(e.cmd)}'''")
        print(f"Return Code: {e.returncode}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        print("   Hint: Make sure your Docker containers are running with 'docker-compose up -d'")
        sys.exit(1)
    print("-" * 50)


# --- Main Test Steps ---
def main():
    """Main function to run the E2E test steps."""
    print("🏁 Starting End-to-End Test for TAT-RAG 🏁")

    # Step 1: Ingest documents (using default LangChain parser)
    print("\n--- Step 1: Document Ingestion (LangChain Parser) ---")
    ingest_command = [
        "docker-compose", "exec", "rag-app",
        "python", "main.py", "ingest", INGEST_FILE_PATH
    ]
    run_command(ingest_command)

    # Step 1b: Test Fitz parser (if PDF files exist)
    # Note: This is optional and will be skipped if no PDF files are available
    print("\n--- Step 1b: Testing Fitz Parser (Optional) ---")
    print("   Checking for PDF files to test Fitz parser...")
    check_pdf_command = [
        "docker-compose", "exec", "rag-app",
        "sh", "-c", "ls /app/data/*.pdf 2>/dev/null | head -1"
    ]
    try:
        result = subprocess.run(
            check_pdf_command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=os.environ.copy()
        )
        if result.stdout.strip():
            pdf_file = result.stdout.strip()
            print(f"   Found PDF: {pdf_file}")
            print("   Testing Fitz parser on this PDF...")
            fitz_ingest_command = [
                "docker-compose", "exec", "rag-app",
                "python", "main.py", "ingest", pdf_file,
                "--file-type", "pdf",
                "--parser", "fitz"
            ]
            run_command(fitz_ingest_command)
        else:
            print("   No PDF files found in /app/data/. Skipping Fitz parser test.")
    except Exception as e:
        print(f"   Could not check for PDFs: {e}")
        print("   Skipping Fitz parser test.")

    # Step 2: Run evaluation from CSV
    print("\n--- Step 2: RAGAS Evaluation ---")
    evaluate_command = [
        "docker-compose", "exec", "rag-app",
        "python", "main.py", "evaluate",
        "--csv-path", EVAL_CSV_PATH,
        "--output", EVAL_OUTPUT_PATH
    ]
    run_command(evaluate_command)

    print(f"\n🎉 End-to-End Test Complete! 🎉")
    print(f"📊 Evaluation report saved to the 'output' directory inside the container.")
    print(f"   You can find it locally in the './output' folder on your host machine.")
    print(f"\n💡 Tip: To test Fitz parser specifically, place a PDF in ./data/ and run again.")

if __name__ == "__main__":
    main()
