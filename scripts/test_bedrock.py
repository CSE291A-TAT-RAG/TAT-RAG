import json

import boto3

# Fill these with your AWS Bedrock credentials.
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_SESSION_TOKEN = ""  # leave empty if not using temporary credentials

REGION = "us-west-2"
MODEL_ID = "amazon.titan-text-express-v1"


def main() -> None:
    client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN or None,
    )

    payload = {
        "inputText": (
            "You are a diagnostic health check. "
            "Reply with only the lowercase word 'pong' and nothing else."
        ),
        "textGenerationConfig": {
            "maxTokenCount": 200,
            "temperature": 0,
            "topP": 0.1,
        },
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
    )
    result = json.loads(response["body"].read())
    output_text = result.get("results", [{}])[0].get("outputText", "")
    print(output_text.strip() or result)


if __name__ == "__main__":
    main()
