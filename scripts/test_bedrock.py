import json

import boto3

# Fill these with your AWS Bedrock credentials.
AWS_ACCESS_KEY_ID = "ASIA2GORDRDSQG7SLWQH"
AWS_SECRET_ACCESS_KEY = "NhmQEHqwPhOikfE/NigGjuM6s89mphZBAburM8Wd"
AWS_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEGYaCXVzLXdlc3QtMiJHMEUCIHP2VGg7DHU78pRL9ewFxURNWON2+Z/izJI6MimOsabPAiEAz7b+BLY/ViXQIAhpH7SSL3CVOub58iAL+/RuJTQondIqlAIIHxAAGgw3MDEwNTUwNzY1ODEiDJGJ90jQ9+MORby0jyrxAWOBkN1m3R0JkndxOn4UyS56BU5Us7p6zHSHEbKhJg5BZKa7av1XAZsXxaXIBnkfOMQKhw/5d7xNHoNZFAgBiqd1yre9Q0KIx7oS7+QSkmpylVbNydtxIQZRTffz2C5CIuW/e/V6PTH0vF+dt6F0TCXZrRF3X/8L/4JiiCptgMIcbKcWePdWuFe7zciGjkPzp4yvTb8bnMRe4aziRDkyRfuApO1H+z6yoHTOEnyXC4JpOVujM0i78SQGX8PntnnOGRtW7qM/77KQYhwKmTtN1b8UcpRPYYCxBwA7I7J01fNOIjwZdIpr4m593GVl22aWuWwwpPnfxwY6nQFWfrygMiWZvXl78oKbfA+1exDBkTdKq6l9zpFJXLLD9eCAR0AFgBQzURUh11s7oWeoBwwg+y9R1zc4BBqpCuOHd4nzKiHKDR9ivUXN7MjUrfsSqxf70Da3OjQWZUryO5JFhjjMRpYvIWD+CVaiNhqbnsMzyM0JfELkrHDssZ/do6LNS+FQTmGxIVKJ9s+8RMu1YPJCfYYMPpYUIp/J"  # leave empty if not using temporary credentials

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
