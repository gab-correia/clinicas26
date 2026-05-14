import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Defina a variável de ambiente OPENAI_API_KEY no .env ou no ambiente.")
        return

    client = OpenAI(api_key=api_key)

    resposta = client.responses.create(
        model="gpt-4.1-mini",
        input="Responda em uma frase: API funcionando?",
    )

    print(resposta.output_text)


if __name__ == "__main__":
    main()
