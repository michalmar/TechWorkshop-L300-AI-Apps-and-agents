from azure.identity import DefaultAzureCredential
from azure.ai.evaluation.red_team import RedTeam, AttackStrategy
from pyrit.prompt_target import OpenAIChatTarget
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

azure_ai_project = os.getenv("FOUNDRY_ENDPOINT")
if not azure_ai_project:
    raise ValueError("FOUNDRY_ENDPOINT is required")

red_team_agent = RedTeam(
    azure_ai_project=azure_ai_project,
    credential=DefaultAzureCredential(),
    custom_attack_seed_prompts="data/custom_attack_prompts.json",
)

endpoint = (os.getenv("gpt_endpoint") or "").rstrip("/")
deployment = os.getenv("gpt_deployment")
api_key = os.getenv("gpt_api_key")
api_version = os.getenv("gpt_api_version")

if not endpoint:
    raise ValueError("gpt_endpoint is required")
if not deployment:
    raise ValueError("gpt_deployment is required")
if not api_key:
    raise ValueError("gpt_api_key is required")
if not api_version:
    raise ValueError("gpt_api_version is required")

chat_target = OpenAIChatTarget(
    model_name=deployment,
    endpoint=f"{endpoint}/openai/deployments/{deployment}/chat/completions",
    api_key=api_key,
    api_version=api_version,
)


async def main():
    result = await red_team_agent.scan(
        target=chat_target,
        scan_name="Red Team Scan - Easy-Moderate Strategies",
        attack_strategies=[
            AttackStrategy.Flip,
            AttackStrategy.ROT13,
            AttackStrategy.Base64,
            AttackStrategy.AnsiAttack,
            AttackStrategy.Tense,
        ],
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
