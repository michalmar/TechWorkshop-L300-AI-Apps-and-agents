import logging
import os
import json
from collections.abc import AsyncIterable
from typing import Any

import openai
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


def _get_azure_openai_client() -> openai.AsyncAzureOpenAI:
    endpoint = os.getenv('gpt_endpoint')
    deployment_name = os.getenv('gpt_deployment')
    api_version = os.getenv('gpt_api_version')
    api_key = os.getenv('gpt_api_key')

    if not endpoint:
        raise ValueError('gpt_endpoint is required')
    if not deployment_name:
        raise ValueError('gpt_deployment is required')
    if not api_version:
        raise ValueError('gpt_api_version is required')

    if not api_key:
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential,
            'https://cognitiveservices.azure.com/.default',
        )
        return openai.AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )

    return openai.AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def get_products(question: str) -> list[dict[str, Any]]:
    product_dict = [
        {
            'id': '1',
            'name': 'Eco-Friendly Paint Roller',
            'type': 'Paint Roller',
            'description': 'A high-quality, eco-friendly paint roller for smooth finishes.',
            'punchLine': 'Roll with the best, paint with the rest!',
            'price': 15.99,
        },
        {
            'id': '2',
            'name': 'Premium Paint Brush Set',
            'type': 'Paint Brush',
            'description': 'A set of premium paint brushes for detailed work and fine finishes.',
            'punchLine': 'Brush up your skills with our premium set!',
            'price': 25.49,
        },
        {
            'id': '3',
            'name': 'All-Purpose Paint Tray',
            'type': 'Paint Tray',
            'description': 'A durable paint tray suitable for all types of rollers and brushes.',
            'punchLine': 'Tray it, paint it, love it!',
            'price': 9.99,
        },
    ]
    _ = question
    return product_dict


class AgentFrameworkProductManagementAgent:
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    class _DelegatedAgent:
        def __init__(self, name: str, instructions: str, runner):
            self.name = name
            self.instructions = instructions
            self._runner = runner

        def as_tool(self) -> dict[str, Any]:
            return {
                'type': 'function',
                'function': {
                    'name': self.name,
                    'description': f'Delegate the request to {self.name}.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'user_input': {
                                'type': 'string',
                                'description': 'User request to be handled by this agent.',
                            }
                        },
                        'required': ['user_input'],
                        'additionalProperties': False,
                    },
                },
            }

        async def run(self, user_input: str, session_id: str) -> str:
            return await self._runner(user_input, session_id)

    def __init__(self):
        self.client = _get_azure_openai_client()
        self.deployment = os.getenv('gpt_deployment')
        self.sessions: dict[str, list[dict[str, str]]] = {}

        self.manager_instructions = (
            "You are the ProductManagerAgent for Zava. Decide whether to use ProductAgent, MarketingAgent, or RankerAgent behavior. "
            "Always return concise, helpful answers."
        )

        self.product_instructions = (
            "You are ProductAgent. Use only the provided product catalog and do not invent products."
        )

        self.marketing_instructions = (
            "You are MarketingAgent. Improve product messaging and sales copy while remaining truthful to provided catalog data."
        )

        self.ranker_instructions = (
            "You are RankerAgent. Recommend and rank products from the provided catalog based on the user need."
        )

        self.product_agent = self._DelegatedAgent(
            name='ProductAgent',
            instructions=self.product_instructions,
            runner=self._run_product_agent,
        )
        self.marketing_agent = self._DelegatedAgent(
            name='MarketingAgent',
            instructions=self.marketing_instructions,
            runner=self._run_marketing_agent,
        )
        self.ranker_agent = self._DelegatedAgent(
            name='RankerAgent',
            instructions=self.ranker_instructions,
            runner=self._run_ranker_agent,
        )

    def _agent_tool_for_products(self) -> list[dict[str, Any]]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'get_products',
                    'description': 'Retrieves a set of products based on a natural language user query.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'question': {
                                'type': 'string',
                                'description': 'Natural language query, for example: What kinds of paint rollers do you have in stock?',
                            }
                        },
                        'required': ['question'],
                        'additionalProperties': False,
                    },
                },
            }
        ]

    async def _chat(self, system_prompt: str, user_input: str, session_id: str) -> tuple[str, list[dict[str, str]]]:
        history = self.sessions.get(session_id, [])
        messages: list[dict[str, str]] = [{'role': 'system', 'content': system_prompt}] + history[-8:] + [
            {'role': 'user', 'content': user_input}
        ]

        completion = await self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_completion_tokens=700,
            reasoning_effort='low',
        )
        text = completion.choices[0].message.content or ''
        new_history = history + [
            {'role': 'user', 'content': user_input},
            {'role': 'assistant', 'content': text},
        ]
        return text, new_history

    async def _product_chat_with_tool(self, user_input: str, session_id: str) -> tuple[str, list[dict[str, str]]]:
        history = self.sessions.get(session_id, [])
        system_prompt = (
            f"{self.manager_instructions}\n"
            f"{self.product_instructions}\n"
            "If the user asks about product availability, types, pricing, descriptions, or recommendations, "
            "you must call get_products and then answer using only that catalog data."
        )

        base_messages: list[dict[str, Any]] = [
            {'role': 'system', 'content': system_prompt},
            *history[-8:],
            {'role': 'user', 'content': user_input},
        ]

        first = await self.client.chat.completions.create(
            model=self.deployment,
            messages=base_messages,
            tools=self._agent_tool_for_products(),
            tool_choice='auto',
            max_completion_tokens=700,
            reasoning_effort='low',
        )

        first_message = first.choices[0].message
        tool_calls = first_message.tool_calls or []

        if not tool_calls:
            text = first_message.content or ''
            return text, history + [
                {'role': 'user', 'content': user_input},
                {'role': 'assistant', 'content': text},
            ]

        tool_messages: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if tool_call.function.name != 'get_products':
                continue
            try:
                arguments = json.loads(tool_call.function.arguments or '{}')
            except json.JSONDecodeError:
                arguments = {'question': user_input}

            question = arguments.get('question') or user_input
            products = get_products(question)
            tool_messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': json.dumps(products),
                }
            )

        follow_up_messages: list[dict[str, Any]] = base_messages + [
            {
                'role': 'assistant',
                'content': first_message.content,
                'tool_calls': [tc.model_dump() for tc in tool_calls],
            },
            *tool_messages,
        ]

        second = await self.client.chat.completions.create(
            model=self.deployment,
            messages=follow_up_messages,
            max_completion_tokens=700,
            reasoning_effort='low',
        )
        final_text = second.choices[0].message.content or ''

        return final_text, history + [
            {'role': 'user', 'content': user_input},
            {'role': 'assistant', 'content': final_text},
        ]

    async def _run_product_agent(self, user_input: str, session_id: str) -> str:
        logger.info('Delegating to ProductAgent', extra={'session_id': session_id})
        result_text, _ = await self._product_chat_with_tool(user_input, session_id)
        return result_text

    async def _run_marketing_agent(self, user_input: str, session_id: str) -> str:
        logger.info('Delegating to MarketingAgent', extra={'session_id': session_id})
        catalog_text = f"Catalog: {get_products(user_input)}"
        system_prompt = (
            f"{self.manager_instructions}\n"
            f"{self.marketing_instructions}\n"
            f"{catalog_text}"
        )
        result_text, _ = await self._chat(system_prompt, user_input, session_id)
        return result_text

    async def _run_ranker_agent(self, user_input: str, session_id: str) -> str:
        logger.info('Delegating to RankerAgent', extra={'session_id': session_id})
        catalog_text = f"Catalog: {get_products(user_input)}"
        system_prompt = (
            f"{self.manager_instructions}\n"
            f"{self.ranker_instructions}\n"
            f"{catalog_text}"
        )
        result_text, _ = await self._chat(system_prompt, user_input, session_id)
        return result_text

    async def invoke(self, user_input: str, session_id: str) -> dict[str, Any]:
        try:
            product_agent = self.product_agent
            marketing_agent = self.marketing_agent
            ranker_agent = self.ranker_agent

            tool_runners = {
                product_agent.name: product_agent,
                marketing_agent.name: marketing_agent,
                ranker_agent.name: ranker_agent,
            }

            history = self.sessions.get(session_id, [])
            manager_prompt = (
                "Your role is to carefully analyze the user's request and delegate to the right specialized tool. "
                "Use ProductAgent for product lookup/catalog questions, MarketingAgent for improving descriptions and promotional copy, "
                "and RankerAgent for recommendation/ranking tasks."
            )
            messages: list[dict[str, Any]] = [
                {'role': 'system', 'content': manager_prompt},
                *history[-8:],
                {'role': 'user', 'content': user_input},
            ]

            tools = [
                product_agent.as_tool(),
                marketing_agent.as_tool(),
                ranker_agent.as_tool(),
            ]

            manager_reply = await self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                tools=tools,
                tool_choice='auto',
                max_completion_tokens=300,
                reasoning_effort='low',
            )

            manager_message = manager_reply.choices[0].message
            tool_calls = manager_message.tool_calls or []

            if tool_calls:
                selected_tools = [tool_call.function.name for tool_call in tool_calls]
                logger.info(
                    'ProductManagerAgent selected delegated tools',
                    extra={'session_id': session_id, 'tools': selected_tools},
                )
            else:
                logger.info(
                    'ProductManagerAgent responded directly without tool delegation',
                    extra={'session_id': session_id},
                )

            if tool_calls:
                parts: list[str] = []
                for tool_call in tool_calls:
                    agent_tool = tool_runners.get(tool_call.function.name)
                    if not agent_tool:
                        continue
                    try:
                        arguments = json.loads(tool_call.function.arguments or '{}')
                    except json.JSONDecodeError:
                        arguments = {'user_input': user_input}
                    delegated_input = arguments.get('user_input') or user_input
                    delegated_result = await agent_tool.run(delegated_input, session_id)
                    if delegated_result:
                        parts.append(delegated_result)
                result_text = '\n\n'.join(parts).strip()
            else:
                result_text = (manager_message.content or '').strip()

            if not result_text:
                result_text = await self._run_product_agent(user_input, session_id)

            self.sessions[session_id] = history + [
                {'role': 'user', 'content': user_input},
                {'role': 'assistant', 'content': result_text},
            ]

            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': result_text,
            }
        except Exception as e:
            logger.error('Product manager invoke failed: %s', e)
            return {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f'Error: {e}',
            }

    async def stream(self, user_input: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
        yield await self.invoke(user_input, session_id)
