import asyncio

from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from tradingagents.llm_adapters.codex_cli_adapter import ChatCodexCLI


@tool
def lookup_price(ticker: str) -> str:
    """查询股价"""
    return f"{ticker}=123.45"


def main() -> None:
    llm = ChatCodexCLI(
        model="gpt-5.4",
        reasoning_effort="medium",
        fast_mode=False,
        ask_for_approval="never",
        sandbox_mode="read-only",
        request_timeout=180,
    )

    messages = [HumanMessage(content="请先调用 lookup_price 查询 AAPL，然后再回答。")]

    bound = llm.bind_tools(
        [lookup_price],
        tool_choice={"type": "function", "function": {"name": "lookup_price"}},
        parallel_tool_calls=False,
    )

    invoke_msg = bound.invoke(messages)
    print("invoke:", invoke_msg.content, invoke_msg.tool_calls)

    stream_chunks = list(
        llm._stream(
            messages,
            tools=[lookup_price],
            tool_choice="lookup_price",
            parallel_tool_calls=False,
        )
    )
    stream_msg = generate_from_stream(iter(stream_chunks)).generations[0].message
    print("stream:", stream_msg.content, stream_msg.tool_calls)

    async def run_astream() -> None:
        chunks = [
            chunk async for chunk in llm._astream(
                messages,
                tools=[lookup_price],
                tool_choice="lookup_price",
                parallel_tool_calls=False,
            )
        ]
        msg = generate_from_stream(iter(chunks)).generations[0].message
        print("astream:", msg.content, msg.tool_calls)

    asyncio.run(run_astream())


if __name__ == "__main__":
    main()
