import argparse
import json

from mcp_integration.client import MCPClientAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal local MCP client demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List tools exposed by the MCP server")
    call_parser = subparsers.add_parser("call", help="Call one MCP tool")
    call_parser.add_argument("tool_name")
    call_parser.add_argument("arguments", help="JSON object containing tool arguments")
    args = parser.parse_args()

    client = MCPClientAdapter()
    if args.command == "list":
        print(json.dumps([tool.__dict__ for tool in client.list_tools()], indent=2))
        return

    arguments = json.loads(args.arguments)
    if not isinstance(arguments, dict):
        parser.error("arguments must decode to a JSON object")
    print(json.dumps(client.call_tool(args.tool_name, arguments), indent=2))


if __name__ == "__main__":
    main()
