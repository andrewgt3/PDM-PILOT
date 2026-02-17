#!/usr/bin/env python3
"""
OPC UA Node Browser
Utility to explore OPC UA servers and find NodeIDs for configuration.
"""

import asyncio
import logging
import argparse
from asyncua import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("opcua_browser")

async def browse_recursive(node, depth=0, max_depth=3):
    """Recursively browse nodes."""
    if depth > max_depth:
        return

    try:
        children = await node.get_children()
        for child in children:
            name = await child.read_display_name()
            node_id = child.nodeid.to_string()
            node_class = await child.read_node_class()
            
            indent = "  " * depth
            print(f"{indent}- {name.Text} ({node_id}) [{node_class}]")
            
            # Recurse for Objects and Folders
            if str(node_class) in ['NodeClass.Object', 'NodeClass.Folder']: # Check class type carefully
                await browse_recursive(child, depth + 1, max_depth)
                
    except Exception as e:
        indent = "  " * depth
        print(f"{indent}  (Error browsing children: {e})")

async def main():
    parser = argparse.ArgumentParser(description="Browse OPC UA Server Nodes")
    parser.add_argument("--url", default="opc.tcp://localhost:4840", help="OPC UA Server URL")
    parser.add_argument("--depth", type=int, default=2, help="Browsing depth")
    args = parser.parse_args()

    print(f"Connecting to {args.url}...")
    
    try:
        async with Client(url=args.url) as client:
            print("Connected.")
            namespace_array = await client.get_namespace_array()
            print(f"Namespaces: {namespace_array}")
            
            # Start browsing from Objects folder
            objects = client.nodes.objects
            print("\nBrowsing Objects:")
            await browse_recursive(objects, max_depth=args.depth)
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
