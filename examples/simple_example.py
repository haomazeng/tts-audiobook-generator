#!/usr/bin/env python3
"""
Simple example of using TTS Audiobook Generator programmatically
"""

import asyncio
from src.config import Config
from src.converter import AudioConverter

async def main():
    # Load config
    config = Config.load("config.yaml")

    # Create converter
    converter = AudioConverter(config, work_dir="./output")

    # Convert a file
    result = await converter.convert_file("document.pdf")
    print(f"Audio saved to: {result}")

if __name__ == "__main__":
    asyncio.run(main())
